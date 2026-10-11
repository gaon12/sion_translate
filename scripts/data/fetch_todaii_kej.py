#!/usr/bin/env python3
"""Build an authorized Korean-English-Japanese Todaii sentence corpus.

The collector uses only Todaii's unauthenticated public endpoints.  It fetches
the current article catalogue, then joins each Japanese sentence with the
Korean locale translation and the API's English translation from the same
sentence object.  Collection is resumable at an article-batch boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://api2.easyjapanese.net/api"
SOURCE = "Todaii Japanese"
SOURCE_URL = "https://japanese.todaiinews.com/"
NEWS_SITEMAP_URL = "https://japanese.todaiinews.com/sitemap/news/sitemap.xml"
USER_AGENT = "sion-translate-authorized-corpus-refresh/1.0"
_SPACE_RE = re.compile(r"\s+")
_SITEMAP_NEWS_RE = re.compile(
    r"<loc>https://japanese\.todaiinews\.com/en/news/([0-9a-f]{32})</loc>"
)
_TRAINING_MARKUP_RE = re.compile(r"<(?:ruby|rb|rt|rp|span)\b", flags=re.IGNORECASE)
_HANGUL_RE = re.compile(r"[가-힣]")
_JAPANESE_RE = re.compile(r"[ぁ-ゖァ-ヺ一-龯]")
_LATIN_RE = re.compile(r"[A-Za-z]")
NEWS_TOPICS = (
    "tomo_diary",
    "new_features",
    "travel",
    "culture",
    "society_life",
    "short-stories",
    "health",
    "education-learning",
    "cuisine",
    "lifestyle",
    "economy_business",
    "science-technology",
    "sports",
    "japan_journey",
    "unique_japan",
    "100_japanese_culture_keywords",
    "perspectives_and_analysis",
    "delicious_japanese_food",
    "do_you_know",
    "japan_check_in",
    "trends",
    "animals_plants",
    "natural_geography",
)


class _JapaneseHTMLParser(HTMLParser):
    """Keep visible base text while discarding ruby pronunciation markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"rt", "rp"}:
            self.ignored_depth += 1
        elif tag == "br" and self.ignored_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"rt", "rp"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.ignored_depth == 0:
            self.parts.append(data)


def normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def japanese_text(value: str) -> str:
    parser = _JapaneseHTMLParser()
    parser.feed(value)
    parser.close()
    return normalize_text("".join(parser.parts))


@dataclass(frozen=True)
class Article:
    article_id: str
    date: str
    topic: str
    level: int | None

    @classmethod
    def from_api(cls, value: dict[str, Any]) -> Article:
        article_id = value.get("id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError("catalogue entry is missing an article id")
        details = value.get("value")
        details = details if isinstance(details, dict) else {}
        date = value.get("time")
        topic = details.get("topic")
        level = details.get("level")
        return cls(
            article_id=article_id,
            date=date if isinstance(date, str) else "",
            topic=topic if isinstance(topic, str) else "",
            level=level if isinstance(level, int) else None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "date": self.date,
            "topic": self.topic,
            "level": self.level,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Article:
        return cls(
            article_id=str(value["article_id"]),
            date=str(value.get("date", "")),
            topic=str(value.get("topic", "")),
            level=value.get("level") if isinstance(value.get("level"), int) else None,
        )


class RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_request - now
            if delay > 0:
                time.sleep(delay)
            self.next_request = max(now, self.next_request) + self.interval


class JsonClient:
    def __init__(
        self,
        *,
        timeout: float,
        retries: int,
        request_interval: float,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.rate_limiter = RateLimiter(request_interval)

    def get_bytes(self, url: str) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                self.rate_limiter.wait()
                request = Request(
                    url,
                    headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                )
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt >= self.retries:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.0
            except (TimeoutError, URLError, json.JSONDecodeError):
                if attempt >= self.retries:
                    raise
                delay = 0.0
            time.sleep(max(delay, min(30.0, 0.75 * (2**attempt))) + random.random() * 0.25)
        raise AssertionError("retry loop exited unexpectedly")

    def get(self, url: str) -> dict[str, Any]:
        payload = json.loads(self.get_bytes(url))
        if not isinstance(payload, dict):
            raise ValueError(f"API returned a non-object payload: {url}")
        return payload

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode("utf-8")


def fetch_catalogue(
    fetch_json: Callable[[str], dict[str, Any]],
    *,
    page_size: int,
    max_articles: int | None = None,
    filters: dict[str, str] | None = None,
) -> list[Article]:
    articles: list[Article] = []
    seen: set[str] = set()
    page = 1
    while True:
        parameters = {"page": page, "limit": page_size, "is_web": "true"}
        parameters.update(filters or {})
        query = urlencode(parameters)
        payload = fetch_json(f"{API_ROOT}/v2/news?{query}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"catalogue page {page} has no data object")
        raw_entries = data.get("data")
        if not isinstance(raw_entries, list):
            raise ValueError(f"catalogue page {page} has no article list")
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            article = Article.from_api(raw_entry)
            if article.article_id in seen:
                continue
            seen.add(article.article_id)
            articles.append(article)
            if max_articles is not None and len(articles) >= max_articles:
                return articles

        pagination = data.get("pagination")
        pagination = pagination if isinstance(pagination, dict) else {}
        last_page = pagination.get("last_page")
        total = pagination.get("total")
        if not raw_entries:
            break
        if isinstance(total, int) and len(articles) >= total:
            break
        if isinstance(last_page, int) and page >= last_page:
            break
        page += 1
    return articles


def catalogue_total(payload: dict[str, Any]) -> int:
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    pagination = data.get("pagination")
    pagination = pagination if isinstance(pagination, dict) else {}
    total = pagination.get("total")
    if not isinstance(total, int):
        raise ValueError("catalogue response has no integer total")
    return total


def parse_news_sitemap(value: str) -> list[Article]:
    seen: set[str] = set()
    articles: list[Article] = []
    for match in _SITEMAP_NEWS_RE.finditer(value):
        article_id = match.group(1)
        if article_id in seen:
            continue
        seen.add(article_id)
        articles.append(Article(article_id=article_id, date="", topic="", level=None))
    if not articles:
        raise ValueError("news sitemap contains no article ids")
    return articles


def fetch_partitioned_catalogue(
    fetch_json: Callable[[str], dict[str, Any]],
    *,
    fetch_text: Callable[[str], str] | None = None,
    page_size: int,
    workers: int,
) -> list[Article]:
    """Enumerate the public catalogue through the UI's topic/level filters.

    The backing search service limits one result window to 1,000 items.  Every
    topic/JLPT partition is below that public limit, so the same filters exposed
    by the website can enumerate the complete catalogue without a private API.
    """

    summary_query = urlencode({"page": 1, "limit": 1, "is_web": "true"})
    expected_minimum = catalogue_total(fetch_json(f"{API_ROOT}/v2/news?{summary_query}"))
    partitions = [
        {"topic": topic, "level": str(level)} for topic in NEWS_TOPICS for level in range(1, 6)
    ]
    by_id: dict[str, Article] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_catalogue,
                fetch_json,
                page_size=page_size,
                filters=partition,
            ): partition
            for partition in partitions
        }
        for future in as_completed(futures):
            partition = futures[future]
            articles = future.result()
            if len(articles) > 1_000:
                raise ValueError(f"catalogue partition exceeds public window: {partition}")
            for article in articles:
                by_id.setdefault(article.article_id, article)

    if fetch_text is not None:
        for article in parse_news_sitemap(fetch_text(NEWS_SITEMAP_URL)):
            by_id.setdefault(article.article_id, article)

    if len(by_id) < expected_minimum:
        raise ValueError(
            "public catalogue partitions and sitemap did not cover the advertised catalogue: "
            f"found={len(by_id)} advertised={expected_minimum}"
        )
    return sorted(
        by_id.values(), key=lambda article: (article.date, article.article_id), reverse=True
    )


def detail_url(article_id: str) -> str:
    query = urlencode({"news_id": article_id, "lang": "ko", "furigana": "false"})
    return f"{API_ROOT}/news/get-list?{query}"


def extract_article_rows(article: Article, payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"article {article.article_id} has no result object")
    content = result.get("content")
    if not isinstance(content, list):
        raise ValueError(f"article {article.article_id} has no sentence list")

    rows: list[dict[str, Any]] = []
    for sentence_index, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        raw_japanese = item.get("text")
        translations = item.get("translate")
        if not isinstance(raw_japanese, str) or not isinstance(translations, dict):
            continue
        raw_korean = translations.get("translate_lang")
        raw_english = translations.get("translate_en")
        if not isinstance(raw_korean, str) or not isinstance(raw_english, str):
            continue
        ko = normalize_text(raw_korean)
        en = normalize_text(raw_english)
        ja = japanese_text(raw_japanese)
        if not ko or not en or not ja:
            continue
        resource_id = f"{article.article_id}:{sentence_index:04d}"
        row: dict[str, Any] = {
            "ko": ko,
            "en": en,
            "ja": ja,
            "source": SOURCE,
            "resource_id": resource_id,
            "document_id": f"todaii/{article.article_id}",
            "family_id": f"todaii:{article.article_id}",
            "domain": "news",
            "original_direction": "ja_parallel_translation",
            "source_revision": article.date or "undated",
        }
        if article.topic:
            row["topic"] = article.topic
        if article.level is not None:
            row["jlpt_level"] = article.level
        rows.append(row)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_catalogue(path: Path, articles: Iterable[Article]) -> None:
    atomic_json(path, [article.to_json() for article in articles])


def read_catalogue(path: Path) -> list[Article]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"catalogue checkpoint must be an array: {path}")
    return [Article.from_json(item) for item in value if isinstance(item, dict)]


def encode_row(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def file_digest(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def validate_jsonl(path: Path) -> dict[str, Any]:
    row_count = 0
    duplicate_triples = 0
    characteristic_misses = {"ko": 0, "en": 0, "ja": 0}
    resource_ids: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    document_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not an object")
            texts: list[str] = []
            for language in ("ko", "en", "ja"):
                text = row.get(language)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"line {line_number} has empty {language} text")
                if _TRAINING_MARKUP_RE.search(text):
                    raise ValueError(f"line {line_number} retains HTML training markup")
                texts.append(text)
            resource_id = row.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError(f"line {line_number} has no resource_id")
            if resource_id in resource_ids:
                raise ValueError(f"duplicate resource_id on line {line_number}: {resource_id}")
            resource_ids.add(resource_id)
            document_id = row.get("document_id")
            if isinstance(document_id, str):
                document_ids.add(document_id)

            triple = (texts[0], texts[1], texts[2])
            if triple in triples:
                duplicate_triples += 1
            else:
                triples.add(triple)
            if not _HANGUL_RE.search(texts[0]):
                characteristic_misses["ko"] += 1
            if not _LATIN_RE.search(texts[1]):
                characteristic_misses["en"] += 1
            if not _JAPANESE_RE.search(texts[2]):
                characteristic_misses["ja"] += 1
            row_count += 1

    size, digest = file_digest(path)
    return {
        "row_count": row_count,
        "unique_resource_ids": len(resource_ids),
        "document_count": len(document_ids),
        "duplicate_triples": duplicate_triples,
        "characteristic_script_misses": characteristic_misses,
        "output_bytes": size,
        "sha256": digest,
    }


def row_integrity_error(row: dict[str, Any]) -> str | None:
    for language in ("ko", "en", "ja"):
        text = row.get(language)
        if not isinstance(text, str) or not text.strip():
            return f"empty_{language}"
        if _TRAINING_MARKUP_RE.search(text):
            return f"html_markup_{language}"
        if not any(character.isalnum() for character in text):
            return f"non_content_{language}"
    return None


def sanitize_existing_jsonl(path: Path) -> dict[str, Any]:
    bad_documents: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not an object")
            document_id = row.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"line {line_number} has no document_id")
            error = row_integrity_error(row)
            if error:
                bad_documents.setdefault(document_id, set()).add(error)

    temporary = path.with_suffix(path.suffix + ".sanitize.tmp")
    kept_rows = 0
    removed_rows = 0
    with path.open(encoding="utf-8") as source, temporary.open("wb") as target:
        for line in source:
            row = json.loads(line)
            if row["document_id"] in bad_documents:
                removed_rows += 1
                continue
            target.write(encode_row(row))
            kept_rows += 1
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
    return {
        "removed_document_count": len(bad_documents),
        "removed_row_count": removed_rows,
        "kept_row_count": kept_rows,
        "reasons": {document_id: sorted(reasons) for document_id, reasons in bad_documents.items()},
    }


def collect(
    *,
    client: JsonClient,
    articles: list[Article],
    output: Path,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_suffix(output.suffix + ".part")
    state_path = output.with_suffix(output.suffix + ".state.json")
    state = {
        "next_article": 0,
        "rows": 0,
        "empty_articles": 0,
        "failed_articles": [],
    }
    if state_path.exists():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid state checkpoint: {state_path}")
        state.update(loaded)
    elif part_path.exists():
        raise ValueError(f"partial output exists without a state checkpoint: {part_path}")

    start = int(state["next_article"])
    rows_written = int(state["rows"])
    empty_articles = int(state["empty_articles"])
    raw_failures = state.get("failed_articles", [])
    failed_articles = list(raw_failures) if isinstance(raw_failures, list) else []
    mode = "ab" if start else "wb"
    with part_path.open(mode) as handle, ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(start, len(articles), batch_size):
            batch = articles[batch_start : batch_start + batch_size]
            futures = {
                executor.submit(client.get, detail_url(article.article_id)): article
                for article in batch
            }
            payloads: dict[str, dict[str, Any]] = {}
            for future in as_completed(futures):
                article = futures[future]
                try:
                    payloads[article.article_id] = future.result()
                except HTTPError as error:
                    if error.code not in {400, 404, 410}:
                        raise
                    failed_articles.append(
                        {"article_id": article.article_id, "error": f"HTTP {error.code}"}
                    )

            for article in batch:
                payload = payloads.get(article.article_id)
                if payload is None:
                    continue
                try:
                    rows = extract_article_rows(article, payload)
                except ValueError as error:
                    failed_articles.append({"article_id": article.article_id, "error": str(error)})
                    continue
                if not rows:
                    empty_articles += 1
                for row in rows:
                    handle.write(encode_row(row))
                    rows_written += 1
            handle.flush()
            os.fsync(handle.fileno())
            next_article = batch_start + len(batch)
            state = {
                "next_article": next_article,
                "rows": rows_written,
                "empty_articles": empty_articles,
                "failed_articles": failed_articles,
            }
            atomic_json(state_path, state)
            print(
                f"articles={next_article}/{len(articles)} rows={rows_written} "
                f"empty_articles={empty_articles} failed_articles={len(failed_articles)}",
                flush=True,
            )

    part_path.replace(output)
    state_path.unlink(missing_ok=True)
    return {
        "article_count": len(articles),
        "row_count": rows_written,
        "empty_articles": empty_articles,
        "failed_articles": failed_articles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/data32_kej.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=0.04)
    parser.add_argument("--max-articles", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--sanitize-existing", action="store_true")
    parser.add_argument(
        "--permission-confirmed",
        action="store_true",
        help="Confirm that the dataset owner granted permission for AI training use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.sanitize_existing:
        if not output.exists():
            raise SystemExit(f"output does not exist: {output}")
        sanitization = sanitize_existing_jsonl(output)
        validation = validate_jsonl(output)
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"invalid manifest: {manifest_path}")
            manifest["row_count"] = validation["row_count"]
            manifest["output_bytes"] = validation["output_bytes"]
            manifest["sha256"] = validation["sha256"]
            manifest["validation"] = validation
            manifest["sanitization"] = sanitization
            atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {"sanitization": sanitization, "validation": validation},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.validate_only:
        if not output.exists():
            raise SystemExit(f"output does not exist: {output}")
        print(json.dumps(validate_jsonl(output), ensure_ascii=False, indent=2))
        return
    if not args.permission_confirmed:
        raise SystemExit("Refusing collection without --permission-confirmed")
    if args.workers < 1 or args.batch_size < 1 or not 1 <= args.page_size <= 50:
        raise SystemExit("workers and batch-size must be positive; page-size must be in [1, 50]")

    catalogue_path = output.with_suffix(output.suffix + ".catalog.json")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    errors_path = output.with_suffix(output.suffix + ".errors.json")
    part_path = output.with_suffix(output.suffix + ".part")
    state_path = output.with_suffix(output.suffix + ".state.json")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {output}; pass --overwrite to rebuild")
        output.unlink()
        manifest_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)
        catalogue_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    client = JsonClient(
        timeout=args.timeout,
        retries=args.retries,
        request_interval=args.request_interval,
    )
    if catalogue_path.exists():
        articles = read_catalogue(catalogue_path)
    else:
        if args.max_articles is None:
            articles = fetch_partitioned_catalogue(
                client.get,
                fetch_text=client.get_text,
                page_size=args.page_size,
                workers=min(args.workers, 8),
            )
        else:
            articles = fetch_catalogue(
                client.get,
                page_size=args.page_size,
                max_articles=args.max_articles,
            )
        if not articles:
            raise SystemExit("Todaii catalogue is empty")
        write_catalogue(catalogue_path, articles)
    print(f"catalogue_articles={len(articles)}", flush=True)

    started_at = datetime.now(UTC)
    stats = collect(
        client=client,
        articles=articles,
        output=output,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    size, digest = file_digest(output)
    validation = validate_jsonl(output)
    failures = stats.pop("failed_articles")
    if failures:
        atomic_json(errors_path, failures)
    dates = [article.date for article in articles if article.date]
    manifest = {
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "api_root": API_ROOT,
        "permission_basis": "direct permission from the dataset owner, confirmed by the project owner",
        "languages": ["ko", "en", "ja"],
        "alignment": "Todaii article sentence object",
        "catalogue_newest_date": max(dates) if dates else None,
        "catalogue_oldest_date": min(dates) if dates else None,
        **stats,
        "failed_article_count": len(failures),
        "errors": str(errors_path) if failures else None,
        "validation": validation,
        "output_bytes": size,
        "sha256": digest,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "output": str(output),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
