#!/usr/bin/env python3
"""Rebuild the verified Korean-Japanese sources used in the 2026-07-27 cleanup.

This script intentionally keeps source-specific provenance and performs only
deterministic joins:

* NICT QE/APE: Japanese ``*.source`` and Korean human ``*.ref`` rows joined by ID.
* Bible.com: Korean RNKSV and Japanese Bible 1819 verses joined by ``data-usfm``.
* MASSIVE: optional audit-only rebuild of train rows joined by ID after removing
  locale-specific slot localization. It is not rebuilt by default because equal
  intent IDs are not guaranteed to be literal translations across locales.
* Firefox/VS Code: translations joined by repository path and resource key.

The script uses only the Python standard library plus ``git`` for sparse
checkouts of the two localization repositories.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import statistics
import subprocess
import tarfile
import time
from typing import Any, Iterable, Iterator
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


USER_AGENT = "sion-translate-dataset-remediation/2026-07-27"

NICT_URL = "https://paraphrasing.org/~fujita/resources/NICT-QEAPE-0.201710.zip"
NICT_SHA256 = "f1a8669ec9aaf183b435547294c7fb509e2d245afb0f98f55316ae3f0990d3a9"
NICT_ARCHIVE = "NICT-QEAPE-0.201710.zip"

MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MASSIVE_ARCHIVE = "amazon-massive-dataset-1.1.tar.gz"

FIREFOX_REPOSITORY = "https://github.com/mozilla-l10n/firefox-l10n.git"
FIREFOX_COMMIT = "4ba6d99269dfe826c3441160c374bf7c1397e14a"
VSCODE_REPOSITORY = "https://github.com/microsoft/vscode-loc.git"
VSCODE_COMMIT = "da6509eed60b550e0e785d0d78ac05be46d5e982"

BIBLE_VERSIONS = {
    "ko": {
        "id": "142",
        "title": "RNKSV",
        "start": "GEN.1",
        "locale": "ko",
        "slug": "RNKSV",
    },
    "ja": {
        "id": "1819",
        "title": "新共同訳",
        "start": "GEN.1",
        "locale": "ja",
        "slug": "新共同訳",
    },
}

_BIBLE_CHAPTER_COUNTS = (
    ("GEN", 50),
    ("EXO", 40),
    ("LEV", 27),
    ("NUM", 36),
    ("DEU", 34),
    ("JOS", 24),
    ("JDG", 21),
    ("RUT", 4),
    ("1SA", 31),
    ("2SA", 24),
    ("1KI", 22),
    ("2KI", 25),
    ("1CH", 29),
    ("2CH", 36),
    ("EZR", 10),
    ("NEH", 13),
    ("EST", 10),
    ("JOB", 42),
    ("PSA", 150),
    ("PRO", 31),
    ("ECC", 12),
    ("SNG", 8),
    ("ISA", 66),
    ("JER", 52),
    ("LAM", 5),
    ("EZK", 48),
    ("DAN", 12),
    ("HOS", 14),
    ("JOL", 3),
    ("AMO", 9),
    ("OBA", 1),
    ("JON", 4),
    ("MIC", 7),
    ("NAM", 3),
    ("HAB", 3),
    ("ZEP", 3),
    ("HAG", 2),
    ("ZEC", 14),
    ("MAL", 4),
    ("MAT", 28),
    ("MRK", 16),
    ("LUK", 24),
    ("JHN", 21),
    ("ACT", 28),
    ("ROM", 16),
    ("1CO", 16),
    ("2CO", 13),
    ("GAL", 6),
    ("EPH", 6),
    ("PHP", 4),
    ("COL", 4),
    ("1TH", 5),
    ("2TH", 3),
    ("1TI", 6),
    ("2TI", 4),
    ("TIT", 3),
    ("PHM", 1),
    ("HEB", 13),
    ("JAS", 5),
    ("1PE", 5),
    ("2PE", 3),
    ("1JN", 5),
    ("2JN", 1),
    ("3JN", 1),
    ("JUD", 1),
    ("REV", 22),
)

TARGET_NAMES = {
    "nict": "data16.jsonl",
    "bible": "data18.jsonl",
    "massive": "data37.jsonl",
    "ui": "data39.jsonl",
}
DEFAULT_SOURCES = ("bible", "nict", "ui")

_SPACE_RE = re.compile(r"\s+")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_DIGIT_RE = re.compile(r"\d+")
_PLACEHOLDER_RE = re.compile(
    r"""
    \{\s*[$-]?[A-Za-z_][A-Za-z0-9_.:-]*\s*\}
    |\{\d+\}
    |%(?:\d+\$)?[A-Za-z]
    |\$\{[A-Za-z_][A-Za-z0-9_.:-]*\}
    |</?[A-Za-z][^>]*>
    |&[A-Za-z][A-Za-z0-9]+;
    """,
    re.VERBOSE,
)
_FTL_MESSAGE_RE = re.compile(r"^(-?[A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.*)$")
_FTL_ATTRIBUTE_RE = re.compile(r"^\s+\.([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.*)$")
_BIBLE_INLINE_NOTE_MARKER_RE = re.compile(r"(?<=[\u3040-\u309f])\d+(?=\s+[「『])")
_BIBLE_TEXT_CORRECTIONS = {
    ("ja", "AMO.7.10"): ("言った 「", "言った。「"),
}


def log(message: str) -> None:
    print(message, flush=True)


def canonical_text(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def clean_bible_verse(language: str, verse_id: str, value: str) -> str:
    """Remove inline note markers and apply narrowly audited source corrections."""

    cleaned = canonical_text(_BIBLE_INLINE_NOTE_MARKER_RE.sub("", value))
    replacement = _BIBLE_TEXT_CORRECTIONS.get((language, verse_id))
    if replacement is not None:
        before, after = replacement
        cleaned = cleaned.replace(before, after)
    return cleaned


def pair_key(ko: str, ja: str) -> str:
    return f"{canonical_text(ko)}\0{canonical_text(ja)}"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target: Path, expected_sha256: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == expected_sha256:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    temporary.replace(target)
    return target


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count, sha256_file(path)


@dataclass
class BuildResult:
    source: str
    target: str
    candidates: list[dict[str, Any]]
    raw_rows: int
    rejected: Counter[str] = field(default_factory=Counter)
    details: dict[str, Any] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "raw_rows": self.raw_rows,
            "candidate_rows": len(self.candidates),
            "rejected": dict(sorted(self.rejected.items())),
            "details": self.details,
        }


def _split_tsv(line: str, minimum_fields: int) -> list[str]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < minimum_fields:
        raise ValueError(f"Expected at least {minimum_fields} TSV fields: {line[:120]!r}")
    return fields


def build_nict(cache_dir: Path) -> BuildResult:
    archive_path = download_verified(
        NICT_URL,
        cache_dir / NICT_ARCHIVE,
        NICT_SHA256,
    )
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    raw_rows = 0

    with zipfile.ZipFile(archive_path) as archive:
        for subset in ("hospital", "travel"):
            source_member = f"NICT-QEAPE-0.201710/ja/{subset}.source"
            reference_member = f"NICT-QEAPE-0.201710/ko/{subset}.ref"
            source_lines = archive.read(source_member).decode("utf-8-sig").splitlines()
            reference_lines = archive.read(reference_member).decode("utf-8-sig").splitlines()
            if len(source_lines) != len(reference_lines):
                raise ValueError(
                    f"NICT {subset}: source/reference count mismatch "
                    f"({len(source_lines)} != {len(reference_lines)})"
                )
            for source_line, reference_line in zip(
                source_lines,
                reference_lines,
                strict=True,
            ):
                raw_rows += 1
                source_fields = _split_tsv(source_line, 2)
                reference_fields = _split_tsv(reference_line, 3)
                source_id, ja = source_fields[0], "\t".join(source_fields[1:])
                reference_id = reference_fields[0]
                ko = "\t".join(reference_fields[2:])
                if source_id != reference_id:
                    raise ValueError(
                        f"NICT {subset}: ID mismatch {source_id!r} != {reference_id!r}"
                    )
                ko, ja = canonical_text(ko), canonical_text(ja)
                if not ko or not ja:
                    rejected["empty"] += 1
                    continue
                key = pair_key(ko, ja)
                if key in seen_pairs:
                    rejected["internal_exact_duplicate"] += 1
                    continue
                seen_pairs.add(key)
                rows.append(
                    {
                        "ko": ko,
                        "ja": ja,
                        "source": "NICT-QEAPE-0.201710",
                        "resource_id": source_id,
                        "subset": subset,
                        "target_kind": "human_reference",
                        "document_id": f"nict:{source_id.split(':', 1)[0]}",
                        "family_id": f"nict:{source_id}",
                        "domain": "medical" if subset == "hospital" else "travel",
                        "original_direction": "ja_to_ko",
                        "source_revision": NICT_SHA256,
                    }
                )

    return BuildResult(
        source="NICT QE/APE Japanese source + Korean human reference",
        target=TARGET_NAMES["nict"],
        candidates=rows,
        raw_rows=raw_rows,
        rejected=rejected,
        details={
            "url": NICT_URL,
            "archive_sha256": NICT_SHA256,
            "join": "source/reference ID",
            "excluded_fields": ["MT hypothesis", "post-edit", "quality labels"],
        },
    )


class VerseParser(HTMLParser):
    """Extract only visible verse content, excluding labels, headings and notes."""

    _BLOCKED_CLASSES = {"label", "heading", "note", "body", "ft"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.verses: dict[str, list[str]] = defaultdict(list)
        self.current_verse: str | None = None
        self.collect_depth = 0
        self.blocked_depth = 0
        self._stack: list[tuple[str | None, int, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack.append((self.current_verse, self.collect_depth, self.blocked_depth))
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if "verse" in classes and attributes.get("data-usfm"):
            self.current_verse = str(attributes["data-usfm"])
        if classes & self._BLOCKED_CLASSES:
            self.blocked_depth += 1
        if "content" in classes and self.current_verse and self.blocked_depth == 0:
            self.collect_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.current_verse and self.collect_depth and self.blocked_depth == 0:
            self.verses[self.current_verse].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self.current_verse, self.collect_depth, self.blocked_depth = self._stack.pop()


def _parse_next_data(page: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        page,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Bible.com page did not contain __NEXT_DATA__")
    return json.loads(match.group(1))


def _next_usfm(chapter_info: dict[str, Any]) -> str | None:
    next_info = chapter_info.get("next")
    if not isinstance(next_info, dict):
        return None
    usfm = next_info.get("usfm")
    if isinstance(usfm, list) and usfm and isinstance(usfm[0], str):
        return usfm[0]
    if isinstance(usfm, str):
        return usfm
    return None


def _fetch_bible_chapter(
    language: str,
    usfm: str,
    cache_dir: Path,
) -> dict[str, Any]:
    spec = BIBLE_VERSIONS[language]
    version_id = str(spec["id"])
    chapter_cache = cache_dir / "bible" / version_id / f"{usfm}.json"
    if chapter_cache.exists():
        return json.loads(chapter_cache.read_text(encoding="utf-8"))

    slug = quote(str(spec["slug"]), safe="")
    url = (
        f"https://www.bible.com/{spec['locale']}/bible/{version_id}/{quote(usfm, safe='.')}.{slug}"
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                page = response.read().decode("utf-8")
            payload = _parse_next_data(page)
            chapter_info = payload["props"]["pageProps"]["chapterInfo"]
            parser = VerseParser()
            parser.feed(str(chapter_info["content"]))
            verses = {}
            for verse_id, parts in parser.verses.items():
                verse = clean_bible_verse(language, verse_id, "".join(parts))
                if verse:
                    verses[verse_id] = verse
            result = {
                "usfm": usfm,
                "next": _next_usfm(chapter_info),
                "verses": verses,
                "url": url,
            }
            chapter_cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = chapter_cache.with_suffix(".json.part")
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            temporary.replace(chapter_cache)
            return result
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def _bible_chapter_refs(language: str) -> list[str]:
    counts = dict(_BIBLE_CHAPTER_COUNTS)
    if language == "ja":
        # 新共同訳 follows the Hebrew chapter division in these two books.
        counts["JOL"] = 4
        counts["MAL"] = 3
    return [
        f"{book}.{chapter}"
        for book, _ in _BIBLE_CHAPTER_COUNTS
        for chapter in range(1, counts[book] + 1)
    ]


def _crawl_bible_version(
    language: str,
    cache_dir: Path,
) -> tuple[dict[str, str], list[str], int]:
    verses: dict[str, str] = {}
    order: list[str] = []
    chapter_refs = _bible_chapter_refs(language)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(chapter_refs), 100):
            batch_refs = chapter_refs[start : start + 100]
            chapters = list(
                executor.map(
                    lambda ref: _fetch_bible_chapter(language, ref, cache_dir),
                    batch_refs,
                )
            )
            for expected_ref, chapter in zip(batch_refs, chapters, strict=True):
                if chapter.get("usfm") != expected_ref or not chapter.get("verses"):
                    raise ValueError(
                        f"Bible.com returned invalid chapter for {language}:{expected_ref}"
                    )
                for verse_id, text in chapter["verses"].items():
                    if verse_id not in verses:
                        order.append(verse_id)
                        verses[verse_id] = text
                    elif text and text != verses[verse_id]:
                        verses[verse_id] = canonical_text(f"{verses[verse_id]} {text}")
            log(
                f"Bible {language}: {min(start + 100, len(chapter_refs))} chapters / "
                f"{len(verses)} verse IDs"
            )

    return verses, order, len(chapter_refs)


def build_bible(cache_dir: Path) -> BuildResult:
    with ThreadPoolExecutor(max_workers=2) as executor:
        ko_future = executor.submit(_crawl_bible_version, "ko", cache_dir)
        ja_future = executor.submit(_crawl_bible_version, "ja", cache_dir)
        ko_verses, ko_order, ko_chapters = ko_future.result()
        ja_verses, _, ja_chapters = ja_future.result()

    ko_by_chapter: dict[str, set[str]] = defaultdict(set)
    ja_by_chapter: dict[str, set[str]] = defaultdict(set)
    for verse_id in ko_verses:
        ko_by_chapter[".".join(verse_id.split(".")[:2])].add(verse_id)
    for verse_id in ja_verses:
        ja_by_chapter[".".join(verse_id.split(".")[:2])].add(verse_id)

    all_chapters = set(ko_by_chapter) | set(ja_by_chapter)
    mismatched_chapters = {
        chapter
        for chapter in all_chapters
        if ko_by_chapter.get(chapter, set()) != ja_by_chapter.get(chapter, set())
    }
    # Psalm superscriptions are numbered as verses in one versification and as
    # headings in the other. Equal-looking IDs therefore still contain offsets.
    excluded_books = {"PSA", "JOL", "MAL"}
    excluded_chapters = mismatched_chapters | {
        chapter for chapter in all_chapters if chapter.split(".", 1)[0] in excluded_books
    }
    common_ids = {
        verse_id
        for verse_id in set(ko_verses) & set(ja_verses)
        if ".".join(verse_id.split(".")[:2]) not in excluded_chapters
    }
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    for verse_id in ko_order:
        if verse_id not in common_ids:
            continue
        ko, ja = ko_verses[verse_id], ja_verses[verse_id]
        if (
            len(canonical_text(ko)) < 2
            or len(canonical_text(ja)) < 2
            or not _HANGUL_RE.search(ko)
            or not _JAPANESE_RE.search(ja)
        ):
            rejected["missing_expected_script_or_too_short"] += 1
            continue
        key = pair_key(ko, ja)
        if key in seen_pairs:
            rejected["internal_exact_duplicate"] += 1
            continue
        seen_pairs.add(key)
        rows.append(
            {
                "ko": ko,
                "ja": ja,
                "source": "Bible.com RNKSV 142 + 新共同訳 1819",
                "resource_id": verse_id,
                "document_id": ".".join(verse_id.split(".")[:2]),
                "family_id": f"bible:{verse_id}",
                "domain": "religion",
                "original_direction": "parallel_verse",
                "source_revision": "bible.com:142+1819:2026-07-27",
            }
        )

    ko_only = [verse_id for verse_id in ko_order if verse_id not in ja_verses]
    ja_only = sorted(set(ja_verses) - set(ko_verses))
    rejected["ko_only_verse_id"] = len(ko_only)
    rejected["ja_only_verse_id"] = len(ja_only)
    return BuildResult(
        source="Bible.com RNKSV 142 + 新共同訳 1819",
        target=TARGET_NAMES["bible"],
        candidates=rows,
        raw_rows=max(len(ko_verses), len(ja_verses)),
        rejected=rejected,
        details={
            "join": (
                "exact data-usfm verse ID only inside chapters whose Korean and "
                "Japanese verse-ID sets are identical"
            ),
            "ko_chapters": ko_chapters,
            "ja_chapters": ja_chapters,
            "ko_verse_ids": len(ko_verses),
            "ja_verse_ids": len(ja_verses),
            "common_verse_ids": len(common_ids),
            "mismatched_chapters_excluded": len(mismatched_chapters),
            "mismatched_chapter_sample": sorted(mismatched_chapters)[:50],
            "systemic_versification_books_excluded": sorted(excluded_books),
            "all_excluded_chapters": len(excluded_chapters),
            "ko_only_sample": ko_only[:20],
            "ja_only_sample": ja_only[:20],
            "ko_start_url": "https://www.bible.com/bible/142/GEN.1",
            "ja_start_url": "https://www.bible.com/bible/1819/GEN.1",
        },
    )


def _massive_member(archive: tarfile.TarFile, locale: str) -> tarfile.TarInfo:
    suffix = f"/{locale}.jsonl"
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"MASSIVE archive: expected one {locale}.jsonl, got {len(matches)}")
    return matches[0]


def _read_massive_locale(archive: tarfile.TarFile, locale: str) -> dict[str, dict[str, Any]]:
    member = _massive_member(archive, locale)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"MASSIVE archive could not open {member.name}")
    records: dict[str, dict[str, Any]] = {}
    for raw_line in handle:
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("partition") == "train":
            records[str(record["id"])] = record
    return records


def _slot_methods(record: dict[str, Any]) -> list[dict[str, Any]]:
    methods = record.get("slot_method", [])
    return methods if isinstance(methods, list) else []


def _slot_labels(record: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(item.get("slot"))
        for item in _slot_methods(record)
        if isinstance(item, dict) and item.get("slot")
    )


def _slot_method_signature(record: dict[str, Any]) -> Counter[tuple[str, str]]:
    return Counter(
        (str(item.get("slot")), str(item.get("method")))
        for item in _slot_methods(record)
        if isinstance(item, dict) and item.get("slot") and item.get("method")
    )


def _has_localization(record: dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("method") == "localization"
        for item in _slot_methods(record)
    )


def _judgments_pass(record: dict[str, Any]) -> bool:
    judgments = record.get("judgments")
    if not isinstance(judgments, list) or len(judgments) < 2:
        return False
    valid = [item for item in judgments if isinstance(item, dict)]
    if len(valid) < 2:
        return False
    if sum(item.get("intent_score") in (1, 2) for item in valid) < 2:
        return False
    if statistics.median(float(item.get("grammar_score", 0)) for item in valid) < 3:
        return False
    if statistics.median(float(item.get("spelling_score", 0)) for item in valid) < 1:
        return False
    if sum(item.get("language_identification") == "target" for item in valid) < 2:
        return False
    labels = _slot_labels(record)
    if labels and sum(item.get("slots_score") in (1, 2) for item in valid) < 2:
        return False
    return True


def build_massive(cache_dir: Path) -> BuildResult:
    archive_path = download_verified(
        MASSIVE_URL,
        cache_dir / MASSIVE_ARCHIVE,
        MASSIVE_SHA256,
    )
    rejected: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()

    with tarfile.open(archive_path, "r:gz") as archive:
        ko_by_id = _read_massive_locale(archive, "ko-KR")
        ja_by_id = _read_massive_locale(archive, "ja-JP")

    common_ids = sorted(set(ko_by_id) & set(ja_by_id), key=int)
    for record_id in common_ids:
        ko_record, ja_record = ko_by_id[record_id], ja_by_id[record_id]
        if (
            ko_record.get("scenario"),
            ko_record.get("intent"),
        ) != (
            ja_record.get("scenario"),
            ja_record.get("intent"),
        ):
            rejected["scenario_or_intent_mismatch"] += 1
            continue
        if _has_localization(ko_record) or _has_localization(ja_record):
            rejected["locale_specific_slot_localization"] += 1
            continue
        if _slot_labels(ko_record) != _slot_labels(ja_record):
            rejected["slot_label_mismatch"] += 1
            continue
        if _slot_method_signature(ko_record) != _slot_method_signature(ja_record):
            rejected["slot_method_mismatch"] += 1
            continue
        if not _judgments_pass(ko_record) or not _judgments_pass(ja_record):
            rejected["judgment_quality"] += 1
            continue
        ko = canonical_text(str(ko_record.get("utt", "")))
        ja = canonical_text(str(ja_record.get("utt", "")))
        if not ko or not ja:
            rejected["empty"] += 1
            continue
        key = pair_key(ko, ja)
        if key in seen_pairs:
            rejected["internal_exact_duplicate"] += 1
            continue
        seen_pairs.add(key)
        rows.append(
            {
                "ko": ko,
                "ja": ja,
                "source": "Amazon MASSIVE 1.1",
                "resource_id": record_id,
                "scenario": str(ko_record["scenario"]),
                "intent": str(ko_record["intent"]),
                "slot_policy": "no_locale_specific_localization",
                "document_id": f"massive:{ko_record['scenario']}",
                "family_id": f"massive:{record_id}",
                "domain": "voice_assistant",
                "original_direction": "en_localized_to_ko_and_ja",
                "source_revision": MASSIVE_SHA256,
            }
        )

    return BuildResult(
        source="Amazon MASSIVE 1.1 train",
        target=TARGET_NAMES["massive"],
        candidates=rows,
        raw_rows=len(common_ids),
        rejected=rejected,
        details={
            "url": MASSIVE_URL,
            "archive_sha256": MASSIVE_SHA256,
            "join": "train partition + exact utterance ID",
            "quality_policy": (
                "same scenario/intent and slot labels; no slot_method=localization; "
                "both locales pass majority intent/language and median grammar/spelling checks"
            ),
        },
    )


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def ensure_sparse_checkout(
    target: Path,
    repository: str,
    commit: str,
    sparse_paths: list[str],
) -> Path:
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repository,
                str(target),
            ]
        )
    _run_git(["sparse-checkout", "set", *sparse_paths], cwd=target)
    try:
        _run_git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=target)
    except subprocess.CalledProcessError:
        _run_git(["fetch", "--depth", "1", "origin", commit], cwd=target)
    _run_git(["checkout", "--detach", commit], cwd=target)
    actual = _run_git(["rev-parse", "HEAD"], cwd=target)
    if actual != commit:
        raise ValueError(f"Git checkout mismatch for {repository}: {actual} != {commit}")
    return target


def _finalize_ftl_entry(
    output: dict[str, str],
    key: str | None,
    parts: list[str],
    *,
    complex_entry: bool,
) -> None:
    if key is None or complex_entry:
        return
    value = canonical_text(" ".join(parts))
    if value:
        output[key] = value


def parse_simple_ftl(path: Path) -> dict[str, str]:
    """Parse literal Fluent messages and attributes; selectors are excluded."""

    output: dict[str, str] = {}
    current_key: str | None = None
    current_parts: list[str] = []
    current_complex = False
    attribute_key: str | None = None
    attribute_parts: list[str] = []
    attribute_complex = False

    def flush_attribute() -> None:
        nonlocal attribute_key, attribute_parts, attribute_complex
        _finalize_ftl_entry(
            output,
            attribute_key,
            attribute_parts,
            complex_entry=attribute_complex,
        )
        attribute_key = None
        attribute_parts = []
        attribute_complex = False

    def flush_message() -> None:
        nonlocal current_key, current_parts, current_complex
        flush_attribute()
        _finalize_ftl_entry(
            output,
            current_key,
            current_parts,
            complex_entry=current_complex,
        )
        current_key = None
        current_parts = []
        current_complex = False

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        if match := _FTL_MESSAGE_RE.match(raw_line):
            flush_message()
            current_key = match.group(1)
            value = match.group(2).strip()
            current_parts = [value] if value else []
            current_complex = "->" in value or value.count("{") != value.count("}")
            continue
        if current_key and (match := _FTL_ATTRIBUTE_RE.match(raw_line)):
            flush_attribute()
            attribute_key = f"{current_key}.{match.group(1)}"
            value = match.group(2).strip()
            attribute_parts = [value] if value else []
            attribute_complex = "->" in value or value.count("{") != value.count("}")
            continue
        if current_key and raw_line[:1].isspace() and raw_line.strip():
            value = raw_line.strip()
            if attribute_key:
                attribute_parts.append(value)
                attribute_complex |= (
                    "->" in value
                    or value.startswith(("[", "*["))
                    or value.count("{") != value.count("}")
                )
            else:
                current_parts.append(value)
                current_complex |= (
                    "->" in value
                    or value.startswith(("[", "*["))
                    or value.count("{") != value.count("}")
                )
            continue
        if raw_line and not raw_line[:1].isspace():
            flush_message()
    flush_message()
    return output


def parse_properties(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    logical_lines: list[str] = []
    pending = ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.rstrip()
        pending += line
        if pending.endswith("\\") and not pending.endswith("\\\\"):
            pending = pending[:-1]
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        logical_lines.append(pending)

    for line in logical_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        match = re.match(r"([^:=\s]+)\s*(?:[:=]|\s)\s*(.*)$", stripped)
        if not match:
            continue
        key, value = match.group(1), canonical_text(match.group(2))
        if value:
            output[key] = value
    return output


def _flatten_json_strings(value: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield "/".join(prefix), value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_json_strings(value[key], (*prefix, str(key)))


def _placeholder_signature(text: str) -> Counter[str]:
    normalized: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        token = match.group(0)
        if token.startswith("<"):
            tag_match = re.match(r"</?([A-Za-z][A-Za-z0-9:-]*)", token)
            token = f"<{tag_match.group(1).lower()}>" if tag_match else token
        normalized.append(token.replace(" ", ""))
    return Counter(normalized)


def _ui_pair_rejection(ko: str, ja: str, resource_id: str) -> str | None:
    if not ko or not ja:
        return "empty"
    if len(ko) > 800 or len(ja) > 800:
        return "too_long"
    if not _HANGUL_RE.search(ko) or not _JAPANESE_RE.search(ja):
        return "missing_expected_script"
    visible_ko = len(re.sub(r"\W", "", ko))
    visible_ja = len(re.sub(r"\W", "", ja))
    if min(visible_ko, visible_ja) < 2:
        return "too_short"
    ratio = len(ko) / max(len(ja), 1)
    if ratio < 0.2 or ratio > 5.0:
        return "length_ratio"
    if _placeholder_signature(ko) != _placeholder_signature(ja):
        return "placeholder_mismatch"
    if Counter(_DIGIT_RE.findall(ko)) != Counter(_DIGIT_RE.findall(ja)):
        return "number_mismatch"
    lowered = resource_id.lower()
    if (
        any(term in lowered for term in ("accesskey", ".key", "/keybinding"))
        and max(
            len(ko),
            len(ja),
        )
        <= 3
    ):
        return "access_key"
    if canonical_text(ko).casefold() == canonical_text(ja).casefold():
        return "identical"
    return None


def _collect_firefox_candidates(repository: Path) -> Iterator[tuple[str, str, str]]:
    ko_root, ja_root = repository / "ko", repository / "ja"
    ko_files = {
        path.relative_to(ko_root).as_posix(): path
        for path in ko_root.rglob("*")
        if path.is_file() and path.suffix in {".ftl", ".properties"}
    }
    ja_files = {
        path.relative_to(ja_root).as_posix(): path
        for path in ja_root.rglob("*")
        if path.is_file() and path.suffix in {".ftl", ".properties"}
    }
    for relative in sorted(set(ko_files) & set(ja_files)):
        parser = parse_simple_ftl if Path(relative).suffix == ".ftl" else parse_properties
        ko_messages = parser(ko_files[relative])
        ja_messages = parser(ja_files[relative])
        for key in sorted(set(ko_messages) & set(ja_messages)):
            yield f"{relative}::{key}", ko_messages[key], ja_messages[key]


def _collect_vscode_candidates(repository: Path) -> Iterator[tuple[str, str, str]]:
    ko_root = repository / "i18n" / "vscode-language-pack-ko" / "translations"
    ja_root = repository / "i18n" / "vscode-language-pack-ja" / "translations"
    ko_files = {
        path.relative_to(ko_root).as_posix(): path
        for path in ko_root.rglob("*.json")
        if path.is_file()
    }
    ja_files = {
        path.relative_to(ja_root).as_posix(): path
        for path in ja_root.rglob("*.json")
        if path.is_file()
    }
    for relative in sorted(set(ko_files) & set(ja_files)):
        ko_payload = json.loads(ko_files[relative].read_text(encoding="utf-8-sig"))
        ja_payload = json.loads(ja_files[relative].read_text(encoding="utf-8-sig"))
        ko_messages = dict(_flatten_json_strings(ko_payload.get("contents", {})))
        ja_messages = dict(_flatten_json_strings(ja_payload.get("contents", {})))
        for key in sorted(set(ko_messages) & set(ja_messages)):
            yield f"{relative}::{key}", ko_messages[key], ja_messages[key]


def build_ui(cache_dir: Path) -> BuildResult:
    firefox = ensure_sparse_checkout(
        cache_dir / "firefox-l10n",
        FIREFOX_REPOSITORY,
        FIREFOX_COMMIT,
        ["ja", "ko"],
    )
    vscode = ensure_sparse_checkout(
        cache_dir / "vscode-loc",
        VSCODE_REPOSITORY,
        VSCODE_COMMIT,
        [
            "i18n/vscode-language-pack-ja",
            "i18n/vscode-language-pack-ko",
        ],
    )

    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    raw_rows = 0
    source_counts: Counter[str] = Counter()

    sources = (
        ("Mozilla Firefox l10n", FIREFOX_COMMIT, _collect_firefox_candidates(firefox)),
        ("Microsoft VS Code l10n", VSCODE_COMMIT, _collect_vscode_candidates(vscode)),
    )
    for source, commit, candidates in sources:
        for resource_id, raw_ko, raw_ja in candidates:
            raw_rows += 1
            ko, ja = canonical_text(raw_ko), canonical_text(raw_ja)
            reason = _ui_pair_rejection(ko, ja, resource_id)
            if reason:
                rejected[f"{source}:{reason}"] += 1
                continue
            key = pair_key(ko, ja)
            if key in seen_pairs:
                rejected[f"{source}:internal_exact_duplicate"] += 1
                continue
            seen_pairs.add(key)
            rows.append(
                {
                    "ko": ko,
                    "ja": ja,
                    "source": source,
                    "resource_id": resource_id,
                    "source_commit": commit,
                    "document_id": resource_id.split("::", 1)[0],
                    "family_id": f"{source}:{resource_id}",
                    "domain": "software_ui",
                    "original_direction": "parallel_localization",
                    "source_revision": commit,
                }
            )
            source_counts[source] += 1

    return BuildResult(
        source="Pinned Firefox and VS Code Korean/Japanese localization",
        target=TARGET_NAMES["ui"],
        candidates=rows,
        raw_rows=raw_rows,
        rejected=rejected,
        details={
            "join": "same repository-relative path and resource key",
            "firefox": {
                "repository": FIREFOX_REPOSITORY,
                "commit": FIREFOX_COMMIT,
            },
            "vscode": {
                "repository": VSCODE_REPOSITORY,
                "commit": VSCODE_COMMIT,
            },
            "written_by_source_before_root_dedup": dict(source_counts),
            "filters": [
                "expected Korean/Japanese scripts",
                "matching placeholders",
                "matching ASCII numeric tokens",
                "0.2 <= character-length ratio <= 5.0",
                "no literal-identical or access-key-only rows",
                "Fluent selectors excluded",
            ],
        },
    )


def remove_existing_root_duplicates(
    results: list[BuildResult],
    data_dir: Path,
) -> dict[str, int]:
    owners: dict[str, tuple[BuildResult, dict[str, Any]]] = {}
    internal_cross_source = Counter()
    for result in results:
        deduplicated: list[dict[str, Any]] = []
        for row in result.candidates:
            key = pair_key(str(row["ko"]), str(row["ja"]))
            if key in owners:
                result.rejected["cross_new_source_exact_duplicate"] += 1
                internal_cross_source[result.target] += 1
                continue
            owners[key] = (result, row)
            deduplicated.append(row)
        result.candidates = deduplicated

    candidate_keys = set(owners)
    existing_hits: Counter[str] = Counter()
    found_existing_keys: set[str] = set()
    target_paths = {data_dir / result.target for result in results}
    for path in sorted(data_dir.glob("*.jsonl")):
        if path in target_paths:
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                ko, ja = record.get("ko"), record.get("ja")
                if not isinstance(ko, str) or not isinstance(ja, str):
                    continue
                key = pair_key(ko, ja)
                owner = owners.get(key)
                if owner is not None and key not in found_existing_keys:
                    result, _ = owner
                    existing_hits[result.target] += 1
                    found_existing_keys.add(key)
                    candidate_keys.discard(key)
        log(
            f"existing-root dedup: {path.name} scanned; "
            f"{sum(existing_hits.values())} candidate hits"
        )

    for result in results:
        before = len(result.candidates)
        result.candidates = [
            row
            for row in result.candidates
            if pair_key(str(row["ko"]), str(row["ja"])) in candidate_keys
        ]
        removed = before - len(result.candidates)
        if removed:
            result.rejected["existing_root_exact_duplicate"] += removed
    return dict(existing_hits)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        ko, ja = row.get("ko"), row.get("ja")
        if not isinstance(ko, str) or not isinstance(ja, str) or not ko or not ja:
            raise ValueError(f"Invalid row {index}: missing ko/ja strings")
        key = pair_key(ko, ja)
        if key in seen:
            raise ValueError(f"Invalid row {index}: exact duplicate")
        seen.add(key)


def build_selected(
    selected: list[str],
    *,
    data_dir: Path,
    cache_dir: Path,
    report_path: Path,
    scan_existing: bool,
) -> dict[str, Any]:
    builders = {
        "nict": build_nict,
        "bible": build_bible,
        "massive": build_massive,
        "ui": build_ui,
    }
    results: list[BuildResult] = []
    for name in selected:
        log(f"building {name}")
        results.append(builders[name](cache_dir))
        log(f"{name}: {len(results[-1].candidates)} candidates")

    existing_hits: dict[str, int] = {}
    if scan_existing:
        existing_hits = remove_existing_root_duplicates(results, data_dir)

    output_reports: list[dict[str, Any]] = []
    for result in results:
        validate_rows(result.candidates)
        output_path = data_dir / result.target
        count, digest = write_jsonl(output_path, result.candidates)
        item = result.report()
        item.update(
            {
                "written_rows": count,
                "output_sha256": digest,
                "output_bytes": output_path.stat().st_size,
            }
        )
        output_reports.append(item)
        log(f"wrote {output_path}: {count} rows / {digest}")

    report = {
        "schema": "sion-verified-parallel-rebuild-v1",
        "date": "2026-07-27",
        "selected": selected,
        "existing_root_scan": scan_existing,
        "existing_root_hits": existing_hits,
        "outputs": output_reports,
        "total_written": sum(item["written_rows"] for item in output_reports),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    log(f"report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(TARGET_NAMES),
        default=list(DEFAULT_SOURCES),
        help=(
            "Sources to rebuild. MASSIVE is opt-in because matching intent IDs "
            "do not guarantee translation-equivalent utterances."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/source_cache/remediation_20260727"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/dataset_remediation_20260727.json"),
    )
    parser.add_argument(
        "--no-scan-existing",
        action="store_true",
        help="Do not remove exact pairs already present in other data/*.jsonl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_selected(
        args.sources,
        data_dir=args.data_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        report_path=args.report.resolve(),
        scan_existing=not args.no_scan_existing,
    )


if __name__ == "__main__":
    main()
