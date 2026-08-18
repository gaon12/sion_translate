#!/usr/bin/env python3
"""Build the English and language-consistent reasoning foundation corpora.

The generated files deliberately live under ``data/corpus/<language>``.
Reasoning rows retain ``text`` for tokenizer sampling and human audit, while the
mixed-objective foundation reader consumes their explicit
``prompt``/``think``/``answer`` fields. Korean and Japanese traces are rejected
when an otherwise native ``think`` section changes to English prose; formula
variables and product names are not treated as a language switch.
"""

# The acquisition script reads third-party Arrow schemas and JSON payloads.
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, TextIO
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SourceFile:
    repo: str
    revision: str
    path: str
    size: int
    license: str
    language: str
    category: str
    sha256: str | None = None

    @property
    def url(self) -> str:
        encoded_path = "/".join(quote(part, safe="") for part in self.path.split("/"))
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{encoded_path}"


FINEWEB_SOURCE = SourceFile(
    repo="mdonigian/fineweb-edu-curated",
    revision="e300861750a006af42cb1d9ec2846368a6e0a4ee",
    path="data_0000.parquet",
    size=3_301_516_976,
    license="ODC-By-1.0",
    language="en",
    category="multidomain_educational",
)

REASONING_SOURCES = (
    SourceFile(
        repo="Jongsim/claude-opus-4.6-reasoning-12k-ko-filtered-v2",
        revision="7f4d46a49d6ace960f10fc0fcf5d576ee2e95e7c",
        path="reasoning_merged_ko_filtered_v2.parquet",
        size=13_498_647,
        license="Apache-2.0",
        language="ko",
        category="mixed",
    ),
    SourceFile(
        repo="elyza/JaMARD",
        revision="82e107d209dec19e17a76d76425452c81b192755",
        path="data/train.parquet",
        size=231_287_917,
        license="MIT",
        language="ja",
        category="math_reasoning",
        sha256="f05f958e30a4e67baf3fb2b1ef6d68017ecaf8a5ea9d6bbde6e4e09817ae7810",
    ),
    SourceFile(
        repo="DeL-TaiseiOzaki/Tengentoppa-sft-reasoning-ja",
        revision="90c8ca1cbc7e8800c2965c9792faf5b012e42efd",
        path="final_outputs.jsonl",
        size=9_391_738,
        license="Apache-2.0",
        language="ja",
        category="open_ended",
    ),
    *(
        SourceFile(
            repo="Daemontatox/alpaca_reasoning_COT",
            revision="abd9c19cc1d5f4f33e329ff8a00e9ff0f064f43f",
            path=f"{category}_train.parquet",
            size=size,
            license="Apache-2.0",
            language="en",
            category=category,
        )
        for category, size in (
            ("aqua", 504_638),
            ("creak", 943_601),
            ("ecqa", 1_334_840),
            ("esnli", 4_919_257),
            ("gsm8k", 2_221_738),
            ("qasc", 207_121),
            ("qed", 2_881_923),
            ("sensemaking", 687_548),
            ("strategyqa", 382_263),
        )
    ),
)

# Physical JSONL budgets.  The total (6.30 GB) is deliberately between the
# current Japanese 6.37 GB and Korean 8.51 GB corpora, rather than letting one
# English web source dominate the model.
ENGLISH_CATEGORY_BUDGETS = {
    "general": 2_200_000_000,
    "mathematics": 500_000_000,
    "computer_science": 500_000_000,
    "ml_ai": 350_000_000,
    "physical_sciences": 400_000_000,
    "life_sciences": 400_000_000,
    "engineering_tech": 450_000_000,
    "environmental": 350_000_000,
    "medicine_health": 400_000_000,
    "business_economics": 400_000_000,
    "law_government": 350_000_000,
}

_GROUP_ALIASES = {
    "Mathematics": "mathematics",
    "Computer Science": "computer_science",
    "ML/AI": "ml_ai",
    "Physical Sciences": "physical_sciences",
    "Life Sciences": "life_sciences",
    "Engineering/Tech": "engineering_tech",
    "Environmental Sci": "environmental",
    "Medicine/Health": "medicine_health",
    "Business/Economics": "business_economics",
    "Law/Government": "law_government",
    "General Knowledge": "general",
    **{name: name for name in ENGLISH_CATEGORY_BUDGETS},
}

_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SPAN_SPLIT = re.compile(r"(?:\n+|(?<=[.!?。！？])\s+)")
_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_ENGLISH_FUNCTION_WORDS = frozenset(
    "a an and are as at be because been but by can could did do does for from "
    "given had has have he her here how if in into is it its let may more must "
    "not of on one or our should so than that the their then there therefore "
    "these they this to was we were what when where which while who will with "
    "would you your".split()
)
_PROMPT_SUFFIX = re.compile(
    r"\s*(?:let'?s think|h+m+|step[- ]by[- ]step|chain of thought|"
    r"reasoning process|some random reasoning|stream of consciousness)[:.\s]*$",
    re.IGNORECASE,
)
_FINAL_MARKER = re.compile(
    r"(?:therefore,?\s*|so,?\s*)?(?:the\s+)?final answer\s*(?:is|:)\s*",
    re.IGNORECASE,
)
_JAMARD_FINAL_MARKERS = ("回答:", "####", "答え:")


def _script_counts(text: str) -> tuple[int, int, int]:
    return len(_HANGUL.findall(text)), len(_JAPANESE.findall(text)), len(_LATIN.findall(text))


def reasoning_language_issue(language: str, think: str) -> str | None:
    """Return why a reasoning trace changes language, or ``None`` when valid."""

    think = think.strip()
    if len(think) < 20:
        return "think_too_short"
    hangul, japanese, latin = _script_counts(think)
    target = hangul if language == "ko" else japanese if language == "ja" else latin
    competing = latin if language in {"ko", "ja"} else hangul + japanese
    if target < 12:
        if language == "en" and competing == 0 and len(re.findall(r"\d", think)) >= 3:
            return None
        return "think_target_script_missing"

    if language in {"ko", "ja"}:
        target_pattern = _HANGUL if language == "ko" else _JAPANESE
        for span in _SPAN_SPLIT.split(think):
            words = [word.lower() for word in _ENGLISH_WORD.findall(span)]
            if len(words) < 6 or len(target_pattern.findall(span)) >= 4:
                continue
            function_words = sum(word in _ENGLISH_FUNCTION_WORDS for word in words)
            # Technical identifiers and equations rarely contain several English
            # function words.  A prose switch almost always does.
            if function_words >= 2 or len(words) >= 14:
                return "think_english_prose_switch"
    if target / max(1, target + competing) < (0.52 if language in {"ko", "ja"} else 0.80):
        return "think_language_ratio"
    return None


def text_language_issue(language: str, text: str, *, field: str) -> str | None:
    hangul, japanese, latin = _script_counts(text)
    target = hangul if language == "ko" else japanese if language == "ja" else latin
    other = latin if language in {"ko", "ja"} else hangul + japanese
    if target < 3:
        # Short final answers are often numbers, option letters, symbols, or
        # proper names.  They are safe when they contain no competing script;
        # the language-switch requirement applies most strictly to ``think``.
        if field == "answer" and other == 0 and text.strip():
            return None
        return f"{field}_target_script_missing"
    threshold = 0.35 if language in {"ko", "ja"} else 0.75
    if target / max(1, target + other) < threshold:
        return f"{field}_language_ratio"
    return None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_file(source: SourceFile, path: Path) -> None:
    actual_size = path.stat().st_size
    if actual_size != source.size:
        raise RuntimeError(
            f"download size mismatch for {source.repo}/{source.path}: "
            f"{actual_size} != {source.size}"
        )
    if source.sha256 is not None:
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != source.sha256:
            raise RuntimeError(
                f"download SHA-256 mismatch for {source.repo}/{source.path}: "
                f"{actual_sha256} != {source.sha256}"
            )


def _download(source: SourceFile, root: Path) -> Path:
    destination = root / source.repo.replace("/", "--") / source.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.size:
        _validate_source_file(source, destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = Request(source.url, headers={"User-Agent": "sion-foundation-builder/1"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urlopen(request, timeout=120) as response:
        append = existing > 0 and getattr(response, "status", 200) == 206
        with partial.open("ab" if append else "wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    _validate_source_file(source, partial)
    partial.replace(destination)
    return destination


def _iter_parquet(path: Path, *, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("pyarrow is required to read the source parquet files") from error
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _write_jsonl(handle: TextIO, row: dict[str, Any]) -> int:
    rendered = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    handle.write(rendered)
    return len(rendered.encode("utf-8"))


def _choose_group(groups: object, used: dict[str, int]) -> str | None:
    candidates = []
    if isinstance(groups, list):
        candidates = [
            _GROUP_ALIASES[group]
            for group in groups
            if isinstance(group, str) and group in _GROUP_ALIASES
        ]
    if not candidates:
        candidates = ["general"]
    available = [
        group
        for group in dict.fromkeys(candidates)
        if used[group] < ENGLISH_CATEGORY_BUDGETS[group]
    ]
    if not available:
        return None
    return min(available, key=lambda group: used[group] / ENGLISH_CATEGORY_BUDGETS[group])


def build_english_corpus(source_path: Path, corpus_root: Path) -> dict[str, Any]:
    output_dir = corpus_root / "en"
    output_dir.mkdir(parents=True, exist_ok=True)
    used = {group: 0 for group in ENGLISH_CATEGORY_BUDGETS}
    rows: Counter[str] = Counter()
    with ExitStack() as stack:
        handles = {
            group: stack.enter_context(
                (output_dir / f"fineweb_edu_curated_{group}.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )
            )
            for group in ENGLISH_CATEGORY_BUDGETS
        }
        for row in _iter_parquet(source_path):
            rows["seen"] += 1
            text = row.get("text")
            if not isinstance(text, str) or len(text.strip()) < 80:
                rows["invalid_text"] += 1
                continue
            text = text.strip()
            if text_language_issue("en", text, field="document"):
                rows["non_english"] += 1
                continue
            group = _choose_group(row.get("assigned_groups"), used)
            if group is None:
                rows["category_full"] += 1
                continue
            rendered_size = len(
                (
                    json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
            )
            if used[group] + rendered_size > ENGLISH_CATEGORY_BUDGETS[group]:
                used[group] = ENGLISH_CATEGORY_BUDGETS[group]
                rows["category_full"] += 1
                continue
            used[group] += _write_jsonl(handles[group], {"text": text})
            rows[f"accepted_{group}"] += 1
            if all(used[name] >= budget for name, budget in ENGLISH_CATEGORY_BUDGETS.items()):
                break
    return {
        "source": vars(FINEWEB_SOURCE),
        "rows": dict(sorted(rows.items())),
        "bytes_by_category": used,
        "total_output_bytes": sum(used.values()),
    }


def _split_english_output(output: str) -> tuple[str, str]:
    matches = list(_FINAL_MARKER.finditer(output))
    if matches:
        match = matches[-1]
        return output[: match.start()].strip(), output[match.end() :].strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", output) if part.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:-1]), sentences[-1]
    return output, output


def _jamard_reasoning_rows(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    """Select one verified, shortest CoT per unique JaMARD train prompt."""

    selected: dict[str, dict[str, str]] = {}
    stats: Counter[str] = Counter()
    for raw_index, row in enumerate(_iter_parquet(path)):
        stats["seen"] += 1
        prompt = str(row.get("instruction") or "").strip()
        answer = str(row.get("gold_answer") or "").strip()
        candidates: list[tuple[int, str, int, str, str]] = []
        for response_index, raw_response in enumerate(row.get("true_responses") or []):
            if not isinstance(raw_response, str) or not raw_response.strip():
                continue
            response = raw_response.strip()
            marker_matches = [
                (response.rfind(marker), marker)
                for marker in _JAMARD_FINAL_MARKERS
                if response.rfind(marker) >= 0
            ]
            if not marker_matches:
                continue
            marker_position, marker = max(marker_matches)
            think = response[:marker_position].strip()
            if think:
                candidates.append((len(response), response, response_index, think, marker))

        if not prompt or not answer or not candidates:
            stats["no_usable_verified_response"] += 1
            continue

        response_length, _, response_index, think, marker = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        candidate = {
            "prompt": prompt,
            "think": think,
            "answer": answer,
            "category": "math_reasoning",
            "source_id": f"train:{raw_index}:{response_index}",
            "seed_source": str(row.get("source") or "unknown"),
            "_answer_marker": marker,
            "_response_length": str(response_length),
        }
        previous = selected.get(prompt)
        if previous is not None:
            stats["duplicate_prompt"] += 1
            previous_key = (
                int(previous["_response_length"]),
                previous["think"],
                previous["source_id"],
            )
            candidate_key = (response_length, think, candidate["source_id"])
            if candidate_key >= previous_key:
                continue
        selected[prompt] = candidate

    rows = sorted(selected.values(), key=lambda row: row["source_id"])
    for row in rows:
        stats[f"selected_marker_{row.pop('_answer_marker')}"] += 1
        row.pop("_response_length")
    stats["selected_unique_prompts"] = len(rows)
    return rows, stats


def _reasoning_rows(source: SourceFile, path: Path) -> Iterator[dict[str, str]]:
    rows = _iter_jsonl(path) if path.suffix == ".jsonl" else _iter_parquet(path)
    for index, row in enumerate(rows):
        prompt = think = answer = ""
        category = source.category
        source_id = str(row.get("id", index))
        if source.language == "ko":
            raw_messages = row.get("messages")
            try:
                messages = json.loads(raw_messages) if isinstance(raw_messages, str) else []
            except json.JSONDecodeError:
                messages = []
            user = next((item for item in messages if item.get("role") == "user"), {})
            assistant = next((item for item in messages if item.get("role") == "assistant"), {})
            prompt = str(user.get("content", ""))
            think = str(assistant.get("reasoning", ""))
            answer = str(assistant.get("content", ""))
            category = str(row.get("domain") or category)
        elif source.repo.startswith("DeL-"):
            prompt = str(row.get("instruction", ""))
            think = str(row.get("reasoning", ""))
            answer = str(row.get("final_output", ""))
        else:
            prompt = _PROMPT_SUFFIX.sub("", str(row.get("instruction", ""))).strip()
            if row.get("input"):
                prompt = f"{prompt}\n{row['input']}".strip()
            think, answer = _split_english_output(str(row.get("output", "")))
        yield {
            "prompt": prompt.strip(),
            "think": think.strip(),
            "answer": answer.strip(),
            "category": category,
            "source_id": source_id,
        }


def build_reasoning_corpora(
    source_paths: Iterable[tuple[SourceFile, Path]],
    corpus_root: Path,
) -> dict[str, Any]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    source_manifests: list[dict[str, Any]] = []
    seen: dict[str, set[str]] = defaultdict(set)
    output_paths: dict[tuple[str, str], Path] = {}
    with ExitStack() as stack:
        handles: dict[tuple[str, str], TextIO] = {}
        for source, path in source_paths:
            source_key = f"{source.repo}/{source.path}"
            source_stats = stats[source_key]
            if source.repo == "elyza/JaMARD":
                source_rows, normalization_stats = _jamard_reasoning_rows(path)
                source_stats.update(normalization_stats)
                count_seen_in_loop = False
            else:
                source_rows = _reasoning_rows(source, path)
                count_seen_in_loop = True
            output_key = (source.language, source.repo.split("/")[-1].lower())
            if output_key not in handles:
                output_dir = corpus_root / source.language
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"reasoning_{output_key[1]}.jsonl"
                output_paths[output_key] = output_path
                handles[output_key] = stack.enter_context(
                    output_path.open("w", encoding="utf-8", newline="\n")
                )
            for row in source_rows:
                if count_seen_in_loop:
                    source_stats["seen"] += 1
                if not row["prompt"] or not row["think"] or not row["answer"]:
                    source_stats["missing_field"] += 1
                    continue
                issue = text_language_issue(source.language, row["prompt"], field="prompt")
                issue = issue or reasoning_language_issue(source.language, row["think"])
                issue = issue or text_language_issue(source.language, row["answer"], field="answer")
                if issue:
                    source_stats[issue] += 1
                    continue
                digest = sha256(
                    f"{row['prompt']}\0{row['think']}\0{row['answer']}".encode("utf-8")
                ).hexdigest()
                if digest in seen[source.language]:
                    source_stats["duplicate"] += 1
                    continue
                seen[source.language].add(digest)
                normalized = {
                    "text": (
                        f"<question>\n{row['prompt']}\n</question>\n"
                        f"<think>\n{row['think']}\n</think>\n"
                        f"<answer>\n{row['answer']}\n</answer>"
                    ),
                    "prompt": row["prompt"],
                    "think": row["think"],
                    "answer": row["answer"],
                    "language": source.language,
                    "category": row["category"],
                    "source": source.repo,
                    "source_id": row["source_id"],
                    "license": source.license,
                    "source_revision": source.revision,
                }
                if row.get("seed_source"):
                    normalized["seed_source"] = row["seed_source"]
                _write_jsonl(handles[output_key], normalized)
                source_stats["accepted"] += 1
            source_manifests.append({"source": vars(source), "stats": dict(source_stats)})
    return {
        "sources": source_manifests,
        "accepted_by_language": {
            language: len(digests) for language, digests in sorted(seen.items())
        },
        "outputs": {
            f"{language}/{slug}": {
                "path": path.as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for (language, slug), path in sorted(output_paths.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "english", "reasoning"), default="all")
    parser.add_argument("--corpus-root", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--download-root", type=Path, default=Path(".codex_work/foundation_expansion_sources")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/foundation_expansion.json"))
    parser.add_argument("--keep-downloads", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest: dict[str, Any] = {"schema": "sion-foundation-expansion-v1"}
    if args.manifest.is_file():
        existing = json.loads(args.manifest.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing.get("schema") == manifest["schema"]:
            manifest.update(existing)
    downloaded: list[Path] = []
    if args.mode in {"all", "english"}:
        fineweb = _download(FINEWEB_SOURCE, args.download_root)
        downloaded.append(fineweb)
        manifest["english"] = build_english_corpus(fineweb, args.corpus_root)
    if args.mode in {"all", "reasoning"}:
        sources: list[tuple[SourceFile, Path]] = []
        for source in REASONING_SOURCES:
            path = _download(source, args.download_root)
            downloaded.append(path)
            sources.append((source, path))
        manifest["reasoning"] = build_reasoning_corpora(sources, args.corpus_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.keep_downloads:
        for path in downloaded:
            path.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
