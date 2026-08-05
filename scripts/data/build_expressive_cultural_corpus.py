#!/usr/bin/env python3
"""Build leakage-separated expressive/cultural ko-ja training and challenge data.

The seed file is deliberately small and human-authored.  This builder does not
paraphrase or randomly combine fragments: it validates the pragmatic metadata,
writes only ``split=train`` pairs to the training JSONL, and expands every
``split=challenge`` pair into explicit ko->ja and ja->ko comparison cases.

Example::

    python scripts/data/build_expressive_cultural_corpus.py \
      --training-output data/synthetic_expressive_cultural.jsonl \
      --challenge-output examples/expressive_cultural_cases.jsonl \
      --report reports/expressive-cultural-build.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import unicodedata


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_PATH = REPOSITORY_ROOT / "examples" / "expressive_cultural_seed_pairs.jsonl"
ALLOWED_CATEGORIES = frozenset(
    {
        "profanity_slang",
        "interjection_moan",
        "idiom_culture",
    }
)
ALLOWED_SPLITS = frozenset({"train", "challenge"})
SOURCE_NAME = "human_curated_expressive_cultural_v1"


@dataclass(frozen=True, slots=True)
class SeedPair:
    id: str
    ko: str
    ja: str
    category: str
    subcategory: str
    intensity: int
    register: str
    localization_strategy: str
    split: str


@dataclass(frozen=True, slots=True)
class BuildResult:
    seed_pairs: int
    training_pairs: int
    challenge_pairs: int
    challenge_cases: int
    seed_sha256: str
    training_by_category: dict[str, int]
    challenge_by_category: dict[str, int]


def _text(row: dict[str, Any], key: str, *, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {key} must be a non-empty string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if "\n" in normalized or "\r" in normalized:
        raise ValueError(f"{location}: {key} must stay on one JSONL line")
    return normalized


def _seed_pair(row: object, *, location: str) -> SeedPair:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: each seed must be a JSON object")
    intensity = row.get("intensity")
    if not isinstance(intensity, int) or isinstance(intensity, bool) or not 1 <= intensity <= 5:
        raise ValueError(f"{location}: intensity must be an integer in [1, 5]")
    category = _text(row, "category", location=location)
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"{location}: unknown category {category!r}; expected {sorted(ALLOWED_CATEGORIES)}"
        )
    split = _text(row, "split", location=location)
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"{location}: split must be train or challenge")
    return SeedPair(
        id=_text(row, "id", location=location),
        ko=_text(row, "ko", location=location),
        ja=_text(row, "ja", location=location),
        category=category,
        subcategory=_text(row, "subcategory", location=location),
        intensity=intensity,
        register=_text(row, "register", location=location),
        localization_strategy=_text(row, "localization_strategy", location=location),
        split=split,
    )


def load_seed_pairs(path: str | Path) -> list[SeedPair]:
    """Load, normalize, and deterministically order curated seed pairs."""

    path = Path(path)
    pairs: list[SeedPair] = []
    seen_ids: set[str] = set()
    seen_surfaces: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            pair = _seed_pair(row, location=location)
            if pair.id in seen_ids:
                raise ValueError(f"{location}: duplicate id {pair.id!r}")
            surface = (pair.ko.casefold(), pair.ja.casefold())
            if surface in seen_surfaces:
                raise ValueError(f"{location}: duplicate ko-ja surface pair")
            seen_ids.add(pair.id)
            seen_surfaces.add(surface)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path}: no seed pairs")
    return sorted(pairs, key=lambda pair: pair.id)


def _jsonl_text(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _training_row(pair: SeedPair) -> dict[str, Any]:
    return {
        "bidirectional": True,
        "category": pair.category,
        "intensity": pair.intensity,
        "ja": pair.ja,
        "ko": pair.ko,
        "localization_strategy": pair.localization_strategy,
        "pair_id": pair.id,
        "provenance": SOURCE_NAME,
        "register": pair.register,
        "subcategory": pair.subcategory,
        # Human-authored seed data is still synthetic rather than naturally
        # occurring parallel text, so the normal synthetic downweight applies.
        "synthetic": True,
    }


def _challenge_rows(pair: SeedPair) -> list[dict[str, Any]]:
    metadata = {
        "category": pair.category,
        "intensity": pair.intensity,
        "localization_strategy": pair.localization_strategy,
        "register": pair.register,
        "seed_id": pair.id,
        "subcategory": pair.subcategory,
    }
    return [
        {
            **metadata,
            "id": f"ko-ja-{pair.id}",
            "reference": pair.ja,
            "source": pair.ko,
            "source_language": "ko",
            "target_language": "ja",
        },
        {
            **metadata,
            "id": f"ja-ko-{pair.id}",
            "reference": pair.ko,
            "source": pair.ja,
            "source_language": "ja",
            "target_language": "ko",
        },
    ]


def build_corpus(
    seed_path: str | Path,
    training_output: str | Path,
    challenge_output: str | Path,
    *,
    report_output: str | Path | None = None,
) -> BuildResult:
    """Build deterministic train/challenge JSONL files with zero seed overlap."""

    pairs = load_seed_pairs(seed_path)
    training_pairs = [pair for pair in pairs if pair.split == "train"]
    challenge_pairs = [pair for pair in pairs if pair.split == "challenge"]
    if not training_pairs or not challenge_pairs:
        raise ValueError("seed file must contain both train and challenge pairs")

    training_rows = [_training_row(pair) for pair in training_pairs]
    challenge_rows = [row for pair in challenge_pairs for row in _challenge_rows(pair)]
    canonical_seed = _jsonl_text([asdict(pair) for pair in pairs])
    seed_sha256 = hashlib.sha256(canonical_seed.encode("utf-8")).hexdigest()

    training_output = Path(training_output)
    challenge_output = Path(challenge_output)
    training_output.parent.mkdir(parents=True, exist_ok=True)
    challenge_output.parent.mkdir(parents=True, exist_ok=True)
    training_output.write_text(_jsonl_text(training_rows), encoding="utf-8", newline="\n")
    challenge_output.write_text(_jsonl_text(challenge_rows), encoding="utf-8", newline="\n")

    result = BuildResult(
        seed_pairs=len(pairs),
        training_pairs=len(training_pairs),
        challenge_pairs=len(challenge_pairs),
        challenge_cases=len(challenge_rows),
        seed_sha256=seed_sha256,
        training_by_category=dict(
            sorted(Counter(pair.category for pair in training_pairs).items())
        ),
        challenge_by_category=dict(
            sorted(Counter(pair.category for pair in challenge_pairs).items())
        ),
    )
    if report_output is not None:
        report_path = Path(report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--challenge-output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_corpus(
        args.seed,
        args.training_output,
        args.challenge_output,
        report_output=args.report,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
