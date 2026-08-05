from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "data" / "build_expressive_cultural_corpus.py"
SEED_PATH = REPOSITORY_ROOT / "examples" / "expressive_cultural_seed_pairs.jsonl"
CHALLENGE_PATH = REPOSITORY_ROOT / "examples" / "expressive_cultural_cases.jsonl"

SPEC = importlib.util.spec_from_file_location("build_expressive_cultural_corpus", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builder_is_deterministic_and_matches_the_shipped_challenge(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = BUILD.build_corpus(
        SEED_PATH,
        first / "train.jsonl",
        first / "challenge.jsonl",
        report_output=first / "report.json",
    )
    BUILD.build_corpus(
        SEED_PATH,
        second / "train.jsonl",
        second / "challenge.jsonl",
        report_output=second / "report.json",
    )

    for name in ("train.jsonl", "challenge.jsonl", "report.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "challenge.jsonl").read_bytes() == CHALLENGE_PATH.read_bytes()
    assert result.training_pairs == 18
    assert result.challenge_pairs == 12
    assert result.challenge_cases == 24
    assert len(result.seed_sha256) == 64


def test_training_and_challenge_splits_do_not_leak(tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    challenge = tmp_path / "challenge.jsonl"
    result = BUILD.build_corpus(SEED_PATH, training, challenge)
    train_rows = _rows(training)
    challenge_rows = _rows(challenge)

    assert result.training_by_category == {
        "idiom_culture": 6,
        "interjection_moan": 6,
        "profanity_slang": 6,
    }
    assert all(row["synthetic"] is True and row["bidirectional"] is True for row in train_rows)
    assert all(1 <= row["intensity"] <= 5 for row in train_rows)
    assert all(row["register"] and row["localization_strategy"] for row in train_rows)
    train_surfaces = {(row["ko"], row["ja"]) for row in train_rows}
    challenge_surfaces = {
        (row["source"], row["reference"])
        if row["source_language"] == "ko"
        else (row["reference"], row["source"])
        for row in challenge_rows
    }
    assert train_surfaces.isdisjoint(challenge_surfaces)
    assert {row["source_language"] for row in challenge_rows} == {"ko", "ja"}
    assert all(
        row["seed_id"] not in {item["pair_id"] for item in train_rows} for row in challenge_rows
    )


def test_seed_validation_rejects_bad_intensity_and_duplicate_ids(tmp_path: Path) -> None:
    base = {
        "id": "pair-1",
        "ko": "아!",
        "ja": "あっ！",
        "category": "interjection_moan",
        "subcategory": "reaction",
        "intensity": 1,
        "register": "casual_spoken",
        "localization_strategy": "acoustic_pragmatic_equivalent",
        "split": "train",
    }
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps({**base, "intensity": 0}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intensity"):
        BUILD.load_seed_pairs(invalid)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "".join(json.dumps(base, ensure_ascii=False) + "\n" for _ in range(2)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        BUILD.load_seed_pairs(duplicate)
