from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_review_queue.py"
SPEC = importlib.util.spec_from_file_location("build_review_queue_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD_REVIEW_QUEUE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD_REVIEW_QUEUE
SPEC.loader.exec_module(BUILD_REVIEW_QUEUE)

REPAIR_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "apply_contamination_repairs.py"
)
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "apply_contamination_repairs_test", REPAIR_SCRIPT_PATH
)
assert REPAIR_SPEC is not None and REPAIR_SPEC.loader is not None
APPLY_REPAIRS = importlib.util.module_from_spec(REPAIR_SPEC)
sys.modules[REPAIR_SPEC.name] = APPLY_REPAIRS
REPAIR_SPEC.loader.exec_module(APPLY_REPAIRS)


def test_review_queue_cli_requires_an_explicit_language_graph() -> None:
    parser = BUILD_REVIEW_QUEUE.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "parallel.jsonl", "--output", "review.jsonl"])


def test_contamination_repair_cli_requires_an_explicit_direction() -> None:
    parser = APPLY_REPAIRS.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "parallel.jsonl"])
    args = parser.parse_args(
        [
            "--input",
            "parallel.jsonl",
            "--source-language",
            "KO-kr",
            "--target-language",
            "ja-JP",
        ]
    )
    assert args.source_language == "KO-kr"
    assert args.target_language == "ja-JP"


def test_review_queue_rejects_a_partially_supported_graph_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "review.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_review_queue.py",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(output),
            "--language-pairs",
            "ko",
            "ja",
            "--language-pairs",
            "en",
            "fr",
        ],
    )

    with pytest.raises(SystemExit, match="en→fr"):
        BUILD_REVIEW_QUEUE.main()

    assert not output.exists()
