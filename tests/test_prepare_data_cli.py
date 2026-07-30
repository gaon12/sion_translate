"""Dataset preparation CLI defaults."""

from __future__ import annotations

from sion_translate.cli.prepare_data import build_parser
from sion_translate.config import DataConfig


def test_approximate_split_is_the_safe_default_with_explicit_legacy_opt_out() -> None:
    assert DataConfig().approximate_split is True

    required = [
        "--input",
        "data/*.jsonl",
        "--tokenizer",
        "tokenizer.model",
        "--output-dir",
        "dataset",
    ]
    parser = build_parser()
    assert parser.parse_args(required).approximate_split is True
    assert parser.parse_args([*required, "--exact-split"]).approximate_split is False
    assert parser.parse_args([*required, "--no-approximate-split"]).approximate_split is False
