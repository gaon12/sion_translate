"""Dataset preparation CLI defaults."""

from __future__ import annotations

import pytest

from sion_translate.cli.prepare_data import build_parser
from sion_translate.config import DataConfig


def test_approximate_split_is_the_safe_default_with_explicit_legacy_opt_out() -> None:
    assert DataConfig().approximate_split is True

    base = [
        "--input",
        "data/*.jsonl",
        "--tokenizer",
        "tokenizer.model",
        "--output-dir",
        "dataset",
    ]
    required = [*base, "--language-pair", "ko", "ja"]
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    assert parser.parse_args(required).approximate_split is True
    assert parser.parse_args([*required, "--exact-split"]).approximate_split is False
    assert parser.parse_args([*required, "--no-approximate-split"]).approximate_split is False


def test_language_pair_interfaces_are_explicit_and_mutually_exclusive() -> None:
    parser = build_parser()
    base = [
        "--input",
        "data/*.jsonl",
        "--tokenizer",
        "tokenizer.model",
        "--output-dir",
        "dataset",
    ]

    args = parser.parse_args(
        [*base, "--language-pairs", "sw", "ar", "--language-pairs", "ar", "tr"]
    )
    assert args.language_pair is None
    assert args.language_pairs == [["sw", "ar"], ["ar", "tr"]]
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *base,
                "--language-pair",
                "sw",
                "ar",
                "--language-pairs",
                "ar",
                "tr",
            ]
        )
