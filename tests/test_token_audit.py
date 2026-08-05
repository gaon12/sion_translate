from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.token_audit import audit_token_exposure
from sion_translate.tokenizer import train_tokenizer


def _corpus(path: Path) -> None:
    rows = [
        {"ko": "헉, 정말 깜짝 놀랐어!", "ja": "えっ、本当にびっくりした！"},
        {"ko": "둘은 붕어빵처럼 닮았다.", "ja": "二人は瓜二つだ。"},
        {"ko": "젠장, 또 늦었네.", "ja": "ちくしょう、また遅れた。"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def corpus_and_tokenizer(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "parallel.jsonl"
    _corpus(corpus)
    tokenizer = train_tokenizer(
        [str(corpus)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=0,
        seed_sentencepiece_size=100,
        validation_fraction=0.0,
        test_fraction=0.0,
        num_workers=1,
        num_threads=1,
    )
    return corpus, tokenizer


def test_audit_reports_each_translation_direction(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure([str(corpus)], tokenizer, rare_threshold=2)

    assert report["complete_scan"] is True
    assert report["physical_pairs"] == 3
    assert report["virtual_translation_examples"] == 6
    assert report["directions"]["ko-ja"]["examples"] == 3
    assert report["directions"]["ja-ko"]["examples"] == 3
    assert report["languages"]["ko"]["target_enabled"] is True
    assert report["languages"]["ja"]["target_enabled"] is True
    assert report["languages"]["ko"]["byte_fallback_rate"] >= 0.0
    assert report["global_target_frequency"]["observed_pieces"] > 0
    assert (
        report["global_target_frequency"]["eligible_pieces"]
        == report["global_target_frequency"]["observed_pieces"]
        + report["global_target_frequency"]["unused_pieces"]
    )
    assert report["lowest_global_target_exposure"]


def test_source_only_language_is_not_misreported_as_undertrained_target(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        source_only_languages=["ko"],
        rare_threshold=2,
    )

    assert set(report["directions"]) == {"ko-ja"}
    assert report["virtual_translation_examples"] == 3
    korean = report["languages"]["ko"]
    assert korean["target_enabled"] is False
    assert korean["target_frequency"]["eligible_pieces"] == 0
    assert korean["lowest_target_exposure"] == []


def test_prefix_scan_is_labelled_incomplete(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        max_physical_pairs=1,
    )
    assert report["physical_pairs"] == 1
    assert report["complete_scan"] is False


def test_invalid_audit_limits_are_rejected(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    with pytest.raises(ValueError, match="rare_threshold"):
        audit_token_exposure([str(corpus)], tokenizer, rare_threshold=0)
    with pytest.raises(ValueError, match="max_physical_pairs"):
        audit_token_exposure([str(corpus)], tokenizer, max_physical_pairs=-1)
