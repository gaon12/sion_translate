"""foundation 코퍼스의 디코더 타깃 노출 감사.

복원 과제의 정답은 원문 전체이므로 단일어 코퍼스의 **모든 토큰이 디코더
타깃**입니다. 병렬 감사만 보고 어휘를 판단하면 두 방향으로 틀립니다 —
foundation 이 충분히 노출시키는 조각을 위험하다고 하거나, 단일어가 어휘에
밀어 넣은 조각이 번역 학습에서 안 나오는 것을 놓칩니다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sion_translate.data.monolingual import discover_monolingual_sources
from sion_translate.token_audit import (
    audit_monolingual_token_exposure,
    combine_target_exposure,
)
from sion_translate.tokenizer import train_tokenizer


@pytest.fixture(scope="module")
def tokenizer_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("audit_tokenizer")
    shard = directory / "pairs.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(400):
            handle.write(
                json.dumps(
                    {
                        "ko": f"한국어 문장 {index} 입니다 그리고 조금 더 깁니다",
                        "ja": f"日本語の文 {index} です そしてもう少し長いです",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return train_tokenizer(
        [str(shard)],
        directory / "out",
        vocab_size=700,
        num_workers=1,
        language_pair=["ko", "ja"],
    )


def _corpus(tmp_path, ko_lines=120, ja_lines=80):
    root = tmp_path / "corpus"
    (root / "ko").mkdir(parents=True)
    (root / "ja").mkdir(parents=True)
    (root / "ko" / "a.txt").write_text(
        "\n".join(f"한국어 단일어 문장 {i} 입니다 조금 더 깁니다" for i in range(ko_lines)) + "\n",
        encoding="utf-8",
    )
    (root / "ja" / "a.txt").write_text(
        "\n".join(f"日本語の単言語文 {i} です もう少し長いです" for i in range(ja_lines)) + "\n",
        encoding="utf-8",
    )
    return discover_monolingual_sources(root, ["ko", "ja"])


def test_every_language_is_counted(tmp_path, tokenizer_model) -> None:
    report = audit_monolingual_token_exposure(_corpus(tmp_path), tokenizer_model)
    assert report["accepted_lines"] == {"ko": 120, "ja": 80}
    assert set(report["per_language"]) == {"ko", "ja"}
    assert report["decoder_target_totals"]["total_occurrences"] > 0
    assert report["complete_scan"] is True


def test_short_and_long_lines_are_dropped_by_reason(tmp_path, tokenizer_model) -> None:
    root = tmp_path / "corpus"
    (root / "ko").mkdir(parents=True)
    (root / "ko" / "a.txt").write_text(
        "\n".join(["짧다", "충분히 긴 한국어 문장입니다", "가" * 9000]) + "\n",
        encoding="utf-8",
    )
    report = audit_monolingual_token_exposure(
        discover_monolingual_sources(root, ["ko"]),
        tokenizer_model,
    )
    assert report["accepted_lines"]["ko"] == 1
    assert report["dropped_lines"]["ko:too_short"] == 1
    assert report["dropped_lines"]["ko:too_long"] == 1


def test_a_prefix_sample_is_labelled_as_incomplete(tmp_path, tokenizer_model) -> None:
    """표본은 빠른 preflight 용이지 어휘가 안전하다는 선언이 아니다."""
    report = audit_monolingual_token_exposure(
        _corpus(tmp_path),
        tokenizer_model,
        max_lines_per_language=10,
    )
    assert report["complete_scan"] is False
    assert report["accepted_lines"] == {"ko": 10, "ja": 10}


def test_special_pieces_are_excluded_from_the_verdict(tmp_path, tokenizer_model) -> None:
    report = audit_monolingual_token_exposure(_corpus(tmp_path), tokenizer_model)
    totals = report["decoder_target_totals"]
    assert totals["eligible_pieces"] < report["vocab_size"]


def test_an_empty_corpus_is_refused(tmp_path, tokenizer_model) -> None:
    with pytest.raises(ValueError, match="읽을 수 있는 파일이 없습니다"):
        audit_monolingual_token_exposure(
            discover_monolingual_sources(tmp_path / "absent", ["ko"]),
            tokenizer_model,
        )


# ── 두 단계를 합쳐야 답이 나온다 ────────────────────────────────────────


def test_foundation_rescues_pieces_the_parallel_corpus_barely_targets(
    tmp_path,
    tokenizer_model,
) -> None:
    """이 함수가 존재하는 이유.

    foundation 이 먼저 돌면 출력 임베딩은 두 단계 모두에서 신호를 받습니다.
    병렬 감사만 보고 판정하면 조각을 잘못 죽입니다.
    """
    monolingual = audit_monolingual_token_exposure(_corpus(tmp_path), tokenizer_model)
    vocab_size = monolingual["vocab_size"]

    # 병렬 코퍼스가 거의 만들지 않는 조각을 흉내 낸다.
    parallel = np.zeros(vocab_size, dtype=np.uint64)
    parallel[:] = 1

    combined = combine_target_exposure(
        parallel,
        monolingual["counts"],
        tokenizer_model,
        rare_threshold=25,
    )
    assert combined["rescued_by_foundation"] > 0
    assert combined["totals"]["total_occurrences"] > int(
        monolingual["decoder_target_totals"]["total_occurrences"]
    )


def test_pieces_neither_stage_targets_are_still_reported(tmp_path, tokenizer_model) -> None:
    monolingual = audit_monolingual_token_exposure(_corpus(tmp_path), tokenizer_model)
    vocab_size = monolingual["vocab_size"]
    combined = combine_target_exposure(
        np.zeros(vocab_size, dtype=np.uint64),
        monolingual["counts"],
        tokenizer_model,
    )
    assert combined["still_below_threshold"] > 0
    assert combined["lowest_target_exposure"]


def test_mismatched_vocabularies_are_refused(tmp_path, tokenizer_model) -> None:
    with pytest.raises(ValueError, match="same vocabulary"):
        combine_target_exposure(
            np.zeros(10, dtype=np.uint64),
            np.zeros(11, dtype=np.uint64),
            tokenizer_model,
        )


def test_a_count_vector_of_the_wrong_size_is_refused(tmp_path, tokenizer_model) -> None:
    with pytest.raises(ValueError, match="tokenizer vocabulary size"):
        combine_target_exposure(
            np.zeros(10, dtype=np.uint64),
            np.zeros(10, dtype=np.uint64),
            tokenizer_model,
        )
