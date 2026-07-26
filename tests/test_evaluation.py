"""평가(sion-evaluate) 로직 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.data.prepare import prepare_dataset
from sion_translate.evaluation import (
    DirectionResult,
    load_benchmark_pairs,
    load_split_pairs,
    number_preservation,
    numeric_tokens,
    results_as_markdown,
    save_results,
    score_translations,
)
from sion_translate.tokenizer import SionTokenizer, train_tokenizer

from tests.test_language_pair import write_en_de_jsonl


def test_score_translations_identity_is_perfect() -> None:
    # 정답과 완전히 같은 번역은 만점이어야 한다.
    chrf, bleu, tokenize = score_translations(
        ["오늘 날씨가 좋다"], ["오늘 날씨가 좋다"], target_language="ja"
    )
    assert chrf == 100.0
    assert round(bleu) == 100
    assert tokenize == "char"  # 일본어는 문자 단위 BLEU
    _, _, tokenize_en = score_translations(["hello"], ["hello"], target_language="en")
    assert tokenize_en == "13a"  # 라틴 문자 언어는 표준 토큰화
    # 완전히 다른 번역은 점수가 낮아야 한다.
    chrf_bad, _, _ = score_translations(
        ["전혀 다른 문장"], ["오늘 날씨가 좋다"], target_language="ja"
    )
    assert chrf_bad < 30.0


def test_load_benchmark_pairs_builds_both_directions(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    rows = [{"ko": f"한국어 {i}", "ja": f"日本語 {i}"} for i in range(5)]
    benchmark.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pairs = load_benchmark_pairs([benchmark], ("ko", "ja"), max_samples_per_direction=3)
    assert len(pairs[("ko", "ja")]) == 3  # max_samples 상한 적용
    # 역방향은 (원문, 정답)이 뒤집혀야 한다.
    assert pairs[("ja", "ko")][0] == (pairs[("ko", "ja")][0][1], pairs[("ko", "ja")][0][0])


def test_load_split_pairs_round_trips_text(tmp_path: Path) -> None:
    """holdout split 의 토큰 id 가 다시 읽을 수 있는 텍스트로 복원되어야 한다."""
    source = tmp_path / "corpus.jsonl"
    write_en_de_jsonl(source)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pair=("en", "de"),
    )
    tokenizer = SionTokenizer(model_path)
    dataset_dir = tmp_path / "dataset"
    stats = prepare_dataset(
        [str(source)],
        model_path,
        dataset_dir,
        validation_fraction=0.15,
        test_fraction=0.15,
        dedup_backend="memory",
        language_pair=("en", "de"),
    )
    assert stats.test > 0
    pairs = load_split_pairs(dataset_dir, "test", tokenizer, max_samples_per_direction=10)
    forward = pairs[("en", "de")]
    backward = pairs[("de", "en")]
    assert len(forward) == min(stats.test, 10)
    assert len(backward) == len(forward)
    # 복원된 텍스트에 실제 단어가 들어 있어야 한다 (디코딩 성공 확인).
    assert any("number" in source_text for source_text, _ in forward)
    assert any("Nummer" in reference for _, reference in forward)


def test_numeric_tokens_ignores_digit_grouping_and_width() -> None:
    # 콤마 표기와 전각/반각 차이는 값이 같으므로 오역으로 세면 안 된다.
    assert numeric_tokens("38,720원") == numeric_tokens("38720원") == ["38720"]
    assert numeric_tokens("２０２６년 ４월") == ["2026", "4"]
    assert numeric_tokens("숫자가 없는 문장") == []


def test_numeric_tokens_sees_counters_but_not_identifiers() -> None:
    # 조사·단위가 붙은 한 자리 숫자(날짜, 복용 횟수)도 값으로 잡아야 한다.
    assert numeric_tokens("1회 250mg씩, 48시간 간격") == ["1", "250", "48"]
    assert numeric_tokens("접수는 4월 30일까지") == ["4", "30"]
    # 식별자 안의 숫자는 값이 아니므로 세지 않는다.
    assert numeric_tokens("config.json의 retry_limit") == []
    assert numeric_tokens("utf8 인코딩") == []


def test_number_preservation_catches_altered_values() -> None:
    # 실제 관측된 오역: 용량과 금액이 그럴듯한 다른 값으로 바뀐다.
    references = [
        "1회 250mg씩 복용하세요.",
        "합계 금액은 38,720엔입니다.",
        "48시간 이상 간격을 두세요.",
    ]
    corrupted = [
        "1회 1200mg씩 복용하세요.",
        "합계 금액은 38,000엔입니다.",
        "48시간 이상 간격을 두세요.",
    ]
    f1, exact = number_preservation(corrupted, references)
    assert exact == 1
    assert f1 < 70.0

    perfect_f1, perfect_exact = number_preservation(references, references)
    assert perfect_f1 == pytest.approx(100.0)
    assert perfect_exact == len(references)


def test_number_preservation_scores_missing_and_invented_numbers() -> None:
    # 누락(재현율)과 환각(정밀도)이 모두 F1 을 낮춰야 한다.
    dropped = number_preservation(["금액은 미정입니다."], ["금액은 38,720엔입니다."])[0]
    invented = number_preservation(
        ["금액은 38,720엔이고 수량은 50개입니다."], ["금액은 38,720엔입니다."]
    )[0]
    assert dropped == pytest.approx(0.0)
    assert 0.0 < invented < 100.0


def test_number_preservation_treats_numberless_pairs_as_clean() -> None:
    f1, exact = number_preservation(["번역문입니다."], ["정답 문장입니다."])
    assert f1 == pytest.approx(100.0)
    assert exact == 1


def test_number_preservation_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="수가 다릅니다"):
        number_preservation(["a"], ["a", "b"])


def test_results_saved_as_json_and_markdown(tmp_path: Path) -> None:
    results = [
        DirectionResult("sion", "ko-ja", 100, 55.5, 22.2, "char", 91.0, 88),
        DirectionResult("deepl", "ko-ja", 100, 66.6, 33.3, "char", 99.5, 99),
    ]
    table = results_as_markdown(results)
    assert "| sion | ko-ja | 100 | 55.50 | 22.20 | 91.00 | 88/100 |" in table
    save_results(results, tmp_path / "eval", metadata={"model": "test"})
    payload = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "test"
    assert len(payload["results"]) == 2
    assert (tmp_path / "eval.md").read_text(encoding="utf-8").startswith("| system |")
