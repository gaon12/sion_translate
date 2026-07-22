"""평가(kjx-evaluate) 로직 검증."""

from __future__ import annotations

import json
from pathlib import Path

from kjx.data.prepare import prepare_dataset
from kjx.evaluation import (
    DirectionResult,
    load_benchmark_pairs,
    load_split_pairs,
    results_as_markdown,
    save_results,
    score_translations,
)
from kjx.tokenizer import KJTokenizer, train_tokenizer

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
    tokenizer = KJTokenizer(model_path)
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


def test_results_saved_as_json_and_markdown(tmp_path: Path) -> None:
    results = [
        DirectionResult("kjx", "ko-ja", 100, 55.5, 22.2, "char"),
        DirectionResult("deepl", "ko-ja", 100, 66.6, 33.3, "char"),
    ]
    table = results_as_markdown(results)
    assert "| kjx | ko-ja | 100 | 55.50 | 22.20 |" in table
    save_results(results, tmp_path / "eval", metadata={"model": "test"})
    payload = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "test"
    assert len(payload["results"]) == 2
    assert (tmp_path / "eval.md").read_text(encoding="utf-8").startswith("| system |")
