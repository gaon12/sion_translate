"""평가(sion-evaluate) 로직 검증."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import sion_translate.cli.evaluate as evaluate_cli
import sion_translate.cli.translate as translate_cli
from sion_translate.cli.evaluate import resolve_evaluation_directions
from sion_translate.cli.translate import (
    resolve_translation_direction,
    resolve_translation_target,
)
from sion_translate.data.prepare import prepare_dataset
from sion_translate.evaluation import (
    DirectionResult,
    load_benchmark_pairs,
    load_split_pairs,
    number_preservation,
    number_preservation_details,
    numeric_corruption,
    numeric_tokens,
    results_as_markdown,
    save_results,
    score_translations,
    structured_tokens,
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


@pytest.mark.parametrize("target_language", ["ja-JP", "zh-Hant", "ko-KR", "th-TH"])
def test_score_translations_uses_script_profiles_for_bcp47_variants(
    target_language: str,
) -> None:
    _chrf, _bleu, tokenize = score_translations(
        ["同じ翻訳です"],
        ["同じ翻訳です"],
        target_language=target_language,
    )

    assert tokenize == "char"


def test_score_translations_honors_an_explicit_latin_script_subtag() -> None:
    _chrf, _bleu, tokenize = score_translations(
        ["isti prevod"],
        ["isti prevod"],
        target_language="sr-Latn-RS",
    )

    assert tokenize == "13a"


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


def test_load_benchmark_pairs_expands_multiple_language_pairs(tmp_path: Path) -> None:
    benchmark = tmp_path / "multilingual.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "ko": "한국어 문장입니다.",
                "ja": "日本語の文です。",
                "en": "An English sentence.",
                "ru": "Русское предложение.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pairs = load_benchmark_pairs(
        [benchmark],
        (("ko", "ja"), ("en", "ru")),
        max_samples_per_direction=10,
    )
    assert set(pairs) == {
        ("ko", "ja"),
        ("ja", "ko"),
        ("en", "ru"),
        ("ru", "en"),
    }
    assert pairs[("en", "ru")] == [("An English sentence.", "Русское предложение.")]


def test_load_benchmark_pairs_respects_an_exact_mixed_direction_graph(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "mixed.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "de": "Ein deutscher Satz.",
                "fr": "Une phrase française.",
                "sw": "Sentensi ya Kiswahili.",
                "ar": "جملة عربية.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    directions = (("de", "fr"), ("fr", "de"), ("sw", "ar"))

    pairs = load_benchmark_pairs(
        [benchmark],
        (("de", "fr"), ("sw", "ar")),
        translation_directions=directions,
        max_samples_per_direction=10,
    )

    assert tuple(pairs) == directions
    assert ("ar", "sw") not in pairs
    assert pairs[("sw", "ar")] == [("Sentensi ya Kiswahili.", "جملة عربية.")]


def test_cli_direction_resolution_uses_only_the_model_graph() -> None:
    directions = (("de", "fr"), ("fr", "de"), ("sw", "ar"))

    assert resolve_evaluation_directions("both", directions) == list(directions)
    assert resolve_evaluation_directions("sw-ar", directions) == [("sw", "ar")]
    with pytest.raises(SystemExit, match="SOURCE TARGET"):
        resolve_evaluation_directions("ar-sw", directions)


def test_cli_direction_resolution_supports_structured_bcp47_pairs() -> None:
    directions = (("zh-Hant", "x-acme"), ("x-acme", "zh-Hant"))

    assert resolve_evaluation_directions(
        ["ZH-hant", "X-ACME"],
        directions,
    ) == [("zh-Hant", "x-acme")]
    with pytest.raises(SystemExit, match="SOURCE TARGET"):
        resolve_evaluation_directions("zh-Hant-x-acme", directions)

    parsed = evaluate_cli.build_parser().parse_args(["--direction", "ZH-hant", "X-ACME"])
    assert parsed.direction == ["ZH-hant", "X-ACME"]
    assert evaluate_cli._direction_label("x-a", "x-b-x-c") == "x-a→x-b-x-c"
    assert evaluate_cli._direction_label("x-a-x-b", "x-c") == "x-a-x-b→x-c"


def test_cli_direction_resolution_rejects_canonical_model_duplicates() -> None:
    directions = (("zh-hant", "X-ACME"), ("zh-Hant", "x-acme"))

    with pytest.raises(SystemExit, match="중복"):
        resolve_evaluation_directions(None, directions)


def test_comparison_output_requires_an_exact_line_count(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.txt"
    comparison.write_text("first\nsecond\nthird\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"3줄 != 평가쌍 2개"):
        evaluate_cli._read_comparison_lines(comparison, expected_count=2)

    comparison.write_text("first\nsecond\n", encoding="utf-8")
    assert evaluate_cli._read_comparison_lines(comparison, expected_count=2) == [
        "first",
        "second",
    ]


def test_comparison_specs_require_distinct_trimmed_names_and_paths() -> None:
    assert evaluate_cli._parse_comparison_specs([" Service = output.txt "]) == [
        ("Service", "output.txt")
    ]
    with pytest.raises(SystemExit, match="형식"):
        evaluate_cli._parse_comparison_specs([" =output.txt"])
    with pytest.raises(SystemExit, match="중복"):
        evaluate_cli._parse_comparison_specs(["Service=a.txt", " service =b.txt"])
    with pytest.raises(SystemExit, match="예약"):
        evaluate_cli._parse_comparison_specs(["SION=external.txt"])


def test_result_markdown_escapes_untrusted_labels() -> None:
    rendered = results_as_markdown(
        [
            DirectionResult(
                "A|B\nC",
                "x-a|x-b",
                1,
                100.0,
                100.0,
                "13a",
                100.0,
                1,
                1,
                0,
            )
        ]
    )

    assert "| A\\|B<br>C | x-a\\|x-b | 1 |" in rendered


def test_translate_target_resolution_uses_reachable_model_edges() -> None:
    directions = (("de", "fr"), ("fr", "de"), ("sw", "ar"))

    with pytest.raises(SystemExit, match="--from LANG 또는 --to LANG"):
        resolve_translation_target(None, None, directions)
    assert resolve_translation_target(None, None, (("de", "fr"),)) == "fr"
    assert resolve_translation_target(None, "sw", directions) == "ar"
    assert resolve_translation_target("de", "fr", directions) == "de"
    with pytest.raises(SystemExit, match="출발하는 학습 방향"):
        resolve_translation_target(None, "ar", directions)
    with pytest.raises(SystemExit, match="학습된 target"):
        resolve_translation_target("sw", None, directions)


def test_translate_target_resolution_requires_target_for_branching_source() -> None:
    directions = (("zh-Hant", "x-acme"), ("zh-Hant", "x-other"))

    with pytest.raises(SystemExit, match="--to LANG"):
        resolve_translation_target(None, "ZH-hant", directions)
    assert resolve_translation_target("X-OTHER", "zh-Hant", directions) == "x-other"


def test_translate_direction_resolution_fills_only_unambiguous_endpoints() -> None:
    directions = (("ko", "ja"), ("en", "de"), ("fr", "de"))

    with pytest.raises(SystemExit, match="--from LANG 또는 --to LANG"):
        resolve_translation_direction(None, None, directions)
    assert resolve_translation_direction(None, None, (("ko", "ja"),)) == ("ko", "ja")
    assert resolve_translation_direction("ja", None, directions) == ("ko", "ja")
    assert resolve_translation_direction(None, "EN", directions) == ("en", "de")
    with pytest.raises(SystemExit, match="--from LANG"):
        resolve_translation_direction("de", None, directions)


def test_translate_target_resolution_canonicalizes_bcp47_cli_identities() -> None:
    directions = (("zh-Hant", "x-acme"), ("de", "zh-Hant"))

    assert resolve_translation_target(None, "ZH-hant", directions) == "x-acme"
    assert resolve_translation_target("X-ACME", "zh-Hant", directions) == "x-acme"

    with pytest.raises(SystemExit, match="학습되지 않은 방향"):
        resolve_translation_target("zh-Hant", "x-acme", directions)


def test_translate_target_resolution_rejects_canonical_model_duplicates() -> None:
    directions = (("zh-hant", "X-ACME"), ("zh-Hant", "x-acme"))

    with pytest.raises(SystemExit, match="중복"):
        resolve_translation_target(None, None, directions)


def test_translate_main_passes_canonical_cli_languages_to_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTranslator:
        translation_directions = (("zh-Hant", "x-acme"),)

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def translate(self, _lines: object, **kwargs: object) -> list[str]:
            captured.update(kwargs)
            return ["translated"]

    config = SimpleNamespace(
        training=SimpleNamespace(output_dir="exports"),
        data=SimpleNamespace(tokenizer_model="tokenizer.model", glossary=None),
    )
    monkeypatch.setattr(translate_cli, "configure_stdio", lambda: None)
    monkeypatch.setattr(translate_cli, "load_raw_config", lambda *_args: {})
    monkeypatch.setattr(translate_cli, "config_from_raw", lambda _raw: config)
    monkeypatch.setattr(translate_cli, "Translator", FakeTranslator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sion-translate",
            "--model",
            "model.pt",
            "--from",
            "ZH-hant",
            "--to",
            "X-ACME",
            "hello",
        ],
    )

    translate_cli.main()

    assert captured["source_language"] == "zh-Hant"
    assert captured["target_language"] == "x-acme"


def test_translate_main_passes_a_uniquely_implied_source_to_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTranslator:
        translation_directions = (("ko", "ja"), ("en", "de"))

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def translate(self, _lines: object, **kwargs: object) -> list[str]:
            captured.update(kwargs)
            return ["translated"]

    config = SimpleNamespace(
        training=SimpleNamespace(output_dir="exports"),
        data=SimpleNamespace(tokenizer_model="tokenizer.model", glossary=None),
    )
    monkeypatch.setattr(translate_cli, "configure_stdio", lambda: None)
    monkeypatch.setattr(translate_cli, "load_raw_config", lambda *_args: {})
    monkeypatch.setattr(translate_cli, "config_from_raw", lambda _raw: config)
    monkeypatch.setattr(translate_cli, "Translator", FakeTranslator)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sion-translate", "--model", "model.pt", "--to", "DE", "hello"],
    )

    translate_cli.main()

    assert captured["source_language"] == "en"
    assert captured["target_language"] == "de"


def test_evaluate_main_records_model_owned_language_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTranslator:
        language_pairs = (("sw", "ar"),)
        translation_directions = (("sw", "ar"),)
        tokenizer = object()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def translate(self, *_args: object, **_kwargs: object) -> list[str]:
            return ["الهدف"]

    config = SimpleNamespace(
        training=SimpleNamespace(output_dir="exports"),
        data=SimpleNamespace(tokenizer_model="tokenizer.model", glossary=None),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(evaluate_cli, "load_raw_config", lambda *_args: {})
    monkeypatch.setattr(evaluate_cli, "config_from_raw", lambda _raw: config)
    monkeypatch.setattr(evaluate_cli, "Translator", FakeTranslator)
    monkeypatch.setattr(
        evaluate_cli,
        "load_benchmark_pairs",
        lambda *_args, **_kwargs: {("sw", "ar"): [("chanzo", "الهدف")]},
    )
    monkeypatch.setattr(
        evaluate_cli,
        "score_translations",
        lambda *_args, **_kwargs: (100.0, 100.0, "char"),
    )

    def capture_results(
        _results: object,
        _output: object,
        *,
        metadata: dict[str, object],
    ) -> None:
        captured.update(metadata)

    monkeypatch.setattr(evaluate_cli, "save_results", capture_results)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sion-evaluate",
            "--benchmark",
            "benchmark.jsonl",
            "--model",
            "model.pt",
            "--output",
            str(tmp_path / "evaluation"),
        ],
    )

    evaluate_cli.main()

    assert captured["language_pairs"] == [["sw", "ar"]]


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
    # Simulate an indexed generation that predates manifest-owned language
    # identity while preserving its authenticated payload inventory. The model
    # graph is the only trustworthy identity for this legacy holdout.
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "sion-indexed-parallel-v5"
    manifest.pop("language_pair")
    manifest.pop("language_pairs")
    manifest.pop("languages")
    manifest.pop("language_to_id")
    manifest.pop("preprocessing_schema")
    manifest["fingerprint"].pop("preprocessing_schema")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    raw_fingerprint_path = dataset_dir / "raw_fingerprint.json"
    raw_fingerprint = json.loads(raw_fingerprint_path.read_text(encoding="utf-8"))
    raw_fingerprint.pop("preprocessing_schema")
    raw_fingerprint_path.write_text(json.dumps(raw_fingerprint), encoding="utf-8")

    pairs = load_split_pairs(
        dataset_dir,
        "test",
        tokenizer,
        model_language_pairs=(("en", "de"),),
        max_samples_per_direction=10,
    )
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


def test_numeric_corruption_counts_values_nothing_licenses() -> None:
    # 실측 결함: 원래 값을 남긴 채 새 값을 덧붙이는 발명.
    assert numeric_corruption("1개 주세요", "1つください", "1, 999つください") == (1, 0)
    # 값 자체가 바뀌면 발명 하나와 누락 하나다.
    assert numeric_corruption("가격 100", "価格 100円", "価格 200円") == (1, 1)
    # 원문과 정답이 합의한 값을 빠뜨린 경우.
    assert numeric_corruption("250mg씩 48시간", "250mgずつ48時間", "250mgずつ") == (0, 1)
    # 값을 지킨 번역은 변조가 없다.
    assert numeric_corruption("가격 100", "価格 100円", "価格 100円") == (0, 0)


def test_a_number_only_the_reference_spells_out_is_not_an_invention() -> None:
    """한국어는 수를 한글로 자주 적는다. 원문만으로 판정하면 정상 번역이 걸린다."""

    assert numeric_corruption("하루 두 번", "1日2回", "1日2回") == (0, 0)
    # 정답에도 원문에도 없는 값은 여전히 발명이다.
    assert numeric_corruption("하루 두 번", "1日2回", "1日5回") == (1, 0)


def test_structured_tokens_share_the_reversible_protection_parser() -> None:
    source = (
        "https://example.com/path의 user@example.com에게 "
        "{account_name} 값 retry_limit과 250mg을 전송"
    )
    tokens = structured_tokens(source)

    assert any(token.startswith("url\0") for token in tokens)
    assert any(token.startswith("email\0") for token in tokens)
    assert any(token.startswith("placeholder\0") for token in tokens)
    assert any(token.startswith("identifier\0") for token in tokens)
    assert not any(token.startswith("number\0") for token in tokens)


def test_number_preservation_catches_altered_values() -> None:
    # 실제 관측된 오역: 용량과 금액이 그럴듯한 다른 값으로 바뀐다.
    sources = [
        "1회 250mg씩 복용하세요.",
        "합계 금액은 38,720엔입니다.",
        "48시간 이상 간격을 두세요.",
    ]
    corrupted = [
        "1회 1200mg씩 복용하세요.",
        "합계 금액은 38,000엔입니다.",
        "48시간 이상 간격을 두세요.",
    ]
    f1, exact = number_preservation(corrupted, sources)
    assert exact == 1
    assert f1 < 70.0

    perfect_f1, perfect_exact = number_preservation(sources, sources)
    assert perfect_f1 == pytest.approx(100.0)
    assert perfect_exact == len(sources)


def test_number_preservation_scores_missing_and_invented_numbers() -> None:
    # 누락(재현율)과 환각(정밀도)이 모두 F1 을 낮춰야 한다.
    dropped = number_preservation(["금액은 미정입니다."], sources=["금액은 38,720엔입니다."])[0]
    invented = number_preservation(
        ["금액은 38,720엔이고 수량은 50개입니다."],
        sources=["금액은 38,720엔입니다."],
    )[0]
    assert dropped == pytest.approx(0.0)
    assert 0.0 < invented < 100.0


def test_number_preservation_treats_numberless_pairs_as_clean() -> None:
    f1, exact = number_preservation(["번역문입니다."], sources=["원문입니다."])
    assert f1 == pytest.approx(100.0)
    assert exact == 0


def test_number_preservation_does_not_dilute_numeric_failures() -> None:
    hypotheses = ["숫자 없는 번역"] * 19 + ["용량은 1200mg입니다."]
    sources = ["숫자 없는 원문"] * 19 + ["용량은 250mg입니다."]

    result = number_preservation_details(hypotheses, sources=sources)

    assert result.f1 == pytest.approx(0.0)
    assert result.exact == 0
    assert result.samples == 1
    assert result.inventions == 1


def test_number_preservation_reports_invented_numbers_separately() -> None:
    result = number_preservation_details(
        ["금액은 38,720엔입니다."],
        sources=["금액은 미정입니다."],
    )

    assert result.f1 == pytest.approx(0.0)
    assert result.exact == 0
    assert result.samples == 1
    assert result.inventions == 1


def test_number_preservation_reports_additional_invention_when_source_has_number() -> None:
    result = number_preservation_details(
        ["수량은 1개이고 가격은 999엔입니다."],
        sources=["수량은 1개입니다."],
    )

    assert 0.0 < result.f1 < 100.0
    assert result.exact == 0
    assert result.samples == 1
    assert result.inventions == 1


def test_number_preservation_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="수가 다릅니다"):
        number_preservation(["a"], sources=["a", "b"])


def test_results_saved_as_json_and_markdown(tmp_path: Path) -> None:
    results = [
        DirectionResult("sion", "ko-ja", 100, 55.5, 22.2, "char", 91.0, 8, 10, 1),
        DirectionResult("deepl", "ko-ja", 100, 66.6, 33.3, "char", 99.5, 9, 10, 0),
    ]
    table = results_as_markdown(results)
    assert "| sion | ko-ja | 100 | 55.50 | 22.20 | 91.00 | 8/10 | 1 |" in table
    save_results(results, tmp_path / "eval", metadata={"model": "test"})
    payload = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "test"
    assert len(payload["results"]) == 2
    assert (tmp_path / "eval.md").read_text(encoding="utf-8").startswith("| system |")
