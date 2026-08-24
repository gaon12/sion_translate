"""Verify the shared ``sion-evaluate`` quality-measurement logic."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import sion_translate.evaluation as evaluation_module
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
    # A hypothesis identical to the reference must receive a perfect score.
    chrf, bleu, tokenize = score_translations(
        ["오늘 날씨가 좋다"], ["오늘 날씨가 좋다"], target_language="ja"
    )
    assert chrf == 100.0
    assert round(bleu) == 100
    assert tokenize == "char"  # Japanese uses character-level BLEU.
    _, _, tokenize_en = score_translations(["hello"], ["hello"], target_language="en")
    assert tokenize_en == "13a"  # Latin-script languages use standard tokenization.
    # Unrelated text must receive a low score.
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
    assert len(pairs[("ko", "ja")]) == 3  # Apply the configured sample cap.
    # Reverse-direction pairs must exchange source and reference.
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

    with pytest.raises(SystemExit, match="duplicate"):
        resolve_evaluation_directions(None, directions)


def test_comparison_output_requires_an_exact_line_count(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.txt"
    comparison.write_text("first\nsecond\nthird\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"3 translation lines != 2 evaluation pairs"):
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
    with pytest.raises(SystemExit, match="NAME=FILE format"):
        evaluate_cli._parse_comparison_specs([" =output.txt"])
    with pytest.raises(SystemExit, match="Duplicate"):
        evaluate_cli._parse_comparison_specs(["Service=a.txt", " service =b.txt"])
    with pytest.raises(SystemExit, match="reserved"):
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

    with pytest.raises(SystemExit, match="--from LANG or --to LANG"):
        resolve_translation_target(None, None, directions)
    assert resolve_translation_target(None, None, (("de", "fr"),)) == "fr"
    assert resolve_translation_target(None, "sw", directions) == "ar"
    assert resolve_translation_target("de", "fr", directions) == "de"
    with pytest.raises(SystemExit, match="No trained direction starts"):
        resolve_translation_target(None, "ar", directions)
    with pytest.raises(SystemExit, match="not a trained target"):
        resolve_translation_target("sw", None, directions)


def test_translate_target_resolution_requires_target_for_branching_source() -> None:
    directions = (("zh-Hant", "x-acme"), ("zh-Hant", "x-other"))

    with pytest.raises(SystemExit, match="--to LANG"):
        resolve_translation_target(None, "ZH-hant", directions)
    assert resolve_translation_target("X-OTHER", "zh-Hant", directions) == "x-other"


def test_translate_direction_resolution_fills_only_unambiguous_endpoints() -> None:
    directions = (("ko", "ja"), ("en", "de"), ("fr", "de"))

    with pytest.raises(SystemExit, match="--from LANG or --to LANG"):
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

    with pytest.raises(SystemExit, match="not a trained direction"):
        resolve_translation_target("zh-Hant", "x-acme", directions)


def test_translate_target_resolution_rejects_canonical_model_duplicates() -> None:
    directions = (("zh-hant", "X-ACME"), ("zh-Hant", "x-acme"))

    with pytest.raises(SystemExit, match="duplicate"):
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
    """Holdout token IDs must decode back into readable text."""
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
    # A real word in the reconstructed text confirms successful decoding.
    assert any("number" in source_text for source_text, _ in forward)
    assert any("Nummer" in reference for _, reference in forward)


def test_numeric_tokens_ignores_digit_grouping_and_width() -> None:
    # Grouping commas and full-/half-width forms do not change the numeric value.
    assert numeric_tokens("38,720원") == numeric_tokens("38720원") == ["38720"]
    assert numeric_tokens("２０２６년 ４월") == ["2026", "4"]
    assert numeric_tokens("숫자가 없는 문장") == []


def test_numeric_tokens_sees_counters_but_not_identifiers() -> None:
    # Capture single digits followed by particles or units, including dates and doses.
    assert numeric_tokens("1회 250mg씩, 48시간 간격") == ["1", "250", "48"]
    assert numeric_tokens("접수는 4월 30일까지") == ["4", "30"]
    # Digits embedded in identifiers are not standalone values.
    assert numeric_tokens("config.json의 retry_limit") == []
    assert numeric_tokens("utf8 인코딩") == []


def test_numeric_corruption_counts_values_nothing_licenses() -> None:
    # Measured defect: retain the original value while inventing an additional one.
    assert numeric_corruption("1개 주세요", "1つください", "1, 999つください") == (1, 0)
    # Replacing a value creates one invention and one drop.
    assert numeric_corruption("가격 100", "価格 100円", "価格 200円") == (1, 1)
    # Dropping a value agreed upon by source and reference counts as omission.
    assert numeric_corruption("250mg씩 48시간", "250mgずつ48時間", "250mgずつ") == (0, 1)
    # A hypothesis that preserves the value has no numeric corruption.
    assert numeric_corruption("가격 100", "価格 100円", "価格 100円") == (0, 0)


def test_a_number_only_the_reference_spells_out_is_not_an_invention() -> None:
    """Reference evidence permits digits translated from source number words."""

    assert numeric_corruption("하루 두 번", "1日2回", "1日2回") == (0, 0)
    # A value absent from both source and reference remains an invention.
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
    # Observed defect: doses and amounts change to plausible but incorrect values.
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
    # Both omissions (recall) and inventions (precision) must reduce F1.
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
    with pytest.raises(ValueError, match="does not match source count"):
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


def test_result_staging_failure_preserves_existing_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "eval.json"
    markdown_path = tmp_path / "eval.md"
    json_path.write_text('{"old": true}\n', encoding="utf-8")
    markdown_path.write_text("old report\n", encoding="utf-8")
    calls = 0

    def fail_second_fsync(_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")

    monkeypatch.setattr(evaluation_module.os, "fsync", fail_second_fsync)
    results = [DirectionResult("sion", "sw-ar", 1, 50.0, 20.0, "13a")]

    with pytest.raises(OSError, match="injected staging failure"):
        save_results(results, tmp_path / "eval", metadata={"model": "new"})

    assert json_path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert markdown_path.read_text(encoding="utf-8") == "old report\n"
    assert list(tmp_path.glob(".eval.*.tmp")) == []
