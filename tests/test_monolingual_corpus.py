"""Discovery and reading contracts for monolingual corpora.

This stage consumes the most GPU time, while input mistakes fail silently. One
wrong key can reduce a single file to zero sentences and remain hidden until
training ends. These tests therefore emphasize reporting skipped input with a
reason, not just reading valid input successfully.
"""

from __future__ import annotations

import json

import pytest

from sion_translate.data.monolingual import (
    MonolingualDiscovery,
    ReadStats,
    assess_language_balance,
    discover_monolingual_sources,
    foundation_languages,
    iter_monolingual_lines,
    language_sampling_weights,
    render_discovery_report,
)


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    _write(root / "ko" / "wiki.txt", ["첫 문장입니다", "둘째 문장입니다"])
    _write(
        root / "ja" / "news.jsonl",
        [json.dumps({"text": "日本語の文です"}, ensure_ascii=False)],
    )
    return root


def test_language_folders_become_sources(tmp_path) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path), ["ko", "ja"])
    assert set(discovery.languages) == {"ko", "ja"}
    assert len(discovery.paths_for("ko")) == 1
    assert len(discovery.paths_for("ja")) == 1
    assert discovery.languages_without_data == ()


def test_nested_folders_inside_a_language_are_scanned(tmp_path) -> None:
    root = _corpus(tmp_path)
    _write(root / "ko" / "2026" / "news.txt", ["중첩된 문장"])
    discovery = discover_monolingual_sources(root, ["ko", "ja"])
    assert len(discovery.paths_for("ko")) == 2


def test_a_folder_that_is_not_a_configured_language_is_reported_not_read(tmp_path) -> None:
    """Cover a real layout such as ``data/corpus/korean_tech_corpus_130m/``.

    If it is silently ignored, an operator can believe that its 5.3 GB entered
    training.
    """
    root = _corpus(tmp_path)
    _write(root / "korean_tech_corpus_130m" / "ui.txt", ["기술 문장"])

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    assert "korean_tech_corpus_130m" not in discovery.languages
    assert discovery.unconfigured_languages == ()
    reasons = {entry.path.name: entry.reason for entry in discovery.skipped}
    assert "not a valid language tag" in reasons["korean_tech_corpus_130m"]


def test_a_valid_language_code_that_is_not_configured_is_named(tmp_path) -> None:
    root = _corpus(tmp_path)
    _write(root / "en" / "wiki.txt", ["an english sentence"])

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    assert discovery.unconfigured_languages == ("en",)
    assert "en" not in discovery.languages


def test_script_and_region_language_folders_are_canonicalized(tmp_path) -> None:
    root = tmp_path / "corpus"
    _write(root / "pt-br" / "wiki.txt", ["uma frase em português"])
    _write(root / "ZH-hant" / "wiki.txt", ["一個繁體中文句子"])

    discovery = discover_monolingual_sources(root, ["pt-BR", "zh-Hant"])

    assert discovery.languages == ("pt-BR", "zh-Hant")
    assert len(discovery.paths_for("PT-br")) == 1
    assert discovery.languages_without_data == ()


def test_configured_extension_order_alias_directories_fail_before_scanning(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    first = root / "en-a-aaa-b-ccc-bbb"
    second = root / "en-b-ccc-bbb-a-aaa"
    _write(first / "first.txt", ["first corpus"])
    _write(second / "second.txt", ["second corpus"])

    def reject_scan(*_args, **_kwargs):
        raise AssertionError("corpus files were scanned before alias validation")

    monkeypatch.setattr(type(root), "rglob", reject_scan)

    with pytest.raises(ValueError) as exc_info:
        discover_monolingual_sources(root, ["en-a-aaa-b-ccc-bbb"])

    message = str(exc_info.value)
    assert "same configured language" in message
    assert str(first) in message
    assert str(second) in message


def test_unsupported_extensions_are_reported(tmp_path) -> None:
    root = _corpus(tmp_path)
    _write(root / "ko" / "notes.md", ["마크다운"])
    (root / "ko" / "model.bin").write_bytes(b"\x00\x01")

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    skipped = {entry.path.name for entry in discovery.skipped}
    assert {"notes.md", "model.bin"} <= skipped
    assert len(discovery.paths_for("ko")) == 1


def test_stray_top_level_files_are_reported(tmp_path) -> None:
    """Cover real stray files such as ``data/corpus/a.py`` and ``data.txt``."""
    root = _corpus(tmp_path)
    (root / "a.py").write_text("print()\n", encoding="utf-8")
    (root / "data.txt").write_text("떠 있는 파일\n", encoding="utf-8")

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    reasons = {entry.path.name: entry.reason for entry in discovery.skipped}
    assert "not a language directory" in reasons["a.py"]
    assert "not a language directory" in reasons["data.txt"]


def test_empty_files_are_reported_not_silently_dropped(tmp_path) -> None:
    root = _corpus(tmp_path)
    (root / "ko" / "empty.txt").write_text("", encoding="utf-8")

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    reasons = {entry.path.name: entry.reason for entry in discovery.skipped}
    assert reasons["empty.txt"] == "empty file"


def test_a_missing_root_reports_every_language_as_empty(tmp_path) -> None:
    discovery = discover_monolingual_sources(tmp_path / "absent", ["ko", "ja"])
    assert not discovery
    assert discovery.languages_without_data == ("ko", "ja")


def test_a_language_folder_without_readable_files_is_named(tmp_path) -> None:
    root = _corpus(tmp_path)
    (root / "ja" / "news.jsonl").unlink()

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    assert discovery.languages_without_data == ("ja",)


def test_text_files_yield_one_line_each(tmp_path) -> None:
    path = _write(tmp_path / "a.txt", ["첫 줄", "", "  둘째 줄  ", "셋째 줄"])
    stats = ReadStats()
    assert list(iter_monolingual_lines(path, stats=stats)) == ["첫 줄", "둘째 줄", "셋째 줄"]
    assert stats.accepted == 3
    assert stats.blank == 1


def test_jsonl_files_read_the_text_key(tmp_path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            json.dumps({"text": "좋은 줄"}, ensure_ascii=False),
            json.dumps({"text": "  공백 포함  "}, ensure_ascii=False),
        ],
    )
    assert list(iter_monolingual_lines(path)) == ["좋은 줄", "공백 포함"]


def test_jsonl_rejections_are_counted_by_reason(tmp_path) -> None:
    """Count every rejection reason so missing rows cannot remain silent."""
    path = _write(
        tmp_path / "a.jsonl",
        [
            json.dumps({"text": "좋은 줄"}, ensure_ascii=False),
            "{ 깨진 json",
            json.dumps({"sentence": "키 이름이 다름"}, ensure_ascii=False),
            json.dumps({"text": 12345}),
            json.dumps({"text": "   "}, ensure_ascii=False),
            json.dumps(["객체가 아님"], ensure_ascii=False),
        ],
    )
    stats = ReadStats()

    assert list(iter_monolingual_lines(path, stats=stats)) == ["좋은 줄"]

    assert stats.accepted == 1
    assert stats.malformed_json == 1
    assert stats.non_string_text == 1
    assert stats.blank == 1
    # One missing key plus one non-object record.
    assert stats.missing_text_key == 2
    assert stats.rejected == 5
    assert set(stats.reasons()) == {
        "blank",
        "malformed_json",
        "missing_text_key",
        "non_string_text",
    }


def test_reading_an_unsupported_extension_is_an_error(tmp_path) -> None:
    path = _write(tmp_path / "a.csv", ["가,나"])
    with pytest.raises(ValueError, match=r"\.txt"):
        list(iter_monolingual_lines(path))


def test_source_only_languages_are_excluded_from_foundation() -> None:
    """Monolingual reconstruction teaches the decoder to emit that language.

    Source-only means that a language must never appear as a translation result,
    so foundation training must not teach the opposite behavior first.
    """
    assert foundation_languages(
        ["kj", "kd", "jd", "ko", "ja"],
        source_only_languages=["kj", "kd", "jd"],
    ) == ("ko", "ja")


def test_foundation_languages_keeps_order_and_deduplicates() -> None:
    assert foundation_languages(["ko", "ja", "ko"]) == ("ko", "ja")


def test_temperature_sampling_flattens_but_does_not_equalize() -> None:
    counts = {"ko": 1_000_000, "ja": 10_000}
    proportional = language_sampling_weights(counts, alpha=1.0)
    flattened = language_sampling_weights(counts, alpha=0.7)

    assert proportional["ja"] == pytest.approx(10_000 / 1_010_000)
    assert flattened["ja"] > proportional["ja"]
    assert flattened["ja"] < flattened["ko"]
    assert sum(flattened.values()) == pytest.approx(1.0)


def test_a_language_with_no_data_gets_zero_weight_not_a_share() -> None:
    weights = language_sampling_weights({"ko": 1_000, "ja": 0})
    assert weights["ja"] == 0.0
    assert weights["ko"] == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [0.0, -0.5, 1.5])
def test_alpha_outside_the_unit_interval_is_rejected(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        language_sampling_weights({"ko": 1}, alpha=alpha)


def test_a_language_with_no_data_produces_a_warning() -> None:
    """Model a corpus where one configured language has no data."""
    report = assess_language_balance({"ko": 18_000_000, "ja": 0})
    assert not report.is_balanced()
    assert any("ja" in warning for warning in report.warnings)


def test_a_thin_language_produces_a_warning() -> None:
    report = assess_language_balance({"ko": 10_000_000, "ja": 100}, alpha=1.0)
    assert not report.is_balanced()
    assert any("batch share" in warning for warning in report.warnings)


def test_a_balanced_corpus_produces_no_warning() -> None:
    report = assess_language_balance({"ko": 1_000_000, "ja": 900_000})
    assert report.is_balanced()
    assert report.weights["ja"] > 0.4


def test_the_report_names_every_skipped_path(tmp_path) -> None:
    root = _corpus(tmp_path)
    _write(root / "ko" / "notes.md", ["마크다운"])
    (root / "a.py").write_text("print()\n", encoding="utf-8")

    lines = render_discovery_report(discover_monolingual_sources(root, ["ko", "ja"]))
    rendered = "\n".join(lines)

    assert "notes.md" in rendered
    assert "a.py" in rendered
    assert "ko:" in rendered and "ja:" in rendered


def test_an_empty_discovery_is_falsy() -> None:
    assert not MonolingualDiscovery(root=__import__("pathlib").Path("x"))


# ── Tokenizer sample caps ───────────────────────────────────────────────


def test_budget_follows_the_parallel_corpus_proportions() -> None:
    """Cap each language at its parallel sentence count times the ratio."""
    from sion_translate.data.monolingual import monolingual_budgets

    budgets = monolingual_budgets({"ko": 1000, "ja": 1000}, ["ko", "ja"], ratio=1.0)
    assert budgets == {"ko": 1000, "ja": 1000}

    halved = monolingual_budgets({"ko": 1000, "ja": 1000}, ["ko", "ja"], ratio=0.5)
    assert halved == {"ko": 500, "ja": 500}


def test_a_language_without_parallel_data_still_gets_a_budget() -> None:
    """Monolingual data without a translation pair is a valid intermediate state.

    A zero budget would exclude the language from the tokenizer completely and
    force tokenizer retraining when a translation pair is added later.
    """
    from sion_translate.data.monolingual import monolingual_budgets

    budgets = monolingual_budgets({"ko": 1000, "ja": 500}, ["ko", "ja", "en"], ratio=1.0)
    assert budgets["en"] == 750  # Mean of the ko and ja budgets.
    assert budgets["ko"] == 1000


def test_a_negative_ratio_is_rejected() -> None:
    from sion_translate.data.monolingual import monolingual_budgets

    with pytest.raises(ValueError, match="ratio"):
        monolingual_budgets({"ko": 10}, ["ko"], ratio=-1.0)


def test_sampling_respects_the_budget(tmp_path) -> None:
    path = _write(tmp_path / "a.txt", [f"문장 번호 {index} 입니다" for index in range(1000)])
    from sion_translate.data.monolingual import sample_monolingual_sentences

    sampled = list(sample_monolingual_sentences([path], 100))
    assert 0 < len(sampled) <= 100


def test_sampling_spreads_across_the_file_instead_of_truncating(tmp_path) -> None:
    """Prefix truncation can imprint one source's bias into the vocabulary."""
    path = _write(tmp_path / "a.txt", [f"문장 {index:05d} 번입니다" for index in range(2000)])
    from sion_translate.data.monolingual import sample_monolingual_sentences

    sampled = list(sample_monolingual_sentences([path], 200))
    indices = [int(text.split()[1]) for text in sampled]

    assert len(sampled) > 50
    # A prefix sample would have a maximum index near the sample size.
    assert max(indices) > 1500
    assert min(indices) < 500


def test_sampling_is_deterministic(tmp_path) -> None:
    path = _write(tmp_path / "a.txt", [f"문장 {index} 입니다" for index in range(500)])
    from sion_translate.data.monolingual import sample_monolingual_sentences

    assert list(sample_monolingual_sentences([path], 50)) == list(
        sample_monolingual_sentences([path], 50)
    )


def test_a_zero_budget_yields_nothing(tmp_path) -> None:
    path = _write(tmp_path / "a.txt", ["문장 하나"])
    from sion_translate.data.monolingual import sample_monolingual_sentences

    assert list(sample_monolingual_sentences([path], 0)) == []


# ── Long-document segmentation without dropping or truncating text ─────


def test_a_short_text_is_returned_unchanged() -> None:
    from sion_translate.data.monolingual import segment_text

    assert segment_text("짧은 문장입니다.", maximum_characters=100) == ["짧은 문장입니다."]


def test_a_long_document_is_split_at_sentence_boundaries() -> None:
    """Splitting mid-sentence would teach an incomplete sentence as a full target."""
    from sion_translate.data.monolingual import segment_text

    document = " ".join(f"이것은 {index}번째 문장입니다." for index in range(20))
    segments = segment_text(document, maximum_characters=60)

    assert len(segments) > 1
    assert all(len(segment) <= 60 for segment in segments)
    # Sentence-boundary segmentation leaves each piece ending in punctuation.
    assert all(segment.endswith(".") for segment in segments)
    # No content is lost.
    assert "".join(segments).replace(" ", "") == document.replace(" ", "")


def test_a_single_sentence_longer_than_the_cap_is_hard_split() -> None:
    from sion_translate.data.monolingual import segment_text

    segments = segment_text("가" * 250, maximum_characters=100)
    assert [len(segment) for segment in segments] == [100, 100, 50]


def test_segments_below_the_minimum_are_dropped() -> None:
    from sion_translate.data.monolingual import segment_text

    segments = segment_text(
        "아. " + "긴 문장입니다. " * 20, maximum_characters=40, minimum_characters=5
    )
    assert all(len(segment) >= 5 for segment in segments)


def test_an_empty_document_yields_nothing() -> None:
    from sion_translate.data.monolingual import segment_text

    assert segment_text("   ", maximum_characters=100) == []


def test_a_non_positive_cap_is_rejected() -> None:
    from sion_translate.data.monolingual import segment_text

    with pytest.raises(ValueError, match="maximum_characters"):
        segment_text("가나다", maximum_characters=0)


def test_japanese_sentence_enders_are_boundaries() -> None:
    """Japanese data suffered most from the old behavior: e_gov lost 97.3%."""
    from sion_translate.data.monolingual import segment_text

    document = "".join(f"これは{index}番目の文です。" for index in range(20))
    segments = segment_text(document, maximum_characters=60)
    assert len(segments) > 1
    assert all(segment.endswith("。") for segment in segments)
