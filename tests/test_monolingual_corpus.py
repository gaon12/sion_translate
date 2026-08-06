"""단일어 코퍼스 탐색·읽기 규격.

이 단계는 GPU 시간을 가장 많이 쓰는데 입력 오류는 조용합니다. 키 이름 하나가
틀리면 그 파일만 0문장이 되고, 그 사실은 학습이 끝난 뒤에야 드러납니다.
그래서 여기 테스트는 "잘 읽는다" 보다 "건너뛴 것을 이유와 함께 돌려준다" 에
무게를 둡니다.
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
    """`data/corpus/korean_tech_corpus_130m/` 같은 실제 사례.

    조용히 무시하면 사용자는 그 5.3 GB 가 학습에 들어갔다고 믿습니다.
    """
    root = _corpus(tmp_path)
    _write(root / "korean_tech_corpus_130m" / "ui.txt", ["기술 문장"])

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    assert "korean_tech_corpus_130m" not in discovery.languages
    assert discovery.unconfigured_languages == ()
    reasons = {entry.path.name: entry.reason for entry in discovery.skipped}
    assert "언어 코드 형식이 아닌" in reasons["korean_tech_corpus_130m"]


def test_a_valid_language_code_that_is_not_configured_is_named(tmp_path) -> None:
    root = _corpus(tmp_path)
    _write(root / "en" / "wiki.txt", ["an english sentence"])

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    assert discovery.unconfigured_languages == ("en",)
    assert "en" not in discovery.languages


def test_unsupported_extensions_are_reported(tmp_path) -> None:
    root = _corpus(tmp_path)
    _write(root / "ko" / "notes.md", ["마크다운"])
    (root / "ko" / "model.bin").write_bytes(b"\x00\x01")

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    skipped = {entry.path.name for entry in discovery.skipped}
    assert {"notes.md", "model.bin"} <= skipped
    assert len(discovery.paths_for("ko")) == 1


def test_stray_top_level_files_are_reported(tmp_path) -> None:
    """`data/corpus/a.py` 와 `data/corpus/data.txt` 같은 실제 사례."""
    root = _corpus(tmp_path)
    (root / "a.py").write_text("print()\n", encoding="utf-8")
    (root / "data.txt").write_text("떠 있는 파일\n", encoding="utf-8")

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    reasons = {entry.path.name: entry.reason for entry in discovery.skipped}
    assert "언어 폴더가 아닌" in reasons["a.py"]
    assert "언어 폴더가 아닌" in reasons["data.txt"]


def test_empty_files_are_reported_not_silently_dropped(tmp_path) -> None:
    root = _corpus(tmp_path)
    (root / "ko" / "empty.txt").write_text("", encoding="utf-8")

    discovery = discover_monolingual_sources(root, ["ko", "ja"])

    reasons = {entry.path.name: entry.reason for entry in discovery.skipped}
    assert reasons["empty.txt"] == "빈 파일"


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
    """무엇이 몇 줄 빠졌는지 셀 수 없으면 조용한 손실이 된다."""
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
    # 키 없음 + 객체 아님
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
    """단일어 복원은 그 언어를 디코더 출력으로 만드는 학습이다.

    source-only 는 '번역 결과로 나오면 안 되는 언어' 라는 뜻이므로, foundation
    이 먼저 그 반대를 가르치면 안 됩니다.
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
    """현재 저장소의 실제 상태: ko 5.3 GB, ja 0."""
    report = assess_language_balance({"ko": 18_000_000, "ja": 0})
    assert not report.is_balanced()
    assert any("ja" in warning for warning in report.warnings)


def test_a_thin_language_produces_a_warning() -> None:
    report = assess_language_balance({"ko": 10_000_000, "ja": 100}, alpha=1.0)
    assert not report.is_balanced()
    assert any("비중" in warning for warning in report.warnings)


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
