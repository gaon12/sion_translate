"""FLORES 벤치마크 변환(sion_translate.benchmark) 검증."""

from __future__ import annotations

from pathlib import Path

import pytest

from sion_translate.benchmark import (
    flores_code,
    find_flores_file,
    pairs_from_local_flores,
    write_jsonl,
)
from sion_translate.evaluation import load_benchmark_pairs


def test_flores_code_lookup_and_override() -> None:
    assert flores_code("ko") == "kor_Hang"
    assert flores_code("ja") == "jpn_Jpan"
    assert flores_code("en") == "eng_Latn"
    # override 우선
    assert flores_code("xx", "xxx_Yyyy") == "xxx_Yyyy"
    # 모르는 언어 + override 없음 → 친절한 에러
    with pytest.raises(ValueError, match="FLORES 코드"):
        flores_code("xx")


def _write_flores_split(root: Path, split: str) -> None:
    """FLORES 배포판을 흉내내어 언어별 텍스트 파일을 만든다."""
    (root / split).mkdir(parents=True, exist_ok=True)
    ko = [f"한국어 문장 {i}" for i in range(5)]
    ja = [f"日本語の文 {i}" for i in range(5)]
    (root / split / f"kor_Hang.{split}").write_text("\n".join(ko) + "\n", encoding="utf-8")
    (root / split / f"jpn_Jpan.{split}").write_text("\n".join(ja) + "\n", encoding="utf-8")


def test_find_flores_file_locations(tmp_path: Path) -> None:
    _write_flores_split(tmp_path, "devtest")
    found = find_flores_file(tmp_path, "kor_Hang", "devtest")
    assert found.name == "kor_Hang.devtest"
    with pytest.raises(FileNotFoundError, match="FLORES 파일"):
        find_flores_file(tmp_path, "deu_Latn", "devtest")


def test_pairs_from_local_flores_end_to_end(tmp_path: Path) -> None:
    _write_flores_split(tmp_path, "devtest")
    pairs = pairs_from_local_flores(tmp_path, ("ko", "ja"), split="devtest")
    assert len(pairs) == 5
    assert pairs[0] == {"ko": "한국어 문장 0", "ja": "日本語の文 0"}

    # 변환한 JSONL 이 sion-evaluate 로더로 그대로 읽혀야 한다.
    output = tmp_path / "flores.jsonl"
    count = write_jsonl(pairs, output)
    assert count == 5
    loaded = load_benchmark_pairs([output], ("ko", "ja"), max_samples_per_direction=10)
    assert len(loaded[("ko", "ja")]) == 5
    assert loaded[("ja", "ko")][0] == ("日本語の文 0", "한국어 문장 0")


def test_line_count_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "kor_Hang.dev").write_text("한 줄\n두 줄\n", encoding="utf-8")
    (tmp_path / "dev" / "jpn_Jpan.dev").write_text("一行\n", encoding="utf-8")
    with pytest.raises(ValueError, match="문장 수가 다릅니다"):
        pairs_from_local_flores(tmp_path, ("ko", "ja"), split="dev")


def test_custom_language_pair_with_override(tmp_path: Path) -> None:
    (tmp_path / "devtest").mkdir()
    (tmp_path / "devtest" / "eng_Latn.devtest").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "devtest" / "deu_Latn.devtest").write_text("hallo\nWelt\n", encoding="utf-8")
    pairs = pairs_from_local_flores(
        tmp_path,
        ("en", "de"),
        split="devtest",
        code_overrides={"en": "eng_Latn", "de": "deu_Latn"},
    )
    assert pairs == [{"en": "hello", "de": "hallo"}, {"en": "world", "de": "Welt"}]
