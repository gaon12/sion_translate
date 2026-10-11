from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "add_wutheringwaves_english_to_data13.py"
)
SPEC = importlib.util.spec_from_file_location(
    "add_wutheringwaves_english_to_data13_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
ADD_ENGLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADD_ENGLISH
SPEC.loader.exec_module(ADD_ENGLISH)


def test_canonical_member_id_removes_only_language_prefix() -> None:
    assert (
        ADD_ENGLISH.canonical_member_id("日语 - Japanese/1号演员/ja_vo_Character_Brant_20_1.lab")
        == "1号演员/vo_Character_Brant_20_1.lab"
    )


def test_add_english_rows_uses_three_way_id_intersection() -> None:
    rows = [
        {"ko": "첫째", "ja": "一つ目"},
        {"ko": "둘째", "ja": "二つ目"},
    ]
    japanese = ["speaker/first.lab", "speaker/second.lab", "speaker/jp-only.lab"]
    korean = ["speaker/first.lab", "speaker/second.lab", "speaker/kr-only.lab"]
    english = {"speaker/second.lab": "Second"}

    output, stats = ADD_ENGLISH.add_english_rows(rows, japanese, korean, english)

    assert len(output) == 1
    assert list(output[0])[:3] == ["ko", "en", "ja"]
    assert output[0]["ko"] == "둘째"
    assert output[0]["en"] == "Second"
    assert output[0]["resource_id"] == "speaker/second.lab"
    assert stats["jp_kr_intersection"] == 2
    assert stats["three_way_intersection"] == 1
    assert stats["jp_kr_without_english"] == 1


def test_add_english_rows_rejects_unexplained_row_count() -> None:
    with pytest.raises(ValueError, match="rows=1 intersection=2"):
        ADD_ENGLISH.add_english_rows(
            [{"ko": "한", "ja": "一"}],
            ["a.lab", "b.lab"],
            ["a.lab", "b.lab"],
            {"a.lab": "A", "b.lab": "B"},
        )
