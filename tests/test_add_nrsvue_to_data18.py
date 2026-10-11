from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "add_nrsvue_to_data18.py"
SPEC = importlib.util.spec_from_file_location("add_nrsvue_to_data18_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ADD_NRSVUE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADD_NRSVUE
SPEC.loader.exec_module(ADD_NRSVUE)


def test_add_english_rows_preserves_order_and_updates_provenance() -> None:
    rows = [
        {
            "ko": "태초에 하나님이 천지를 창조하셨다.",
            "ja": "初めに、神は天地を創造された。",
            "source": "old",
            "resource_id": "GEN.1.1",
            "source_revision": "old",
        }
    ]

    output = ADD_NRSVUE.add_english_rows(rows, {"GEN.1.1": "When God began to create"})

    assert list(output[0])[:3] == ["ko", "en", "ja"]
    assert output[0]["en"] == "When God began to create"
    assert output[0]["source"] == ADD_NRSVUE.SOURCE
    assert output[0]["source_revision"] == ADD_NRSVUE.SOURCE_REVISION


def test_add_english_rows_rejects_missing_exact_id() -> None:
    rows = [{"ko": "가", "ja": "あ", "resource_id": "GEN.1.1"}]

    with pytest.raises(ValueError, match="missed 1 rows: GEN.1.1"):
        ADD_NRSVUE.add_english_rows(rows, {})


def test_add_english_rows_merges_compound_nrsvue_span() -> None:
    rows = [
        {
            "ko": "아무것도 받을 생각을 하지 마십시오.",
            "ja": "何かいただけると思ってはなりません。",
            "resource_id": "JAS.1.7",
            "family_id": "bible:JAS.1.7",
        },
        {
            "ko": "그는 두 마음을 품은 사람입니다.",
            "ja": "心が定まらない人です。",
            "resource_id": "JAS.1.8",
            "family_id": "bible:JAS.1.8",
        },
    ]
    english = {
        "JAS.1.7+JAS.1.8": (
            "For the doubter, being double-minded, must not expect to receive anything."
        )
    }

    output = ADD_NRSVUE.add_english_rows(rows, english)

    assert len(output) == 1
    assert output[0]["resource_id"] == "JAS.1.7+JAS.1.8"
    assert output[0]["family_id"] == "bible:JAS.1.7+JAS.1.8"
    assert output[0]["ko"] == "아무것도 받을 생각을 하지 마십시오. 그는 두 마음을 품은 사람입니다."
    assert output[0]["ja"] == "何かいただけると思ってはなりません。心が定まらない人です。"
