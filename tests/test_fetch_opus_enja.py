from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "fetch_opus_enja.py"
SPEC = importlib.util.spec_from_file_location("fetch_opus_enja_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_clean_pair_accepts_bilingual_text() -> None:
    assert MODULE.clean_pair(" Hello, world! ", " こんにちは、世界！ ") == (
        "Hello, world!",
        "こんにちは、世界！",
    )


def test_clean_pair_rejects_wrong_scripts_and_extreme_ratio() -> None:
    assert MODULE.clean_pair("日本語", "English") is None
    assert MODULE.clean_pair("Hi", "あ" * 100) is None
