from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "build_jparacrawl_enja.py"
SPEC = importlib.util.spec_from_file_location("build_jparacrawl_enja_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_clean_pair_uses_strict_script_and_ratio_filters() -> None:
    assert MODULE.clean_pair("A useful English sentence.", "役に立つ日本語の文です。") == (
        "A useful English sentence.",
        "役に立つ日本語の文です。",
    )
    assert MODULE.clean_pair("English only", "Latin only") is None
    assert MODULE.clean_pair("Tiny", "あ" * 100) is None
