from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "rebuild_verified_parallel.py"
)
SPEC = importlib.util.spec_from_file_location("rebuild_verified_parallel_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REBUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REBUILD
SPEC.loader.exec_module(REBUILD)


def test_bible_inline_note_cleanup_is_narrow_and_reproducible() -> None:
    assert (
        REBUILD.clean_bible_verse(
            "ja",
            "AMO.7.10",
            "人を遣わして言った1 「本文",
        )
        == "人を遣わして言った。「本文"
    )
    assert REBUILD.clean_bible_verse("ja", "OTHER.1.1", "第1 「本文") == "第1 「本文"


def test_massive_is_not_rebuilt_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["rebuild_verified_parallel.py"])
    args = REBUILD.parse_args()

    assert args.sources == ["bible", "nict", "ui"]
    assert "massive" not in args.sources
