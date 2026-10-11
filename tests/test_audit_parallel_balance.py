from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "audit_parallel_balance.py"
SPEC = importlib.util.spec_from_file_location("audit_parallel_balance_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_three_language_rows_count_toward_all_pairs(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"ko": "한국어", "ja": "日本語"},
        {"ko": "한국어", "en": "English", "ja": "日本語"},
        {"en": "English", "ja": "日本語"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    report = MODULE.audit([path])

    assert report["effective_pairs"] == {"en-ja": 2, "ko-en": 1, "ko-ja": 2}
    assert report["relative_to_ko_ja"] == {"en-ja": 1.0, "ko-en": 0.5, "ko-ja": 1.0}
