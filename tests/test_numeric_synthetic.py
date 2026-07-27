from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_numeric_synthetic.py"
)
SPEC = importlib.util.spec_from_file_location("build_numeric_synthetic_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
NUMERIC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NUMERIC
SPEC.loader.exec_module(NUMERIC)


def test_numeric_generator_is_deterministic_unique_and_marks_provenance() -> None:
    first, first_counts = NUMERIC.build_rows(180, seed=20260727)
    second, second_counts = NUMERIC.build_rows(180, seed=20260727)

    assert first == second
    assert first_counts == second_counts
    assert len(first) == len({(row["ko"], row["ja"]) for row in first}) == 180
    assert set(first_counts) == {name for name, _, _ in NUMERIC.GENERATORS}
    assert all(row["synthetic"] is True for row in first)
    assert all(row["source_revision"] == NUMERIC.GENERATOR_VERSION for row in first)
