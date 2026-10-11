from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "build_balanced_koen.py"
SPEC = importlib.util.spec_from_file_location("build_balanced_koen_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_allocations_preserve_small_sources() -> None:
    assert MODULE.balanced_allocations([2, 10, 10], 12) == [2, 5, 5]


def test_build_selects_exact_target_from_all_sources(tmp_path: Path) -> None:
    inputs = []
    for source_index, count in enumerate((3, 7)):
        path = tmp_path / f"source{source_index}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "ko": f"한국어 문장 {source_index}-{row}",
                        "en": f"English sentence {source_index}-{row}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for row in range(count)
            ),
            encoding="utf-8",
        )
        inputs.append(path)

    output = tmp_path / "balanced.jsonl"
    report = MODULE.build(inputs, output, 8)

    assert report["row_count"] == 8
    assert [source["selected_rows"] for source in report["sources"]] == [3, 5]
    with output.open(encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 8
