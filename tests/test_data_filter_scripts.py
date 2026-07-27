from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "data"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))


def _load(name: str):
    path = SCRIPT_DIRECTORY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SEMANTIC = _load("filter_semantic_pairs")
STRUCTURAL = _load("refilter_ui_structural")


def test_semantic_score_summary_reports_stable_percentiles() -> None:
    summary = SEMANTIC.score_summary([0.1, 0.2, 0.4, 0.8, 0.9])
    assert summary["minimum"] == 0.1
    assert summary["median"] == 0.4
    assert summary["maximum"] == 0.9
    assert summary["mean"] == pytest.approx(0.48)


def test_ui_refilter_removes_structural_number_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "ko": "파일 %s을(를) 12개 처리했습니다.",
                        "ja": "ファイル%sを12件処理しました。",
                        "resource_id": "ok",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "ko": "파일 12개를 처리했습니다.",
                        "ja": "ファイルを13件処理しました。",
                        "resource_id": "bad-number",
                    },
                    ensure_ascii=False,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "filtered.jsonl"
    report_path = tmp_path / "report.json"
    report = STRUCTURAL.filter_file(source, output, report_path)

    assert report["written_rows"] == 1
    assert report["removed_rows"] == 1
    assert report["rejected_by_reason"]["number_mismatch"] == 1
    kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["resource_id"] for row in kept] == ["ok"]
