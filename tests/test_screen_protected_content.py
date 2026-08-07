"""The screening CLI must remove flagged rows and never echo their text.

Exit code 3 without --output is the useful behaviour: a plain check run fails
loudly instead of quietly reporting and moving on.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.usefixtures("configured_content_screen")


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "screen_protected_content.py"
)
SPEC = importlib.util.spec_from_file_location("screen_protected_content_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCREEN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCREEN
SPEC.loader.exec_module(SCREEN)


CLEAN = {"ko": "둘은 그날 밤 함께 있었다", "ja": "二人はその夜一緒にいた"}
ADULT_SEXUAL = {"ko": "성인 두 사람이 섹스를 했다", "ja": "大人二人がセックスをした"}
CHILD_ONLY = {"ko": "초등학교 앞에서 만나자", "ja": "小学校の前で会おう"}
FLAGGED = {"ko": "초등학생과 성관계를 했다", "ja": "小学生と性交した"}


def write_shard(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_only_the_conjunction_is_removed(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [CLEAN, ADULT_SEXUAL, CHILD_ONLY, FLAGGED])
    output = tmp_path / "out.jsonl"
    assert SCREEN.main([str(source), "--output", str(output)]) == 0
    kept = read_shard(output)
    assert len(kept) == 3
    assert FLAGGED["ko"] not in {row["ko"] for row in kept}
    assert ADULT_SEXUAL["ko"] in {row["ko"] for row in kept}
    assert CHILD_ONLY["ko"] in {row["ko"] for row in kept}


def test_a_clean_file_exits_zero_without_output(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [CLEAN, ADULT_SEXUAL, CHILD_ONLY])
    assert SCREEN.main([str(source)]) == 0


def test_a_check_run_fails_when_something_is_flagged(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [CLEAN, FLAGGED])
    assert SCREEN.main([str(source)]) == 3


def test_the_report_names_markers_and_ids_but_not_the_text(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [CLEAN, FLAGGED])
    report_path = tmp_path / "report.json"
    SCREEN.main(
        [str(source), "--output", str(tmp_path / "out.jsonl"), "--report", str(report_path)]
    )
    body = report_path.read_text(encoding="utf-8")
    report = json.loads(body)
    assert report["rows"] == 2
    assert report["flagged"] == 1
    # 小学 is a marker in its own right and is a substring of 小学生, so both
    # fire. Reporting every marker that matched is the intended behaviour.
    assert report["child_marker_counts"] == {"초등학생": 1, "小学生": 1, "小学": 1}
    assert report["flagged_row_ids"] == ["line:2"]
    # The matched sentence must not appear anywhere in the report.
    assert FLAGGED["ko"] not in body
    assert FLAGGED["ja"] not in body


def test_rejected_ids_are_written_when_asked(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [CLEAN, FLAGGED])
    ids_path = tmp_path / "ids.txt"
    SCREEN.main(
        [str(source), "--output", str(tmp_path / "out.jsonl"), "--rejected-ids", str(ids_path)]
    )
    assert ids_path.read_text(encoding="utf-8").strip() == "line:2"


def test_an_existing_resource_id_is_preferred_over_the_line_number(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [dict(FLAGGED, resource_id="scene:0042")])
    report_path = tmp_path / "report.json"
    SCREEN.main(
        [str(source), "--output", str(tmp_path / "out.jsonl"), "--report", str(report_path)]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["flagged_row_ids"] == ["scene:0042"]


def test_the_internal_line_field_is_not_written_to_the_output(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [CLEAN])
    output = tmp_path / "out.jsonl"
    SCREEN.main([str(source), "--output", str(output)])
    assert "_line" not in read_shard(output)[0]


def test_an_unconfigured_language_screens_nothing(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [FLAGGED])
    assert SCREEN.main([str(source), "--source-language", "en", "--target-language", "de"]) == 0


def test_rows_missing_a_side_are_kept_rather_than_silently_dropped(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [{"ko": "혼자 있는 행"}])
    output = tmp_path / "out.jsonl"
    assert SCREEN.main([str(source), "--output", str(output)]) == 0
    assert len(read_shard(output)) == 1


def test_a_missing_input_is_rejected(tmp_path: Path) -> None:
    assert SCREEN.main([str(tmp_path / "nope.jsonl")]) == 2
