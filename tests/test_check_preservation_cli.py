"""--write-passing turns the report into a gate that produces the cleaned file.

data23 renders ``원`` as ``銭`` - an obsolete Japanese currency subunit - in 19
rows, and neither chrF nor an embedding similarity can see a wrong unit. Reporting
the rate is not enough when the point is to keep the rows out of the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from sion_translate.cli.check_preservation import (
    CHECK_NAMES,
    build_parser,
    check_list,
    main,
    pair_passes,
)
import pytest


GOOD_UNIT = {"ko": "가격은 3000원이야", "ja": "値段は3000ウォンだ"}
BAD_UNIT = {"ko": "15000원?너무 비싼 거 아냐?", "ja": "15000銭?ちょっと高くね?"}
BAD_NUMBER = {"ko": "35% 할인이야", "ja": "62.5%割引だ"}
BAD_SCRIPT = {"ko": "이 이상은 무리다", "ja": "これ 이상は無理だ"}


def write_shard(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(argv: list[str]) -> int:
    return main(argv)


def test_source_and_target_fields_are_explicit() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["translated.jsonl"])

    args = parser.parse_args(
        [
            "translated.jsonl",
            "--source-key",
            "sw",
            "--target-key",
            "ar",
        ]
    )
    assert args.source_key == ["sw"]
    assert args.target_key == ["ar"]


def test_source_and_target_fields_cannot_alias(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "self.jsonl", [{"text": "same"}])

    assert run([str(source), "--source-key", "text", "--target-key", "text"]) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--source-key", "", "--target-key", "translation"],
        [
            "--source-key",
            "source",
            "--source-key",
            "source",
            "--target-key",
            "translation",
        ],
    ],
)
def test_source_and_target_fields_reject_empty_or_duplicate_keys(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    source = write_shard(tmp_path / "invalid-keys.jsonl", [GOOD_UNIT])

    assert run([str(source), *arguments]) == 2


def test_a_wrong_currency_unit_is_removed(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [GOOD_UNIT, BAD_UNIT])
    output = tmp_path / "clean.jsonl"
    assert (
        run(
            [
                str(source),
                "--source-key",
                "ko",
                "--target-key",
                "ja",
                "--write-passing",
                str(output),
            ]
        )
        == 0
    )
    kept = read_shard(output)
    assert len(kept) == 1
    assert kept[0]["ja"] == GOOD_UNIT["ja"]


def test_a_corrupted_number_is_removed(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [GOOD_UNIT, BAD_NUMBER])
    output = tmp_path / "clean.jsonl"
    run([str(source), "--source-key", "ko", "--target-key", "ja", "--write-passing", str(output)])
    assert len(read_shard(output)) == 1


def test_the_script_check_only_applies_when_scripts_are_named(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [GOOD_UNIT, BAD_SCRIPT])
    without = tmp_path / "without.jsonl"
    run([str(source), "--source-key", "ko", "--target-key", "ja", "--write-passing", str(without)])
    assert len(read_shard(without)) == 2

    with_scripts = tmp_path / "with.jsonl"
    run(
        [
            str(source),
            "--source-key",
            "ko",
            "--target-key",
            "ja",
            "--target-scripts",
            "ja",
            "--write-passing",
            str(with_scripts),
        ]
    )
    assert len(read_shard(with_scripts)) == 1


def test_checks_can_be_narrowed(tmp_path: Path) -> None:
    # Enforcing only the unit check must let the number defect through.
    source = write_shard(tmp_path / "in.jsonl", [BAD_UNIT, BAD_NUMBER])
    output = tmp_path / "clean.jsonl"
    run(
        [
            str(source),
            "--source-key",
            "ko",
            "--target-key",
            "ja",
            "--checks",
            "unit",
            "--write-passing",
            str(output),
        ]
    )
    kept = read_shard(output)
    assert len(kept) == 1
    assert kept[0]["ja"] == BAD_NUMBER["ja"]


def test_the_report_records_what_was_removed(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [GOOD_UNIT, BAD_UNIT])
    report_path = tmp_path / "report.json"
    run(
        [
            str(source),
            "--source-key",
            "ko",
            "--target-key",
            "ja",
            "--write-passing",
            str(tmp_path / "clean.jsonl"),
            "--json",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))[0]
    assert report["written_rows"] == 1
    assert report["removed_rows"] == 1
    assert report["checks_enforced"] == list(CHECK_NAMES)


def test_unrelated_fields_survive_the_filter(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [dict(GOOD_UNIT, domain="vn", resource_id="r:1")])
    output = tmp_path / "clean.jsonl"
    run([str(source), "--source-key", "ko", "--target-key", "ja", "--write-passing", str(output)])
    row = read_shard(output)[0]
    assert row["domain"] == "vn"
    assert row["resource_id"] == "r:1"


def test_writing_refuses_more_than_one_input(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "a.jsonl", [GOOD_UNIT])
    second = write_shard(tmp_path / "b.jsonl", [GOOD_UNIT])
    assert (
        run(
            [
                str(first),
                str(second),
                "--source-key",
                "ko",
                "--target-key",
                "ja",
                "--write-passing",
                str(tmp_path / "clean.jsonl"),
            ]
        )
        == 2
    )


def test_unknown_check_names_are_rejected() -> None:
    with pytest.raises(Exception, match="unknown check"):
        check_list("unit,fluency")
    with pytest.raises(Exception, match="at least one"):
        check_list(" , ")
    assert check_list("unit, NUMBER ") == ("unit", "number")


def test_pair_passes_agrees_with_the_named_checks() -> None:
    assert pair_passes(GOOD_UNIT["ko"], GOOD_UNIT["ja"], target_scripts=("ja",), checks=CHECK_NAMES)
    assert not pair_passes(BAD_UNIT["ko"], BAD_UNIT["ja"], target_scripts=("ja",), checks=("unit",))
    # The same pair passes when the failing check is not selected.
    assert pair_passes(BAD_UNIT["ko"], BAD_UNIT["ja"], target_scripts=("ja",), checks=("sign",))
