"""A shard whose keys match no configured pair is dropped without a word.

One shard in this corpus arrived with the keys ``한국어``/``일본어`` instead of
``ko``/``ja``. It parses, it passes every audit that reads it directly, and the
training pipeline yields exactly zero sentences from it. 10,075 rows would have
been lost with nothing in the logs to say so.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "check_shard_keys.py"
SPEC = importlib.util.spec_from_file_location("check_shard_keys_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


def write_shard(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


GOOD = [{"ko": f"문장 {index} 입니다", "ja": f"文 {index} です"} for index in range(30)]
KOREAN_KEYS = [
    {"한국어": f"문장 {index} 입니다", "일본어": f"文 {index} です"} for index in range(30)
]


def test_a_readable_shard_passes(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "good.jsonl", GOOD)
    assert CHECK.main([str(shard), "--pair", "ko", "ja"]) == 0


def test_a_shard_with_unmatched_keys_fails(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "renamed.jsonl", KOREAN_KEYS)
    assert CHECK.main([str(shard), "--pair", "ko", "ja"]) == 1


def test_the_report_names_the_keys_that_are_present(tmp_path: Path, capsys) -> None:
    shard = write_shard(tmp_path / "renamed.jsonl", KOREAN_KEYS)
    CHECK.main([str(shard), "--pair", "ko", "ja"])
    output = capsys.readouterr().out
    # The fix is only obvious if the report says what the file actually has.
    assert "한국어" in output
    assert "일본어" in output
    assert "ko" in output and "ja" in output


def test_a_non_ascii_key_cannot_be_declared_as_a_language(tmp_path: Path) -> None:
    # The obvious fix - "just add the pair" - is not available. Language keys are
    # validated as 1-16 ASCII alphanumerics starting with a letter, so 한국어 can
    # never be one and the JSONL itself has to change. The report says so.
    shard = write_shard(tmp_path / "renamed.jsonl", KOREAN_KEYS)
    assert CHECK.main([str(shard), "--pair", "한국어", "일본어"]) == 2


def test_renaming_the_keys_makes_the_shard_readable(tmp_path: Path) -> None:
    renamed = [
        {"ko": row["한국어"], "ja": row["일본어"]}  # the only real fix
        for row in KOREAN_KEYS
    ]
    shard = write_shard(tmp_path / "fixed.jsonl", renamed)
    assert CHECK.main([str(shard), "--pair", "ko", "ja"]) == 0


def test_a_mixed_run_reports_only_the_broken_shard(tmp_path: Path, capsys) -> None:
    good = write_shard(tmp_path / "good.jsonl", GOOD)
    bad = write_shard(tmp_path / "bad.jsonl", KOREAN_KEYS)
    assert CHECK.main([str(good), str(bad), "--pair", "ko", "ja"]) == 1
    output = capsys.readouterr().out
    assert "yields nothing" in output
    assert output.count("yields nothing") == 1


def test_observed_keys_ignores_non_string_values(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "mixed.jsonl",
        [{"ko": "가", "ja": "あ", "synthetic": True, "count": 3}],  # type: ignore[list-item]
    )
    keys = CHECK.observed_keys(shard)
    assert set(keys) == {"ko", "ja"}


def test_observed_keys_survives_a_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('{"ko": "가", "ja": "あ"}\n')
        handle.write("not json at all\n")
        handle.write('{"ko": "나", "ja": "い"}\n')
    assert set(CHECK.observed_keys(path)) == {"ko", "ja"}


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    assert CHECK.main([str(tmp_path / "nope.jsonl"), "--pair", "ko", "ja"]) == 2


def test_an_empty_file_list_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert CHECK.main(["--pair", "ko", "ja"]) == 2
