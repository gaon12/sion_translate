"""The netspeak generator must stay deterministic, capped, and casual-only.

Unlike the dialect shards this writes an ordinary ko/ja pair, so a bad row goes
straight into the main pool with no source-only tag to contain it. The caps and
the casual filter are the whole defence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_netspeak_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_netspeak_corpus_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


CASUAL = [
    ("이거 진짜 대박이다", "これ本当にやばい"),
    ("너무 맛있어", "とてもうまい"),
    ("오늘 날씨 좋아", "今日は天気いい"),
    ("그건 좀 어려워", "それはちょっと難しい"),
    ("이거 재밌어", "これ面白い"),
    ("진짜 피곤해", "マジで疲れた"),
    ("고마워", "ありがとう"),
    ("너 진짜 귀여워", "お前本当にかわいい"),
    ("이 노래 최고야", "この歌最高だよ"),
    ("빨리 가자", "早く行こう"),
]
POLITE = [
    ("정말 죄송합니다", "本当に申し訳ありません"),
    ("조금 기다려 주세요", "少し待ってください"),
]


def write_shard(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for ko, ja in rows:
            handle.write(json.dumps({"ko": ko, "ja": ja}, ensure_ascii=False) + "\n")
    return path


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(source: Path, output: Path, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "source_key": "ko",
        "target_key": "ja",
        "source_language": "ko",
        "target_language": "ja",
        "min_chars": 4,
        "max_chars": 60,
        "max_per_style": 100,
        "max_per_frame": 100,
        "seed": "test",
        "styles": [],
    }
    arguments.update(overrides)
    return BUILD.build([source], output, **arguments)


def test_a_row_records_the_style_and_rules_that_made_it(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", CASUAL)
    output = tmp_path / "out.jsonl"
    run(source, output)
    rows = read_shard(output)
    assert rows
    row = rows[0]
    assert row["ko"] and row["ja"]
    assert row["net_style"]
    assert row["net_rules"]
    assert row["synthetic"] is True
    # No new language tag: the output is an ordinary pair.
    assert "kd" not in row and "jd" not in row


def test_polite_pairs_never_reach_the_output(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", POLITE)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert result.skipped_not_casual == len(POLITE)
    assert result.rows_written == 0


def test_the_style_cap_keeps_the_styles_balanced(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", CASUAL * 20)
    output = tmp_path / "out.jsonl"
    result = run(source, output, max_per_style=2)
    assert result.per_style
    assert max(result.per_style.values()) <= 2


def test_the_frame_cap_stops_one_sentence_shape_repeating(tmp_path: Path) -> None:
    rows = [(f'그건 "{index}"이야', f"それは「{index}」だよ") for index in range(120)]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = run(source, output, max_per_frame=2, max_per_style=1000)
    assert result.skipped_frame_cap > 0


def test_styles_that_converge_on_the_same_output_are_deduplicated(tmp_path: Path) -> None:
    # A sentence with no substitution and no adjective falls back to laughter in
    # several styles, producing the same pair more than once.
    source = write_shard(tmp_path / "in.jsonl", [("빨리 가자", "早く行こうよ")])
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    rows = read_shard(output)
    pairs = {(row["ko"], row["ja"]) for row in rows}
    assert len(pairs) == len(rows)
    assert result.skipped_duplicate > 0


def test_output_is_independent_of_input_order(tmp_path: Path) -> None:
    forward = write_shard(tmp_path / "a.jsonl", CASUAL)
    backward = write_shard(tmp_path / "b.jsonl", list(reversed(CASUAL)))
    first = tmp_path / "a.out.jsonl"
    second = tmp_path / "b.out.jsonl"
    run(forward, first)
    run(backward, second)
    assert read_shard(first) == read_shard(second)


def test_markup_is_never_rewritten(tmp_path: Path) -> None:
    rows = [("{name}야 반가워", "{name}よろしくね")]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert result.skipped_markup == 1
    assert result.rows_written == 0


def test_a_side_in_the_wrong_script_is_skipped(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [("これは 한국어 아니야", "これは韓国語じゃない")])
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert result.skipped_script == 1


def test_styles_can_be_restricted(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", CASUAL)
    output = tmp_path / "out.jsonl"
    result = run(source, output, styles=["lament"])
    assert set(result.per_style) <= {"lament"}


def test_an_unknown_style_is_rejected() -> None:
    try:
        BUILD.style_list("laughter,keyboard-smash")
    except Exception as error:
        assert "unknown style" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a rejection")


def test_the_report_totals_agree_with_the_file(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", CASUAL)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert sum(result.per_style.values()) == result.rows_written
    assert result.rows_written == len(read_shard(output))
    assert result.per_rule


def test_frame_blanks_quotes_and_digits_like_the_audit() -> None:
    assert BUILD.frame('그건 "0.5"이고 3개야') == "그건 <Q>이고 #개야"


def test_variant_is_stable_for_the_same_input() -> None:
    first = BUILD.variant_of("seed", "laughter", "이거 재밌어")
    second = BUILD.variant_of("seed", "laughter", "이거 재밌어")
    assert first == second
    assert BUILD.variant_of("seed", "laughter", "다른 문장") != first
