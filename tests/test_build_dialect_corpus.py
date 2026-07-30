"""The generator must stay deterministic, capped, and honest about refusals.

A dialect corpus is only useful if one region cannot swamp the rest and if the
same inputs always give the same file, because the caps are the only thing
stopping the varieties with the most common endings from dominating.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_dialect_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_dialect_corpus_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


SPOKEN = [
    ("맛있어요", "美味しいです"),
    ("지금 어디 가요?", "今どこに行きますか?"),
    ("밥 먹었니?", "ご飯食べた?"),
    ("뭐 먹었니?", "何食べた?"),
    ("저는 매일 운동해요", "私は毎日運動します"),
    ("그건 제 잘못이에요", "それは私のせいです"),
    ("우리 같이 가요", "一緒に行きましょう"),
    ("잘 모르겠어요", "よくわかりません"),
    ("이번엔 성공했어요", "今回は成功しました"),
    ("조금 기다려 주세요", "少し待ってください"),
]


def write_shard(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for ko, ja in rows:
            handle.write(json.dumps({"ko": ko, "ja": ja}, ensure_ascii=False) + "\n")
    return path


def read_shard(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(source: Path, output: Path, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "source_key": "ko",
        "target_key": "ja",
        "source_language": "ko",
        "target_language": "ja",
        "min_chars": 4,
        "max_chars": 60,
        "max_per_region": 100,
        "max_per_frame": 100,
        "seed": "test",
        "regions": [],
    }
    arguments.update(overrides)
    return BUILD.build([source], output, **arguments)


def outputs_for(output: Path) -> dict[str, list[dict[str, object]]]:
    return {
        language: read_shard(output.with_name(f"{output.stem}_{language}{output.suffix}"))
        for language in ("ko", "ja")
    }


def test_a_row_keeps_the_standard_pair_beside_the_dialect(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN)
    output = tmp_path / "out.jsonl"
    run(source, output)
    rows = outputs_for(output)["ko"]
    assert rows
    row = rows[0]
    assert row["kd"] and row["ko"] and row["ja"]
    assert row["dialect_language"] == "ko"
    assert row["dialect_region"]
    assert row["dialect_label"]
    assert row["synthetic"] is True
    # The dialect rendering must differ from the standard it came from.
    assert row["kd"] != row["ko"]


def test_each_language_gets_its_own_file(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    files = outputs_for(output)
    assert files["ko"] and files["ja"]
    assert all("kd" in row for row in files["ko"])
    assert all("jd" in row for row in files["ja"])
    # A mixed file would leave rows missing a dialect field entirely.
    assert all("jd" not in row for row in files["ko"])
    assert len(result.outputs) == 2


def test_the_region_cap_stops_one_variety_dominating(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN * 20)
    output = tmp_path / "out.jsonl"
    result = run(source, output, max_per_region=3)
    assert result.per_region
    assert max(result.per_region.values()) <= 3
    assert result.skipped_region_cap > 0


def test_the_frame_cap_stops_one_sentence_shape_repeating(tmp_path: Path) -> None:
    # 200 sentences differing only inside a quoted span: one frame.
    rows = [(f'그건 "{index}"이에요', f"それは「{index}」です") for index in range(200)]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = run(source, output, max_per_frame=2, max_per_region=1000)
    assert result.skipped_frame_cap > 0
    for region, count in result.per_region.items():
        assert count <= 2, (region, count)


def test_duplicate_standard_pairs_are_counted_once(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [SPOKEN[0]] * 5)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert result.pairs_eligible == 1
    assert result.skipped_duplicate == 4


def test_output_is_independent_of_input_order(tmp_path: Path) -> None:
    forward = write_shard(tmp_path / "a.jsonl", SPOKEN)
    backward = write_shard(tmp_path / "b.jsonl", list(reversed(SPOKEN)))
    first = tmp_path / "a.out.jsonl"
    second = tmp_path / "b.out.jsonl"
    run(forward, first)
    run(backward, second)
    assert outputs_for(first) == outputs_for(second)


def test_a_different_seed_changes_which_rows_survive_the_cap(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN)
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    run(source, one, max_per_region=1, seed="alpha")
    run(source, two, max_per_region=1, seed="omega")
    assert outputs_for(one) != outputs_for(two)


def test_markup_is_never_dialectalised(tmp_path: Path) -> None:
    # A UI string is not speech.
    rows = [
        ("{name}님 안녕하세요", "{name}さんこんにちは"),
        ("<b>확인해요</b>", "<b>確認します</b>"),
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert result.skipped_markup == 2
    assert result.rows_written == 0


def test_rows_outside_the_length_window_are_skipped(tmp_path: Path) -> None:
    rows = [("응", "うん"), ("가" * 200 + "요", "あ" * 200 + "です")]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = run(source, output, min_chars=5, max_chars=40)
    assert result.skipped_length == 2


def test_a_side_in_the_wrong_script_is_skipped(tmp_path: Path) -> None:
    rows = [("これは 한국어 아니에요", "これは韓国語ではないです")]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert result.skipped_script == 1


def test_regions_can_be_restricted(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN)
    output = tmp_path / "out.jsonl"
    result = run(source, output, regions=["gyeongsang"])
    assert set(result.per_region) == {"gyeongsang"}
    assert outputs_for(output)["ja"] == []


def test_an_unknown_region_is_rejected() -> None:
    try:
        BUILD.region_list("gyeongsang,atlantis")
    except Exception as error:  # argparse.ArgumentTypeError
        assert "unknown region" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a rejection")


def test_frame_blanks_quotes_and_digits_like_the_audit() -> None:
    assert BUILD.frame('값은 "0.5"이고 3개다') == "값은 <Q>이고 #개다"


def test_a_language_without_profiles_is_an_error(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN)
    try:
        run(source, tmp_path / "out.jsonl", source_language="en", target_language="de")
    except ValueError as error:
        assert "no dialect profiles" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a rejection")


def test_the_report_counts_regions_and_endings(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", SPOKEN)
    output = tmp_path / "out.jsonl"
    result = run(source, output)
    assert sum(result.per_region.values()) == result.rows_written
    assert sum(result.per_language.values()) == result.rows_written
    assert result.endings_used
    assert result.samples
