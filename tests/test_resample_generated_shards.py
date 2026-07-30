from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "resample_generated_shards.py"
)
SPEC = importlib.util.spec_from_file_location("resample_generated_shards_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RESAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESAMPLE
SPEC.loader.exec_module(RESAMPLE)


def write_shard(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, target in rows:
            handle.write(json.dumps({"ko": source, "ja": target}, ensure_ascii=False) + "\n")
    return path


def read_shard(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_skeleton_matches_the_audit_definition() -> None:
    assert RESAMPLE.skeleton('값은 "0.5 mg"이고 3개다.') == "값은 <Q>이고 #개다."


def test_cap_limits_rows_per_frame(tmp_path: Path) -> None:
    # One frame, 200 variants distinguished only by a quoted span.
    rows = [(f'표현 "{index}"을 옮긴다.', f"表現「{index}」を訳す。") for index in range(200)]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"

    result = RESAMPLE.resample_shard(source, output, max_per_skeleton=8)

    assert result.rows_in == 200
    assert result.rows_out == 8
    assert result.skeletons_in == 1
    assert result.largest_frame_in == 200
    assert result.largest_frame_out == 8
    assert result.dropped_over_cap == 192
    assert len(read_shard(output)) == 8


def test_distinct_frames_are_all_kept(tmp_path: Path) -> None:
    syllables = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
    rows = [
        (f"{syllables[index // 26]}{syllables[index % 26]} 마을은 조용했다.", "村は静かだった。")
        for index in range(100)
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"

    result = RESAMPLE.resample_shard(source, output, max_per_skeleton=2)

    assert result.skeletons_in == 100
    assert result.rows_out == 100
    assert result.dropped_over_cap == 0


def test_quoted_span_cap_limits_inventory_reuse(tmp_path: Path) -> None:
    """data48 varies its frames but reuses nine quoted spans, so the frame cap
    alone leaves it degenerate."""

    syllables = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
    rows = [
        (
            f'{syllables[index // 26]}{syllables[index % 26]} 담당자는 "결과만큼 설명도 중요하다"라고 했다.',
            "担当者は「結果と同じく説明も大事だ」と述べた。",
        )
        for index in range(200)
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)

    frames_only = RESAMPLE.resample_shard(source, tmp_path / "a.jsonl", max_per_skeleton=8)
    both = RESAMPLE.resample_shard(
        source, tmp_path / "b.jsonl", max_per_skeleton=8, max_per_quoted_span=5
    )

    assert frames_only.quoted_spans_in == 1
    assert frames_only.largest_span_in == 200
    assert frames_only.rows_out == 200  # every frame is distinct, nothing capped
    assert both.rows_out == 5
    assert both.dropped_over_span_cap == 195
    assert both.dropped_span_examples[0][0].strip("\"“”'‘’") == "결과만큼 설명도 중요하다"


def test_span_cap_is_optional(tmp_path: Path) -> None:
    rows = [("따옴표가 없는 문장이다.", "引用のない文だ。")]
    source = write_shard(tmp_path / "in.jsonl", rows)

    result = RESAMPLE.resample_shard(
        source, tmp_path / "out.jsonl", max_per_skeleton=8, max_per_quoted_span=1
    )

    assert result.rows_out == 1
    assert result.quoted_spans_in == 0


def test_span_cap_rejects_non_positive_values(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [("가나다라.", "アイウ。")])

    with pytest.raises(ValueError, match="max_per_quoted_span must be positive"):
        RESAMPLE.resample_shard(
            source, tmp_path / "o.jsonl", max_per_skeleton=1, max_per_quoted_span=0
        )
    assert (
        RESAMPLE.main(["--output-dir", str(tmp_path), "--max-per-quoted-span", "0", str(source)])
        == 2
    )


def test_selection_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    rows = [(f'표현 "{index}"을 옮긴다.', f"表現「{index}」を訳す。") for index in range(60)]
    forward = write_shard(tmp_path / "forward.jsonl", rows)
    backward = write_shard(tmp_path / "backward.jsonl", list(reversed(rows)))

    RESAMPLE.resample_shard(forward, tmp_path / "a.jsonl", max_per_skeleton=5)
    RESAMPLE.resample_shard(backward, tmp_path / "b.jsonl", max_per_skeleton=5)

    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_a_different_seed_selects_a_different_subset(tmp_path: Path) -> None:
    rows = [(f'표현 "{index}"을 옮긴다.', f"表現「{index}」を訳す。") for index in range(60)]
    source = write_shard(tmp_path / "in.jsonl", rows)

    RESAMPLE.resample_shard(source, tmp_path / "a.jsonl", max_per_skeleton=5, seed=1)
    RESAMPLE.resample_shard(source, tmp_path / "b.jsonl", max_per_skeleton=5, seed=2)

    assert (tmp_path / "a.jsonl").read_bytes() != (tmp_path / "b.jsonl").read_bytes()


def test_foreign_script_targets_are_dropped(tmp_path: Path) -> None:
    rows = [("한국어 맞춤법 확인.", "韓国語の誤記「왠지 모르개」を直す。")] * 1
    rows += [("다른 문장이 여기에 있다.", "別の文がここにある。")]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"

    result = RESAMPLE.resample_shard(source, output, max_per_skeleton=8)

    assert result.dropped_foreign_script == 1
    assert result.rows_out == 1
    assert all("왠지" not in row["ja"] for row in read_shard(output))


def test_korean_target_language_drops_kana_instead(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps({"kj": "오늘 スケジュール", "ko": "오늘 일정"}, ensure_ascii=False) + "\n"
        )
        handle.write(
            json.dumps({"kj": "오늘 ランチ", "ko": "오늘 ランチ"}, ensure_ascii=False) + "\n"
        )
    output = tmp_path / "out.jsonl"

    result = RESAMPLE.resample_shard(
        path,
        output,
        max_per_skeleton=8,
        source_key="kj",
        target_key="ko",
        target_language="ko",
    )

    assert result.dropped_foreign_script == 1
    assert result.rows_out == 1


def test_exact_duplicates_are_dropped(tmp_path: Path) -> None:
    rows = [("같은 문장이다.", "同じ文だ。")] * 5
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"

    result = RESAMPLE.resample_shard(source, output, max_per_skeleton=8)

    assert result.dropped_duplicate == 4
    assert result.rows_out == 1


def test_unreadable_rows_are_counted(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps({"ko": "정상 문장이다.", "ja": "正常な文だ。"}, ensure_ascii=False) + "\n"
        )
        handle.write("not json\n")
        handle.write(json.dumps({"ko": 3, "ja": "あ"}) + "\n")
        handle.write(json.dumps({"ko": "   ", "ja": "あ"}) + "\n")
    output = tmp_path / "out.jsonl"

    result = RESAMPLE.resample_shard(path, output, max_per_skeleton=8)

    assert result.rows_in == 4
    assert result.unreadable == 3
    assert result.rows_out == 1


def test_resample_rejects_bad_arguments(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [("가나다라.", "アイウ。")])

    with pytest.raises(ValueError, match="max_per_skeleton must be positive"):
        RESAMPLE.resample_shard(source, tmp_path / "o.jsonl", max_per_skeleton=0)
    with pytest.raises(ValueError, match="target_language must be"):
        RESAMPLE.resample_shard(
            source, tmp_path / "o.jsonl", max_per_skeleton=1, target_language="en"
        )
    with pytest.raises(FileNotFoundError):
        RESAMPLE.resample_shard(
            tmp_path / "missing.jsonl", tmp_path / "o.jsonl", max_per_skeleton=1
        )


def test_main_writes_to_an_output_directory(tmp_path: Path, capsys) -> None:
    rows = [(f'표현 "{index}"을 옮긴다.', f"表現「{index}」を訳す。") for index in range(40)]
    source = write_shard(tmp_path / "data44.jsonl", rows)
    report = tmp_path / "report.json"

    assert (
        RESAMPLE.main(
            [
                "--output-dir",
                str(tmp_path / "resampled"),
                "--max-per-skeleton",
                "4",
                "--report",
                str(report),
                str(source),
            ]
        )
        == 0
    )

    assert len(read_shard(tmp_path / "resampled" / "data44.jsonl")) == 4
    assert len(read_shard(source)) == 40  # the input is untouched
    assert json.loads(report.read_text(encoding="utf-8"))[0]["rows_out"] == 4
    assert "data44.jsonl" in capsys.readouterr().out


def test_main_in_place_keeps_an_original_copy(tmp_path: Path) -> None:
    rows = [(f'표현 "{index}"을 옮긴다.', f"表現「{index}」を訳す。") for index in range(40)]
    source = write_shard(tmp_path / "data44.jsonl", rows)

    assert RESAMPLE.main(["--in-place", "--max-per-skeleton", "4", str(source)]) == 0

    assert len(read_shard(source)) == 4
    assert len(read_shard(tmp_path / "data44.jsonl.orig")) == 40


def test_main_in_place_is_idempotent(tmp_path: Path) -> None:
    rows = [(f'표현 "{index}"을 옮긴다.', f"表現「{index}」を訳す。") for index in range(40)]
    source = write_shard(tmp_path / "data44.jsonl", rows)

    RESAMPLE.main(["--in-place", "--max-per-skeleton", "4", str(source)])
    first = source.read_bytes()
    RESAMPLE.main(["--in-place", "--max-per-skeleton", "4", str(source)])

    # The second run must resample the preserved original, not the reduced file.
    assert source.read_bytes() == first


def test_main_reports_bad_input(tmp_path: Path) -> None:
    source = write_shard(tmp_path / "in.jsonl", [("가나다라.", "アイウ。")])

    assert (
        RESAMPLE.main(["--output-dir", str(tmp_path), "--max-per-skeleton", "0", str(source)]) == 2
    )
    assert (
        RESAMPLE.main(
            ["--output-dir", str(tmp_path), str(tmp_path / "missing.jsonl")],
        )
        == 2
    )
