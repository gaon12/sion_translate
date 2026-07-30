"""Recovery must repair what is repairable and refuse to invent alignments.

``QualityPolicy()`` accepts every row of data23, data25 and data50, so the tool
that decides whether an excluded shard comes back has to justify each drop by a
named defect. These tests pin the two repairs that matter (segmenter spacing and
fan-out resolution) and pin that determinism does not depend on file order.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "recover_shard.py"
SPEC = importlib.util.spec_from_file_location("recover_shard_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RECOVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECOVER
SPEC.loader.exec_module(RECOVER)


def write_shard(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_prepare_repairs_segmenter_spacing_without_dropping_the_row(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [{"ko": "달콤한 향기가 코끝을 스친다", "ja": "甘い 香り が 鼻先 を かすめる"}],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.prepare_shard(
        source,
        output,
        source_key="ko",
        target_key="ja",
        source_scripts=["ko"],
        target_scripts=["ja"],
        repair_spacing=True,
        policy=RECOVER.QualityPolicy(),
        apply_quality=True,
    )
    assert result.rows_out == 1
    assert result.spacing_repaired == 1
    assert result.spaces_removed == 5
    assert read_shard(output)[0]["ja"] == "甘い香りが鼻先をかすめる"


def test_prepare_leaves_the_space_using_side_alone(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [{"ko": "철 원석을 찾는다", "ja": "鉄原石を探す"}],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.prepare_shard(
        source,
        output,
        source_key="ko",
        target_key="ja",
        source_scripts=["ko"],
        target_scripts=["ja"],
        repair_spacing=True,
        policy=RECOVER.QualityPolicy(),
        apply_quality=True,
    )
    assert result.spacing_repaired == 0
    assert read_shard(output)[0]["ko"] == "철 원석을 찾는다"


def test_prepare_drops_rows_whose_target_carries_the_wrong_script(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [
            {"ko": "이 이상은 무리다", "ja": "これ 이상は無理だ"},
            {"ko": "이 이상은 무리다", "ja": "これ以上は無理だ"},
        ],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.prepare_shard(
        source,
        output,
        source_key="ko",
        target_key="ja",
        source_scripts=["ko"],
        target_scripts=["ja"],
        repair_spacing=True,
        policy=RECOVER.QualityPolicy(),
        apply_quality=True,
    )
    assert result.dropped_foreign_script == 1
    assert result.rows_out == 1
    assert read_shard(output)[0]["ja"] == "これ以上は無理だ"


def test_prepare_reports_the_fanout_it_did_not_resolve(tmp_path: Path) -> None:
    # prepare must not silently pick a winner; that needs a score.
    rows = [{"ko": "보상 받으러 왔어요", "ja": f"報酬もらいに来た{index}"} for index in range(5)]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = RECOVER.prepare_shard(
        source,
        output,
        source_key="ko",
        target_key="ja",
        source_scripts=["ko"],
        target_scripts=["ja"],
        repair_spacing=True,
        policy=RECOVER.QualityPolicy(),
        apply_quality=True,
    )
    assert result.rows_out == 5
    assert result.distinct_sources == 1
    assert result.max_targets_per_source == 5


def test_prepare_drops_exact_duplicate_pairs(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [{"ko": "괜찮아", "ja": "大丈夫だ"}] * 4,
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.prepare_shard(
        source,
        output,
        source_key="ko",
        target_key="ja",
        source_scripts=["ko"],
        target_scripts=["ja"],
        repair_spacing=True,
        policy=RECOVER.QualityPolicy(),
        apply_quality=True,
    )
    assert result.dropped_duplicate_pair == 3
    assert result.rows_out == 1


def test_prepare_keeps_unrelated_fields(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [{"ko": "괜찮아", "ja": "大丈夫 だ", "domain": "r18", "resource_id": "line:7"}],
    )
    output = tmp_path / "out.jsonl"
    RECOVER.prepare_shard(
        source,
        output,
        source_key="ko",
        target_key="ja",
        source_scripts=["ko"],
        target_scripts=["ja"],
        repair_spacing=True,
        policy=RECOVER.QualityPolicy(),
        apply_quality=True,
    )
    row = read_shard(output)[0]
    assert row["domain"] == "r18"
    assert row["resource_id"] == "line:7"
    assert row["ja"] == "大丈夫だ"


def test_select_keeps_the_highest_scoring_target_per_source(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [
            {"ko": "보상 받으러 왔어요", "ja": "報酬もらいに来たよ", "semantic_similarity": 0.91},
            {"ko": "보상 받으러 왔어요", "ja": "鉄原石を探す", "semantic_similarity": 0.42},
            {"ko": "보상 받으러 왔어요", "ja": "全然平気だよ", "semantic_similarity": 0.55},
        ],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.select_alignments(
        source,
        output,
        source_key="ko",
        target_key="ja",
        score_key="semantic_similarity",
        min_similarity=0.40,
        unique_source=True,
        unique_target=False,
        seed="t",
    )
    assert result.rows_out == 1
    assert result.dropped_duplicate_source == 2
    assert read_shard(output)[0]["ja"] == "報酬もらいに来たよ"


def test_select_cuts_the_low_similarity_tail(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [
            {"ko": "가", "ja": "行く", "semantic_similarity": 0.88},
            {"ko": "나", "ja": "無関係な文", "semantic_similarity": 0.31},
        ],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.select_alignments(
        source,
        output,
        source_key="ko",
        target_key="ja",
        score_key="semantic_similarity",
        min_similarity=0.80,
        unique_source=True,
        unique_target=False,
        seed="t",
    )
    assert result.rows_out == 1
    assert result.dropped_below_threshold == 1
    assert result.dropped_examples[0]["score"] == 0.31


def test_select_is_independent_of_file_order(tmp_path: Path) -> None:
    rows = [
        {"ko": "같은 문장", "ja": f"候補{index}", "semantic_similarity": 0.5 + index / 100}
        for index in range(6)
    ]
    forward = write_shard(tmp_path / "a.jsonl", rows)
    backward = write_shard(tmp_path / "b.jsonl", list(reversed(rows)))
    first = tmp_path / "a.out.jsonl"
    second = tmp_path / "b.out.jsonl"
    for source, output in ((forward, first), (backward, second)):
        RECOVER.select_alignments(
            source,
            output,
            source_key="ko",
            target_key="ja",
            score_key="semantic_similarity",
            min_similarity=0.0,
            unique_source=True,
            unique_target=False,
            seed="t",
        )
    assert read_shard(first) == read_shard(second)


def test_select_can_also_require_unique_targets(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [
            {"ko": "첫 문장", "ja": "同じ訳", "semantic_similarity": 0.90},
            {"ko": "둘째 문장", "ja": "同じ訳", "semantic_similarity": 0.85},
        ],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.select_alignments(
        source,
        output,
        source_key="ko",
        target_key="ja",
        score_key="semantic_similarity",
        min_similarity=0.0,
        unique_source=True,
        unique_target=True,
        seed="t",
    )
    assert result.rows_out == 1
    assert result.dropped_duplicate_target == 1
    assert read_shard(output)[0]["ko"] == "첫 문장"


def test_select_reports_the_score_distribution(tmp_path: Path) -> None:
    rows = [
        {"ko": f"문장{index}", "ja": f"文{index}", "semantic_similarity": index / 100}
        for index in range(101)
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    result = RECOVER.select_alignments(
        source,
        output,
        source_key="ko",
        target_key="ja",
        score_key="semantic_similarity",
        min_similarity=0.0,
        unique_source=True,
        unique_target=False,
        seed="t",
    )
    assert result.similarity_percentiles["minimum"] == 0.0
    assert result.similarity_percentiles["median"] == 0.5
    assert result.similarity_percentiles["maximum"] == 1.0


def test_select_refuses_rows_without_a_score(tmp_path: Path) -> None:
    source = write_shard(
        tmp_path / "in.jsonl",
        [
            {"ko": "점수 있음", "ja": "有り", "semantic_similarity": 0.9},
            {"ko": "점수 없음", "ja": "無し"},
        ],
    )
    output = tmp_path / "out.jsonl"
    result = RECOVER.select_alignments(
        source,
        output,
        source_key="ko",
        target_key="ja",
        score_key="semantic_similarity",
        min_similarity=0.0,
        unique_source=True,
        unique_target=False,
        seed="t",
    )
    assert result.dropped_missing_score == 1
    assert result.rows_out == 1


def test_cli_runs_both_stages(tmp_path: Path) -> None:
    raw = write_shard(
        tmp_path / "raw.jsonl",
        [
            {"ko": "보상 받으러 왔어요", "ja": "報酬 もらいに 来た"},
            {"ko": "이 이상은 무리다", "ja": "これ 이상は無理だ"},
        ],
    )
    prepared = tmp_path / "prepared.jsonl"
    assert (
        RECOVER.main(
            [
                "prepare",
                str(raw),
                "--output",
                str(prepared),
                "--source-scripts",
                "ko",
                "--target-scripts",
                "ja",
                "--report",
                str(tmp_path / "prepare.json"),
            ]
        )
        == 0
    )
    assert len(read_shard(prepared)) == 1
    assert read_shard(prepared)[0]["ja"] == "報酬もらいに来た"

    scored = write_shard(
        tmp_path / "scored.jsonl",
        [dict(read_shard(prepared)[0], semantic_similarity=0.87)],
    )
    final = tmp_path / "final.jsonl"
    assert (
        RECOVER.main(
            [
                "select",
                str(scored),
                "--output",
                str(final),
                "--min-similarity",
                "0.8",
                "--unique-source",
                "--report",
                str(tmp_path / "select.json"),
            ]
        )
        == 0
    )
    assert len(read_shard(final)) == 1
    report = json.loads((tmp_path / "select.json").read_text(encoding="utf-8"))
    assert report["rows_out"] == 1


def test_cli_rejects_a_missing_input(tmp_path: Path) -> None:
    assert (
        RECOVER.main(
            [
                "prepare",
                str(tmp_path / "nope.jsonl"),
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )
        == 2
    )
