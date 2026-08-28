from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "dedup_corpus.py"
SPEC = importlib.util.spec_from_file_location("dedup_corpus_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DEDUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEDUP
SPEC.loader.exec_module(DEDUP)


def write_shard(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def run(
    tmp_path: Path,
    inputs: list[Path],
    *extra: str,
    pairs: tuple[tuple[str, str], ...] = (("ko", "ja"),),
) -> dict[str, object]:
    report = tmp_path / "report.json"
    arguments = [str(path) for path in inputs]
    for language_a, language_b in pairs:
        arguments += ["--pair", language_a, language_b]
    arguments += ["--report", str(report), "--jobs", "1", *extra]
    assert DEDUP.main(arguments) == 0
    return json.loads(report.read_text(encoding="utf-8"))


def kept_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_loose_identity_ignores_punctuation_but_keeps_symbol_only_lines() -> None:
    assert DEDUP.loose_identity(">>알겠습니다.") == DEDUP.loose_identity("알겠습니다")
    assert DEDUP.loose_identity("……") != DEDUP.loose_identity("!!!")


def test_cleanliness_prefers_the_unquoted_copy() -> None:
    quoted = DEDUP.cleanliness_score([">>알겠습니다."])
    clean = DEDUP.cleanliness_score(["알겠습니다."])

    assert clean > quoted


def test_duplicate_across_shards_is_removed_once(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    second = write_shard(
        tmp_path / "data02.jsonl",
        [{"ko": "가", "ja": "あ"}, {"ko": "나", "ja": "い"}],
    )
    output = tmp_path / "out"

    report = run(tmp_path, [first, second], "--output-dir", str(output))

    assert report["removed"]["duplicate_exact"] == 1
    assert kept_rows(output / "data01.jsonl") == [{"ko": "가", "ja": "あ"}]
    assert kept_rows(output / "data02.jsonl") == [{"ko": "나", "ja": "い"}]


def test_quoted_copy_loses_to_the_clean_copy_even_when_it_comes_first(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [
            {"ko": ">>알겠습니다.", "ja": ">わかりました。"},
            {"ko": "알겠습니다.", "ja": "わかりました。"},
        ],
    )
    output = tmp_path / "out"

    report = run(tmp_path, [shard], "--output-dir", str(output))

    assert report["removed"]["duplicate_loose"] == 1
    assert kept_rows(output / "data01.jsonl") == [{"ko": "알겠습니다.", "ja": "わかりました。"}]


def test_row_survives_when_only_one_of_its_pairs_is_a_duplicate(tmp_path: Path) -> None:
    real = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    dialect = write_shard(
        tmp_path / "data02.jsonl",
        [{"jd": "あー", "ko": "가", "ja": "あ"}],
    )
    output = tmp_path / "out"

    report = run(
        tmp_path,
        [real, dialect],
        "--output-dir",
        str(output),
        pairs=(("ko", "ja"), ("jd", "ko"), ("jd", "ja")),
    )

    assert report["edges"]["ko-ja"]["dropped_duplicate_exact"] == 1
    assert kept_rows(output / "data02.jsonl") == [{"jd": "あー", "ko": "가", "ja": "あ"}]


def test_exact_only_shard_keeps_punctuation_variants(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "synthetic_numeric_data38.jsonl",
        [
            {"ko": "2026-07-31", "ja": "2026-07-31"},
            {"ko": "2026/07/31", "ja": "2026/07/31"},
        ],
    )
    output = tmp_path / "out"

    report = run(tmp_path, [shard], "--output-dir", str(output))

    assert report["removed"]["duplicate_loose"] == 0
    assert len(kept_rows(output / "synthetic_numeric_data38.jsonl")) == 2


def test_generated_shard_loses_precedence_to_real_data(tmp_path: Path) -> None:
    generated = write_shard(tmp_path / "synthetic_a.jsonl", [{"ko": "가", "ja": "あ"}])
    real = write_shard(tmp_path / "zz_real.jsonl", [{"ko": "가", "ja": "あ"}])
    output = tmp_path / "out"

    run(tmp_path, [generated, real], "--output-dir", str(output))

    assert kept_rows(output / "synthetic_a.jsonl") == []
    assert kept_rows(output / "zz_real.jsonl") == [{"ko": "가", "ja": "あ"}]


def test_holdout_sentence_is_removed_from_training(tmp_path: Path) -> None:
    holdout = write_shard(tmp_path / "holdout.jsonl", [{"ko": "가", "ja": "あ"}])
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "全然ちがう訳"}, {"ko": "나", "ja": "い"}],
    )
    output = tmp_path / "out"

    report = run(
        tmp_path,
        [shard],
        "--output-dir",
        str(output),
        "--holdout",
        str(holdout),
    )

    assert report["removed"]["holdout_denylist"] == 1
    assert kept_rows(output / "data01.jsonl") == [{"ko": "나", "ja": "い"}]


def test_one_to_many_cap_keeps_the_best_translations(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [
            {"ko": "가", "ja": "あ"},
            {"ko": "가", "ja": "い"},
            {"ko": "가", "ja": "う"},
        ],
    )
    output = tmp_path / "out"

    report = run(
        tmp_path,
        [shard],
        "--output-dir",
        str(output),
        "--max-targets-per-source",
        "2",
    )

    assert report["removed"]["one_to_many_cap"] == 1
    assert len(kept_rows(output / "data01.jsonl")) == 2


def test_removed_rows_are_archived_with_a_reason(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "あ"}, {"ko": "가", "ja": "あ"}],
    )
    output = tmp_path / "out"
    archive = tmp_path / "archive"

    run(tmp_path, [shard], "--output-dir", str(output), "--archive-dir", str(archive))

    archived = kept_rows(archive / "data01.removed.jsonl")
    assert len(archived) == 1
    assert archived[0]["reason"] == "duplicate_exact"
    assert archived[0]["line"] == 2
    assert archived[0]["row"] == {"ko": "가", "ja": "あ"}


def test_in_place_replaces_the_shard_and_archives_first(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "あ"}, {"ko": "가", "ja": "あ"}],
    )
    archive = tmp_path / "archive"

    run(tmp_path, [shard], "--in-place", "--archive-dir", str(archive))

    assert kept_rows(shard) == [{"ko": "가", "ja": "あ"}]
    assert (archive / "data01.removed.jsonl").exists()


def test_report_only_run_leaves_every_file_untouched(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "あ"}, {"ko": "가", "ja": "あ"}],
    )
    before = shard.read_text(encoding="utf-8")

    report = run(tmp_path, [shard])

    assert report["removed"]["duplicate_exact"] == 1
    assert shard.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("*.jsonl")) == [shard]


def test_rows_without_a_configured_pair_are_kept_by_default(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "あ"}, {"note": "no language keys"}],
    )
    output = tmp_path / "out"

    run(tmp_path, [shard], "--output-dir", str(output))

    assert len(kept_rows(output / "data01.jsonl")) == 2


def test_rows_without_a_configured_pair_can_be_dropped(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "あ"}, {"note": "no language keys"}],
    )
    output = tmp_path / "out"

    run(tmp_path, [shard], "--output-dir", str(output), "--drop-rows-without-a-pair")

    assert kept_rows(output / "data01.jsonl") == [{"ko": "가", "ja": "あ"}]


def test_in_place_and_output_dir_are_mutually_exclusive(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])

    code = DEDUP.main(
        [str(shard), "--pair", "ko", "ja", "--in-place", "--output-dir", str(tmp_path / "out")]
    )

    assert code == 2


def test_shard_emptied_by_deduplication_is_reported(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    second = write_shard(tmp_path / "data02.jsonl", [{"ko": "가", "ja": "あ"}])
    output = tmp_path / "out"

    report = run(tmp_path, [first, second], "--output-dir", str(output))

    assert report["emptied_shards"] == ["data02.jsonl"]


def test_invalid_json_line_is_reported_with_its_location(tmp_path: Path) -> None:
    shard = tmp_path / "data01.jsonl"
    shard.write_text('{"ko": "가", "ja": "あ"}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="data01.jsonl:2"):
        DEDUP.main([str(shard), "--pair", "ko", "ja", "--jobs", "1"])


def test_reuse_staging_skips_rescanning_an_unchanged_shard(tmp_path: Path) -> None:
    shard = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": "가", "ja": "あ"}, {"ko": "가", "ja": "あ"}],
    )
    staging = tmp_path / "staging"

    first = run(tmp_path, [shard], "--staging-dir", str(staging))
    second = run(tmp_path, [shard], "--staging-dir", str(staging), "--reuse-staging")

    assert first["removed"] == second["removed"]


def test_reuse_staging_rescans_a_changed_shard(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    staging = tmp_path / "staging"

    run(tmp_path, [shard], "--staging-dir", str(staging))
    write_shard(shard, [{"ko": "가", "ja": "あ"}, {"ko": "가", "ja": "あ"}])
    report = run(tmp_path, [shard], "--staging-dir", str(staging), "--reuse-staging")

    assert report["removed"]["duplicate_exact"] == 1


def test_trilingual_row_outranks_the_bilingual_copy_of_the_same_sentence(tmp_path: Path) -> None:
    bilingual = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    trilingual = write_shard(tmp_path / "data02.jsonl", [{"ko": "가", "en": "a", "ja": "あ"}])
    output = tmp_path / "out"

    run(
        tmp_path,
        [bilingual, trilingual],
        "--output-dir",
        str(output),
        pairs=(("ko", "ja"), ("ko", "en"), ("en", "ja")),
    )

    assert kept_rows(output / "data01.jsonl") == []
    assert kept_rows(output / "data02.jsonl") == [{"ko": "가", "en": "a", "ja": "あ"}]


def test_cleanliness_is_scored_per_pair_not_per_row(tmp_path: Path) -> None:
    quoted_trilingual = write_shard(
        tmp_path / "data01.jsonl",
        [{"ko": ">>가.", "en": ">>a.", "ja": ">>あ。"}],
    )
    clean_trilingual = write_shard(
        tmp_path / "data02.jsonl",
        [{"ko": "가.", "en": "a.", "ja": "あ。"}],
    )
    output = tmp_path / "out"

    run(
        tmp_path,
        [quoted_trilingual, clean_trilingual],
        "--output-dir",
        str(output),
        pairs=(("ko", "ja"), ("ko", "en"), ("en", "ja")),
    )

    assert kept_rows(output / "data01.jsonl") == []
    assert kept_rows(output / "data02.jsonl") == [{"ko": "가.", "en": "a.", "ja": "あ。"}]


def test_reuse_staging_rejects_an_older_staging_format(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    staging = tmp_path / "staging"
    run(tmp_path, [shard], "--staging-dir", str(staging))

    summary_path = staging / "data01.jsonl.shard.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["format"] = "dedup-corpus-staging-v0"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert DEDUP.reuse_scan(shard, staging) is None


FAST_PATH_PAIRS = (("ko", "ja"), ("ko", "en"), ("en", "ja"), ("jd", "ko"), ("jd", "ja"))
FAST_PATH_LANGUAGES = frozenset(language for pair in FAST_PATH_PAIRS for language in pair)


def fast(row: object) -> list[tuple[str, str, str, str]] | None:
    return DEDUP.expand_flat_record(row, FAST_PATH_PAIRS, FAST_PATH_LANGUAGES)


def reference(row: object) -> list[tuple[str, str, str, str]]:
    return DEDUP.reference_expansion(row, [list(pair) for pair in FAST_PATH_PAIRS])


@pytest.mark.parametrize(
    "row",
    [
        {"ko": "가", "ja": "あ"},
        {"ko": "가", "en": "a", "ja": "あ"},
        {"jd": "あー", "ko": "가", "ja": "あ"},
        {"ko": "가", "ja": "あ", "domain": "일상생활", "style": "구어체"},
        {"en": ["one", "two"], "ja": ["いち", "に"]},
        {"en": ["one", "one"], "ja": ["いち", "いち"]},
        {"ko": "가", "ja": "  "},
        {"ko": "가"},
        {"note": "no language keys"},
    ],
)
def test_fast_path_matches_the_reference_expansion(row: dict[str, object]) -> None:
    expanded = fast(row)

    assert expanded is not None
    assert expanded == reference(row)


@pytest.mark.parametrize(
    "row",
    [
        {"ko-ja": {"ko": "가", "ja": "あ"}},
        {"records": [{"ko": "가", "ja": "あ"}]},
        {"source_language": "ko", "target_language": "ja", "source": "가", "target": "あ"},
        {"en": ["one", "two"], "ja": ["いち"]},
        {"ko": 5, "ja": "あ"},
    ],
)
def test_layouts_outside_the_flat_shape_defer_to_the_reference(row: dict[str, object]) -> None:
    assert fast(row) is None


def test_verify_sample_reports_the_offending_line(tmp_path: Path, monkeypatch) -> None:
    shard = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    monkeypatch.setattr(DEDUP, "reference_expansion", lambda row, pairs: [])

    with pytest.raises(ValueError, match="data01.jsonl:1"):
        DEDUP.scan_shard((str(shard), str(tmp_path / "staging"), [["ko", "ja"]], 500))


def test_verify_sample_can_be_switched_off(tmp_path: Path, monkeypatch) -> None:
    shard = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    monkeypatch.setattr(DEDUP, "reference_expansion", lambda row, pairs: [])

    summary = DEDUP.scan_shard((str(shard), str(tmp_path / "staging"), [["ko", "ja"]], 0))

    assert summary["rows"] == 1


def test_a_shard_that_loses_nothing_is_left_untouched(tmp_path: Path) -> None:
    untouched = write_shard(tmp_path / "data01.jsonl", [{"ko": "가", "ja": "あ"}])
    losing = write_shard(
        tmp_path / "data02.jsonl",
        [{"ko": "가", "ja": "あ"}, {"ko": "나", "ja": "い"}],
    )
    before = untouched.stat().st_mtime_ns

    report = run(tmp_path, [untouched, losing], "--in-place", "--archive-dir", str(tmp_path / "a"))

    assert untouched.stat().st_mtime_ns == before
    assert report["unchanged_shards"] == 1
    assert report["files"]["data01.jsonl"]["rewritten"] is False
    assert report["files"]["data02.jsonl"]["rewritten"] is True
    assert kept_rows(losing) == [{"ko": "나", "ja": "い"}]
