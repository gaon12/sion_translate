"""The skeleton cap is what keeps template generation from becoming restatement.

data44 and data45 lost 90% of their rows to one frame restated thousands of times.
This generator is allowed to use templates because the domains are formulaic, so
the cap has to be enforced rather than assumed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_formulaic_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_formulaic_corpus_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(output: Path, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "source_key": "ko",
        "target_key": "ja",
        "max_per_skeleton": 1,
        "max_combinations": 500,
        "seed": "test",
        "domains": [],
    }
    arguments.update(overrides)
    return BUILD.build(output, **arguments)


def test_one_row_per_skeleton_gives_a_perfect_skelttr(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    result = run(output, max_per_skeleton=1)
    rows = read_shard(output)
    skeletons = {BUILD.frame_skeleton(str(row["ko"])) for row in rows}
    assert len(skeletons) == len(rows)
    assert result.rows_written == len(rows)


def test_raising_the_cap_raises_rows_per_skeleton(tmp_path: Path) -> None:
    one = run(tmp_path / "one.jsonl", max_per_skeleton=1)
    three = run(tmp_path / "three.jsonl", max_per_skeleton=3)
    assert three.rows_written > one.rows_written
    rows = read_shard(tmp_path / "three.jsonl")
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['domain']}|{BUILD.frame_skeleton(str(row['ko']))}"
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 3


def test_a_row_records_its_domain_and_frame(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    run(output)
    row = read_shard(output)[0]
    assert row["ko"] and row["ja"]
    assert row["domain"]
    assert row["domain_label"]
    assert isinstance(row["frame_index"], int)
    assert row["synthetic"] is True


def test_no_placeholder_survives_into_the_output(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    result = run(output)
    for row in read_shard(output):
        assert "{" not in str(row["ko"]), row
        assert "{" not in str(row["ja"]), row
    assert result.skipped_unresolved == 0


def test_no_particle_alternation_survives_into_the_output(tmp_path: Path) -> None:
    # `방지을/를` in a shipped row would be worse than the wrong particle.
    output = tmp_path / "out.jsonl"
    run(output)
    for row in read_shard(output):
        assert "/" not in str(row["ko"]) or "http" in str(row["ko"]), row


def test_output_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    run(first)
    run(second)
    assert read_shard(first) == read_shard(second)


def test_a_different_seed_changes_which_rows_are_kept(tmp_path: Path) -> None:
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    run(one, seed="alpha", max_combinations=20)
    run(two, seed="omega", max_combinations=20)
    assert read_shard(one) != read_shard(two)


def test_domains_can_be_restricted(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    result = run(output, domains=["medical"])
    assert set(result.per_domain) == {"medical"}
    assert {row["domain"] for row in read_shard(output)} == {"medical"}


def test_an_unknown_domain_is_rejected() -> None:
    try:
        BUILD.domain_list("legal,finance")
    except Exception as error:
        assert "unknown domain" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a rejection")


def test_an_invalid_lexicon_stops_the_build(tmp_path: Path) -> None:
    bad = BUILD.lexicon.Domain(
        code="bad",
        label="bad",
        frames=(BUILD.lexicon.Frame("{days}일", "{days}日"),),
    )
    original = BUILD.lexicon.DOMAINS
    try:
        BUILD.lexicon.DOMAINS = (bad,)
        try:
            run(tmp_path / "out.jsonl")
        except ValueError as error:
            assert "lexicon is invalid" in str(error)
        else:  # pragma: no cover
            raise AssertionError("expected a rejection")
    finally:
        BUILD.lexicon.DOMAINS = original


def test_the_report_totals_agree_with_the_file(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    result = run(output)
    assert sum(result.per_domain.values()) == result.rows_written
    assert result.rows_written == len(read_shard(output))
    for code, count in result.per_domain.items():
        # With a cap of one, rows and skeletons must match exactly.
        assert result.distinct_skeletons[code] == count


def test_frame_skeleton_matches_the_audit_definition() -> None:
    assert BUILD.frame_skeleton('값은 "0.5"이고 3개다') == "값은 <Q>이고 #개다"
