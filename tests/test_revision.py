"""초안 수정 학습 데이터 생성 검증."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from sion_translate.evaluation import has_excessive_repetition, numeric_tokens
from sion_translate.revision import (
    DEFAULT_CORRUPTIONS,
    DRAFT_SEPARATOR,
    RevisionExample,
    build_revision_examples,
    corrupt_target,
    parse_revision_input,
    serialize_revision_input,
    write_revision_examples,
)
from sion_translate.tokenizer import OPTIONAL_CONTROL_SYMBOLS


def _rng() -> random.Random:
    return random.Random(20260726)


def test_serialize_round_trips() -> None:
    serialized = serialize_revision_input("원문입니다.", "初稿です。")
    assert DRAFT_SEPARATOR in serialized
    assert parse_revision_input(serialized) == ("원문입니다.", "初稿です。")


def test_parse_rejects_input_without_the_separator() -> None:
    with pytest.raises(ValueError, match=DRAFT_SEPARATOR):
        parse_revision_input("구분자가 없는 문장")


def test_draft_separator_is_a_reserved_control_symbol() -> None:
    # 구분자가 여러 조각으로 쪼개지면 학습 때와 다른 입력이 된다.
    assert DRAFT_SEPARATOR in OPTIONAL_CONTROL_SYMBOLS


def test_number_corruption_changes_a_value_and_keeps_the_rest() -> None:
    target = "1回250mgずつ、48時間間隔で服用してください。"
    draft = corrupt_target("1회 250mg씩, 48시간 간격", target, "number", _rng())
    assert draft != target
    # 숫자만 달라져야 한다 — 나머지가 함께 망가지면 무엇을 고쳐야 하는지 불분명해진다.
    assert numeric_tokens(draft) != numeric_tokens(target)
    assert len(numeric_tokens(draft)) == len(numeric_tokens(target))


def test_number_corruption_is_a_no_op_without_digits() -> None:
    target = "숫자가 없는 문장입니다."
    assert corrupt_target("원문", target, "number", _rng()) == target


def test_drop_clause_removes_one_clause() -> None:
    target = "検査の結果は良好です。来週また確認します。念のため記録します。"
    draft = corrupt_target("원문", target, "drop_clause", _rng())
    assert len(draft) < len(target)
    assert draft in target or all(part in target for part in draft.split("。") if part)


def test_truncate_keeps_only_a_prefix() -> None:
    target = "電車が遅れていなければ間に合ったはずだが、駅に着いた時には受付が終わっていた。"
    draft = corrupt_target("원문", target, "truncate", _rng())
    assert target.startswith(draft)
    assert draft != target


def test_repeat_produces_a_collapse_the_reward_can_detect() -> None:
    target = "二つの恋が進化する。本当だよ。"
    draft = corrupt_target("원문", target, "repeat", _rng())
    assert len(draft) > len(target)
    assert has_excessive_repetition(draft)


def test_copy_source_returns_the_untranslated_source() -> None:
    assert corrupt_target("원문 그대로", "正しい訳", "copy_source", _rng()) == "원문 그대로"


def test_swap_reorders_clauses_without_losing_them() -> None:
    target = "まず原因を確認する。次に対策を決める。"
    draft = corrupt_target("원문", target, "swap", _rng())
    assert draft != target
    # 조각은 그대로 있고 순서만 바뀌어야 한다.
    assert sorted(draft.split("。")) == sorted(target.split("。"))


def test_identity_leaves_the_target_alone() -> None:
    assert corrupt_target("원문", "正しい訳", "identity", _rng()) == "正しい訳"


def test_unknown_corruption_is_rejected() -> None:
    with pytest.raises(ValueError, match="알 수 없는 손상 유형"):
        corrupt_target("원문", "번역", "shuffle_everything", _rng())


def _pairs(count: int = 200) -> list[tuple[str, str]]:
    return [
        (
            f"{index}번 항목의 금액은 {1000 + index}원입니다. 확인해 주세요.",
            f"{index}番の項目の金額は{1000 + index}ウォンです。ご確認ください。",
        )
        for index in range(count)
    ]


def test_build_produces_one_example_per_pair_with_the_clean_target() -> None:
    pairs = _pairs()
    examples, stats = build_revision_examples(pairs, seed=1)
    assert len(examples) == len(pairs) == stats.written
    for (serialized, target), (source, clean) in zip(examples, pairs, strict=True):
        parsed_source, _ = parse_revision_input(serialized)
        assert parsed_source == source
        # 정답은 항상 손상되지 않은 번역이어야 한다.
        assert target == clean


def test_identity_examples_exist_so_correct_drafts_are_left_alone() -> None:
    _, stats = build_revision_examples(_pairs(), seed=2)
    assert stats.by_corruption.get("identity", 0) > 0
    # 손상이 실제로 통하지 않은 경우까지 포함해 "그대로 두기" 예제가 있어야 한다.
    assert stats.unchanged >= stats.by_corruption.get("identity", 0)


def test_all_default_corruptions_appear_over_enough_pairs() -> None:
    _, stats = build_revision_examples(_pairs(600), seed=3)
    assert set(stats.by_corruption) == set(DEFAULT_CORRUPTIONS)


def test_weights_can_select_a_single_corruption() -> None:
    _, stats = build_revision_examples(_pairs(50), weights={"number": 1.0}, seed=4)
    assert set(stats.by_corruption) <= {"number", "identity"}


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="알 수 없는 손상 유형"):
        build_revision_examples(_pairs(4), weights={"bogus": 1.0})
    with pytest.raises(ValueError, match="가중치의 합"):
        build_revision_examples(_pairs(4), weights={"number": 0.0})


def test_same_seed_is_reproducible() -> None:
    first, _ = build_revision_examples(_pairs(40), seed=11)
    second, _ = build_revision_examples(_pairs(40), seed=11)
    third, _ = build_revision_examples(_pairs(40), seed=12)
    assert first == second
    assert first != third


def test_written_file_is_a_plain_translation_pair(tmp_path: Path) -> None:
    """데이터 파이프라인을 고치지 않고 쓸 수 있어야 한다."""
    examples, _ = build_revision_examples(_pairs(5), seed=5)
    output = tmp_path / "revise_synthetic.jsonl"
    assert write_revision_examples(output, examples, ("ko", "ja")) == 5
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(set(row) == {"ko", "ja", "synthetic", "training_direction"} for row in rows)
    assert all(DRAFT_SEPARATOR in row["ko"] for row in rows)
    assert all(row["synthetic"] is True for row in rows)
    assert all(row["training_direction"] == ["ko", "ja"] for row in rows)


def test_written_revision_direction_and_keys_are_canonical_bcp47(tmp_path: Path) -> None:
    output = tmp_path / "revise_variants.jsonl"
    assert (
        write_revision_examples(
            output,
            [("fonte <draft> rascunho", "譯文")],
            ("PT-br", "ZH-hant"),
        )
        == 1
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row == {
        "pt-BR": "fonte <draft> rascunho",
        "zh-Hant": "譯文",
        "synthetic": True,
        "training_direction": ["pt-BR", "zh-Hant"],
    }


def test_revision_write_failure_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "revise_atomic.jsonl"
    output.write_text("existing output\n", encoding="utf-8")

    def failing_examples():
        yield "source <draft> draft", "target"
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        write_revision_examples(output, failing_examples(), ("sw", "ar"))

    assert output.read_text(encoding="utf-8") == "existing output\n"
    assert list(tmp_path.glob(".revise_atomic.jsonl.*.tmp")) == []


def test_revision_rejects_reverse_scoped_input_without_replacing_output(tmp_path: Path) -> None:
    output = tmp_path / "revise_direction.jsonl"
    output.write_text("existing output\n", encoding="utf-8")
    example = RevisionExample(
        "source <draft> draft",
        "target",
        {"training_direction": ["ar", "sw"]},
        source_identifier="bt_rows.jsonl:7",
    )

    with pytest.raises(ValueError, match="does not match the requested revision direction"):
        write_revision_examples(output, [example], ("sw", "ar"))

    assert output.read_text(encoding="utf-8") == "existing output\n"


def test_revision_preserves_input_provenance_for_matching_direction(tmp_path: Path) -> None:
    output = tmp_path / "revise_provenance.jsonl"
    example = RevisionExample(
        "source <draft> draft",
        "target",
        {
            "training_direction": ["PT-br", "ZH-hant"],
            "domain": "literary",
            "provenance": {"dataset": "fixture", "row": 9},
        },
        source_identifier="fixture.jsonl:9",
    )

    assert write_revision_examples(output, [example], ("pt-BR", "zh-Hant")) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["training_direction"] == ["pt-BR", "zh-Hant"]
    assert row["domain"] == "literary"
    assert row["provenance"] == {
        "transformation": "revision",
        "input": {
            "source": "fixture.jsonl:9",
            "provenance": {"dataset": "fixture", "row": 9},
        },
    }
