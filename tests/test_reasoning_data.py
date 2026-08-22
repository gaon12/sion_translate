from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sion_translate.data.reasoning import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    ReasoningDataError,
    ReasoningReadStats,
    ReasoningRecord,
    is_reasoning_jsonl,
    iter_reasoning_records,
    parse_reasoning_row,
    reasoning_task_symbol,
    serialize_reasoning_record,
)


class _Processor:
    def __init__(self, pieces: dict[str, int]):
        self.by_id = {token_id: piece for piece, token_id in pieces.items()}

    def id_to_piece(self, token_id: int) -> str:
        return self.by_id.get(token_id, "<unk>")


class _Tokenizer:
    def __init__(self, *, omit: str | None = None):
        symbols = ["<reason_ja>", THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE]
        self.pieces = {symbol: index + 10 for index, symbol in enumerate(symbols) if symbol != omit}
        self.processor = _Processor(self.pieces)

    def piece_id(self, piece: str) -> int:
        # SentencePiece returns unk rather than -1 for an absent normal piece.
        return self.pieces.get(piece, 0)

    def encode(self, text: str) -> list[int]:
        return [100 + ord(char) for char in text if not char.isspace()]


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "二つの数を足してください。",
        "think": "一つずつ確認してから合計を計算する。",
        "answer": "答えは四です。",
        "language": "ja",
        "category": "math",
        "source": "fixture/reasoning",
        "source_id": 7,
        "license": "Apache-2.0",
    }
    row.update(overrides)
    return row


def test_reasoning_task_symbol_validates_language_keys() -> None:
    assert reasoning_task_symbol("ja") == "<reason_ja>"
    assert reasoning_task_symbol("zh-hant") == "<reason_zh-Hant>"
    with pytest.raises(ReasoningDataError, match="BCP 47"):
        reasoning_task_symbol("日本語")


def test_reasoning_rows_compare_canonical_language_identities() -> None:
    record = parse_reasoning_row(_row(language="pt-br"), expected_language="pt-BR")
    assert record.language == "pt-BR"


def test_parse_reasoning_row_keeps_structured_fields_and_metadata() -> None:
    record = parse_reasoning_row(_row(), expected_language="ja")
    assert record.prompt == "二つの数を足してください。"
    assert record.think == "一つずつ確認してから合計を計算する。"
    assert record.answer == "答えは四です。"
    assert record.source_id == "7"
    assert record.category == "math"


@pytest.mark.parametrize("field", ["prompt", "think", "answer"])
def test_parse_reasoning_row_rejects_missing_or_blank_training_fields(field: str) -> None:
    with pytest.raises(ReasoningDataError, match=field):
        parse_reasoning_row(_row(**{field: "  "}), expected_language="ja")


def test_parse_reasoning_row_rejects_language_mismatch_and_marker_injection() -> None:
    with pytest.raises(ReasoningDataError, match="does not match"):
        parse_reasoning_row(_row(language="en"), expected_language="ja")
    with pytest.raises(ReasoningDataError, match="reserved trace marker"):
        parse_reasoning_row(_row(think=f"途中で {ANSWER_OPEN} を出す"))


def test_reasoning_files_are_opted_in_by_name_not_their_text_field(tmp_path: Path) -> None:
    assert is_reasoning_jsonl(tmp_path / "reasoning_math.jsonl")
    assert not is_reasoning_jsonl(tmp_path / "wiki.jsonl")
    assert not is_reasoning_jsonl(tmp_path / "reasoning_math.txt")


def test_jsonl_reader_accounts_for_rejections_without_hiding_valid_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reasoning_math.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps(_row(), ensure_ascii=False),
                "{broken",
                json.dumps(["not", "an", "object"], ensure_ascii=False),
                json.dumps(_row(answer=""), ensure_ascii=False),
                "",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    stats = ReasoningReadStats()
    records = list(iter_reasoning_records(source, expected_language="ja", stats=stats))
    assert len(records) == 1
    assert stats.physical_lines == 5
    assert stats.accepted == 1
    assert stats.malformed_json == 1
    assert stats.non_object == 1
    assert stats.invalid_record == 1
    assert stats.blank == 1
    assert stats.rejected == 4


def test_jsonl_reader_strict_mode_names_the_file_and_line(tmp_path: Path) -> None:
    source = tmp_path / "reasoning_math.jsonl"
    source.write_text(json.dumps(_row(answer=""), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ReasoningDataError, match=r"reasoning_math\.jsonl:1"):
        list(iter_reasoning_records(source, expected_language="ja", strict=True))


def test_serialization_separates_encoder_task_and_decoder_trace() -> None:
    tokenizer = _Tokenizer()
    encoded = serialize_reasoning_record(
        parse_reasoning_row(_row()),
        tokenizer,
        max_source_tokens=128,
        max_target_tokens=128,
    )
    assert encoded.source_ids[0] == tokenizer.piece_id("<reason_ja>")
    assert encoded.target_ids[0] == tokenizer.piece_id(THINK_OPEN)
    think_close = encoded.target_ids.index(tokenizer.piece_id(THINK_CLOSE))
    assert encoded.target_ids[think_close + 1] == tokenizer.piece_id(ANSWER_OPEN)
    assert encoded.target_ids[-1] == tokenizer.piece_id(ANSWER_CLOSE)
    assert not encoded.prompt_truncated
    assert not encoded.think_truncated
    assert not encoded.answer_truncated


def test_truncation_preserves_all_markers_and_both_sections() -> None:
    tokenizer = _Tokenizer()
    encoded = serialize_reasoning_record(
        ReasoningRecord(
            prompt="問" * 50,
            think="考" * 50,
            answer="答" * 50,
            language="ja",
        ),
        tokenizer,
        max_source_tokens=8,
        max_target_tokens=14,
        think_fraction=0.7,
    )
    assert len(encoded.source_ids) == 8
    assert len(encoded.target_ids) == 14
    marker_ids = [
        tokenizer.piece_id(symbol)
        for symbol in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)
    ]
    assert [token_id for token_id in encoded.target_ids if token_id in marker_ids] == marker_ids
    think_close = encoded.target_ids.index(tokenizer.piece_id(THINK_CLOSE))
    answer_open = encoded.target_ids.index(tokenizer.piece_id(ANSWER_OPEN))
    assert think_close > 1
    assert encoded.target_ids[-1] == tokenizer.piece_id(ANSWER_CLOSE)
    assert len(encoded.target_ids[answer_open + 1 : -1]) >= 1
    assert encoded.prompt_truncated
    assert encoded.think_truncated
    assert encoded.answer_truncated


def test_serialization_refuses_a_tokenizer_without_exact_reserved_symbols() -> None:
    with pytest.raises(ReasoningDataError, match=THINK_CLOSE):
        serialize_reasoning_record(
            parse_reasoning_row(_row()),
            _Tokenizer(omit=THINK_CLOSE),
            max_source_tokens=64,
            max_target_tokens=64,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_source_tokens": 1, "max_target_tokens": 16}, "max_source_tokens"),
        ({"max_source_tokens": 16, "max_target_tokens": 5}, "max_target_tokens"),
        (
            {"max_source_tokens": 16, "max_target_tokens": 16, "think_fraction": 1.0},
            "think_fraction",
        ),
    ],
)
def test_serialization_validates_token_budgets(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        serialize_reasoning_record(parse_reasoning_row(_row()), _Tokenizer(), **kwargs)
