"""Structured reasoning examples for the encoder-decoder auxiliary task.

Reasoning is deliberately a different task from translation and monolingual
denoising.  The encoder input starts with ``<reason_xx>`` and the decoder is
supervised on a fully delimited trace::

    <think> ... </think> <answer> ... </answer>

This module owns validation and serialization only.  It does not silently
reinterpret the existing ``text`` field used by monolingual denoising: callers
must opt in by reading the structured ``prompt``/``think``/``answer`` fields.
That separation keeps legacy translation and foundation inputs unchanged when
no reasoning corpus is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Protocol, cast

from sion_translate.tokenizer import normalize_text


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
TRACE_SYMBOLS = (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)
_LANGUAGE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,15}$")


class ReasoningTokenizer(Protocol):
    """Small tokenizer surface needed by :func:`serialize_reasoning_record`."""

    def encode(self, text: str) -> list[int]: ...

    def piece_id(self, piece: str) -> int: ...


class ReasoningDataError(ValueError):
    """A structured reasoning row or tokenizer contract is invalid."""


@dataclass(frozen=True)
class ReasoningRecord:
    prompt: str
    think: str
    answer: str
    language: str
    category: str = ""
    source: str = ""
    source_id: str = ""
    license: str = ""


@dataclass
class ReasoningReadStats:
    physical_lines: int = 0
    accepted: int = 0
    blank: int = 0
    malformed_json: int = 0
    non_object: int = 0
    invalid_record: int = 0

    @property
    def rejected(self) -> int:
        return self.blank + self.malformed_json + self.non_object + self.invalid_record


@dataclass(frozen=True)
class SerializedReasoningRecord:
    """Tokenized row ready to be written to an ordinary indexed shard.

    ``source_ids`` intentionally includes the task token as its first item.
    The collator removes that item and uses it as the encoder prefix.  The four
    decoder delimiters stay in ``target_ids`` even when content is truncated.
    """

    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    prompt_truncated: bool
    think_truncated: bool
    answer_truncated: bool


def reasoning_task_symbol(language: str) -> str:
    """Return the reserved encoder task symbol for a validated language key."""

    if not _LANGUAGE_KEY.fullmatch(language):
        raise ReasoningDataError(
            "reasoning language must be a 1-16 character ASCII key beginning "
            f"with a letter; got {language!r}"
        )
    return f"<reason_{language}>"


def is_reasoning_jsonl(path: str | Path) -> bool:
    """Whether a corpus path opts into the structured reasoning contract."""

    candidate = Path(path)
    return candidate.suffix.lower() == ".jsonl" and candidate.name.lower().startswith("reasoning_")


def _clean_required_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ReasoningDataError(f"reasoning field {field!r} must be a string")
    text = normalize_text(value).strip()
    if not text:
        raise ReasoningDataError(f"reasoning field {field!r} must not be blank")
    injected = next((symbol for symbol in TRACE_SYMBOLS if symbol in text), None)
    if injected is not None:
        raise ReasoningDataError(
            f"reasoning field {field!r} contains reserved trace marker {injected!r}"
        )
    return text


def _clean_optional_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, (str, int)):
        raise ReasoningDataError(f"reasoning metadata {field!r} must be scalar text")
    return normalize_text(str(value)).strip()


def parse_reasoning_row(
    row: Mapping[str, object],
    *,
    expected_language: str | None = None,
) -> ReasoningRecord:
    """Validate and normalize one ``prompt``/``think``/``answer`` mapping."""

    raw_language = row.get("language", expected_language)
    if not isinstance(raw_language, str):
        raise ReasoningDataError("reasoning field 'language' must be a string")
    language = raw_language.strip()
    reasoning_task_symbol(language)
    if expected_language is not None and language != expected_language:
        raise ReasoningDataError(
            "reasoning row language does not match its corpus directory: "
            f"row={language!r}, expected={expected_language!r}"
        )
    return ReasoningRecord(
        prompt=_clean_required_text(row, "prompt"),
        think=_clean_required_text(row, "think"),
        answer=_clean_required_text(row, "answer"),
        language=language,
        category=_clean_optional_text(row, "category"),
        source=_clean_optional_text(row, "source"),
        source_id=_clean_optional_text(row, "source_id"),
        license=_clean_optional_text(row, "license"),
    )


def iter_reasoning_records(
    path: str | Path,
    *,
    expected_language: str | None = None,
    stats: ReasoningReadStats | None = None,
    strict: bool = False,
) -> Iterator[ReasoningRecord]:
    """Yield valid JSONL rows and account for every rejected physical line.

    The default mirrors the existing monolingual reader: malformed rows are
    skipped but exposed through ``stats``.  ``strict=True`` is useful for data
    builders and tests that must fail at the first invalid training example.
    """

    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise ReasoningDataError(f"reasoning corpus must be JSONL: {source}")
    counters = stats if stats is not None else ReasoningReadStats()
    with source.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            counters.physical_lines += 1
            raw = line.strip()
            if not raw:
                counters.blank += 1
                continue
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError as error:
                counters.malformed_json += 1
                if strict:
                    raise ReasoningDataError(
                        f"invalid reasoning JSON at {source}:{line_number}: {error.msg}"
                    ) from error
                continue
            if not isinstance(value, dict):
                counters.non_object += 1
                if strict:
                    raise ReasoningDataError(
                        f"reasoning row must be an object at {source}:{line_number}"
                    )
                continue
            try:
                record = parse_reasoning_row(
                    cast(dict[str, object], value),
                    expected_language=expected_language,
                )
            except ReasoningDataError as error:
                counters.invalid_record += 1
                if strict:
                    raise ReasoningDataError(
                        f"invalid reasoning row at {source}:{line_number}: {error}"
                    ) from error
                continue
            counters.accepted += 1
            yield record


def _required_symbol_id(tokenizer: ReasoningTokenizer, symbol: str) -> int:
    token_id = int(tokenizer.piece_id(symbol))
    processor = getattr(tokenizer, "processor", None)
    exact_piece = None
    if processor is not None:
        id_to_piece = getattr(processor, "id_to_piece", None)
        if callable(id_to_piece) and token_id >= 0:
            exact_piece = str(id_to_piece(token_id))
    if token_id < 0 or (exact_piece is not None and exact_piece != symbol):
        raise ReasoningDataError(
            f"tokenizer is missing required reasoning control symbol {symbol!r}"
        )
    return token_id


def _content_budgets(
    think_length: int,
    answer_length: int,
    total: int,
    *,
    think_fraction: float,
) -> tuple[int, int]:
    """Allocate bounded space while preserving both trace sections."""

    preferred_think = min(
        think_length,
        max(1, min(total - 1, round(total * think_fraction))),
    )
    answer_budget = min(answer_length, total - preferred_think)
    if answer_budget < 1:
        answer_budget = 1
        preferred_think = total - 1
    remaining = total - preferred_think - answer_budget
    if remaining:
        think_extra = min(think_length - preferred_think, remaining)
        preferred_think += think_extra
        remaining -= think_extra
        answer_budget += min(answer_length - answer_budget, remaining)
    return preferred_think, answer_budget


def serialize_reasoning_record(
    record: ReasoningRecord,
    tokenizer: ReasoningTokenizer,
    *,
    max_source_tokens: int,
    max_target_tokens: int,
    think_fraction: float = 0.75,
) -> SerializedReasoningRecord:
    """Serialize one auxiliary example without ever truncating its delimiters.

    The limits exclude BOS/EOS, which are supplied by the standard collator.
    At least one content token is retained in both ``think`` and ``answer``.
    When both sections are long, ``think_fraction`` controls their token budget;
    unused space is automatically donated to the other section.
    """

    if max_source_tokens < 2:
        raise ValueError("max_source_tokens must leave room for a task tag and prompt")
    if max_target_tokens < len(TRACE_SYMBOLS) + 2:
        raise ValueError(
            "max_target_tokens must leave room for four trace markers and both sections"
        )
    if not 0.0 < think_fraction < 1.0:
        raise ValueError("think_fraction must be in (0, 1)")

    task_id = _required_symbol_id(tokenizer, reasoning_task_symbol(record.language))
    think_open, think_close, answer_open, answer_close = (
        _required_symbol_id(tokenizer, symbol) for symbol in TRACE_SYMBOLS
    )
    prompt_ids = tokenizer.encode(record.prompt)
    think_ids = tokenizer.encode(record.think)
    answer_ids = tokenizer.encode(record.answer)
    if not prompt_ids or not think_ids or not answer_ids:
        raise ReasoningDataError("reasoning fields must produce at least one tokenizer token")

    prompt_budget = max_source_tokens - 1
    content_budget = max_target_tokens - len(TRACE_SYMBOLS)
    think_budget, answer_budget = _content_budgets(
        len(think_ids),
        len(answer_ids),
        content_budget,
        think_fraction=think_fraction,
    )
    source_ids = (task_id, *prompt_ids[:prompt_budget])
    target_ids = (
        think_open,
        *think_ids[:think_budget],
        think_close,
        answer_open,
        *answer_ids[:answer_budget],
        answer_close,
    )
    return SerializedReasoningRecord(
        source_ids=source_ids,
        target_ids=target_ids,
        prompt_truncated=len(prompt_ids) > prompt_budget,
        think_truncated=len(think_ids) > think_budget,
        answer_truncated=len(answer_ids) > answer_budget,
    )


__all__ = [
    "ANSWER_CLOSE",
    "ANSWER_OPEN",
    "ReasoningDataError",
    "ReasoningReadStats",
    "ReasoningRecord",
    "SerializedReasoningRecord",
    "THINK_CLOSE",
    "THINK_OPEN",
    "TRACE_SYMBOLS",
    "is_reasoning_jsonl",
    "iter_reasoning_records",
    "parse_reasoning_row",
    "reasoning_task_symbol",
    "serialize_reasoning_record",
]
