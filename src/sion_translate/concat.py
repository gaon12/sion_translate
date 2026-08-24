"""Concatenate unrelated pairs into long, multi-sentence training examples.

A model trained only on single sentences can omit later sentences, merge their
content, or stop generation early when it receives a multi-sentence source.
Concatenation teaches these behaviors even when every original corpus row holds
only one sentence:

- process source and target sequences several times longer than one sentence;
- preserve the same sentence count on both sides;
- retain aligned sentence order across languages;
- continue generation instead of treating an early EOS as correct; and
- recognize an explicit or whitespace sentence boundary.

The sentences are intentionally unrelated. A coherent paragraph lets the model
guess later text from earlier context instead of learning to translate every
sentence without omission. Unrelated rows require attention to each source.

Output names start with ``concat_``. Include that prefix in
``prepare_dataset.train_only_prefixes`` so synthetic examples cannot leak into
validation or test splits.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence, TypeAlias, cast, overload

from sion_translate.data.record_metadata import (
    RECORD_METADATA_FIELDS,
    resolve_record_training_direction,
)
from sion_translate.data.records import expand_parallel_record
from sion_translate.language_tags import canonicalize_language_pair
from sion_translate.synthetic import DEFAULT_SYNTHETIC_PREFIXES, synthetic_path, synthetic_record

# ``<seg>`` is a tokenizer-reserved control token. A whitespace separator
# resembles ordinary pasted multi-sentence input, while ``<seg>`` teaches an
# explicit boundary.
SEPARATORS = {"space": " ", "seg": " <seg> "}


@dataclass(frozen=True, slots=True)
class ConcatRecord:
    """One physical parallel row plus the metadata needed for safe derivation."""

    language_a: str
    text_a: str
    language_b: str
    text_b: str
    metadata: Mapping[str, object]
    synthetic: bool = False
    source_identifier: str | None = None


TextPair: TypeAlias = tuple[str, str]
ConcatInput: TypeAlias = TextPair | ConcatRecord

# Unscoped concatenations derived solely from real bitext remain valid for both
# directions, so ``concat_`` itself is intentionally excluded.  The remaining
# built-in prefixes identify synthetic sources whose generation direction must
# be explicit before another transformation can consume them.
_DIRECTION_REQUIRED_SYNTHETIC_PREFIXES = tuple(
    prefix for prefix in DEFAULT_SYNTHETIC_PREFIXES if prefix != "concat_"
)


@dataclass
class ConcatStats:
    """Summarize one concatenation run."""

    source_pairs: int
    written: int
    skipped_too_long: int
    sentences_per_example: dict[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_pairs": self.source_pairs,
            "written": self.written,
            "skipped_too_long": self.skipped_too_long,
            "sentences_per_example": {
                str(count): total for count, total in sorted(self.sentences_per_example.items())
            },
        }


def read_records(
    paths: Sequence[str | Path],
    language_pair: Sequence[str],
) -> Iterator[ConcatRecord]:
    """Read canonical rows while retaining direction and provenance metadata.

    Invalid JSON and incomplete records retain the historical skip behavior.
    Ambiguous canonical language aliases and invalid row-scoped directions are
    security boundaries, however, so they fail closed instead of disappearing.
    """

    normalized_pair = canonicalize_language_pair(
        language_pair,
        field="concat language_pair",
    )
    reverse_pair = (normalized_pair[1], normalized_pair[0])
    allowed_directions = frozenset((normalized_pair, reverse_pair))
    for path in paths:
        source_path = Path(path)
        path_is_synthetic = synthetic_path(
            source_path,
            _DIRECTION_REQUIRED_SYNTHETIC_PREFIXES,
        )
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                expansion = expand_parallel_record(row, (normalized_pair,))
                ambiguous_issues = {
                    "duplicate_language_key",
                    "duplicate_pair_container",
                }.intersection(expansion.issues)
                if ambiguous_issues:
                    raise ValueError(
                        f"{source_path}:{line_number}: ambiguous canonical language aliases: "
                        f"{sorted(ambiguous_issues)!r}"
                    )
                is_synthetic = path_is_synthetic or _contains_synthetic_record(row)
                for pair in expansion.pairs:
                    try:
                        direction = resolve_record_training_direction(
                            pair.metadata,
                            normalized_pair,
                            allowed_directions,
                        )
                    except ValueError as error:
                        raise ValueError(f"{source_path}:{line_number}: {error}") from error
                    if is_synthetic and direction is None:
                        raise ValueError(
                            f"{source_path}:{line_number}: synthetic records require an explicit "
                            "training_direction before concatenation"
                        )
                    metadata = dict(pair.metadata)
                    if direction is not None:
                        metadata["training_direction"] = list(direction)
                    yield ConcatRecord(
                        language_a=pair.language_a,
                        text_a=pair.text_a.strip(),
                        language_b=pair.language_b,
                        text_b=pair.text_b.strip(),
                        metadata=metadata,
                        synthetic=is_synthetic,
                        source_identifier=f"{source_path.as_posix()}:{line_number}",
                    )


def _contains_synthetic_record(node: object) -> bool:
    """Recognize nested synthetic rows without trusting only their container."""

    if synthetic_record(node):
        return True
    if isinstance(node, Mapping):
        mapping = cast(Mapping[object, object], node)
        for key, value in mapping.items():
            # A provenance payload may describe an upstream synthetic source;
            # it does not make the current row synthetic by itself.
            if key != "provenance" and _contains_synthetic_record(value):
                return True
        return False
    if isinstance(node, (list, tuple)):
        items = cast(Sequence[object], node)
        return any(_contains_synthetic_record(item) for item in items)
    return False


def read_pairs(
    paths: Sequence[str | Path],
    language_pair: Sequence[str],
) -> Iterator[tuple[str, str]]:
    """Compatibility view over :func:`read_records` without row metadata."""

    for record in read_records(paths, language_pair):
        yield record.text_a, record.text_b


def _canonical_record(record: ConcatRecord) -> tuple[ConcatRecord, tuple[str, str] | None]:
    physical_pair = canonicalize_language_pair(
        (record.language_a, record.language_b),
        field="concat record language pair",
    )
    reverse_pair = (physical_pair[1], physical_pair[0])
    allowed_directions = frozenset((physical_pair, reverse_pair))
    direction = resolve_record_training_direction(
        record.metadata,
        physical_pair,
        allowed_directions,
    )
    if record.synthetic and direction is None:
        raise ValueError("synthetic concat records require an explicit training_direction")
    metadata = dict(record.metadata)
    if direction is not None:
        metadata["training_direction"] = list(direction)
    return (
        ConcatRecord(
            language_a=physical_pair[0],
            text_a=record.text_a,
            language_b=physical_pair[1],
            text_b=record.text_b,
            metadata=metadata,
            synthetic=record.synthetic,
            source_identifier=record.source_identifier,
        ),
        direction,
    )


def _combined_metadata(
    records: Sequence[ConcatRecord],
    direction: tuple[str, str] | None,
) -> dict[str, object]:
    """Preserve safe common annotations and an ordered input provenance trail."""

    metadata: dict[str, object] = {}
    common_fields = tuple(
        field
        for field in RECORD_METADATA_FIELDS
        if field not in {"provenance", "training_direction"}
    )
    for field in common_fields:
        if not records or any(field not in record.metadata for record in records):
            continue
        first = records[0].metadata[field]
        if all(record.metadata[field] == first for record in records[1:]):
            metadata[field] = deepcopy(first)
    if direction is not None:
        metadata["training_direction"] = list(direction)

    provenance_inputs: list[dict[str, object]] = []
    for record in records:
        item: dict[str, object] = {}
        if record.source_identifier is not None:
            item["source"] = record.source_identifier
        if "provenance" in record.metadata:
            item["provenance"] = deepcopy(record.metadata["provenance"])
        if item:
            provenance_inputs.append(item)
    if provenance_inputs:
        metadata["provenance"] = {
            "transformation": "concatenation",
            "inputs": provenance_inputs,
        }
    return metadata


def _too_long(
    joined_a: str,
    joined_b: str,
    *,
    max_tokens: int | None,
    max_chars: int,
    count_tokens: Callable[[str], int] | None,
) -> bool:
    if len(joined_a) > max_chars or len(joined_b) > max_chars:
        return True
    if max_tokens is not None and count_tokens is not None:
        if count_tokens(joined_a) > max_tokens or count_tokens(joined_b) > max_tokens:
            return True
    return False


@overload
def build_concatenations(
    pairs: Sequence[TextPair],
    *,
    count: int,
    min_sentences: int = 2,
    max_sentences: int = 4,
    separator: str = "space",
    max_chars: int = 480,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    seed: int = 20260726,
) -> tuple[list[TextPair], ConcatStats]: ...


@overload
def build_concatenations(
    pairs: Sequence[ConcatRecord],
    *,
    count: int,
    min_sentences: int = 2,
    max_sentences: int = 4,
    separator: str = "space",
    max_chars: int = 480,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    seed: int = 20260726,
) -> tuple[list[ConcatRecord], ConcatStats]: ...


def build_concatenations(
    pairs: Sequence[ConcatInput],
    *,
    count: int,
    min_sentences: int = 2,
    max_sentences: int = 4,
    separator: str = "space",
    max_chars: int = 480,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    seed: int = 20260726,
) -> tuple[Sequence[ConcatInput], ConcatStats]:
    """Combine unrelated pairs into multi-sentence examples.

    A pair is never reused inside one result, so a sentence cannot be joined to
    itself. Combinations beyond the length limits are discarded because the
    training shard would truncate them anyway.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if min_sentences < 2:
        raise ValueError("min_sentences must be at least 2 for concatenation")
    if max_sentences < min_sentences:
        raise ValueError("max_sentences must be at least min_sentences")
    if separator not in SEPARATORS:
        raise ValueError(f"separator must be one of {sorted(SEPARATORS)}")
    if len(pairs) < min_sentences:
        raise ValueError(
            f"Only {len(pairs)} pairs are available; cannot build {min_sentences} sentences"
        )

    rich_mode = isinstance(pairs[0], ConcatRecord)
    if any(isinstance(pair, ConcatRecord) != rich_mode for pair in pairs):
        raise ValueError("plain text pairs and metadata-bearing concat records cannot be mixed")

    record_groups: dict[
        tuple[str, str, tuple[str, str] | None],
        list[ConcatRecord],
    ] = {}
    if rich_mode:
        for item in pairs:
            record, direction = _canonical_record(cast(ConcatRecord, item))
            group_key = (record.language_a, record.language_b, direction)
            record_groups.setdefault(group_key, []).append(record)
        eligible_groups = [
            (key, records)
            for key, records in record_groups.items()
            if len(records) >= min_sentences
        ]
        if not eligible_groups:
            raise ValueError(
                "no single canonical training_direction group contains enough rows "
                f"for min_sentences={min_sentences}"
            )
    else:
        eligible_groups = []

    joiner = SEPARATORS[separator]
    rng = random.Random(seed)
    built: list[ConcatInput] = []
    histogram: dict[int, int] = {}
    skipped = 0
    # Bound attempts because strict length limits may make the requested count impossible.
    attempts = 0
    attempt_budget = max(count * 20, 100)
    while len(built) < count and attempts < attempt_budget:
        attempts += 1
        selected_records: list[ConcatRecord] = []
        selected_group: tuple[str, str, tuple[str, str] | None] | None = None
        if rich_mode:
            selected_group, candidates = rng.choices(
                eligible_groups,
                weights=[len(group_records) for _, group_records in eligible_groups],
                k=1,
            )[0]
            wanted = rng.randint(min_sentences, min(max_sentences, len(candidates)))
            selected_records = rng.sample(candidates, wanted)
            joined_a = joiner.join(record.text_a for record in selected_records)
            joined_b = joiner.join(record.text_b for record in selected_records)
        else:
            wanted = rng.randint(min_sentences, min(max_sentences, len(pairs)))
            selected_indexes = rng.sample(range(len(pairs)), wanted)
            plain_pairs = cast(Sequence[TextPair], pairs)
            joined_a = joiner.join(plain_pairs[index][0] for index in selected_indexes)
            joined_b = joiner.join(plain_pairs[index][1] for index in selected_indexes)
        if _too_long(
            joined_a,
            joined_b,
            max_tokens=max_tokens,
            max_chars=max_chars,
            count_tokens=count_tokens,
        ):
            skipped += 1
            continue
        if rich_mode:
            assert selected_group is not None
            language_a, language_b, direction = selected_group
            built.append(
                ConcatRecord(
                    language_a=language_a,
                    text_a=joined_a,
                    language_b=language_b,
                    text_b=joined_b,
                    metadata=_combined_metadata(selected_records, direction),
                    synthetic=any(record.synthetic for record in selected_records),
                )
            )
        else:
            built.append((joined_a, joined_b))
        histogram[wanted] = histogram.get(wanted, 0) + 1
    return built, ConcatStats(len(pairs), len(built), skipped, histogram)


def write_concatenations(
    output_path: str | Path,
    examples: Iterable[ConcatInput],
    language_pair: Sequence[str],
) -> int:
    """Write the format consumed by ``prepare_dataset``."""
    key_a, key_b = canonicalize_language_pair(
        language_pair,
        field="concat language_pair",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for example in examples:
                row: dict[str, object]
                if isinstance(example, ConcatRecord):
                    record, direction = _canonical_record(example)
                    if frozenset((record.language_a, record.language_b)) != frozenset(
                        (key_a, key_b)
                    ):
                        raise ValueError(
                            "concat example language pair does not match output language_pair: "
                            f"example={(record.language_a, record.language_b)!r}, "
                            f"output={(key_a, key_b)!r}"
                        )
                    if (record.language_a, record.language_b) == (key_a, key_b):
                        text_a, text_b = record.text_a, record.text_b
                    else:
                        text_a, text_b = record.text_b, record.text_a
                    row = {key_a: text_a, key_b: text_b}
                    for field in RECORD_METADATA_FIELDS:
                        if field in record.metadata:
                            row[field] = deepcopy(record.metadata[field])
                    if direction is not None:
                        row["training_direction"] = list(direction)
                    if record.synthetic:
                        row["synthetic"] = True
                else:
                    text_a, text_b = example
                    row = {key_a: text_a, key_b: text_b}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return written
