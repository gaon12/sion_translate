"""Generate training data for complete-draft sequence revision.

Learning ``source + initial translation -> corrected translation`` gives the
model a second pass for sentences it cannot translate correctly at once. It
reuses the encoder-decoder structure and builds data from existing source and
target pairs.

The source and draft are serialized as ``source <draft> draft``. The existing
pipeline can therefore process them as ordinary translation pairs without
special indexing, shard, or collator changes.

An ideal draft would come from a trained model, but requiring that model creates
a bootstrap cycle. Instead, this module deliberately corrupts gold targets in
ways that imitate errors observed from the project model:

- ``number`` changes a value, such as 250mg to 1200mg. This is the most severe
  observed defect and receives the largest default weight. The model must read
  the source to recover the correct value.
- ``drop_clause`` removes a complete clause or sentence.
- ``truncate`` imitates premature termination by removing the end.
- ``repeat`` repeats a short fragment to imitate generation collapse.
- ``copy_source`` leaves the source untranslated.
- ``swap`` reverses two adjacent fragments to imitate alignment failure.
- ``identity`` leaves the draft correct. The model must learn not to modify an
  already correct draft, or revision itself can introduce errors.
"""

# Revision rows are loaded from JSON and normalized immediately.
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeAlias

from sion_translate.data.record_metadata import RECORD_METADATA_FIELDS
from sion_translate.language_tags import canonicalize_language_pair

DRAFT_SEPARATOR = "<draft>"

# Default corruption mix. See the module documentation for the identity share.
DEFAULT_CORRUPTIONS: dict[str, float] = {
    "number": 0.26,
    "drop_clause": 0.16,
    "truncate": 0.12,
    "repeat": 0.10,
    "copy_source": 0.06,
    "swap": 0.10,
    "identity": 0.20,
}

_NUMBER_RUN = re.compile(r"\d+")
# Treat punctuation used across supported scripts as clause boundaries.
_CLAUSE_SPLIT = re.compile(r"(?<=[。．.!?！？、,，])\s*")


@dataclass
class RevisionStats:
    """Summarize generated examples by corruption type."""

    written: int
    by_corruption: dict[str, int]
    unchanged: int

    def as_dict(self) -> dict:
        return {
            "written": self.written,
            "unchanged_drafts": self.unchanged,
            "by_corruption": dict(sorted(self.by_corruption.items())),
        }


@dataclass(frozen=True, slots=True)
class RevisionExample:
    """A revision target with authenticated source-row annotations."""

    serialized_source: str
    target: str
    metadata: Mapping[str, object]
    source_identifier: str | None = None


RevisionOutput: TypeAlias = tuple[str, str] | RevisionExample


def _validated_revision_components(source: str, draft: str) -> tuple[str, str]:
    normalized_source = source.strip()
    normalized_draft = draft.strip()
    if not normalized_source:
        raise ValueError("revision source must be nonblank")
    if not normalized_draft:
        raise ValueError("revision draft must be nonblank")
    if DRAFT_SEPARATOR in normalized_source or DRAFT_SEPARATOR in normalized_draft:
        raise ValueError(
            f"revision source and draft cannot contain the reserved {DRAFT_SEPARATOR} separator"
        )
    return normalized_source, normalized_draft


def serialize_revision_input(source: str, draft: str) -> str:
    """Serialize a source and draft as ``source <draft> draft``."""
    normalized_source, normalized_draft = _validated_revision_components(source, draft)
    return f"{normalized_source} {DRAFT_SEPARATOR} {normalized_draft}"


def parse_revision_input(text: str) -> tuple[str, str]:
    """Parse serialized revision input into ``(source, draft)``."""
    separator_count = text.count(DRAFT_SEPARATOR)
    if separator_count != 1:
        raise ValueError(
            f"revision input must contain exactly one {DRAFT_SEPARATOR} separator; "
            f"found={separator_count}: {text[:60]}"
        )
    source, _, draft = text.partition(DRAFT_SEPARATOR)
    return _validated_revision_components(source, draft)


def _clauses(text: str) -> list[str]:
    parts = [part for part in _CLAUSE_SPLIT.split(text) if part.strip()]
    return parts if len(parts) > 1 else []


def _corrupt_number(target: str, rng: random.Random) -> str:
    """Replace one number with a plausible but incorrect value."""
    normalized = unicodedata.normalize("NFKC", target)
    matches = list(_NUMBER_RUN.finditer(normalized))
    if not matches:
        return normalized
    match = rng.choice(matches)
    digits = match.group(0)
    # Preserve the width or add one digit to resemble observed errors such as 250 -> 1200.
    if rng.random() < 0.5 and len(digits) > 1:
        replacement = list(digits)
        position = rng.randrange(len(digits))
        choices = [d for d in "0123456789" if d != digits[position]]
        replacement[position] = rng.choice(choices)
        new_digits = "".join(replacement)
    else:
        new_digits = str(rng.randint(1, 9)) + digits[: max(1, len(digits) - 1)]
    return normalized[: match.start()] + new_digits + normalized[match.end() :]


def _corrupt_drop_clause(target: str, rng: random.Random) -> str:
    parts = _clauses(target)
    if not parts:
        return target
    removed = rng.randrange(len(parts))
    return "".join(part for index, part in enumerate(parts) if index != removed).strip()


def _corrupt_truncate(target: str, rng: random.Random) -> str:
    if len(target) < 8:
        return target
    keep = rng.randint(len(target) // 4, max(len(target) // 4 + 1, (len(target) * 3) // 4))
    return target[:keep].strip()


def _corrupt_repeat(target: str, rng: random.Random) -> str:
    if len(target) < 6:
        return target
    span = min(len(target), rng.randint(2, 6))
    start = rng.randrange(max(1, len(target) - span))
    fragment = target[start : start + span]
    return target[: start + span] + fragment * rng.randint(4, 6) + target[start + span :]


def _corrupt_swap(target: str, rng: random.Random) -> str:
    parts = _clauses(target)
    if len(parts) < 2:
        return target
    position = rng.randrange(len(parts) - 1)
    parts[position], parts[position + 1] = parts[position + 1], parts[position]
    return "".join(parts).strip()


_CORRUPTIONS = {
    "number": _corrupt_number,
    "drop_clause": _corrupt_drop_clause,
    "truncate": _corrupt_truncate,
    "repeat": _corrupt_repeat,
    "swap": _corrupt_swap,
}


def corrupt_target(
    source: str,
    target: str,
    kind: str,
    rng: random.Random,
) -> str:
    """Create a draft by applying corruption ``kind`` to the gold target."""
    if kind == "identity":
        return target
    if kind == "copy_source":
        return source
    if kind not in _CORRUPTIONS:
        raise ValueError(
            f"unknown corruption type: {kind} (available: {sorted(DEFAULT_CORRUPTIONS)})"
        )
    return _CORRUPTIONS[kind](target, rng)


def build_revision_examples(
    pairs: Sequence[tuple[str, str]],
    *,
    weights: dict[str, float] | None = None,
    seed: int = 20260726,
) -> tuple[list[tuple[str, str]], RevisionStats]:
    """Build one ``(source <draft> draft, gold target)`` example per pair.

    A no-op corruption, such as ``number`` on text without digits, is counted
    under ``unchanged``. It remains useful because leaving a correct draft
    unchanged is a required revision behavior.
    """
    weights = weights or DEFAULT_CORRUPTIONS
    unknown = set(weights) - set(DEFAULT_CORRUPTIONS)
    if unknown:
        raise ValueError(f"unknown corruption types: {sorted(unknown)}")
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("the sum of corruption weights must be positive")

    kinds = list(weights)
    probabilities = [weights[kind] for kind in kinds]
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    histogram: dict[str, int] = {}
    unchanged = 0
    for source, target in pairs:
        kind = rng.choices(kinds, weights=probabilities, k=1)[0]
        draft = corrupt_target(source, target, kind, rng)
        if not draft.strip():
            # An empty draft contains no evidence that revision can use.
            draft = target
            kind = "identity"
        if draft.strip() == target.strip():
            unchanged += 1
        histogram[kind] = histogram.get(kind, 0) + 1
        examples.append((serialize_revision_input(source, draft), target))
    return examples, RevisionStats(len(examples), histogram, unchanged)


def write_revision_examples(
    output_path: str | Path,
    examples: Iterable[RevisionOutput],
    language_pair: Sequence[str],
) -> int:
    """Write examples in the authenticated format consumed by ``prepare_dataset``.

    ``source <draft> draft`` occupies the source field, so the data pipeline can
    treat each row as an ordinary translation pair. Every row carries revision
    provenance independently of its filename, allowing the indexed loader to
    authenticate the training objective.
    """
    key_a, key_b = canonicalize_language_pair(
        language_pair,
        field="revision language_pair",
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
                if isinstance(example, RevisionExample):
                    serialized, target = example.serialized_source, example.target
                    metadata = dict(example.metadata)
                    raw_direction = metadata.get("training_direction")
                    if raw_direction is not None:
                        input_direction = canonicalize_language_pair(
                            raw_direction,
                            field="revision input training_direction",
                        )
                        if input_direction != (key_a, key_b):
                            location = (
                                f" at {example.source_identifier}"
                                if example.source_identifier is not None
                                else ""
                            )
                            raise ValueError(
                                "revision input training_direction does not match the requested "
                                f"revision direction{location}: input={input_direction!r}, "
                                f"requested={(key_a, key_b)!r}"
                            )
                else:
                    serialized, target = example
                    metadata = {}
                source, draft = parse_revision_input(serialized)
                serialized = serialize_revision_input(source, draft)
                provenance: dict[str, object] = {"transformation": "revision"}
                row: dict[str, object] = {
                    key_a: serialized,
                    key_b: target,
                    "synthetic": True,
                    "training_direction": [key_a, key_b],
                    "provenance": provenance,
                }
                if isinstance(example, RevisionExample):
                    for field in RECORD_METADATA_FIELDS:
                        if field not in {"provenance", "training_direction"} and field in metadata:
                            row[field] = deepcopy(metadata[field])
                    provenance_input: dict[str, object] = {}
                    if example.source_identifier is not None:
                        provenance_input["source"] = example.source_identifier
                    if "provenance" in metadata:
                        provenance_input["provenance"] = deepcopy(metadata["provenance"])
                    if provenance_input:
                        provenance["input"] = provenance_input
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return written
