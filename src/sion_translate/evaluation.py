"""Shared translation-quality evaluation logic.

The ``sion-evaluate`` CLI uses this module to compute chrF and BLEU on a fixed
evaluation set and to compare the model with external systems such as DeepL,
Google, or Papago on exactly the same examples.

The metrics serve different purposes:

- **chrF**, a character n-gram F-score, does not require word tokenization and
  remains comparable across languages with different spacing conventions. It
  is the primary metric.
- **BLEU** is included as a conventional secondary metric. Languages registered
  for character tokenization use ``tokenize="char"``; other languages use the
  standard 13a tokenizer.
- **Number preservation** catches severe value corruption that chrF and BLEU
  barely penalize when most character n-grams still overlap, such as changing
  ``250mg`` to ``1200mg``. A one-digit error can invalidate an amount, dose, or
  date. F1 reports both omission and invention, while exact-match counts show
  how many number-bearing sentences preserve every value.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sion_translate.data import IndexedParallelDataset
from sion_translate.data.records import (
    expand_parallel_record,
    normalize_language_pairs,
    normalize_translation_directions,
)
from sion_translate.scripts_registry import uses_character_tokenization
from sion_translate.structured import structured_signature
from sion_translate.tokenizer import SionTokenizer

# Numeric runs representing values such as amounts, doses, dates, and versions.
# This definition matches post-training rewards so optimization and evaluation
# do not measure different targets.
#
# Boundaries consider only ASCII letters, digits, and underscores. Python's
# ``\w`` also includes Hangul and kana, so using ``(?![\w])`` would omit single
# digits followed by particles or units, including ``4월``, ``1회``, and ``5개``.
# Conversely, digits in ``utf8``, ``config2``, or ``HTTP429`` belong to an
# identifier and are excluded by adjacent ASCII characters. The first branch
# still captures ``250`` in ``250mg``.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d[\d,.:/%+\-]*\d|(?<![A-Za-z0-9_])[-+]?\d(?![0-9])"
)


def normalized_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    """Return NFKC-normalized matches, treating full-width digits as equivalent."""
    normalized = unicodedata.normalize("NFKC", text)
    return [match.group(0).casefold().rstrip(".,;:!?") for match in pattern.finditer(normalized)]


def multiset_f1(expected: Sequence[object], actual: Sequence[object]) -> float:
    """Compute duplicate-aware F1, returning 1 when both multisets are empty."""
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    if not expected_counts and not actual_counts:
        return 1.0
    if not expected_counts or not actual_counts:
        return 0.0
    overlap = sum((expected_counts & actual_counts).values())
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def structured_tokens(text: str) -> list[str]:
    """Return structure tokens that must survive translation, excluding numbers."""

    return list(structured_signature(text, include_numbers=False).elements())


def has_excessive_repetition(text: str) -> bool:
    """Detect generation collapse dominated by one character or repeated phrase."""
    surface = [char for char in text if not char.isspace()]
    if len(surface) < 12:
        return False
    if Counter(surface).most_common(1)[0][1] / len(surface) >= 0.70:
        return True
    return re.search(r"(.{1,8})\1{4,}", "".join(surface)) is not None


def numeric_tokens(text: str) -> list[str]:
    """Extract numeric values, ignoring commas used as digit-group separators.

    ``38,720`` and ``38720`` denote the same value. Treating formatting alone
    as mistranslation would penalize a locale's notation convention.
    """
    return [token.replace(",", "") for token in normalized_matches(NUMBER_PATTERN, text)]


def numeric_corruption(source: str, reference: str, hypothesis: str) -> tuple[int, int]:
    """Return ``(invented, dropped)`` counts for unjustified numeric changes.

    ``multiset_f1`` turns corruption into a ratio. A candidate that invents one
    value can resemble a candidate that drops one value from a number-heavy
    sentence, allowing a small chrF gain to dominate under a 0.10 reward weight.
    That behavior changed values in 8 of 10 deployment-holdout sentences. This
    function therefore returns absolute counts for use as hard penalties.

    The decision combines evidence from both source and reference:

    * An **invention** appears in the hypothesis but in neither source nor
      reference, so neither side justifies it.
    * A **drop** is present in both source and reference but absent from the
      hypothesis, so both sides agree that it is required.

    Source-only checks fail when a language spells out numbers. In ``하루 두 번``
    to ``1日2回``, the digit ``2`` appears only in the valid reference and is not
    an invention. Reference-only checks fail when the reference is corrupted;
    source evidence can still license the original value.
    """

    source_values = Counter(numeric_tokens(source))
    reference_values = Counter(numeric_tokens(reference))
    hypothesis_values = Counter(numeric_tokens(hypothesis))
    licensed = source_values | reference_values  # Multiset union: maximum count per value.
    required = source_values & reference_values  # Multiset intersection: minimum count.
    invented = sum((hypothesis_values - licensed).values())
    dropped = sum((required - hypothesis_values).values())
    return invented, dropped


@dataclass(frozen=True, slots=True)
class NumberPreservationResult:
    """Report preservation only across sentences that contain numeric values."""

    f1: float
    exact: int
    samples: int
    inventions: int


def number_preservation_details(
    hypotheses: Sequence[str],
    sources: Sequence[str],
) -> NumberPreservationResult:
    """Measure source-number preservation and count sentences with inventions.

    Counting many clean, number-free sentences as correct would hide actual
    numeric errors. ``samples`` therefore includes only sentences where the
    source or hypothesis contains a number. ``inventions`` counts sentences
    whose hypothesis numeric multiset exceeds the source multiset.

    References score translation quality through chrF and BLEU, but are not the
    preservation baseline here. Values that must survive translation should be
    compared directly between source and hypothesis so reference formatting or
    reference errors cannot distort this metric.
    """

    if len(hypotheses) != len(sources):
        raise ValueError(
            f"hypothesis count {len(hypotheses)} does not match source count {len(sources)}"
        )

    scores: list[float] = []
    exact = 0
    inventions = 0
    for hypothesis, source in zip(hypotheses, sources, strict=True):
        expected = numeric_tokens(source)
        actual = numeric_tokens(hypothesis)
        if not expected and not actual:
            continue
        scores.append(multiset_f1(expected, actual))
        expected_counts = Counter(expected)
        actual_counts = Counter(actual)
        if expected_counts == actual_counts:
            exact += 1
        if actual_counts - expected_counts:
            inventions += 1

    if not scores:
        return NumberPreservationResult(f1=100.0, exact=0, samples=0, inventions=0)
    return NumberPreservationResult(
        f1=100.0 * sum(scores) / len(scores),
        exact=exact,
        samples=len(scores),
        inventions=inventions,
    )


def number_preservation(
    hypotheses: Sequence[str],
    sources: Sequence[str],
) -> tuple[float, int]:
    """Return the backward-compatible ``(number F1, exact sentence count)`` pair.

    Hypotheses are compared directly with sources and both values cover only
    number-bearing sentences. Use :func:`number_preservation_details` when a
    report also needs the exact denominator and invention count.
    """

    result = number_preservation_details(hypotheses, sources=sources)
    return result.f1, result.exact


@dataclass
class DirectionResult:
    """Store one system's evaluation result for one translation direction."""

    system: str  # "sion" or an external system name supplied through --compare.
    direction: str  # Canonical "source-target" form.
    samples: int
    chrf: float  # Primary metric in [0, 100]; higher is better.
    bleu: float
    bleu_tokenize: str  # BLEU tokenizer recorded for reproducibility.
    number_f1: float = 0.0  # Mean number-preservation F1 in [0, 100].
    number_exact: int = 0  # Number-bearing sentences with every value preserved.
    number_samples: int = 0  # Sentences with a number in source or hypothesis.
    number_inventions: int = 0  # Sentences containing values absent from the source.


def score_translations(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    target_language: str,
) -> tuple[float, float, str]:
    """Return ``(chrF, BLEU, BLEU tokenizer)`` for one target language."""
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypothesis count {len(hypotheses)} does not match reference count {len(references)}"
        )
    from sacrebleu.metrics.bleu import BLEU
    from sacrebleu.metrics.chrf import CHRF

    tokenize = "char" if uses_character_tokenization(target_language) else "13a"
    chrf = CHRF().corpus_score(list(hypotheses), [list(references)]).score
    bleu = BLEU(tokenize=tokenize).corpus_score(list(hypotheses), [list(references)]).score
    return chrf, bleu, tokenize


def load_split_pairs(
    dataset_dir: str | Path,
    split: str,
    tokenizer: SionTokenizer,
    *,
    model_language_pairs: Sequence[Sequence[str]],
    max_samples_per_direction: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Decode a prepared holdout split into ``(source, reference)`` text pairs.

    The result maps ``(source language, target language)`` to text pairs. Shards
    store only token IDs, so the supplied tokenizer reconstructs their text.
    """
    dataset = IndexedParallelDataset(
        dataset_dir,
        split,
        bidirectional=True,
        legacy_language_pairs=model_language_pairs,
    )
    # Source-only languages (한본어 kj) are never a target, so the reachable
    # direction count is smaller than 2x the pair count and the early exit
    # below has to use the real number or it never fires.
    expected_directions = dataset.direction_count
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        direction = (item["src_language"], item["target_language"])
        bucket = pairs.setdefault(direction, [])
        if len(bucket) >= max_samples_per_direction:
            # Stop after every reachable direction reaches its sample cap.
            if (
                all(len(existing) >= max_samples_per_direction for existing in pairs.values())
                and len(pairs) == expected_directions
            ):
                break
            continue
        bucket.append(
            (tokenizer.decode(item["src"].tolist()), tokenizer.decode(item["tgt"].tolist()))
        )
    return pairs


def load_benchmark_pairs(
    paths: Sequence[str | Path],
    language_pair: Sequence[str] | Sequence[Sequence[str]],
    *,
    translation_directions: Sequence[Sequence[str]] | None = None,
    max_samples_per_direction: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Load external benchmark JSONL, such as converted FLORES, as text pairs.

    The format matches training data, with one language-keyed JSON object per
    line. Every explicitly configured direction can be included.
    """
    if language_pair and isinstance(language_pair[0], str):
        language_pairs = normalize_language_pairs(language_pair)  # type: ignore[arg-type]
    else:
        language_pairs = normalize_language_pairs(
            language_pairs=language_pair  # type: ignore[arg-type]
        )
    directions = normalize_translation_directions(
        language_pairs,
        translation_directions,
    )
    output: dict[tuple[str, str], list[tuple[str, str]]] = {
        direction: [] for direction in directions
    }
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                expansion = expand_parallel_record(row, language_pairs)
                for pair in expansion.pairs:
                    forward = output.get((pair.language_a, pair.language_b))
                    reverse = output.get((pair.language_b, pair.language_a))
                    if forward is not None and len(forward) < max_samples_per_direction:
                        forward.append((pair.text_a, pair.text_b))
                    if reverse is not None and len(reverse) < max_samples_per_direction:
                        reverse.append((pair.text_b, pair.text_a))
                if all(len(samples) >= max_samples_per_direction for samples in output.values()):
                    break
    return output


def results_as_markdown(results: Sequence[DirectionResult]) -> str:
    """Render a human-readable Markdown comparison table."""
    lines = [
        "| system | direction | samples | chrF | BLEU | number F1 | number exact | number inventions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        number_f1 = f"{result.number_f1:.2f}" if result.number_samples else "-"
        exact = f"{result.number_exact}/{result.number_samples}" if result.number_samples else "-"
        lines.append(
            f"| {_markdown_cell(result.system)} | {_markdown_cell(result.direction)} "
            f"| {result.samples} "
            f"| {result.chrf:.2f} | {result.bleu:.2f} "
            f"| {number_f1} | {exact} | {result.number_inventions} |"
        )
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _stage_text(path: Path, text: str) -> Path:
    """Write and synchronize a private sibling file without changing ``path``."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _fsync_parent(path: Path) -> None:
    """Persist a published directory entry where the platform supports it."""

    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Windows and some network filesystems do not expose directory fsync.
        pass
    finally:
        os.close(descriptor)


def save_results(
    results: Sequence[DirectionResult],
    output_path: str | Path,
    *,
    metadata: dict[str, object],
) -> None:
    """Save machine-readable JSON and human-readable Markdown results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "metadata": metadata,
        "results": [asdict(result) for result in results],
    }
    json_path = output_path.with_suffix(".json")
    markdown_path = output_path.with_suffix(".md")
    staged_json: Path | None = None
    staged_markdown: Path | None = None
    try:
        # Stage and synchronize both representations before replacing either
        # existing report. A failed render or full disk therefore preserves the
        # complete previous pair rather than truncating one public file.
        staged_json = _stage_text(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        staged_markdown = _stage_text(markdown_path, results_as_markdown(results) + "\n")
        os.replace(staged_markdown, markdown_path)
        staged_markdown = None
        _fsync_parent(markdown_path)
        # Publish machine-readable JSON last so it remains the authoritative
        # completion signal for consumers that inspect both files.
        os.replace(staged_json, json_path)
        staged_json = None
        _fsync_parent(json_path)
    finally:
        if staged_json is not None:
            staged_json.unlink(missing_ok=True)
        if staged_markdown is not None:
            staged_markdown.unlink(missing_ok=True)
