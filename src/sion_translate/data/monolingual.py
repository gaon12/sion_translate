"""Discover and read monolingual corpora for foundation pretraining.

Foundation pretraining runs before translation training and consumes monolingual
text grouped by language rather than parallel sentence pairs. Arrange the files in
language-code directories::

    data/corpus/
      ko/
        kowiki_corpus.txt          # One sentence or paragraph per line
        news.jsonl                 # One {"text": "..."} object per line
      ja/
        wiki.txt

Only ``.txt`` and ``.jsonl`` are accepted. This deliberately narrow contract catches
quiet input failures before an expensive GPU run. For example, a misspelled JSON key
could otherwise reduce one file to zero usable sentences without being noticed until
training finishes. Discovery therefore returns every skipped entry and its reason so
the caller can report the loss before training starts.
"""

# Monolingual manifests contain heterogeneous JSON statistics.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_tag,
    canonicalize_language_tags,
    is_well_formed_language_tag,
)


class _LanguageDirectoryValidator:
    """Compatibility facade for the former public regex constant."""

    @staticmethod
    def match(value: object) -> bool:
        return is_well_formed_language_tag(value)

    fullmatch = match


LANGUAGE_DIRECTORY_PATTERN = _LanguageDirectoryValidator()

TEXT_SUFFIX = ".txt"
JSONL_SUFFIX = ".jsonl"
ALLOWED_SUFFIXES = (TEXT_SUFFIX, JSONL_SUFFIX)

JSONL_TEXT_KEY = "text"

DEFAULT_CORPUS_DIRECTORY = "data/corpus"

# Temperature exponent for sampling across languages. A value of 1.0 samples in
# direct proportion to sentence counts; smaller values move the distribution toward
# equal language shares. Monolingual corpus sizes commonly differ much more than
# parallel shard sizes, so this default applies stronger smoothing than the shard-level
# default of 0.9.
DEFAULT_LANGUAGE_SAMPLING_ALPHA = 0.7


@dataclass(frozen=True)
class MonolingualSource:
    """One file that will actually contribute to foundation training."""

    language: str
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class SkippedEntry:
    """A skipped path and the reason it was excluded from training."""

    path: Path
    reason: str


@dataclass(frozen=True)
class MonolingualDiscovery:
    root: Path
    sources: tuple[MonolingualSource, ...] = ()
    skipped: tuple[SkippedEntry, ...] = ()
    # Configured languages whose directory is absent or has no usable files.
    languages_without_data: tuple[str, ...] = ()
    # Language directories excluded because the language is not configured.
    unconfigured_languages: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.sources)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(source.language for source in self.sources))

    def paths_for(self, language: str) -> tuple[Path, ...]:
        normalized = canonicalize_language_tag(language)
        return tuple(source.path for source in self.sources if source.language == normalized)

    def bytes_per_language(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for source in self.sources:
            totals[source.language] = totals.get(source.language, 0) + source.size_bytes
        return totals


def foundation_languages(
    languages: Sequence[str],
    source_only_languages: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return the languages that are eligible for foundation training.

    Source-only languages are excluded. A monolingual reconstruction objective teaches
    the decoder to produce the corpus language, whereas a source-only designation means
    that the language must not be emitted as a translation target. Filtering here keeps
    foundation pretraining consistent with the later translation-data contract. The
    rule is derived entirely from configuration and works for arbitrary language tags.
    """

    normalized = canonicalize_language_tags(
        list(languages),
        field="foundation languages",
        reject_duplicates=False,
    )
    excluded = frozenset(
        canonicalize_language_tags(
            list(source_only_languages),
            field="source-only languages",
            reject_duplicates=False,
        )
    )
    return tuple(language for language in normalized if language not in excluded)


def discover_monolingual_sources(
    root: str | Path,
    languages: Sequence[str],
) -> MonolingualDiscovery:
    """Discover usable files in language directories and report every skipped entry."""

    root = Path(root)
    configured = canonicalize_language_tags(
        list(languages),
        field="monolingual languages",
        reject_duplicates=False,
    )
    if not configured:
        raise ValueError("at least one language is required to discover monolingual sources")

    if not root.is_dir():
        return MonolingualDiscovery(
            root=root,
            skipped=(SkippedEntry(root, "corpus directory does not exist"),),
            languages_without_data=configured,
        )

    configured_set = set(configured)
    skipped: list[SkippedEntry] = []
    language_directories: list[tuple[Path, str]] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            skipped.append(SkippedEntry(entry, "top-level entry is not a language directory"))
            continue
        try:
            entry_language = canonicalize_language_tag(
                entry.name,
                field="monolingual language directory",
            )
        except LanguageTagError:
            skipped.append(SkippedEntry(entry, "directory name is not a valid language tag"))
            continue
        language_directories.append((entry, entry_language))

    configured_directories: dict[str, list[Path]] = {}
    for entry, entry_language in language_directories:
        if entry_language in configured_set:
            configured_directories.setdefault(entry_language, []).append(entry)

    alias_collisions = {
        language: paths for language, paths in configured_directories.items() if len(paths) > 1
    }
    if alias_collisions:
        details = "; ".join(
            f"{language}: {', '.join(str(path) for path in paths)}"
            for language, paths in alias_collisions.items()
        )
        raise ValueError(
            "multiple monolingual corpus directories canonicalize to the same "
            f"configured language ({details})"
        )

    sources: list[MonolingualSource] = []
    unconfigured: list[str] = []
    for entry, entry_language in language_directories:
        if entry_language not in configured_set:
            reason = (
                "language directory is not configured "
                f"(configured languages: {', '.join(configured)})"
            )
            skipped.append(SkippedEntry(entry, reason))
            unconfigured.append(entry_language)
            continue
        found = False
        for candidate in sorted(entry.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
                skipped.append(
                    SkippedEntry(
                        candidate,
                        f"unsupported extension (allowed: {', '.join(ALLOWED_SUFFIXES)})",
                    )
                )
                continue
            size = candidate.stat().st_size
            if size == 0:
                skipped.append(SkippedEntry(candidate, "empty file"))
                continue
            sources.append(MonolingualSource(entry_language, candidate, size))
            found = True
        if not found:
            skipped.append(SkippedEntry(entry, "no readable .txt or .jsonl files"))

    present = {source.language for source in sources}
    return MonolingualDiscovery(
        root=root,
        sources=tuple(sources),
        skipped=tuple(skipped),
        languages_without_data=tuple(x for x in configured if x not in present),
        unconfigured_languages=tuple(dict.fromkeys(unconfigured)),
    )


@dataclass
class ReadStats:
    """Per-file rejection counts that expose otherwise silent data loss."""

    accepted: int = 0
    blank: int = 0
    malformed_json: int = 0
    missing_text_key: int = 0
    non_string_text: int = 0

    @property
    def rejected(self) -> int:
        return self.blank + self.malformed_json + self.missing_text_key + self.non_string_text

    def reasons(self) -> dict[str, int]:
        return {
            name: value
            for name, value in (
                ("blank", self.blank),
                ("malformed_json", self.malformed_json),
                ("missing_text_key", self.missing_text_key),
                ("non_string_text", self.non_string_text),
            )
            if value
        }


def iter_monolingual_lines(
    path: str | Path,
    *,
    stats: ReadStats | None = None,
) -> Iterator[str]:
    """Yield lines from ``.txt`` files or ``text`` values from ``.jsonl`` files.

    Blank values are skipped in both formats. When ``stats`` is supplied, it records
    the reason for every rejected row.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = " or ".join(ALLOWED_SUFFIXES)
        raise ValueError(f"monolingual corpora allow only {allowed}: {path}")
    stats = stats if stats is not None else ReadStats()

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        if suffix == TEXT_SUFFIX:
            for line in handle:
                text = line.strip()
                if not text:
                    stats.blank += 1
                    continue
                stats.accepted += 1
                yield text
            return
        for line in handle:
            raw = line.strip()
            if not raw:
                stats.blank += 1
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                stats.malformed_json += 1
                continue
            if not isinstance(row, dict) or JSONL_TEXT_KEY not in row:
                stats.missing_text_key += 1
                continue
            value = row[JSONL_TEXT_KEY]
            if not isinstance(value, str):
                stats.non_string_text += 1
                continue
            text = value.strip()
            if not text:
                stats.blank += 1
                continue
            stats.accepted += 1
            yield text


# Treat common full stops and question/exclamation marks across scripts as boundaries.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。．！？])\s*")


def segment_text(
    text: str,
    *,
    maximum_characters: int,
    minimum_characters: int = 1,
) -> list[str]:
    """Split a long document into segments no longer than ``maximum_characters``.

    The function neither discards an entire long document nor truncates its tail. A
    historical audit found that the former behavior would discard 97.3% of ``e_gov``,
    92.8% of ``aozora``, and 68.0% of ``kowiki`` characters because those corpora often
    store documents as lines longer than 4,000 characters. Token-limit truncation then
    removed 23.9% of the surviving text, for 25.8% total character loss.

    Segmentation prefers sentence boundaries and uses a hard character split only when
    one sentence exceeds the limit. A reconstruction target is the segment itself, so
    preserving sentence boundaries avoids teaching incomplete sentences as complete
    targets whenever possible.
    """

    if maximum_characters < 1:
        raise ValueError("maximum_characters must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= maximum_characters:
        # Apply the minimum on the short fast path too; otherwise a line could bypass
        # the filter merely because it does not need segmentation.
        return [text] if len(text) >= minimum_characters else []

    segments: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > maximum_characters:
            # Hard-split only when a single sentence exceeds the limit.
            if current:
                segments.append(current)
                current = ""
            segments.append(sentence[:maximum_characters])
            sentence = sentence[maximum_characters:].strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > maximum_characters:
            if current:
                segments.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        segments.append(current)
    return [segment for segment in segments if len(segment) >= minimum_characters]


def language_sampling_weights(
    counts: dict[str, int],
    *,
    alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
) -> dict[str, float]:
    """Return temperature-sampling weights by language, normalized to sum to one.

    ``alpha=1`` follows corpus size exactly; values closer to zero move nonempty
    languages toward equal shares. A language with no records receives zero weight
    because sampling cannot create missing data; callers should report that condition
    as a warning instead.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("language sampling alpha must be in (0, 1]")
    scaled = {
        language: math.pow(float(count), alpha) for language, count in counts.items() if count > 0
    }
    total = sum(scaled.values())
    if total <= 0:
        return {language: 0.0 for language in counts}
    weights = {language: 0.0 for language in counts}
    weights.update({language: value / total for language, value in scaled.items()})
    return weights


@dataclass
class BalanceReport:
    """Measured language balance and the warnings derived from it."""

    counts: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def is_balanced(self) -> bool:
        return not self.warnings


def assess_language_balance(
    counts: dict[str, int],
    *,
    alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    minimum_share: float = 0.05,
) -> BalanceReport:
    """Report language imbalance that remains after temperature sampling.

    Temperature sampling reduces a size gap but cannot eliminate it, and it cannot
    sample a language with zero records. Foundation training builds language-specific
    encoder and decoder representations, so missing or severely underrepresented
    target languages can produce correspondingly weak translation directions.
    """

    weights = language_sampling_weights(counts, alpha=alpha)
    warnings: list[str] = []
    empty = sorted(language for language, count in counts.items() if count <= 0)
    if empty:
        warnings.append(
            f"Languages with no monolingual data: {', '.join(empty)}. "
            "Foundation training cannot learn these languages, so translation "
            "directions that target them may underperform."
        )
    thin = sorted(
        language
        for language, weight in weights.items()
        if 0.0 < weight < minimum_share and len(counts) > 1
    )
    if thin:
        rendered = ", ".join(f"{language} {weights[language]:.1%}" for language in thin)
        warnings.append(
            f"Languages below a {minimum_share:.0%} batch share after temperature "
            f"sampling (alpha={alpha}): {rendered}. Lower alpha or add data."
        )
    return BalanceReport(counts=dict(counts), weights=weights, warnings=tuple(warnings))


def _estimated_line_count(path: Path, *, probe_lines: int = 500) -> int:
    """Estimate a file's line count from a prefix instead of scanning it in full.

    Reading a multi-gigabyte file an extra time solely to count lines is wasteful. An
    inaccurate estimate cannot exceed the output budget because the sampler applies a
    hard cap. The estimate affects only how evenly samples are distributed.
    """

    size = path.stat().st_size
    if size == 0:
        return 0
    consumed = 0
    lines = 0
    with path.open("rb") as handle:
        for raw in handle:
            consumed += len(raw)
            lines += 1
            if lines >= probe_lines:
                break
    if lines == 0:
        return 0
    return max(1, round(size / (consumed / lines)))


def sample_monolingual_sentences(
    paths: Sequence[Path],
    budget: int,
    *,
    seed: int = 0,
) -> Iterator[str]:
    """Yield at most ``budget`` sentences distributed across the complete input.

    Prefix truncation can select only the first source when a corpus is grouped by
    source, embedding that source bias in the tokenizer vocabulary. Content hashing
    instead makes deterministic selections that span the files: identical inputs and
    seed produce identical samples.
    """

    if budget <= 0:
        return
    estimated = sum(_estimated_line_count(path) for path in paths)
    if estimated <= 0:
        return
    probability = min(1.0, budget / estimated)
    threshold = int(probability * (1 << 64))
    emitted = 0
    for path in paths:
        for text in iter_monolingual_lines(path):
            if emitted >= budget:
                return
            if threshold < (1 << 64):
                digest = hashlib.blake2b(f"{seed}\0{text}".encode("utf-8"), digest_size=8).digest()
                if int.from_bytes(digest, "big") >= threshold:
                    continue
            emitted += 1
            yield text


def monolingual_budgets(
    parallel_counts: dict[str, int],
    languages: Sequence[str],
    *,
    ratio: float,
) -> dict[str, int]:
    """Calculate the tokenizer's monolingual-sample limit for each language.

    A normal limit is ``parallel sentence count * ratio``. Including every monolingual
    sentence can let the largest corpus dominate the vocabulary, while excluding all
    monolingual text forces foundation training to represent corpus-specific words with
    weaker pieces. A parallel-data-relative limit broadens coverage without changing
    the intended language allocation as sharply.

    When a language has monolingual data but no parallel pairs yet, use the mean count
    of languages that do have pairs instead of assigning zero. That state is a valid
    intermediate stage when adding a new configured language, and excluding it entirely
    would make later tokenizer reuse unsafe.
    """

    if ratio < 0:
        raise ValueError("monolingual sample ratio must be non-negative")
    observed = [count for count in parallel_counts.values() if count > 0]
    fallback = round(sum(observed) / len(observed)) if observed else 0
    return {
        language: int(round(ratio * (parallel_counts.get(language, 0) or fallback)))
        for language in languages
    }


def render_discovery_report(
    discovery: MonolingualDiscovery,
    *,
    maximum_skipped: int = 20,
) -> list[str]:
    """Render a human-readable discovery summary for pre-training review."""

    lines = [f"Monolingual corpus root: {discovery.root}"]
    totals = discovery.bytes_per_language()
    if totals:
        for language in sorted(totals):
            files = len(discovery.paths_for(language))
            lines.append(f"  {language}: {files} file(s), {totals[language] / 1e9:.2f} GB")
    else:
        lines.append("  No usable training files")
    if discovery.languages_without_data:
        lines.append(f"  Languages without data: {', '.join(discovery.languages_without_data)}")
    if discovery.unconfigured_languages:
        lines.append(
            "  Unconfigured language directories skipped: "
            f"{', '.join(discovery.unconfigured_languages)}"
        )
    if discovery.skipped:
        lines.append(f"  Skipped entries ({len(discovery.skipped)}):")
        for entry in discovery.skipped[:maximum_skipped]:
            lines.append(f"    - {entry.path}: {entry.reason}")
        remaining = len(discovery.skipped) - maximum_skipped
        if remaining > 0:
            lines.append(f"    ... and {remaining} more")
    return lines


__all__ = [
    "ALLOWED_SUFFIXES",
    "BalanceReport",
    "DEFAULT_CORPUS_DIRECTORY",
    "DEFAULT_LANGUAGE_SAMPLING_ALPHA",
    "JSONL_TEXT_KEY",
    "LANGUAGE_DIRECTORY_PATTERN",
    "MonolingualDiscovery",
    "MonolingualSource",
    "ReadStats",
    "SkippedEntry",
    "assess_language_balance",
    "discover_monolingual_sources",
    "foundation_languages",
    "iter_monolingual_lines",
    "language_sampling_weights",
    "monolingual_budgets",
    "render_discovery_report",
    "segment_text",
    "sample_monolingual_sentences",
]
