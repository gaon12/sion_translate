"""Audit whether challenge sentences already occur in the training corpus.

The earlier leakage guard operated only inside the reviewed seed set. It kept
exact duplicates apart while splitting 30 reviewed pairs into 18 training and
12 challenge pairs, but did not check whether those challenge sentences were
already present in the 8.97-million-row source corpus. Common expressions such
as `세 살 버릇 여든까지 간다` are especially likely to occur there.

Jaccard and MinHash were both measured and rejected for this task:

    case                         J5     J3     C5     C3
    leaked, one insertion        0.32   0.53   0.60   0.83
    leaked inside longer text    0.39   0.45   0.88   0.90
    exact leak                   1.00   1.00   1.00   1.00
    unrelated                    0.00   0.00   0.00   0.00
    topic-only similarity        0.00   0.00   0.00   0.00

Jaccard breaks down on short sentences. Challenge idioms are often 10 to 15
characters, so inserting one character disrupts all five 5-grams that cross
that position. More importantly, a typical leak is an idiom embedded in a
longer sentence, not a similar sentence of equal length. Penalizing the longer
corpus denominator asks the wrong question.

The audit therefore uses containment: the fraction of the challenge's
3-grams found in a corpus row. In the measurements above, leaked examples
score at least 0.83 while unrelated examples score 0.00.

MinHash buckets were also rejected. With ``num_perm=1``, collision probability
equals Jaccard, and a J5 score of 0.32 would discard two of three true candidates.
Instead, the audit builds a reverse index of challenge 3-grams. There are only
dozens of challenge items, so this index stays small and looking up each corpus
row's 3-grams finds every overlapping candidate.

Both containment and exact-match checks use
:func:`~sion_translate.splitting.comparison_key`, which removes punctuation,
symbols, and whitespace. This choice follows a measured failure: the former
deduplication key retained punctuation, so `김칫국부터 마시지 마.` and the corpus
row `김칫국부터 마시지 마…` were treated as different and exact leakage was
reported as zero even though the complete sentence was present.

The ordinary deduplication key remains unchanged because punctuation variants
can be distinct training rows. Leakage auditing asks a different question:
whether the sentence content already occurs in the corpus. Reusing one key for
both questions was the defect.
"""

# Holdout reports consume heterogeneous JSON findings.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from sion_translate.data.quality import canonical_text
from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
)
from sion_translate.language_tags import canonicalize_language_tag
from sion_translate.splitting import character_shingles, comparison_key

# Minimum fraction of challenge 3-grams present in a corpus row. Measured leaks
# scored at least 0.83 and unrelated/topic-only rows scored 0.00. A threshold
# of 0.6 also catches partial quotations.
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# Use 3-grams for short idioms. A 5-gram set is too small and unstable under a
# one-character edit in a 10-to-15-character sentence.
SHINGLE_SIZE = 3


@dataclass(frozen=True)
class HoldoutItem:
    """Represent one challenge sentence included in the audit."""

    identifier: str
    language: str
    text: str
    category: str = ""


@dataclass
class LeakMatch:
    """Represent a similar row found in the training corpus."""

    file: str
    line: int
    text: str
    similarity: float
    exact: bool


@dataclass
class HoldoutFinding:
    item: HoldoutItem
    matches: list[LeakMatch] = field(default_factory=list)

    @property
    def leaked(self) -> bool:
        return bool(self.matches)

    @property
    def worst(self) -> LeakMatch | None:
        return max(self.matches, key=lambda match: match.similarity, default=None)


def load_holdout_items(
    paths: Sequence[str | Path],
    *,
    language_pairs: Sequence[Sequence[str]],
) -> list[HoldoutItem]:
    """Load both source and reference sides of every challenge JSONL row.

    Auditing only one side is insufficient. If a reference occurs in the
    corpus, the model has already been trained to generate that sentence and
    the challenge is still leaked.
    """

    pairs = normalize_language_pairs(language_pairs=language_pairs)
    allowed = set(languages_from_pairs(pairs))
    items: list[HoldoutItem] = []
    seen_identifiers: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            identifier = str(row.get("id") or f"{path.name}:{line_number}")
            category = str(row.get("category", ""))
            for field_name, language_field in (
                ("source", "source_language"),
                ("reference", "target_language"),
            ):
                text = row.get(field_name)
                raw_language = row.get(language_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                if not isinstance(raw_language, str) or not raw_language.strip():
                    raise ValueError(f"{path}:{line_number} {field_name} requires {language_field}")
                language = canonicalize_language_tag(
                    raw_language,
                    field=f"{path}:{line_number} {language_field}",
                )
                if language not in allowed:
                    raise ValueError(
                        f"{path}:{line_number} {language_field}={language!r} is outside "
                        f"the configured language_pairs graph {sorted(allowed)}"
                    )
                item_identifier = f"{identifier}#{field_name}"
                if item_identifier in seen_identifiers:
                    raise ValueError(f"duplicate holdout identifier: {item_identifier!r}")
                seen_identifiers.add(item_identifier)
                items.append(
                    HoldoutItem(
                        identifier=item_identifier,
                        language=language,
                        text=canonical_text(text),
                        category=category,
                    )
                )
    return items


def shingles(text: str) -> set[str]:
    return set(character_shingles(comparison_key(text), size=SHINGLE_SIZE))


def containment(holdout_text: str, corpus_text: str) -> float:
    """Return the fraction of challenge 3-grams contained in a corpus row.

    Containment uses the challenge as its denominator. A typical leak embeds a
    short idiom in a longer row, which must not score lower merely because the
    surrounding corpus sentence contains additional text.
    """

    holdout_shingles = shingles(holdout_text)
    if not holdout_shingles:
        return 0.0
    return len(holdout_shingles & shingles(corpus_text)) / len(holdout_shingles)


def iter_corpus_texts(
    paths: Iterable[Path],
    *,
    language_pairs: Sequence[Sequence[str]],
) -> Iterator[tuple[Path, int, str, str]]:
    """Yield endpoints from every supported parallel-record layout."""

    pairs = normalize_language_pairs(language_pairs=language_pairs)
    for path in paths:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                try:
                    row = json.loads(raw_line.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                expansion = expand_parallel_record(row, pairs)
                seen_endpoints: set[tuple[str, str]] = set()
                for pair in expansion.pairs:
                    for language, value in (
                        (pair.language_a, pair.text_a),
                        (pair.language_b, pair.text_b),
                    ):
                        endpoint = (language, canonical_text(value))
                        if endpoint in seen_endpoints:
                            continue
                        seen_endpoints.add(endpoint)
                        yield path, line_number, language, value


def audit_holdout_leakage(
    items: Sequence[HoldoutItem],
    corpus_paths: Sequence[Path],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    maximum_matches_per_item: int = 5,
    language_pairs: Sequence[Sequence[str]],
) -> list[HoldoutFinding]:
    """Scan the complete corpus for exact and near copies of challenge items."""

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if not items:
        raise ValueError("there are no challenge sentences to audit")
    if not corpus_paths:
        raise ValueError("there are no training corpus files to audit")
    if maximum_matches_per_item < 1:
        raise ValueError("maximum_matches_per_item must be positive")
    pairs = normalize_language_pairs(language_pairs=language_pairs)
    known_languages = set(languages_from_pairs(pairs))
    canonical_item_languages = {
        item.identifier: canonicalize_language_tag(
            item.language,
            field=f"holdout item {item.identifier!r} language",
        )
        for item in items
    }
    unknown_languages = sorted(set(canonical_item_languages.values()) - known_languages)
    if unknown_languages:
        raise ValueError(
            "holdout item languages must appear in language_pairs; "
            f"unknown languages: {unknown_languages}"
        )
    if len({item.identifier for item in items}) != len(items):
        raise ValueError("holdout item identifiers must be unique")

    findings = {item.identifier: HoldoutFinding(item=item) for item in items}
    # The reverse challenge 3-gram index stays small because there are only
    # dozens of items. Looking up each corpus row finds every overlapping item.
    index: dict[tuple[str, str], set[str]] = {}
    for item in items:
        for shingle in shingles(item.text):
            index.setdefault((canonical_item_languages[item.identifier], shingle), set()).add(
                item.identifier
            )

    for path, line_number, language, raw_text in iter_corpus_texts(
        corpus_paths,
        language_pairs=pairs,
    ):
        text = canonical_text(raw_text)
        candidate_ids: set[str] = set()
        for shingle in shingles(text):
            candidate_ids |= index.get((language, shingle), frozenset())
        for identifier in candidate_ids:
            finding = findings[identifier]
            candidate = finding.item
            similarity = containment(candidate.text, text)
            if similarity < similarity_threshold:
                continue
            finding.matches.append(
                LeakMatch(
                    file=str(path),
                    line=line_number,
                    text=text,
                    similarity=similarity,
                    exact=comparison_key(text) == comparison_key(candidate.text),
                )
            )
            # Retain the worst N matches, not the first N in scan order. In a
            # measured case, `호랑이도 제 말 하면 온다더니.` scored 0.91 in
            # data12 and 1.00 in data9. Lexical file ordering visited data12
            # first, so a first-N cap discarded the later exact match and made
            # the safety gate underreport leakage.
            if len(finding.matches) > maximum_matches_per_item:
                finding.matches.sort(key=lambda match: -match.similarity)
                del finding.matches[maximum_matches_per_item:]
    for finding in findings.values():
        finding.matches.sort(key=lambda match: -match.similarity)
    return list(findings.values())


def summarize(findings: Sequence[HoldoutFinding]) -> dict[str, object]:
    leaked = [finding for finding in findings if finding.leaked]
    exact_hits = [finding for finding in leaked if any(match.exact for match in finding.matches)]
    by_category: dict[str, int] = {}
    for finding in leaked:
        by_category[finding.item.category] = by_category.get(finding.item.category, 0) + 1
    return {
        "audited_items": len(findings),
        "leaked_items": len(leaked),
        "exact_leaked_items": len(exact_hits),
        "leak_rate": (len(leaked) / len(findings)) if findings else 0.0,
        "by_category": by_category,
        "note": (
            "Leaked items are not an independent holdout. They may remain useful as a "
            "regression smoke set, but must not be cited as a quality benchmark."
        ),
    }


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "HoldoutFinding",
    "HoldoutItem",
    "LeakMatch",
    "audit_holdout_leakage",
    "iter_corpus_texts",
    "containment",
    "shingles",
    "load_holdout_items",
    "summarize",
]
