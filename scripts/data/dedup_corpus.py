#!/usr/bin/env python3
"""Remove duplicate translation pairs across the whole corpus, not one shard.

``dedup_shard.py`` collapses repeats inside a single file. That is not enough
here: a measured scan of ``data/*.jsonl`` found 914,100 exactly duplicated rows,
and the worst offenders duplicate rows that live in *other* shards -- 50,996
rows of ``data64_kej`` already exist in ``data29_kej``, and every duplicated row
of ``data78_koen`` is already inside ``data60_koen``.

The preparer removes exact duplicate pairs while it builds the indexed dataset,
so this script does not exist to make training correct. It exists so the shards
themselves are correct: row counts, language-pair balance targets, tokenizer
frequency statistics and packaged bundle sizes are all computed from the files
on disk, and every one of them is wrong while those files carry duplicates.

What identity means here
------------------------
Rows are judged as the *pairs* they expand into, using the configured language
pairs and ``expand_parallel_record`` -- the same expansion the preparer uses. A
dialect row carrying ``jd``/``ko``/``ja`` contributes three pairs, and its
standard ``ko``-``ja`` pair being a copy of a real corpus row says nothing about
its dialect pairs. Such a row is kept, because dropping it would discard the
dialect data that is the reason the row exists.

Two identities are applied in sequence:

``exact``
    NFKC plus whitespace folding, the dedup key the preparer itself uses.
``loose``
    ``comparison_key``: NFKC with punctuation, symbols and separators removed.
    This is what collapses the quote-marker repeats that fill discussion-forum
    corpora, where ``>>알겠습니다.`` and ``알겠습니다.`` are the same sentence
    quoted twice. An all-symbol line (``……``, ``❤``) has an empty comparison key
    and would otherwise collapse into one arbitrary row, so those fall back to
    the exact key.

Shards whose subject *is* punctuation and formatting -- generated numeric, unit,
date and identifier drills, and localization resource strings -- opt out of the
loose round through ``--exact-only``. Merging ``2026-07-31`` with ``2026/07/31``
there would erase the distinction the shard was built to teach.

Which copy survives
-------------------
Within a duplicate group the survivor is chosen by, in order: source tier
(``--tier``, real data before generated data, mirroring the preparer); how many
language edges the row carries, because a ``ko``/``en``/``ja`` row strictly
dominates a ``ko``/``ja`` row holding the same two sentences -- this is what
retires the bilingual game-dialogue shards whose lines were later republished
in trilingual form; a cleanliness score over the pair itself, which penalizes
leading quote and bullet markers and prefers intact sentence punctuation; then
file order and line number so that repeated runs agree.

Usage::

    # report only; nothing is written except the report
    python scripts/data/dedup_corpus.py --report reports/dedup.json

    # write cleaned shards somewhere else
    python scripts/data/dedup_corpus.py --output-dir data/deduped

    # replace the shards, archiving every removed row first
    python scripts/data/dedup_corpus.py --in-place

Exit codes: 0 success, 2 bad input.
"""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import glob as globlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

import numpy as np

from sion_translate.data.records import expand_parallel_record, normalize_language_pairs
from sion_translate.splitting import comparison_key, normalized_split_key


DEFAULT_INPUT_GLOB = "data/*.jsonl"
DEFAULT_CONFIG = Path("sion_translate.yaml")
# Shards whose training signal is the formatting itself.
DEFAULT_EXACT_ONLY = ("synthetic_numeric", "data39")
# The preparer gives real rows precedence over generated rows; mirror it.
DEFAULT_TIERS = ("synthetic=100",)
BASE_TIER = 50

REASON_EXACT = 1
REASON_LOOSE = 2
REASON_CAP = 3
REASON_DENYLIST = 4
REASON_NAMES = {
    REASON_EXACT: "duplicate_exact",
    REASON_LOOSE: "duplicate_loose",
    REASON_CAP: "one_to_many_cap",
    REASON_DENYLIST: "holdout_denylist",
}

# A quoted reply keeps the sentence and puts a marker in front of it. The marker
# is not punctuation the model should learn, so a marked copy loses to a clean
# one even when it was seen first.
_LEADING_MARKER = re.compile(r"^[\s>＞|#*•·=~\-–—]+")
_PUNCTUATION_CAP = 50
_SCORE_OFFSET = 1000
# exact hi/lo, loose hi/lo, side-a hi/lo, side-b hi/lo
_KEY_COLUMNS = 8
# row index, cleanliness score, language edges the row carries
_META_COLUMNS = 3
# Bumped whenever the staged array layout changes, so --reuse-staging cannot
# read arrays written by an older version of this script.
_STAGING_FORMAT = "dedup-corpus-staging-v1"
# Keys that give a record a meaning the flat fast path does not implement: the
# explicit source/target layouts and the nested containers of records.py. A
# record carrying any of them, or any key that could be read as a pair label, is
# handed to the reference expansion instead of being expanded by a weaker rule.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "source",
        "src",
        "input",
        "target",
        "tgt",
        "reference",
        "translation",
        "output",
        "source_language",
        "src_language",
        "target_language",
        "tgt_language",
        "records",
        "items",
        "pairs",
        "translations",
    }
)


def exact_identity(text: str) -> str:
    return normalized_split_key(text)


def loose_identity(text: str) -> str:
    """Punctuation-insensitive key, falling back when a line is all symbols."""

    key = comparison_key(text)
    return key if key else normalized_split_key(text)


def _digest(payload: str) -> tuple[int, int]:
    raw = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(raw[:8], "little"), int.from_bytes(raw[8:], "little")


def pair_digest(language_a: str, key_a: str, language_b: str, key_b: str) -> tuple[int, int]:
    return _digest(f"pair\0{language_a}\0{key_a}\0{language_b}\0{key_b}")


def side_digest(language: str, key: str) -> tuple[int, int]:
    return _digest(f"side\0{language}\0{key}")


def cleanliness_score(texts: Sequence[str]) -> int:
    """Prefer an unquoted copy, then one that kept its sentence punctuation."""

    score = 0
    for text in texts:
        marker = _LEADING_MARKER.match(text)
        if marker is not None and marker.group(0).strip():
            score -= 100
        punctuation = sum(1 for character in text if unicodedata.category(character)[0] == "P")
        score += min(punctuation, _PUNCTUATION_CAP)
    return score


def iter_rows(path: Path) -> Iterator[tuple[int, object]]:
    """Yield ``(line number, parsed row)`` for every non-empty physical line."""

    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield number, json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON") from error


def edge_name(language_a: str, language_b: str) -> str:
    return f"{language_a}-{language_b}"


def expand_flat_record(
    row: object,
    pairs: Sequence[tuple[str, str]],
    languages: frozenset[str],
) -> list[tuple[str, str, str, str]] | None:
    """Expand a flat language-keyed record, or return ``None`` to fall back.

    ``expand_parallel_record`` re-validates every configured language tag on
    every call, which a profile puts at 87% of the scan. That cost is worth
    paying for the arbitrary nested layouts it supports, but every shard in this
    corpus is a flat object whose values are strings or equal-length lists of
    strings. This handles exactly that shape and defers anything else -- nested
    containers, pair-labelled keys, explicit source/target fields -- to the
    reference implementation, so no record is expanded by a weaker rule.

    ``--verify-sample`` checks the two against each other on real rows.
    """

    if not isinstance(row, dict):
        return None
    values: dict[str, list[str]] = {}
    for key, value in row.items():
        if not isinstance(key, str):
            return None
        if key not in languages:
            # A container under an unconfigured key may hold nested records, a
            # key such as "ko-ja" is a pair label, and the reserved keys select
            # the explicit source/target layouts. All need the full walk.
            if isinstance(value, (dict, list, tuple)):
                return None
            if key in _RESERVED_RECORD_KEYS or "-" in key:
                return None
            continue
        if isinstance(value, str):
            values[key] = [value]
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            values[key] = list(value)
        else:
            return None

    expanded: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for language_a, language_b in pairs:
        texts_a = values.get(language_a)
        texts_b = values.get(language_b)
        if texts_a is None or texts_b is None:
            continue
        if len(texts_a) != len(texts_b):
            return None
        for text_a, text_b in zip(texts_a, texts_b, strict=True):
            if not text_a.strip() or not text_b.strip():
                continue
            key = (language_a, text_a, language_b, text_b)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(key)
    return expanded


def reference_expansion(row: object, pairs: Sequence[Sequence[str]]) -> list[tuple[str, str, str, str]]:
    return [
        (pair.language_a, pair.text_a, pair.language_b, pair.text_b)
        for pair in expand_parallel_record(row, pairs).pairs
    ]


def scan_shard(job: tuple[str, str, list[list[str]]]) -> dict[str, object]:
    """Hash every configured pair of one shard into per-edge staging arrays."""

    raw_path, raw_staging, raw_pairs, verify_sample = job
    path = Path(raw_path)
    staging = Path(raw_staging)
    pairs = normalize_language_pairs(language_pairs=raw_pairs)
    languages = frozenset(language for pair in pairs for language in pair)
    name = path.name

    # Flat typed arrays, not lists of tuples: the largest shard expands to over
    # eight million pairs, and boxed Python integers cost gigabytes there.
    keys: dict[str, array] = {}
    meta: dict[str, array] = {}
    has_pairs = array("b")
    issues: dict[str, int] = {}
    rows = 0

    fallbacks = 0
    for number, row in iter_rows(path):
        index = rows
        rows += 1
        expanded = expand_flat_record(row, pairs, languages)
        if expanded is None:
            fallbacks += 1
            expansion = expand_parallel_record(row, pairs)
            for issue in expansion.issues:
                issues[issue] = issues.get(issue, 0) + 1
            expanded = [
                (item.language_a, item.text_a, item.language_b, item.text_b)
                for item in expansion.pairs
            ]
        elif number <= verify_sample and expanded != reference_expansion(row, pairs):
            raise ValueError(
                f"{path}:{number} expands differently under the flat-record fast path; "
                "re-run with --verify-sample 0 only after fixing it"
            )
        if not expanded:
            has_pairs.append(0)
            continue
        has_pairs.append(1)
        edges_in_row = len(expanded)
        for language_a, text_a, language_b, text_b in expanded:
            score = cleanliness_score((text_a, text_b))
            exact_a = exact_identity(text_a)
            exact_b = exact_identity(text_b)
            loose_a = loose_identity(text_a)
            loose_b = loose_identity(text_b)
            edge = edge_name(language_a, language_b)
            if edge not in keys:
                keys[edge] = array("Q")
                meta[edge] = array("I")
            keys[edge].extend(
                (
                    *pair_digest(language_a, exact_a, language_b, exact_b),
                    *pair_digest(language_a, loose_a, language_b, loose_b),
                    *side_digest(language_a, loose_a),
                    *side_digest(language_b, loose_b),
                )
            )
            meta[edge].extend((index, score + _SCORE_OFFSET, edges_in_row))

    staging.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for edge, flat in keys.items():
        counts[edge] = len(flat) // _KEY_COLUMNS
        np.save(
            staging / f"{name}.{edge}.keys.npy",
            np.frombuffer(flat, dtype=np.uint64).reshape(-1, _KEY_COLUMNS),
        )
        np.save(
            staging / f"{name}.{edge}.meta.npy",
            np.frombuffer(meta[edge], dtype=np.uint32).reshape(-1, _META_COLUMNS),
        )
    np.save(staging / f"{name}.rows.npy", np.frombuffer(has_pairs, dtype=np.int8).astype(bool))

    stat = path.stat()
    summary = {
        "file": name,
        "rows": rows,
        "rows_without_a_pair": int(len(has_pairs) - sum(has_pairs)),
        "pairs": counts,
        "expansion_issues": issues,
        "reference_expansion_rows": fallbacks,
        "source_bytes": stat.st_size,
        "source_modified_ns": stat.st_mtime_ns,
        "format": _STAGING_FORMAT,
    }
    (staging / f"{name}.shard.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def reuse_scan(path: Path, staging: Path) -> dict[str, object] | None:
    """A staged scan of this exact file, or ``None`` when it cannot be trusted."""

    summary_path = staging / f"{path.name}.shard.json"
    if not summary_path.exists() or not (staging / f"{path.name}.rows.npy").exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("format") != _STAGING_FORMAT:
        return None
    stat = path.stat()
    if (
        summary.get("source_bytes") != stat.st_size
        or summary.get("source_modified_ns") != stat.st_mtime_ns
    ):
        return None
    return summary


def load_denylist(paths: Sequence[Path], pairs: Sequence[Sequence[str]]) -> set[tuple[int, int]]:
    """Side identities of held-out material that must not appear in training."""

    normalized = normalize_language_pairs(language_pairs=[list(pair) for pair in pairs])
    denied: set[tuple[int, int]] = set()
    for path in paths:
        for _, row in iter_rows(path):
            for pair in expand_parallel_record(row, normalized).pairs:
                denied.add(side_digest(pair.language_a, loose_identity(pair.text_a)))
                denied.add(side_digest(pair.language_b, loose_identity(pair.text_b)))
    return denied


def _group_first(identity: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Positions, in ``order``, that open a new identity group."""

    sorted_identity = identity[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = (sorted_identity[1:, 0] != sorted_identity[:-1, 0]) | (
        sorted_identity[1:, 1] != sorted_identity[:-1, 1]
    )
    return first


def winners(identity: np.ndarray, priority: Sequence[np.ndarray]) -> np.ndarray:
    """Mask of the surviving member of every identity group.

    ``priority`` is ordered most significant first and decides which member of a
    duplicate group survives.
    """

    if len(identity) == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((*reversed(list(priority)), identity[:, 1], identity[:, 0]))
    first = _group_first(identity, order)
    mask = np.zeros(len(order), dtype=bool)
    mask[order[first]] = True
    return mask


def cap_by_source(
    source: np.ndarray,
    priority: Sequence[np.ndarray],
    limit: int,
) -> np.ndarray:
    """Mask keeping at most ``limit`` targets per source surface, best first."""

    if len(source) == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((*reversed(list(priority)), source[:, 1], source[:, 0]))
    first = _group_first(source, order)
    group = np.cumsum(first) - 1
    positions = np.arange(len(order), dtype=np.int64)
    group_start = positions[first][group]
    mask = np.zeros(len(order), dtype=bool)
    mask[order[(positions - group_start) < limit]] = True
    return mask


def tier_of(name: str, tiers: Sequence[tuple[str, int]]) -> int:
    for prefix, rank in tiers:
        if name.startswith(prefix):
            return rank
    return BASE_TIER


def resolve(
    staging: Path,
    shards: Sequence[dict[str, object]],
    *,
    tiers: Sequence[tuple[str, int]],
    exact_only: Sequence[str],
    denylist: set[tuple[int, int]],
    cap: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Decide every row of every shard, one language edge at a time."""

    names = [str(shard["file"]) for shard in shards]
    index_of = {name: position for position, name in enumerate(names)}
    tier_ranks = np.array([tier_of(name, tiers) for name in names], dtype=np.uint32)
    loose_allowed = np.array(
        [not any(name.startswith(prefix) for prefix in exact_only) for name in names],
        dtype=bool,
    )

    kept = {
        name: np.zeros(int(shards[position]["rows"]), dtype=bool)
        for position, name in enumerate(names)
    }
    reason = {
        name: np.zeros(int(shards[position]["rows"]), dtype=np.uint8)
        for position, name in enumerate(names)
    }
    denied_hi = np.array(sorted({high for high, _ in denylist}), dtype=np.uint64)

    edges = sorted({edge for shard in shards for edge in dict(shard["pairs"] or {})})
    edge_report: dict[str, object] = {}

    for edge in edges:
        key_blocks = []
        meta_blocks = []
        owner_blocks = []
        for position, name in enumerate(names):
            key_path = staging / f"{name}.{edge}.keys.npy"
            if not key_path.exists():
                continue
            block = np.load(key_path)
            key_blocks.append(block)
            meta_blocks.append(np.load(staging / f"{name}.{edge}.meta.npy"))
            owner_blocks.append(np.full(len(block), position, dtype=np.uint32))
        if not key_blocks:
            continue
        keys = np.concatenate(key_blocks)
        meta = np.concatenate(meta_blocks)
        owner = np.concatenate(owner_blocks)
        del key_blocks, meta_blocks, owner_blocks

        row_index = meta[:, 0].astype(np.int64)
        # More edges and a higher score are both better, so sort on the negation.
        rank_score = -meta[:, 1].astype(np.int64)
        rank_edges = -meta[:, 2].astype(np.int64)
        rank_tier = tier_ranks[owner].astype(np.int64)
        priority = (rank_tier, rank_edges, rank_score, owner.astype(np.int64), row_index)
        total = len(keys)

        vetoed = np.zeros(total, dtype=bool)
        if denylist:
            for column in (4, 6):
                candidate = np.isin(keys[:, column], denied_hi)
                for position in np.flatnonzero(candidate):
                    identity = (int(keys[position, column]), int(keys[position, column + 1]))
                    if identity in denylist:
                        vetoed[position] = True

        alive = ~vetoed
        exact_survivors = np.zeros(total, dtype=bool)
        exact_survivors[alive] = winners(
            keys[alive][:, 0:2], [column[alive] for column in priority]
        )
        dropped_exact = alive & ~exact_survivors

        loose_pool = exact_survivors & loose_allowed[owner]
        loose_survivors = exact_survivors.copy()
        loose_survivors[loose_pool] = winners(
            keys[loose_pool][:, 2:4], [column[loose_pool] for column in priority]
        )
        dropped_loose = exact_survivors & ~loose_survivors

        dropped_cap = np.zeros(total, dtype=bool)
        if cap > 0:
            capped = np.zeros(total, dtype=bool)
            capped[loose_survivors] = cap_by_source(
                keys[loose_survivors][:, 4:6],
                [column[loose_survivors] for column in priority],
                cap,
            )
            dropped_cap = loose_survivors & ~capped
            loose_survivors = capped

        for mask, code in (
            (dropped_exact, REASON_EXACT),
            (dropped_loose, REASON_LOOSE),
            (dropped_cap, REASON_CAP),
            (vetoed, REASON_DENYLIST),
        ):
            for position in np.unique(owner[mask]):
                name = names[int(position)]
                selected = mask & (owner == position)
                target = reason[name]
                rows = row_index[selected]
                np.maximum.at(target, rows, code)

        for position in np.unique(owner[loose_survivors]):
            name = names[int(position)]
            selected = loose_survivors & (owner == position)
            kept[name][row_index[selected]] = True

        # A held-out sentence must not survive anywhere, so its veto outranks a
        # sibling pair on the same row.
        for position in np.unique(owner[vetoed]):
            name = names[int(position)]
            selected = vetoed & (owner == position)
            kept[name][row_index[selected]] = False

        edge_report[edge] = {
            "pairs": int(total),
            "kept": int(loose_survivors.sum()),
            "dropped_duplicate_exact": int(dropped_exact.sum()),
            "dropped_duplicate_loose": int(dropped_loose.sum()),
            "dropped_one_to_many_cap": int(dropped_cap.sum()),
            "dropped_holdout_denylist": int(vetoed.sum()),
        }
        del keys, meta, owner

    return kept, reason, edge_report


class AtomicJsonlWriter:
    """Stream lines into a sibling staging file, then publish it in one step.

    Shards reach several gigabytes, so nothing is buffered in memory, and an
    in-place rewrite must not touch its own input until the read is finished.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        self._temporary = Path(temporary_name)
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self.lines = 0

    def write(self, line: str) -> None:
        self._handle.write(line)
        self.lines += 1

    def publish(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.replace(self._temporary, self.path)

    def discard(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self._temporary.unlink(missing_ok=True)


def apply_decisions(
    path: Path,
    *,
    keep: np.ndarray,
    reason: np.ndarray,
    has_pairs: np.ndarray,
    destination: Path | None,
    archive: Path | None,
    keep_unpaired: bool,
) -> dict[str, int]:
    """Rewrite one shard, archiving the rows it loses."""

    counts = {name: 0 for name in REASON_NAMES.values()}
    counts["rows_without_a_pair_kept"] = 0
    counts["no_configured_pair"] = 0
    kept_writer = AtomicJsonlWriter(destination / path.name) if destination else None
    archive_writer = (
        AtomicJsonlWriter(archive / f"{path.stem}.removed.jsonl") if archive else None
    )
    kept_rows = 0
    removed_rows = 0
    index = 0

    def removal(number: int, label: str, payload: str) -> None:
        nonlocal removed_rows
        removed_rows += 1
        if archive_writer is not None:
            archive_writer.write(
                json.dumps(
                    {
                        "file": path.name,
                        "line": number,
                        "reason": label,
                        "row": json.loads(payload),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                position = index
                index += 1
                if not has_pairs[position]:
                    if keep_unpaired:
                        counts["rows_without_a_pair_kept"] += 1
                        kept_rows += 1
                        if kept_writer is not None:
                            kept_writer.write(stripped + "\n")
                    else:
                        counts["no_configured_pair"] += 1
                        removal(number, "no_configured_pair", stripped)
                    continue
                if keep[position]:
                    kept_rows += 1
                    if kept_writer is not None:
                        kept_writer.write(stripped + "\n")
                    continue
                label = REASON_NAMES.get(int(reason[position]), "duplicate_exact")
                counts[label] += 1
                removal(number, label, stripped)
    except BaseException:
        for writer in (kept_writer, archive_writer):
            if writer is not None:
                writer.discard()
        raise

    # The input is closed now, so an in-place rewrite may replace it.
    if archive_writer is not None:
        if removed_rows:
            archive_writer.publish()
        else:
            archive_writer.discard()
    if kept_writer is not None:
        kept_writer.publish()
    counts["kept"] = kept_rows
    counts["removed"] = removed_rows
    return counts


def configured_pairs(
    config_path: Path | None,
    explicit: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    if explicit:
        return normalize_language_pairs(language_pairs=explicit)
    from sion_translate.config import load_config

    path = config_path or DEFAULT_CONFIG
    return normalize_language_pairs(
        language_pairs=load_config(path).data.configured_language_pairs()
    )


def parse_tiers(values: Sequence[str]) -> tuple[tuple[str, int], ...]:
    tiers: list[tuple[str, int]] = []
    for value in values:
        prefix, separator, rank = value.partition("=")
        if not separator or not prefix:
            raise ValueError(f"--tier expects PREFIX=RANK, got {value!r}")
        tiers.append((prefix, int(rank)))
    return tuple(tiers)


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = globlib.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for match in matches:
            candidate = Path(match)
            if candidate.is_file():
                paths.add(candidate)
    return sorted(paths, key=lambda item: item.name)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[DEFAULT_INPUT_GLOB],
        help=f"shards or globs (default: {DEFAULT_INPUT_GLOB})",
    )
    parser.add_argument("--config", type=Path, help="config to read language pairs from")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("LANG_A", "LANG_B"),
        help="language pair, repeatable; overrides --config",
    )
    parser.add_argument("--output-dir", type=Path, help="write cleaned shards here")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input shards after archiving every removed row",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="where removed rows are stored (default: data/excluded/dedup_<today>)",
    )
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument(
        "--exact-only",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "shard prefix that skips the punctuation-insensitive round "
            f"(default: {', '.join(DEFAULT_EXACT_ONLY)})"
        ),
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="PREFIX=RANK",
        help=f"survivor precedence, lower wins (default: {', '.join(DEFAULT_TIERS)})",
    )
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        metavar="PATH",
        help="evaluation file whose sentences must not remain in training, repeatable",
    )
    parser.add_argument(
        "--max-targets-per-source",
        type=int,
        default=0,
        help="keep at most N translations of one source surface (0 disables)",
    )
    parser.add_argument(
        "--drop-rows-without-a-pair",
        action="store_true",
        help="also remove rows that no configured language pair can read",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel shard scanners",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="where per-shard hashes are staged (default: a temporary directory)",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=500,
        help=(
            "cross-check the flat-record fast path against expand_parallel_record "
            "on this many leading rows of each shard (0 disables)"
        ),
    )
    parser.add_argument(
        "--reuse-staging",
        action="store_true",
        help=(
            "skip scanning a shard whose staged hashes match its current size and "
            "timestamp, so a reported run can be applied without rescanning"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.in_place and args.output_dir is not None:
        print("--in-place and --output-dir are mutually exclusive", file=sys.stderr)
        return 2
    if args.max_targets_per_source < 0:
        print("--max-targets-per-source must not be negative", file=sys.stderr)
        return 2

    paths = expand_inputs(args.inputs)
    if not paths:
        print(f"no shards matched: {args.inputs}", file=sys.stderr)
        return 2
    try:
        pairs = configured_pairs(args.config, args.pair)
        tiers = parse_tiers(args.tier or list(DEFAULT_TIERS))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    exact_only = tuple(args.exact_only) or DEFAULT_EXACT_ONLY

    denylist: set[tuple[int, int]] = set()
    if args.holdout:
        denylist = load_denylist(expand_inputs(args.holdout), pairs)

    staging_context: tempfile.TemporaryDirectory[str] | None = None
    if args.staging_dir is not None:
        staging = args.staging_dir
        staging.mkdir(parents=True, exist_ok=True)
    else:
        staging_context = tempfile.TemporaryDirectory(prefix="dedup-corpus-")
        staging = Path(staging_context.name)

    try:
        shards: list[dict[str, object]] = []
        pending: list[tuple[str, str, list[list[str]]]] = []
        for path in paths:
            reused = reuse_scan(path, staging) if args.reuse_staging else None
            if reused is not None:
                shards.append(reused)
                print(f"reused {path.name} rows={reused['rows']}", flush=True)
            else:
                pending.append(
                    (str(path), str(staging), [list(pair) for pair in pairs], args.verify_sample)
                )

        if args.jobs > 1 and len(pending) > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for result in pool.map(scan_shard, pending):
                    shards.append(result)
                    print(f"scanned {result['file']} rows={result['rows']}", flush=True)
        else:
            for job in pending:
                result = scan_shard(job)
                shards.append(result)
                print(f"scanned {result['file']} rows={result['rows']}", flush=True)
        order = {path.name: position for position, path in enumerate(paths)}
        shards.sort(key=lambda shard: order[str(shard["file"])])

        kept, reason, edge_report = resolve(
            staging,
            shards,
            tiers=tiers,
            exact_only=exact_only,
            denylist=denylist,
            cap=args.max_targets_per_source,
        )

        destination = args.output_dir
        archive = args.archive_dir
        if args.in_place and archive is None:
            archive = Path("data/excluded") / f"dedup_{date.today():%Y%m%d}"
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
        if archive is not None:
            archive.mkdir(parents=True, exist_ok=True)

        per_shard: dict[str, object] = {}
        totals = {name: 0 for name in REASON_NAMES.values()}
        emptied: list[str] = []
        unchanged: list[str] = []
        kept_rows = 0
        for path in paths:
            name = path.name
            has_pairs = np.load(staging / f"{name}.rows.npy")
            # A shard that loses nothing is already the file it would be
            # rewritten into. Skipping it keeps an unchanged shard's timestamp
            # and digest stable, and turns the publish pass from a rewrite of
            # the whole corpus into a rewrite of the shards that changed.
            # Only in place: an --output-dir run must still produce the whole
            # corpus, so an unchanged shard is copied there like any other.
            losses = int(np.count_nonzero(has_pairs & ~kept[name]))
            unpaired = int(np.count_nonzero(~has_pairs))
            if args.in_place and losses == 0 and (
                not args.drop_rows_without_a_pair or unpaired == 0
            ):
                unchanged.append(name)
                rows_kept = int(np.count_nonzero(has_pairs)) + unpaired
                per_shard[name] = {"kept": rows_kept, "removed": 0, "rewritten": False}
                kept_rows += rows_kept
                continue
            target = destination
            if args.in_place:
                target = path.parent
            counts = apply_decisions(
                path,
                keep=kept[name],
                reason=reason[name],
                has_pairs=has_pairs,
                destination=target,
                archive=archive,
                keep_unpaired=not args.drop_rows_without_a_pair,
            )
            counts["rewritten"] = True
            per_shard[name] = counts
            kept_rows += counts["kept"]
            for label in totals:
                totals[label] += counts[label]
            if counts["kept"] == 0 and counts["removed"] > 0:
                emptied.append(name)
                print(
                    f"warning: every row of {name} duplicates a row kept elsewhere",
                    file=sys.stderr,
                )

        report = {
            "shards": len(paths),
            "language_pairs": [list(pair) for pair in pairs],
            "identity": {
                "exact": "NFKC + whitespace fold (normalized_split_key)",
                "loose": "NFKC minus punctuation/symbols/separators (comparison_key)",
                "exact_only_prefixes": list(exact_only),
            },
            "tiers": [f"{prefix}={rank}" for prefix, rank in tiers],
            "max_targets_per_source": args.max_targets_per_source,
            "holdout_side_identities": len(denylist),
            "rows_in": sum(int(shard["rows"]) for shard in shards),
            "rows_out": kept_rows,
            "removed": totals,
            "emptied_shards": emptied,
            "unchanged_shards": len(unchanged),
            "edges": edge_report,
            "files": per_shard,
            "written": (
                "in-place" if args.in_place else (str(destination) if destination else "nothing")
            ),
            "archive": str(archive) if archive else None,
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps({key: report[key] for key in ("rows_in", "rows_out", "removed", "edges")},
                         ensure_ascii=False, indent=2))
    finally:
        if staging_context is not None:
            staging_context.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
