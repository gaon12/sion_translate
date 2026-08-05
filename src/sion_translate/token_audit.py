"""Measure whether tokenizer pieces receive enough translation training signal.

Tokenizer *coverage* and model *exposure* are different problems. Byte fallback
guarantees that every string can be encoded, but a piece seen only a handful of
times as a decoder target still has an effectively untrained output embedding.
This module scans the same parallel JSONL schema as the training pipeline and
reports both failure modes by language and translation direction.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
)
from sion_translate.tokenizer import SionTokenizer, expand_inputs


def _piece_is_special(piece: str) -> bool:
    return piece.startswith("<") and piece.endswith(">") and not piece.startswith("<0x")


def _frequency_summary(counts: np.ndarray, eligible: np.ndarray) -> dict[str, int | float]:
    values = counts[eligible]
    observed = values[values > 0]
    return {
        "eligible_pieces": int(values.size),
        "observed_pieces": int(observed.size),
        "unused_pieces": int(np.count_nonzero(values == 0)),
        "seen_once": int(np.count_nonzero(values == 1)),
        "seen_1_to_9": int(np.count_nonzero((values >= 1) & (values <= 9))),
        "seen_1_to_24": int(np.count_nonzero((values >= 1) & (values <= 24))),
        "median_observed_count": float(np.median(observed)) if observed.size else 0.0,
        "p10_observed_count": float(np.percentile(observed, 10)) if observed.size else 0.0,
    }


def _rare_piece_examples(
    tokenizer: SionTokenizer,
    counts: np.ndarray,
    eligible: np.ndarray,
    *,
    maximum: int,
    include_unused: bool,
) -> list[dict[str, int | str]]:
    candidates = np.flatnonzero(eligible & ((counts >= 0) if include_unused else (counts > 0)))
    ordered = candidates[np.lexsort((candidates, counts[candidates]))]
    return [
        {
            "id": int(token_id),
            "piece": tokenizer.processor.id_to_piece(int(token_id)),
            "count": int(counts[token_id]),
        }
        for token_id in ordered[:maximum]
    ]


def audit_token_exposure(
    input_patterns: Sequence[str],
    tokenizer_model: str | Path,
    *,
    language_pair: Sequence[str] = ("ko", "ja"),
    language_pairs: Sequence[Sequence[str]] | None = None,
    source_only_languages: Sequence[str] = (),
    bidirectional: bool = True,
    max_physical_pairs: int = 0,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
    filter_quality: bool = True,
) -> dict[str, Any]:
    """Audit target-token exposure without materializing an indexed dataset.

    ``max_physical_pairs=0`` performs an exact full scan. A positive value is a
    deterministic prefix sample and is labelled as such in the report; it is
    useful for a quick preflight, not for declaring a vocabulary safe.
    """

    if max_physical_pairs < 0:
        raise ValueError("max_physical_pairs must be non-negative")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")
    if max_piece_examples < 0:
        raise ValueError("max_piece_examples must be non-negative")

    paths = expand_inputs(input_patterns)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {list(input_patterns)}")
    pairs = normalize_language_pairs(language_pair, language_pairs)
    languages = languages_from_pairs(pairs)
    source_only = frozenset(map(str, source_only_languages))
    unknown = sorted(source_only - set(languages))
    if unknown:
        raise ValueError(f"unknown source_only_languages: {unknown}")

    tokenizer = SionTokenizer(tokenizer_model)
    missing_tags = sorted(set(languages) - set(tokenizer.languages))
    if missing_tags:
        raise ValueError(f"tokenizer is missing configured language tags: {missing_tags}")

    vocab_size = len(tokenizer)
    physical_counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in languages}
    source_counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in languages}
    target_counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in languages}
    language_totals: dict[str, Counter[str]] = {language: Counter() for language in languages}
    direction_totals: dict[str, Counter[str]] = {}
    invalid = Counter()
    physical_pairs = 0
    virtual_examples = 0
    policy = QualityPolicy()

    def add_counts(target: np.ndarray, token_ids: list[int]) -> None:
        if token_ids:
            target += np.bincount(token_ids, minlength=vocab_size).astype(np.uint64, copy=False)

    stop = False
    for path in paths:
        with path.open("rb") as handle:
            for raw_line in handle:
                try:
                    row = json.loads(raw_line.decode("utf-8-sig"))
                except UnicodeDecodeError:
                    invalid["invalid_utf8"] += 1
                    continue
                except json.JSONDecodeError:
                    invalid["invalid_json"] += 1
                    continue
                expansion = expand_parallel_record(row, pairs)
                invalid.update(expansion.issues)
                for pair in expansion.pairs:
                    text_a = canonical_text(pair.text_a)
                    text_b = canonical_text(pair.text_b)
                    if (
                        filter_quality
                        and not assess_pair(
                            text_a,
                            text_b,
                            policy,
                            languages=(pair.language_a, pair.language_b),
                        ).accepted
                    ):
                        invalid["quality_filtered"] += 1
                        continue
                    ids_a = tokenizer.encode(text_a)
                    ids_b = tokenizer.encode(text_b)
                    add_counts(physical_counts[pair.language_a], ids_a)
                    add_counts(physical_counts[pair.language_b], ids_b)
                    physical_pairs += 1

                    directions = [(pair.language_a, ids_a, text_a, pair.language_b, ids_b, text_b)]
                    if bidirectional:
                        directions.append(
                            (pair.language_b, ids_b, text_b, pair.language_a, ids_a, text_a)
                        )
                    for src_lang, src_ids, src_text, tgt_lang, tgt_ids, tgt_text in directions:
                        if tgt_lang in source_only:
                            continue
                        add_counts(source_counts[src_lang], src_ids)
                        add_counts(target_counts[tgt_lang], tgt_ids)
                        virtual_examples += 1
                        direction = f"{src_lang}-{tgt_lang}"
                        totals = direction_totals.setdefault(direction, Counter())
                        totals["examples"] += 1
                        totals["source_tokens"] += len(src_ids)
                        totals["target_tokens"] += len(tgt_ids)
                        totals["source_characters"] += sum(not c.isspace() for c in src_text)
                        totals["target_characters"] += sum(not c.isspace() for c in tgt_text)
                    for language, ids, text in (
                        (pair.language_a, ids_a, text_a),
                        (pair.language_b, ids_b, text_b),
                    ):
                        totals = language_totals[language]
                        totals["physical_sentences"] += 1
                        totals["physical_tokens"] += len(ids)
                        totals["physical_characters"] += sum(not c.isspace() for c in text)
                        totals["byte_fallback_tokens"] += sum(
                            tokenizer.processor.id_to_piece(token_id).startswith("<0x")
                            for token_id in ids
                        )
                        totals["unknown_tokens"] += sum(
                            token_id == tokenizer.unk_id for token_id in ids
                        )

                    if max_physical_pairs and physical_pairs >= max_physical_pairs:
                        stop = True
                        break
                if stop:
                    break
        if stop:
            break

    special = np.array(
        [_piece_is_special(tokenizer.processor.id_to_piece(i)) for i in range(vocab_size)],
        dtype=np.bool_,
    )
    byte = np.array(
        [tokenizer.processor.id_to_piece(i).startswith("<0x") for i in range(vocab_size)],
        dtype=np.bool_,
    )
    eligible = ~(special | byte)
    global_physical = np.zeros(vocab_size, dtype=np.uint64)
    global_target = np.zeros(vocab_size, dtype=np.uint64)
    for language in languages:
        global_physical += physical_counts[language]
        global_target += target_counts[language]

    language_reports: dict[str, Any] = {}
    for language in languages:
        totals = language_totals[language]
        tokens = totals["physical_tokens"]
        characters = totals["physical_characters"]
        target_enabled = language not in source_only
        target_eligible = (
            eligible & (physical_counts[language] > 0)
            if target_enabled
            else np.zeros(vocab_size, dtype=np.bool_)
        )
        target_summary = _frequency_summary(target_counts[language], target_eligible)
        target_summary["rare_threshold"] = rare_threshold
        target_summary["rare_observed_pieces"] = int(
            np.count_nonzero(
                target_eligible
                & (target_counts[language] > 0)
                & (target_counts[language] < rare_threshold)
            )
        )
        language_reports[language] = {
            **{name: int(value) for name, value in totals.items()},
            "target_enabled": target_enabled,
            "tokens_per_character": round(tokens / max(characters, 1), 6),
            "byte_fallback_rate": round(totals["byte_fallback_tokens"] / max(tokens, 1), 8),
            "target_frequency": target_summary,
            "lowest_target_exposure": _rare_piece_examples(
                tokenizer,
                target_counts[language],
                target_eligible,
                maximum=max_piece_examples,
                include_unused=True,
            ),
        }

    direction_report = {}
    for direction, totals in sorted(direction_totals.items()):
        target_tokens = totals["target_tokens"]
        target_characters = totals["target_characters"]
        direction_report[direction] = {
            **{name: int(value) for name, value in totals.items()},
            "target_tokens_per_character": round(target_tokens / max(target_characters, 1), 6),
            "mean_target_tokens": round(target_tokens / max(totals["examples"], 1), 6),
        }

    global_corpus_observed = eligible & (global_physical > 0)
    # The global view deliberately covers the complete learnable vocabulary.
    # Restricting it to pieces present in this corpus would hide the most severe
    # failure mode: an ordinary SentencePiece token that never receives a decoder
    # update at all. Per-language summaries remain corpus-conditioned so Japanese
    # pieces are not misleadingly labelled as unused Korean targets (and vice versa).
    global_summary = _frequency_summary(global_target, eligible)
    global_summary["corpus_observed_pieces"] = int(np.count_nonzero(global_corpus_observed))
    global_summary["rare_threshold"] = rare_threshold
    global_summary["rare_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_target > 0) & (global_target < rare_threshold))
    )
    return {
        "schema": "sion-token-exposure-audit-v1",
        "complete_scan": max_physical_pairs == 0 or not stop,
        "parameters": {
            "tokenizer_model": str(Path(tokenizer_model).resolve()),
            "language_pairs": [list(pair) for pair in pairs],
            "source_only_languages": sorted(source_only),
            "bidirectional": bidirectional,
            "max_physical_pairs": max_physical_pairs,
            "rare_threshold": rare_threshold,
            "filter_quality": filter_quality,
        },
        "vocab_size": vocab_size,
        "physical_pairs": physical_pairs,
        "virtual_translation_examples": virtual_examples,
        "invalid_or_filtered": dict(sorted(invalid.items())),
        "directions": direction_report,
        "languages": language_reports,
        "global_target_frequency": global_summary,
        "lowest_global_target_exposure": _rare_piece_examples(
            tokenizer,
            global_target,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
    }


__all__ = ["audit_token_exposure"]
