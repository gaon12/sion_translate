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
from typing import Any, Iterator, Sequence

import numpy as np

from sion_translate.data.monolingual import MonolingualDiscovery, iter_monolingual_lines
from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
)
from sion_translate.fingerprint import file_sha256
from sion_translate.tokenizer import SionTokenizer, expand_inputs


def _piece_is_special(piece: str) -> bool:
    return piece.startswith("<") and piece.endswith(">") and not piece.startswith("<0x")


def _frequency_summary(counts: np.ndarray, eligible: np.ndarray) -> dict[str, int | float]:
    values = counts[eligible]
    observed = values[values > 0]
    return {
        "total_occurrences": int(values.sum(dtype=np.uint64)),
        "eligible_pieces": int(values.size),
        "observed_pieces": int(observed.size),
        "unused_pieces": int(np.count_nonzero(values == 0)),
        "seen_once": int(np.count_nonzero(values == 1)),
        "seen_1_to_9": int(np.count_nonzero((values >= 1) & (values <= 9))),
        "seen_1_to_24": int(np.count_nonzero((values >= 1) & (values <= 24))),
        "median_observed_count": float(np.median(observed)) if observed.size else 0.0,
        "p10_observed_count": float(np.percentile(observed, 10)) if observed.size else 0.0,
    }


def _load_indexed_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Indexed dataset manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid indexed dataset manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Indexed dataset manifest must contain a JSON object: {manifest_path}")
    return manifest


def _indexed_tokenizer_identity(
    manifest: dict[str, Any],
    tokenizer_model: Path,
) -> dict[str, str | bool | None]:
    """Verify the tokenizer mapping when an indexed manifest has an identity."""

    expected_sha256 = None
    identity_source = None
    fingerprint = manifest.get("fingerprint")
    if isinstance(fingerprint, dict) and isinstance(fingerprint.get("tokenizer_sha256"), str):
        expected_sha256 = fingerprint["tokenizer_sha256"].lower()
        identity_source = "manifest.fingerprint.tokenizer_sha256"
    else:
        recorded_path = manifest.get("tokenizer_model")
        if isinstance(recorded_path, str) and Path(recorded_path).is_file():
            expected_sha256 = file_sha256(recorded_path).lower()
            identity_source = "manifest.tokenizer_model"

    actual_sha256 = file_sha256(tokenizer_model).lower()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "Tokenizer SHA-256 does not match the indexed dataset: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return {
        "sha256": actual_sha256,
        "verified_against_manifest": expected_sha256 is not None,
        "identity_source": identity_source,
    }


def _indexed_languages(manifest: dict[str, Any], *, modern: bool) -> tuple[str, ...]:
    raw_languages = manifest.get("languages")
    if isinstance(raw_languages, list) and raw_languages:
        languages = tuple(str(language) for language in raw_languages)
    else:
        raw_pair = manifest.get("language_pair", ["ko", "ja"])
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError("Indexed dataset manifest has no valid language metadata")
        languages = tuple(dict.fromkeys(str(language) for language in raw_pair))
    if any(not language for language in languages) or len(set(languages)) != len(languages):
        raise ValueError("Indexed dataset languages must be unique, non-empty strings")
    if modern and len(languages) > np.iinfo(np.uint16).max:
        raise ValueError("Indexed dataset has too many languages for uint16 language ids")
    return languages


def _row_blocks(lengths: np.ndarray, maximum_tokens: int = 4_000_000) -> Iterator[slice]:
    """Yield row slices whose expanded token metadata stays memory-bounded."""

    if len(lengths) == 0:
        return
    cumulative = np.cumsum(lengths, dtype=np.uint64)
    start = 0
    while start < len(lengths):
        before = 0 if start == 0 else int(cumulative[start - 1])
        end = int(np.searchsorted(cumulative, before + maximum_tokens, side="right"))
        end = max(start + 1, end)
        yield slice(start, min(end, len(lengths)))
        start = end


def _open_indexed_token_store(
    path: Path,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Indexed token shard not found: {path}")
    byte_size = path.stat().st_size
    if byte_size % np.dtype(np.uint32).itemsize:
        raise ValueError(f"Indexed token shard has a partial uint32 value: {path}")
    token_count = byte_size // np.dtype(np.uint32).itemsize
    expected_offsets = np.cumsum(
        np.concatenate((np.zeros(1, dtype=np.uint64), lengths[:-1].astype(np.uint64))),
        dtype=np.uint64,
    )
    if not np.array_equal(offsets.astype(np.uint64), expected_offsets):
        raise ValueError(f"Indexed token offsets are not contiguous in {path}")
    expected_tokens = int(lengths.sum(dtype=np.uint64))
    if token_count != expected_tokens:
        raise ValueError(
            f"Indexed token count does not match its index in {path}: "
            f"{token_count} != {expected_tokens}"
        )
    if token_count == 0:
        return np.empty(0, dtype=np.uint32)
    return np.memmap(path, dtype=np.uint32, mode="r")


def _accumulate_indexed_side(
    store: np.ndarray,
    lengths: np.ndarray,
    language_ids: np.ndarray,
    target_rows: np.ndarray,
    physical_counts: list[np.ndarray],
    target_counts: list[np.ndarray],
    *,
    vocab_size: int,
) -> None:
    """Count a stored side once physically and when it is a decoder target."""

    token_offset = 0
    for block in _row_blocks(lengths):
        block_lengths = lengths[block].astype(np.int64, copy=False)
        block_size = int(block_lengths.sum(dtype=np.int64))
        block_tokens = np.asarray(store[token_offset : token_offset + block_size])
        token_offset += block_size
        if block_size == 0:
            continue
        maximum_id = int(block_tokens.max(initial=0))
        if maximum_id >= vocab_size:
            raise ValueError(
                f"Indexed token id {maximum_id} exceeds tokenizer vocabulary size {vocab_size}"
            )
        row_languages = language_ids[block]
        row_targets = target_rows[block]
        unique_languages = np.unique(row_languages)
        if len(unique_languages) == 1:
            language_id = int(unique_languages[0])
            all_counts = np.bincount(
                block_tokens.astype(np.int64, copy=False),
                minlength=vocab_size,
            ).astype(np.uint64, copy=False)
            physical_counts[language_id] += all_counts
            if bool(row_targets.all()):
                target_counts[language_id] += all_counts
            elif bool(row_targets.any()):
                token_targets = np.repeat(row_targets, block_lengths)
                decoder_tokens = block_tokens[token_targets].astype(np.int64, copy=False)
                target_counts[language_id] += np.bincount(
                    decoder_tokens,
                    minlength=vocab_size,
                ).astype(np.uint64, copy=False)
            continue

        token_languages = np.repeat(row_languages, block_lengths)
        token_targets = None if bool(row_targets.all()) else np.repeat(row_targets, block_lengths)
        for language_id in unique_languages:
            language_mask = token_languages == language_id
            language_tokens = block_tokens[language_mask].astype(np.int64, copy=False)
            all_counts = np.bincount(
                language_tokens,
                minlength=vocab_size,
            ).astype(np.uint64, copy=False)
            physical_counts[int(language_id)] += all_counts
            if token_targets is None:
                target_counts[int(language_id)] += all_counts
                continue
            decoder_mask = language_mask & token_targets
            if bool(decoder_mask.any()):
                decoder_tokens = block_tokens[decoder_mask].astype(np.int64, copy=False)
                target_counts[int(language_id)] += np.bincount(
                    decoder_tokens,
                    minlength=vocab_size,
                ).astype(np.uint64, copy=False)


def _add_direction_totals(
    totals: dict[str, Counter[str]],
    source_languages: np.ndarray,
    target_languages: np.ndarray,
    source_lengths: np.ndarray,
    target_lengths: np.ndarray,
    enabled: np.ndarray,
    languages: Sequence[str],
) -> int:
    examples = 0
    enabled_indices = np.flatnonzero(enabled)
    if not enabled_indices.size:
        return examples
    pair_keys = source_languages[enabled_indices].astype(np.uint64) * np.uint64(
        len(languages)
    ) + target_languages[enabled_indices].astype(np.uint64)
    for pair_key in np.unique(pair_keys):
        selected = enabled_indices[pair_keys == pair_key]
        source_id, target_id = divmod(int(pair_key), len(languages))
        direction = f"{languages[source_id]}-{languages[target_id]}"
        direction_totals = totals.setdefault(direction, Counter())
        direction_totals["examples"] += len(selected)
        direction_totals["source_tokens"] += int(source_lengths[selected].sum(dtype=np.uint64))
        direction_totals["target_tokens"] += int(target_lengths[selected].sum(dtype=np.uint64))
        examples += len(selected)
    return examples


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


def audit_monolingual_token_exposure(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    *,
    minimum_characters: int = 8,
    maximum_characters: int = 4000,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
    max_lines_per_language: int = 0,
) -> dict[str, Any]:
    """foundation 코퍼스가 각 조각을 디코더 타깃으로 몇 번 보여 주는지 센다.

    복원 과제의 정답은 손상되지 않은 원문 전체입니다. 즉 단일어 코퍼스의
    **모든 토큰이 디코더 타깃**이고, 그래서 이 단계는 병렬 코퍼스가 한 번도
    출력으로 만들어 본 적 없는 조각에도 출력 임베딩 학습 신호를 줍니다.

    병렬 감사만 보고 어휘를 판단하면 두 방향으로 틀립니다. foundation 이
    충분히 노출시키는 조각을 위험하다고 하거나, 반대로 단일어 코퍼스가 어휘에
    밀어 넣은 조각이 번역 학습에서 전혀 나오지 않는 것을 놓칩니다.

    ``max_lines_per_language=0`` 이 전량 스캔이고, 양수는 결정적 prefix 표본
    이라 보고서에 그렇게 표시됩니다 — 빠른 preflight 용이지 어휘가 안전하다고
    선언할 근거는 아닙니다.
    """

    if minimum_characters < 1:
        raise ValueError("minimum_characters must be positive")
    if maximum_characters <= minimum_characters:
        raise ValueError("maximum_characters must be greater than minimum_characters")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")
    if max_lines_per_language < 0:
        raise ValueError("max_lines_per_language must be non-negative")
    if not discovery.sources:
        raise ValueError(f"단일어 코퍼스에 읽을 수 있는 파일이 없습니다: {discovery.root}")

    tokenizer = SionTokenizer(tokenizer_model)
    vocab_size = len(tokenizer)
    counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in discovery.languages}
    accepted = Counter()
    dropped = Counter()

    for language in discovery.languages:
        target = counts[language]
        for path in discovery.paths_for(language):
            if max_lines_per_language and accepted[language] >= max_lines_per_language:
                break
            for text in iter_monolingual_lines(path):
                if max_lines_per_language and accepted[language] >= max_lines_per_language:
                    break
                normalized = canonical_text(text)
                if len(normalized) < minimum_characters:
                    dropped[f"{language}:too_short"] += 1
                    continue
                if len(normalized) > maximum_characters:
                    dropped[f"{language}:too_long"] += 1
                    continue
                token_ids = tokenizer.encode(normalized)
                if token_ids:
                    target += np.bincount(token_ids, minlength=vocab_size).astype(
                        np.uint64, copy=False
                    )
                accepted[language] += 1

    combined = np.zeros(vocab_size, dtype=np.uint64)
    for value in counts.values():
        combined += value
    eligible = np.array(
        [
            not _piece_is_special(tokenizer.processor.id_to_piece(index))
            for index in range(vocab_size)
        ]
    )
    return {
        "scan": "monolingual-corpus",
        "root": str(discovery.root),
        "complete_scan": max_lines_per_language == 0,
        "max_lines_per_language": max_lines_per_language,
        "rare_threshold": rare_threshold,
        "vocab_size": vocab_size,
        "languages": list(discovery.languages),
        "accepted_lines": dict(accepted),
        "dropped_lines": dict(dropped),
        "decoder_target_totals": _frequency_summary(combined, eligible),
        "per_language": {
            language: _frequency_summary(counts[language], eligible) for language in counts
        },
        "lowest_target_exposure": _rare_piece_examples(
            tokenizer,
            combined,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
        "counts": combined,
    }


def combine_target_exposure(
    parallel_counts: np.ndarray,
    monolingual_counts: np.ndarray,
    tokenizer_model: str | Path,
    *,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
) -> dict[str, Any]:
    """두 단계를 합쳐야 비로소 "이 조각이 학습되는가"에 답할 수 있다.

    foundation 이 먼저 돌면 출력 임베딩은 두 단계 모두에서 신호를 받습니다.
    한쪽만 보고 판정하면 조각을 잘못 살리거나 잘못 죽입니다.
    """

    if parallel_counts.shape != monolingual_counts.shape:
        raise ValueError("count vectors must describe the same vocabulary")
    tokenizer = SionTokenizer(tokenizer_model)
    vocab_size = len(tokenizer)
    if parallel_counts.shape[0] != vocab_size:
        raise ValueError("count vectors do not match the tokenizer vocabulary size")
    eligible = np.array(
        [
            not _piece_is_special(tokenizer.processor.id_to_piece(index))
            for index in range(vocab_size)
        ]
    )
    combined = parallel_counts.astype(np.uint64) + monolingual_counts.astype(np.uint64)
    rescued = int(
        np.count_nonzero(
            eligible & (parallel_counts < rare_threshold) & (combined >= rare_threshold)
        )
    )
    still_rare = int(np.count_nonzero(eligible & (combined < rare_threshold)))
    return {
        "scan": "combined-stages",
        "rare_threshold": rare_threshold,
        "totals": _frequency_summary(combined, eligible),
        # foundation 이 병렬 코퍼스만으로는 부족했던 조각을 몇 개 구제했는가.
        "rescued_by_foundation": rescued,
        "still_below_threshold": still_rare,
        "lowest_target_exposure": _rare_piece_examples(
            tokenizer,
            combined,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
    }


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
    return_counts: bool = False,
) -> dict[str, Any]:
    """Audit target-token exposure without materializing an indexed dataset.

    ``return_counts`` attaches the raw decoder-target count vector under
    ``global_target_counts`` so a caller can combine it with another stage's
    exposure. It is off by default because the value is a vocabulary-sized
    NumPy array and the report is otherwise JSON-serializable.

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
            "target_tokens": int(target_counts[language].sum(dtype=np.uint64)),
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
    global_summary["all_target_tokens"] = int(global_target.sum(dtype=np.uint64))
    global_summary["corpus_observed_pieces"] = int(np.count_nonzero(global_corpus_observed))
    global_summary["rare_threshold"] = rare_threshold
    global_summary["rare_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_target > 0) & (global_target < rare_threshold))
    )
    report_counts = {"global_target_counts": global_target} if return_counts else {}
    return {
        **report_counts,
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


def audit_indexed_token_exposure(
    dataset_root: str | Path,
    tokenizer_model: str | Path,
    *,
    split: str = "train",
    bidirectional: bool = True,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
) -> dict[str, Any]:
    """Audit exact decoder-target exposure from already indexed token shards.

    The scan follows the indexed dataset's virtual-direction semantics without
    decoding or re-tokenizing text. Side B is a target for the stored forward
    direction. Side A is additionally a target only when bidirectional loading
    is enabled, the row is not ``forward_only``, and its language is not listed
    as source-only in the manifest. Runtime-added BOS/EOS/language control tokens
    are intentionally outside this content-piece audit.
    """

    if not split or split in {".", ".."} or Path(split).name != split:
        raise ValueError("split must be one directory name")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")
    if max_piece_examples < 0:
        raise ValueError("max_piece_examples must be non-negative")

    root = Path(dataset_root)
    manifest = _load_indexed_manifest(root)
    tokenizer_path = Path(tokenizer_model)
    tokenizer_identity = _indexed_tokenizer_identity(manifest, tokenizer_path)
    split_root = root / split
    index_paths = sorted(split_root.glob("*.idx.npy"))
    if not index_paths:
        raise FileNotFoundError(f"No index shards found under {split_root}")

    first_index = np.load(index_paths[0], mmap_mode="r", allow_pickle=False)
    first_fields = frozenset(first_index.dtype.names or ())
    modern = {"src_offset", "src_length", "tgt_offset", "tgt_length"}.issubset(first_fields)
    legacy = {"ko_offset", "ko_length", "ja_offset", "ja_length"}.issubset(first_fields)
    if modern == legacy:
        raise ValueError(
            f"Unsupported indexed shard layout in {index_paths[0]}: {first_index.dtype.descr!r}"
        )

    languages = _indexed_languages(manifest, modern=modern)
    language_to_id = {language: index for index, language in enumerate(languages)}
    raw_source_only = manifest.get("source_only_languages", [])
    if not isinstance(raw_source_only, list):
        raise ValueError("Indexed dataset source_only_languages must be a list")
    source_only = frozenset(str(value) for value in raw_source_only)
    unknown_source_only = sorted(source_only - set(languages))
    if unknown_source_only:
        raise ValueError(
            f"Indexed dataset manifest has unknown source-only languages: {unknown_source_only}"
        )
    source_only_ids = np.asarray(
        [language_to_id[language] for language in source_only],
        dtype=np.uint16,
    )

    if legacy:
        raw_pair = manifest.get("language_pair", ["ko", "ja"])
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError("Legacy indexed dataset requires a two-language language_pair")
        legacy_pair = (str(raw_pair[0]), str(raw_pair[1]))
        missing_pair_languages = sorted(set(legacy_pair) - set(languages))
        if missing_pair_languages:
            raise ValueError(
                "Legacy indexed language_pair is absent from language metadata: "
                f"{missing_pair_languages}"
            )
    else:
        legacy_pair = None

    tokenizer = SionTokenizer(tokenizer_path)
    vocab_size = len(tokenizer)
    physical_counts = [np.zeros(vocab_size, dtype=np.uint64) for _ in languages]
    target_counts = [np.zeros(vocab_size, dtype=np.uint64) for _ in languages]
    physical_sentences = np.zeros(len(languages), dtype=np.uint64)
    direction_totals: dict[str, Counter[str]] = {}
    physical_pairs = 0
    virtual_examples = 0
    forward_only_pairs = 0

    for index_path in index_paths:
        index = np.load(index_path, mmap_mode="r", allow_pickle=False)
        fields = frozenset(index.dtype.names or ())
        shard_modern = {"src_offset", "src_length", "tgt_offset", "tgt_length"}.issubset(fields)
        shard_legacy = {"ko_offset", "ko_length", "ja_offset", "ja_length"}.issubset(fields)
        if shard_modern != modern or shard_legacy != legacy:
            raise ValueError(f"Indexed shard layouts are inconsistent at {index_path}")

        row_count = len(index)
        physical_pairs += row_count
        if modern:
            required_metadata = {"src_language_id", "tgt_language_id"}
            if not required_metadata.issubset(fields):
                raise ValueError(f"Modern indexed shard lacks language ids: {index_path}")
            side_a_offsets = index["src_offset"]
            side_a_lengths = index["src_length"]
            side_b_offsets = index["tgt_offset"]
            side_b_lengths = index["tgt_length"]
            side_a_languages = index["src_language_id"].astype(np.uint16)
            side_b_languages = index["tgt_language_id"].astype(np.uint16)
            prefix = index_path.name.removesuffix(".idx.npy")
            side_a_path = split_root / f"{prefix}.src.bin"
            side_b_path = split_root / f"{prefix}.tgt.bin"
        else:
            assert legacy_pair is not None
            side_a_offsets = index["ko_offset"]
            side_a_lengths = index["ko_length"]
            side_b_offsets = index["ja_offset"]
            side_b_lengths = index["ja_length"]
            side_a_languages = np.full(
                row_count,
                language_to_id[legacy_pair[0]],
                dtype=np.uint16,
            )
            side_b_languages = np.full(
                row_count,
                language_to_id[legacy_pair[1]],
                dtype=np.uint16,
            )
            prefix = index_path.name.removesuffix(".idx.npy")
            side_a_path = split_root / f"{prefix}.{legacy_pair[0]}.bin"
            side_b_path = split_root / f"{prefix}.{legacy_pair[1]}.bin"

        if row_count:
            maximum_language_id = max(
                int(side_a_languages.max(initial=0)),
                int(side_b_languages.max(initial=0)),
            )
            if maximum_language_id >= len(languages):
                raise ValueError(
                    f"Indexed language id {maximum_language_id} exceeds manifest metadata at "
                    f"{index_path}"
                )
        forward_only = (
            index["forward_only"].astype(np.bool_)
            if "forward_only" in fields
            else np.zeros(row_count, dtype=np.bool_)
        )
        forward_only_pairs += int(np.count_nonzero(forward_only))
        side_a_source_only = np.isin(side_a_languages, source_only_ids)
        side_b_source_only = np.isin(side_b_languages, source_only_ids)
        forward_enabled = ~side_b_source_only
        reverse_enabled = (
            np.asarray(bidirectional & ~forward_only & ~side_a_source_only, dtype=np.bool_)
            if row_count
            else np.empty(0, dtype=np.bool_)
        )

        side_a_store = _open_indexed_token_store(
            side_a_path,
            side_a_offsets,
            side_a_lengths,
        )
        side_b_store = _open_indexed_token_store(
            side_b_path,
            side_b_offsets,
            side_b_lengths,
        )
        _accumulate_indexed_side(
            side_a_store,
            side_a_lengths,
            side_a_languages,
            reverse_enabled,
            physical_counts,
            target_counts,
            vocab_size=vocab_size,
        )
        _accumulate_indexed_side(
            side_b_store,
            side_b_lengths,
            side_b_languages,
            forward_enabled,
            physical_counts,
            target_counts,
            vocab_size=vocab_size,
        )

        physical_sentences += np.bincount(
            np.concatenate((side_a_languages, side_b_languages)).astype(np.int64),
            minlength=len(languages),
        ).astype(np.uint64, copy=False)
        virtual_examples += _add_direction_totals(
            direction_totals,
            side_a_languages,
            side_b_languages,
            side_a_lengths,
            side_b_lengths,
            forward_enabled,
            languages,
        )
        virtual_examples += _add_direction_totals(
            direction_totals,
            side_b_languages,
            side_a_languages,
            side_b_lengths,
            side_a_lengths,
            reverse_enabled,
            languages,
        )

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
    for language_id in range(len(languages)):
        global_physical += physical_counts[language_id]
        global_target += target_counts[language_id]

    language_reports: dict[str, Any] = {}
    for language_id, language in enumerate(languages):
        physical = physical_counts[language_id]
        target = target_counts[language_id]
        physical_tokens = int(physical.sum(dtype=np.uint64))
        target_enabled = language not in source_only
        target_eligible = (
            eligible & (physical > 0) if target_enabled else np.zeros(vocab_size, dtype=np.bool_)
        )
        target_summary = _frequency_summary(target, target_eligible)
        target_summary["rare_threshold"] = rare_threshold
        target_summary["rare_observed_pieces"] = int(
            np.count_nonzero(target_eligible & (target > 0) & (target < rare_threshold))
        )
        byte_tokens = int(physical[byte].sum(dtype=np.uint64))
        language_reports[language] = {
            "physical_sentences": int(physical_sentences[language_id]),
            "physical_tokens": physical_tokens,
            "byte_fallback_tokens": byte_tokens,
            "unknown_tokens": int(physical[tokenizer.unk_id]),
            "target_enabled": target_enabled,
            "target_tokens": int(target.sum(dtype=np.uint64)),
            "byte_fallback_rate": round(byte_tokens / max(physical_tokens, 1), 8),
            "target_frequency": target_summary,
            "lowest_target_exposure": _rare_piece_examples(
                tokenizer,
                target,
                target_eligible,
                maximum=max_piece_examples,
                include_unused=True,
            ),
        }

    direction_report = {
        direction: {
            **{name: int(value) for name, value in totals.items()},
            "mean_target_tokens": round(
                totals["target_tokens"] / max(totals["examples"], 1),
                6,
            ),
        }
        for direction, totals in sorted(direction_totals.items())
    }
    global_summary = _frequency_summary(global_target, eligible)
    global_summary["all_target_tokens"] = int(global_target.sum(dtype=np.uint64))
    global_summary["corpus_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_physical > 0))
    )
    global_summary["rare_threshold"] = rare_threshold
    global_summary["rare_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_target > 0) & (global_target < rare_threshold))
    )
    return {
        "schema": "sion-indexed-token-exposure-audit-v1",
        "complete_scan": True,
        "count_basis": "stored_target_content_tokens",
        "runtime_control_tokens_included": False,
        "parameters": {
            "dataset_root": str(root.resolve()),
            "dataset_format": str(manifest.get("format", "unknown")),
            "tokenizer_model": str(tokenizer_path.resolve()),
            "tokenizer_identity": tokenizer_identity,
            "split": split,
            "source_only_languages": sorted(source_only),
            "bidirectional": bidirectional,
            "rare_threshold": rare_threshold,
        },
        "vocab_size": vocab_size,
        "physical_pairs": physical_pairs,
        "forward_only_pairs": forward_only_pairs,
        "virtual_translation_examples": virtual_examples,
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


__all__ = ["audit_indexed_token_exposure", "audit_token_exposure"]
