# SentencePiece metadata is returned as an untyped JSON mapping.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import glob
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, cast

import numpy as np
import sentencepiece as spm

from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
    normalize_translation_directions,
)
from sion_translate.data.monolingual import (
    MonolingualDiscovery,
    monolingual_budgets,
    sample_monolingual_sentences,
)
from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.data.record_metadata import resolve_record_training_direction
from sion_translate.fingerprint import file_sha256
from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_tag,
    canonicalize_language_tags,
)
from sion_translate.performance import bounded_ordered_map, build_cpu_plan
from sion_translate.splitting import (
    TargetSplitGuard,
    choose_split_for_key,
    endpoint_split_digest,
    endpoint_split_key,
)
from sion_translate.synthetic import (
    DEFAULT_SYNTHETIC_PREFIXES,
    normalize_synthetic_prefixes,
    synthetic_path,
    synthetic_record,
)


LEGACY_LANGUAGE_PAIR = ("ko", "ja")
TOKENIZER_METADATA_FILENAME = "tokenizer_metadata.json"
TOKENIZER_METADATA_VERSION = 2
TOKENIZER_TRAINING_SCHEMA = "sion-tokenizer-training-v4"
# Input order affects bounded hashing samples and therefore the learned
# SentencePiece vocabulary. Keep the policy inside the authenticated contract so
# a future ordering change cannot silently reuse tokenizer bytes from an older
# traversal implementation.
TOKENIZER_INPUT_TRAVERSAL_POLICY = "portable-input-order-v1"
SENTENCEPIECE_MULTITHREADED_TRAINING_REGRESSION = "0.2.2"
SENTENCEPIECE_META_PIECE_COUNT = 4  # pad, unknown, beginning, and end
DEFAULT_TOKENIZER_INPUT_SENTENCE_SIZE = 1_000_000
DEFAULT_TOKENIZER_SAMPLING_ALPHA = 0.7
TOKENIZER_ARTIFACT_FILENAMES = (
    "sion.vocab",
    "token_features.npz",
    TOKENIZER_METADATA_FILENAME,
    # The model is published last and acts as the generation commit marker.
    "sion.model",
)
TOKENIZER_STAGING_PREFIX = ".sion-tokenizer-staging-"
TOKENIZER_PUBLISH_PREFIX = ".sion-tokenizer-publish-"

# Language-dependent control tokens: <2xx> requests translation into xx, and
# <denoise_xx> requests reconstruction of text in xx.
SHARED_CONTROL_SYMBOLS = [
    "<doc>",
    "<seg>",
    "<ctx>",
    "<style>",
    "<domain>",
    "<glossary>",
    "<protect>",
    "<mask>",
]
SLOT_SYMBOLS = [f"<slot_{index}>" for index in range(64)]

# These controls were introduced after the first tokenizer release. New
# tokenizers reserve them, but they intentionally stay outside the required
# compatibility list so older tokenizers can still translate.
#
# <draft> marks a first-pass translation that must be revised against source.
OPTIONAL_CONTROL_SYMBOLS = [
    "<draft>",
]
REASONING_TRACE_SYMBOLS = [
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]


def control_symbols(
    languages: Sequence[str],
    *,
    denoise_languages: Sequence[str] | None = None,
    reasoning_languages: Sequence[str] = (),
) -> list[str]:
    """Return every control symbol that must be reserved for the language graph."""

    unique_languages = canonicalize_language_tags(
        list(languages),
        field="translation control languages",
        reject_duplicates=False,
    )
    unique_denoise_languages = canonicalize_language_tags(
        list(unique_languages if denoise_languages is None else denoise_languages),
        field="denoising control languages",
        reject_duplicates=False,
    )
    unique_reasoning_languages = canonicalize_language_tags(
        list(reasoning_languages),
        field="reasoning control languages",
        reject_duplicates=False,
    )
    return (
        [f"<2{language}>" for language in unique_languages]
        + [f"<denoise_{language}>" for language in unique_denoise_languages]
        + [f"<reason_{language}>" for language in unique_reasoning_languages]
        + (REASONING_TRACE_SYMBOLS if unique_reasoning_languages else [])
        + SHARED_CONTROL_SYMBOLS
        + OPTIONAL_CONTROL_SYMBOLS
    )


# Compatibility alias used only when validating the historical two-language model.
BASE_CONTROL_SYMBOLS = control_symbols(LEGACY_LANGUAGE_PAIR)

SCRIPT_SPECIAL = 0
SCRIPT_HANGUL = 1
SCRIPT_HAN = 2
SCRIPT_HIRAGANA = 3
SCRIPT_KATAKANA = 4
SCRIPT_LATIN = 5
SCRIPT_DIGIT = 6
SCRIPT_MIXED = 7
SCRIPT_OTHER = 8


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def tokenizer_metadata_path(model_or_directory: str | Path) -> Path:
    """Resolve the sidecar metadata path from a model, directory, or sidecar path."""

    path = Path(model_or_directory)
    if path.name == TOKENIZER_METADATA_FILENAME:
        return path
    if path.suffix == ".model":
        return path.parent / TOKENIZER_METADATA_FILENAME
    return path / TOKENIZER_METADATA_FILENAME


def load_tokenizer_metadata(model_or_directory: str | Path) -> dict[str, object] | None:
    """Load tokenizer metadata when present, leaving legacy artifacts supported."""

    path = tokenizer_metadata_path(model_or_directory)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Tokenizer metadata must be a JSON object: {path}")
    return value


def tokenizer_split_digits_policy(model_or_directory: str | Path) -> bool | None:
    """Return the explicitly recorded digit policy, or ``None`` for legacy metadata."""

    metadata = load_tokenizer_metadata(model_or_directory)
    if metadata is None:
        return None
    version = metadata.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < TOKENIZER_METADATA_VERSION
    ):
        return None
    split_digits = metadata.get("split_digits")
    if not isinstance(split_digits, bool):
        raise ValueError("Tokenizer metadata split_digits must be a boolean")
    return split_digits


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_tokenizer_metadata(
    model_path: str | Path,
    *,
    split_digits: bool,
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]] | None = None,
    denoise_languages: Sequence[str] | None = None,
    reasoning_languages: Sequence[str] = (),
    monolingual_sentences: dict[str, int] | None = None,
    monolingual_sample_ratio: float = 0.0,
    required_characters: Sequence[str] = (),
    corpus_sentences: int | None = None,
    corpus_sentences_per_language: Mapping[str, int] | None = None,
    sampled_sentences: int | None = None,
    sampled_sentences_per_language: Mapping[str, int] | None = None,
    training_contract: Mapping[str, Any] | None = None,
) -> Path:
    """Write the reproducibility and identity contract for a trained tokenizer."""

    model_path = Path(model_path)
    vocab_path = model_path.with_suffix(".vocab")
    features_path = model_path.parent / "token_features.npz"
    normalized_pairs = normalize_language_pairs(language_pairs=language_pairs)
    normalized_directions = normalize_translation_directions(
        normalized_pairs,
        translation_directions,
    )
    translation_languages = languages_from_pairs(normalized_pairs)
    normalized_denoise_languages = list(
        canonicalize_language_tags(
            list(translation_languages if denoise_languages is None else denoise_languages),
            field="tokenizer metadata denoise_languages",
            reject_duplicates=False,
        )
    )
    normalized_reasoning_languages = list(
        canonicalize_language_tags(
            list(reasoning_languages),
            field="tokenizer metadata reasoning_languages",
            reject_duplicates=False,
        )
    )
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    metadata = {
        "version": TOKENIZER_METADATA_VERSION,
        "split_digits": bool(split_digits),
        "language_pair": list(normalized_pairs[0]),
        "language_pairs": [list(pair) for pair in normalized_pairs],
        "translation_directions": [list(direction) for direction in normalized_directions],
        "denoise_languages": normalized_denoise_languages,
        "reasoning_languages": normalized_reasoning_languages,
        "vocab_size": int(processor.vocab_size()),
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "vocab_file": vocab_path.name,
        "vocab_sha256": file_sha256(vocab_path),
        # Record the actual monolingual contribution so foundation training can
        # prove that its vocabulary saw those language domains.
        "monolingual_sample_ratio": float(monolingual_sample_ratio),
        "monolingual_sentences": dict(monolingual_sentences or {}),
        # The model format does not identify the trainer build. Keep it in the
        # v2 sidecar as an optional field so v1/v2 loading semantics stay intact.
        "sentencepiece_version": str(getattr(spm, "__version__", "unknown")),
    }
    if required_characters:
        rendered_required = "".join(required_characters)
        metadata["required_character_count"] = len(required_characters)
        metadata["required_characters_sha256"] = hashlib.sha256(
            rendered_required.encode("utf-8")
        ).hexdigest()
    if corpus_sentences is not None:
        metadata["corpus_sentences"] = corpus_sentences
        metadata["corpus_sentences_per_language"] = dict(corpus_sentences_per_language or {})
    if sampled_sentences is not None:
        metadata["sampled_sentences"] = sampled_sentences
        metadata["sampled_sentences_per_language"] = dict(sampled_sentences_per_language or {})
    if training_contract is not None:
        contract = dict(training_contract)
        metadata["training_contract"] = contract
        metadata["training_contract_sha256"] = _canonical_json_sha256(contract)
    if features_path.is_file():
        metadata["token_features_file"] = features_path.name
        metadata["token_features_size"] = features_path.stat().st_size
        metadata["token_features_sha256"] = file_sha256(features_path)
    output_path = tokenizer_metadata_path(model_path)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _to_python_string(value: object, *, value_name: str) -> str:
    """Convert string-like scalars to the built-in ``str`` SentencePiece expects."""
    if isinstance(value, str):
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise TypeError(f"{value_name} contains bytes that are not valid UTF-8") from error

    # numpy.str_ and similar scalar wrappers expose their built-in value via item().
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar_value = item_method()
        except (TypeError, ValueError):
            scalar_value = None

        if isinstance(scalar_value, str):
            return str(scalar_value)

        if isinstance(scalar_value, bytes):
            try:
                return scalar_value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise TypeError(
                    f"{value_name} contains scalar bytes that are not valid UTF-8"
                ) from error

    raise TypeError(f"{value_name} must be text; got type={type(value).__name__}, value={value!r}")


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if any(char in pattern for char in "*?[]"):
            paths.update(Path(match) for match in glob.glob(pattern))
        elif candidate.is_dir():
            paths.update(candidate.glob("*.jsonl"))
        else:
            paths.add(candidate)
    resolved = {path.resolve() for path in paths if path.exists()}
    if not resolved:
        return []
    try:
        common_root: Path | None = Path(os.path.commonpath([str(path.parent) for path in resolved]))
    except ValueError:
        # Inputs on different Windows drives have no common root. Their absolute
        # identities are already embedded in the contract, so using them here is
        # deterministic and cannot make an incompatible build look reusable.
        common_root = None

    def portable_key(path: Path) -> tuple[str, str]:
        identity = (
            path.relative_to(common_root).as_posix() if common_root is not None else path.as_posix()
        )
        # Windows path comparisons ignore case while POSIX comparisons do not.
        # An explicit folded key plus an original-spelling tie breaker defines
        # the same order on every supported host, including case-distinct files.
        return identity.casefold(), identity

    return sorted(resolved, key=portable_key)


def _filter_text_batch(
    batch: tuple[
        list[bytes],
        tuple[tuple[str, str], ...],
        float,
        float,
        bool,
        frozenset[tuple[str, str]],
        bool,
    ],
) -> list[tuple[str, str, str, str, str, bytes, bytes]]:
    (
        lines,
        language_pairs,
        validation_fraction,
        test_fraction,
        approximate_split,
        translation_directions,
        source_is_synthetic,
    ) = batch
    policy = QualityPolicy()
    accepted: list[tuple[str, str, str, str, str, bytes, bytes]] = []
    for raw_line in lines:
        try:
            row = json.loads(raw_line.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        record_group_key = hashlib.sha256(raw_line.strip()).hexdigest()
        expansion = expand_parallel_record(row, language_pairs)
        for pair in expansion.pairs:
            text_a, text_b = canonical_text(pair.text_a), canonical_text(pair.text_b)
            language_a, language_b = pair.language_a, pair.language_b
            languages = (language_a, language_b)
            if not assess_pair(text_a, text_b, policy, languages=languages).accepted:
                continue
            row_direction = resolve_record_training_direction(
                pair.metadata,
                (language_a, language_b),
                translation_directions,
            )
            storage_direction = row_direction or (
                (language_a, language_b)
                if (language_a, language_b) in translation_directions
                else (language_b, language_a)
            )
            if (language_a, language_b) != storage_direction:
                language_a, language_b = language_b, language_a
                text_a, text_b = text_b, text_a
            if source_is_synthetic or synthetic_record(row):
                split = "train"
            elif len(language_pairs) > 1:
                candidate_key = f"record\0{record_group_key}"
                split = choose_split_for_key(
                    candidate_key,
                    validation_fraction,
                    test_fraction,
                )
            else:
                candidate_key = endpoint_split_key(
                    language_a,
                    text_a,
                    approximate=approximate_split,
                )
                split = choose_split_for_key(
                    candidate_key,
                    validation_fraction,
                    test_fraction,
                )
            source_digest = endpoint_split_digest(
                language_a,
                text_a,
                approximate=approximate_split,
            )
            target_digest = endpoint_split_digest(
                language_b,
                text_b,
                approximate=approximate_split,
            )
            accepted.append(
                (language_a, text_a, language_b, text_b, split, source_digest, target_digest)
            )
    return accepted


def _raw_batches(paths: Sequence[Path], batch_size: int = 512):
    for path in paths:
        with path.open("rb") as handle:
            batch: list[bytes] = []
            for raw_line in handle:
                batch.append(raw_line)
                if len(batch) >= batch_size:
                    yield path, batch
                    batch = []
            if batch:
                yield path, batch


def iter_parallel_text(
    paths: Sequence[Path],
    *,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
) -> Iterator[str]:
    """Yield train-partition text without first materializing a temporary corpus."""

    for _, text in iter_parallel_text_with_languages(
        paths,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        language_pair=language_pair,
        language_pairs=language_pairs,
        translation_directions=translation_directions,
        approximate_split=approximate_split,
        source_only_languages=source_only_languages,
        train_only_prefixes=train_only_prefixes,
        num_workers=num_workers,
    ):
        yield text


def iter_parallel_text_with_languages(
    paths: Sequence[Path],
    *,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(language, text)`` pairs from the accepted training partition.

    Language labels make later stratified limits possible. ``iter_parallel_text``
    wraps this function so labeled and unlabeled paths cannot drift apart.
    """

    policy = QualityPolicy()
    policy.validate()
    estimated_pairs = max(1, sum(path.stat().st_size for path in paths) // 200)
    target_split_guard = TargetSplitGuard(estimated_pairs, validation_fraction, test_fraction)
    workers = num_workers or build_cpu_plan(input_files=len(paths)).preprocess_workers
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    normalized_directions = normalize_translation_directions(
        normalized_pairs,
        translation_directions,
        source_only_languages=source_only_languages,
    )
    languages = frozenset(languages_from_pairs(normalized_pairs))
    source_only = frozenset(
        canonicalize_language_tags(
            list(source_only_languages),
            field="source_only_languages",
            reject_duplicates=False,
        )
    )
    unknown_source_only = sorted(source_only - languages)
    if unknown_source_only:
        raise ValueError(
            "source_only_languages must appear in the configured language pairs; "
            f"{unknown_source_only} do not"
        )
    if any(set(pair) <= source_only for pair in normalized_pairs):
        raise ValueError("both sides of a language pair cannot be source-only")
    synthetic_prefixes = normalize_synthetic_prefixes(train_only_prefixes)
    inputs = (
        (
            batch,
            normalized_pairs,
            validation_fraction,
            test_fraction,
            approximate_split,
            frozenset(normalized_directions),
            synthetic_path(path, synthetic_prefixes),
        )
        for path, batch in _raw_batches(paths)
    )
    if workers <= 1:
        results = map(_filter_text_batch, inputs)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = bounded_ordered_map(executor, _filter_text_batch, inputs, max_pending=workers * 2)
    try:
        for candidates in results:
            for (
                language_a,
                text_a,
                language_b,
                text_b,
                split,
                source_digest,
                target_digest,
            ) in candidates:
                if not target_split_guard.accept_many(split, (source_digest, target_digest)):
                    continue
                if split != "train":
                    continue
                yield language_a, text_a
                yield language_b, text_b
    finally:
        if executor is not None:
            executor.shutdown()


class SionTokenizer:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)
        self.pad_id = self.processor.pad_id()
        self.unk_id = self.processor.unk_id()
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()

        required_ids = {
            "pad": self.pad_id,
            "unk": self.unk_id,
            "bos": self.bos_id,
            "eos": self.eos_id,
        }
        missing_ids = [name for name, token_id in required_ids.items() if token_id < 0]
        if missing_ids:
            raise ValueError(f"Tokenizer is missing required IDs: {missing_ids}")

        # Discover the language graph from reserved <2xx> pieces. The same code
        # therefore works for any configured pair without a separate language map.
        self.language_tags: dict[str, int] = {}  # {"ja": <2ja> token ID, ...}
        self.denoise_tags: dict[str, int] = {}  # {"ko": <denoise_ko> token ID, ...}
        self.reasoning_tags: dict[str, int] = {}  # {"en": <reason_en> token ID, ...}
        lang_pattern = re.compile(r"^<2([^<>\s]+)>$")
        denoise_pattern = re.compile(r"^<denoise_([^<>\s]+)>$")
        reasoning_pattern = re.compile(r"^<reason_([^<>\s]+)>$")
        byte_pattern = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
        # The reserved region ends at the first byte-fallback piece. SentencePiece
        # orders meta pieces, user symbols, byte pieces, then learned pieces, so
        # stopping there sees every control without misclassifying learned text.
        #
        # A fixed scan limit would silently lose later languages as this reserved
        # region grows; scanning to the byte boundary avoids that failure mode.
        for token_id in range(self.processor.vocab_size()):
            piece = self.processor.id_to_piece(token_id)
            if byte_pattern.match(piece):
                break
            destination: dict[str, int] | None = None
            kind = ""
            match = lang_pattern.fullmatch(piece)
            if match is not None:
                destination = self.language_tags
                kind = "translation"
            else:
                match = denoise_pattern.fullmatch(piece)
                if match is not None:
                    destination = self.denoise_tags
                    kind = "denoising"
                else:
                    match = reasoning_pattern.fullmatch(piece)
                    if match is not None:
                        destination = self.reasoning_tags
                        kind = "reasoning"
            if destination is None or match is None:
                continue
            raw_language = match.group(1)
            try:
                language = canonicalize_language_tag(
                    raw_language,
                    field=f"tokenizer {kind} control symbol",
                )
            except LanguageTagError as error:
                raise ValueError(f"invalid tokenizer control symbol {piece!r}: {error}") from error
            if language != raw_language:
                raise ValueError(
                    f"non-canonical tokenizer control symbol {piece!r}; "
                    f"expected language identity {language!r}"
                )
            if language in destination:
                raise ValueError(f"duplicate tokenizer {kind} control language {language!r}")
            destination[language] = token_id
        if len(self.language_tags) < 2 or not set(self.language_tags).issubset(self.denoise_tags):
            raise ValueError(
                "Tokenizer must reserve at least two <2xx> tags and a matching "
                "<denoise_xx> tag for every translation language; found "
                f"{sorted(self.language_tags)} / {sorted(self.denoise_tags)}"
            )
        self.languages = tuple(sorted(self.language_tags))
        self.denoise_languages = tuple(sorted(self.denoise_tags))
        self.reasoning_languages = tuple(sorted(self.reasoning_tags))
        unknown_reasoning_languages = sorted(set(self.reasoning_tags) - set(self.denoise_tags))
        if unknown_reasoning_languages:
            raise ValueError(
                "Tokenizer reasoning tags require matching denoise languages; found "
                f"reasoning-only languages {unknown_reasoning_languages}"
            )

        required_symbols = SHARED_CONTROL_SYMBOLS + SLOT_SYMBOLS
        symbol_ids = {
            symbol: int(self.processor.piece_to_id(symbol)) for symbol in required_symbols
        }
        missing_symbols = [
            symbol
            for symbol, token_id in symbol_ids.items()
            if token_id < 0 or self.processor.id_to_piece(token_id) != symbol
        ]
        if missing_symbols:
            raise ValueError(f"Tokenizer is missing required symbols: {missing_symbols}")

        self.mask_id = symbol_ids["<mask>"]
        self.slot_ids = [symbol_ids[symbol] for symbol in SLOT_SYMBOLS]

        # Optional controls remain absent on legacy tokenizers without rejecting them.
        self.optional_ids: dict[str, int] = {}
        for symbol in OPTIONAL_CONTROL_SYMBOLS:
            token_id = int(self.processor.piece_to_id(symbol))
            if token_id >= 0 and self.processor.id_to_piece(token_id) == symbol:
                self.optional_ids[symbol] = token_id
        self.draft_id: int | None = self.optional_ids.get("<draft>")
        self.reasoning_trace_ids: dict[str, int] = {}
        for symbol in REASONING_TRACE_SYMBOLS:
            token_id = int(self.processor.piece_to_id(symbol))
            if token_id >= 0 and self.processor.id_to_piece(token_id) == symbol:
                self.reasoning_trace_ids[symbol] = token_id
        if self.reasoning_tags and len(self.reasoning_trace_ids) != len(REASONING_TRACE_SYMBOLS):
            missing = sorted(set(REASONING_TRACE_SYMBOLS) - set(self.reasoning_trace_ids))
            raise ValueError(
                f"Tokenizer reasoning task tags require every trace marker; missing {missing}"
            )
        # Compatibility aliases exist only for the historical two-language tokenizer.
        if {"ko", "ja"} == set(self.language_tags):
            self.ko_to_ja_id = self.language_tags["ja"]
            self.ja_to_ko_id = self.language_tags["ko"]
            self.denoise_ko_id = self.denoise_tags["ko"]
            self.denoise_ja_id = self.denoise_tags["ja"]

    def __len__(self) -> int:
        return self.processor.vocab_size()

    @property
    def splits_digits(self) -> bool:
        """Return whether the tokenizer keeps every digit as one piece.

        SentencePiece does not preserve this trainer flag in an accessible model
        field, so the check encodes a multi-digit probe. Merged number pieces make
        value substitutions in amounts and measurements more likely.
        """
        pieces = self.processor.encode("38720", out_type=str)
        return all(len(piece.replace("▁", "")) <= 1 for piece in pieces)

    def piece_id(self, piece: str) -> int:
        return int(self.processor.piece_to_id(piece))

    def encode(self, text: str) -> list[int]:
        """Normalize text and convert it to SentencePiece token IDs."""
        source_text = _to_python_string(
            text,
            value_name="tokenizer.encode() input",
        )

        normalized_text = normalize_text(source_text)
        normalized_text = _to_python_string(
            normalized_text,
            value_name="normalize_text() return value",
        )

        return list(
            self.processor.encode(
                normalized_text,
                out_type=int,
            )
        )

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode([int(token_id) for token_id in ids])


@dataclass(frozen=True)
class TokenizerSentence:
    """One normalized tokenizer sentence and its sampling stratum."""

    language: str
    text: str
    monolingual: bool = False


def iter_tokenizer_records(
    paths: Sequence[Path],
    *,
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]] | None = None,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
    monolingual_counts: dict[str, int] | None = None,
) -> Iterator[TokenizerSentence]:
    """Yield parallel data followed by bounded monolingual vocabulary data.

    Monolingual examples expose vocabulary needed by the foundation stage. Their
    per-language budgets are derived from parallel counts, preventing a large
    monolingual source from taking over the joint vocabulary. Parallel examples
    must therefore be consumed first.
    """

    parallel_counts: Counter[str] = Counter()
    for language, text in iter_parallel_text_with_languages(
        paths,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        language_pairs=language_pairs,
        translation_directions=translation_directions,
        approximate_split=approximate_split,
        source_only_languages=source_only_languages,
        train_only_prefixes=train_only_prefixes,
        num_workers=num_workers,
    ):
        parallel_counts[language] += 1
        yield TokenizerSentence(language=language, text=text)

    if monolingual is None or not monolingual.sources or monolingual_sample_ratio <= 0:
        return
    budgets = monolingual_budgets(
        dict(parallel_counts),
        monolingual.languages,
        ratio=monolingual_sample_ratio,
    )
    for language in monolingual.languages:
        emitted = 0
        for text in sample_monolingual_sentences(
            monolingual.paths_for(language),
            budgets.get(language, 0),
        ):
            emitted += 1
            yield TokenizerSentence(
                language=language,
                text=canonical_text(text),
                monolingual=True,
            )
        if monolingual_counts is not None:
            monolingual_counts[language] = emitted


def iter_tokenizer_sentences(
    paths: Sequence[Path],
    *,
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]] | None = None,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
    monolingual_counts: dict[str, int] | None = None,
) -> Iterator[str]:
    """Yield all tokenizer sentences while keeping record metadata internal."""

    for record in iter_tokenizer_records(
        paths,
        monolingual=monolingual,
        monolingual_sample_ratio=monolingual_sample_ratio,
        language_pairs=language_pairs,
        translation_directions=translation_directions,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        approximate_split=approximate_split,
        source_only_languages=source_only_languages,
        train_only_prefixes=train_only_prefixes,
        num_workers=num_workers,
        monolingual_counts=monolingual_counts,
    ):
        yield record.text


@dataclass(frozen=True)
class CorpusCounts:
    """Character and language counts produced by one exhaustive corpus pass.

    Character frequencies drive ``required_chars`` while language counts drive
    deterministic sampling. Computing both together prevents pass-to-pass drift.
    """

    characters: Counter[str]
    sentences: int
    sentences_per_language: Counter[str] = field(default_factory=Counter)
    monolingual_sentences_per_language: Counter[str] = field(default_factory=Counter)

    @property
    def character_total(self) -> int:
        return sum(self.characters.values())


def stratified_sentence_quotas(
    counts: Mapping[str, int],
    sentence_limit: int,
    *,
    alpha: float = DEFAULT_TOKENIZER_SAMPLING_ALPHA,
) -> dict[str, int]:
    """Allocate an exact bounded sample across arbitrary language strata.

    ``alpha=1`` follows corpus size. Lower values reserve relatively more space
    for low-resource languages. Every non-empty language receives one sentence
    before the remaining capacity is distributed with deterministic largest
    remainders. No language names or scripts are embedded in this policy.
    """

    if type(sentence_limit) is not int:
        raise TypeError("sentence_limit must be an integer")
    if sentence_limit < 0:
        raise ValueError("sentence_limit must be non-negative")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("sampling alpha must be finite and in (0, 1]")
    normalized: dict[str, int] = {}
    for language, raw_count in counts.items():
        if type(raw_count) is not int:
            raise TypeError(f"sentence count for {language!r} must be an integer")
        if raw_count < 0:
            raise ValueError(f"sentence count for {language!r} must be non-negative")
        if raw_count:
            normalized[str(language)] = raw_count
    total = sum(normalized.values())
    if sentence_limit == 0 or sentence_limit >= total:
        return dict(sorted(normalized.items()))
    if sentence_limit < len(normalized):
        raise ValueError(
            "input_sentence_size must be at least the number of non-empty languages "
            f"({len(normalized)}) so no language disappears from tokenizer training"
        )

    quotas = {language: 1 for language in normalized}
    capacities = {language: normalized[language] - 1 for language in normalized}
    remaining = sentence_limit - len(quotas)
    while remaining:
        active = [language for language in sorted(normalized) if capacities[language] > 0]
        if not active:
            break
        weights = {language: math.pow(float(normalized[language]), alpha) for language in active}
        weight_total = sum(weights.values())
        ideal = {language: remaining * weights[language] / weight_total for language in active}
        assigned = 0
        for language in active:
            addition = min(capacities[language], math.floor(ideal[language]))
            quotas[language] += addition
            capacities[language] -= addition
            assigned += addition
        remaining -= assigned
        if not remaining:
            break
        by_remainder = sorted(
            (language for language in active if capacities[language] > 0),
            key=lambda language: (-(ideal[language] - math.floor(ideal[language])), language),
        )
        if not by_remainder:
            continue
        for language in by_remainder[:remaining]:
            quotas[language] += 1
            capacities[language] -= 1
            remaining -= 1
            if not remaining:
                break
    if sum(quotas.values()) != sentence_limit:
        raise RuntimeError("stratified tokenizer quota allocation did not reach its exact limit")
    return dict(sorted(quotas.items()))


def iter_stratified_tokenizer_sentences(
    records: Iterable[TokenizerSentence],
    counts: Mapping[str, int],
    quotas: Mapping[str, int],
    *,
    seed: int = 0,
    sampled_counts: Counter[str] | None = None,
    sampled_monolingual_counts: Counter[str] | None = None,
) -> Iterator[str]:
    """Select exact systematic samples in one pass and constant memory.

    A language-specific hash rotates evenly spaced selection positions. The
    result is reproducible for a seed, covers the full ordered corpus rather than
    its prefix, and never stores sentence text in a Python reservoir.
    """

    positions: Counter[str] = Counter()
    offsets = {
        language: int.from_bytes(
            hashlib.blake2b(f"{seed}\0{language}".encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        % count
        for language, count in counts.items()
        if count > 0
    }
    emitted: Counter[str] = Counter()
    emitted_monolingual: Counter[str] = Counter()
    for record in records:
        language = record.language
        if language not in counts:
            raise RuntimeError(
                f"tokenizer corpus produced a language absent from its counting pass: {language}"
            )
        count = counts.get(language, 0)
        quota = quotas.get(language, 0)
        positions[language] += 1
        if count <= 0 or quota <= 0:
            continue
        position = positions[language] - 1
        rotated = (position + offsets[language]) % count
        selected = ((rotated + 1) * quota) // count > (rotated * quota) // count
        if not selected:
            continue
        emitted[language] += 1
        if record.monolingual:
            emitted_monolingual[language] += 1
        yield record.text
    expected = {language: quota for language, quota in quotas.items() if quota > 0}
    observed_counts = {language: positions[language] for language in counts}
    if observed_counts != dict(counts):
        raise RuntimeError(
            "tokenizer corpus changed between counting and sampling: "
            f"expected language counts={dict(counts)}, observed={observed_counts}"
        )
    actual = {language: emitted[language] for language in expected}
    if actual != expected:
        raise RuntimeError(
            "tokenizer corpus changed between counting and sampling: "
            f"expected={expected}, emitted={actual}"
        )
    if sampled_counts is not None:
        sampled_counts.update(emitted)
    if sampled_monolingual_counts is not None:
        sampled_monolingual_counts.update(emitted_monolingual)


# ─────────────────────────────────────────────────────────────────────────
# SentencePiece 0.2.2 can segfault during multithreaded normalization of a
# large corpus. Corpus size is not a reliable predictor of the failure.
#
# Measurements from 2026-08-07. Elements equal characters plus one boundary
# element per sentence in ``MakeSeedSentencePieces``:
#
#     sentences   chars   elements  longest word  result
#     20,459,141  0.93 G  0.95 G          16,886  pass
#     20,923,746  1.95 G  1.97 G               ?  pass
#     20,355,467  1.06 G  1.08 G    1,401   SIGSEGV
#     24,954,792  1.95 G  1.98 G        ?   SIGSEGV
#     41,438,233  3.93 G  3.97 G        ?   SIGSEGV
#
# The third corpus is smaller on every measured axis than the first, yet it
# fails. Ruled-out causes include memory (40-50 GiB passing peak versus 256 GiB
# provisioned for a failure), sentence/character/element counts, 4 versus 16
# threads, lines over 4 KiB, the 32,768-character word limit, and block elements.
#
# A source build points to ``PrefixMatcher::GlobalReplace`` through
# ``std::string::_M_append``/glibc malloc. The C++ CLI reproduces it without a
# Python iterator. Exact A/B results:
#
# * 0.2.2, 4 threads: SIGSEGV; 1 thread: pass
# * 0.2.1, 4 threads: pass
# * 0.2.2 with only the new thread pool bypassed: SIGSEGV
# * 0.2.2 with the normalization offset removed by ``de32a1e`` restored: pass
#
# The omitted normalization offset in 0.2.2 is therefore the measured cause.
# Reject that runtime before scanning because the crash arrives after input load.
# ─────────────────────────────────────────────────────────────────────────


def validate_sentencepiece_training_runtime(num_threads: int) -> str:
    """Return the trainer version, rejecting the measured 0.2.2 crash path."""

    version = str(getattr(spm, "__version__", "unknown"))
    if version == SENTENCEPIECE_MULTITHREADED_TRAINING_REGRESSION and num_threads > 1:
        raise RuntimeError(
            "sentencepiece 0.2.2 has a confirmed SIGSEGV regression in its "
            "multithreaded trainer (upstream de32a1e omitted normalization offsets). "
            "Install the pinned sentencepiece==0.2.1, or explicitly use "
            "num_threads=1 for the slower measured workaround."
        )
    return version


def corpus_character_counts(
    paths: Sequence[Path],
    *,
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]] | None = None,
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
) -> CorpusCounts:
    """Count every character, and every sentence, in the training partition."""

    counts: Counter[str] = Counter()
    sentences = 0
    sentences_per_language: Counter[str] = Counter()
    monolingual_per_language: Counter[str] = Counter()
    for record in iter_tokenizer_records(
        paths,
        monolingual=monolingual,
        monolingual_sample_ratio=monolingual_sample_ratio,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        language_pairs=language_pairs,
        translation_directions=translation_directions,
        approximate_split=approximate_split,
        source_only_languages=source_only_languages,
        train_only_prefixes=train_only_prefixes,
        num_workers=num_workers,
    ):
        counts.update(record.text)
        sentences += 1
        sentences_per_language[record.language] += 1
        if record.monolingual:
            monolingual_per_language[record.language] += 1
    return CorpusCounts(
        characters=counts,
        sentences=sentences,
        sentences_per_language=sentences_per_language,
        monolingual_sentences_per_language=monolingual_per_language,
    )


# Characters SentencePiece refuses to accept as required characters. Passing one
# asserts inside the trainer (``trainer_interface.cc``:
# ``[!port::ContainsKey(required_chars_, kUNKChar)]``) *after* it has read the
# whole corpus, which is the expensive half of the run.
#
# U+2047 is ``kUNKChar`` itself. U+2585 was found by bisecting a real failure and
# has no documented reason to be here -- neighbouring block characters are
# accepted, and it is not the U+2581 space symbol. That is the point of
# :func:`acceptable_required_characters`: this list is what we happen to know,
# not what SentencePiece happens to enforce, so the set is verified rather than
# trusted.
SENTENCEPIECE_RESERVED_CHARACTERS = frozenset(
    # U+2047 is kUNKChar itself.
    "⁇"
    # U+2580-U+259F, the Block Elements. SentencePiece treats these as reserved
    # and drops any *sentence* containing one, which it announces as
    # "Reserved chars are found. Skipped: ...". Requiring a character while
    # discarding every sentence that carries it is a contradiction: the
    # vocabulary must hold it, and nothing in the corpus can teach it. They are
    # ASCII-art residue in scraped text, never content worth a reserved slot.
    + "".join(chr(codepoint) for codepoint in range(0x2580, 0x25A0))
)


def required_characters_from_counts(
    counts: Counter[str],
    *,
    min_occurrences: int = 25,
) -> list[str]:
    """Characters frequent enough that byte fallback would be a regression.

    A character seen this often carries content, so splitting it into raw bytes
    costs the model three tokens where one would do. A mixed-script corpus produces
    produce NFC-fused syllables that appear over a thousand times
    yet are absent from any tokenizer trained before that corpus existed; the
    while an older shipped tokenizer renders one as three ``<0x..>`` pieces.

    The threshold is a floor on *content*, not a guess about any one language:
    below it a character is genuinely incidental and byte fallback is the right
    answer, which is what byte fallback is for. Nothing here names a script, so a
    new language pair gets the same protection without a code change.

    :data:`SENTENCEPIECE_RESERVED_CHARACTERS` is removed regardless of how often
    it occurs. Those characters are not ours to reserve, and leaving them in
    fails the training run rather than the vocabulary: SentencePiece asserts on
    them, and only after it has read the whole corpus.
    """

    if min_occurrences < 1:
        raise ValueError("min_occurrences must be positive")
    return sorted(
        character
        for character, count in counts.items()
        if count >= min_occurrences
        and not character.isspace()
        and character not in SENTENCEPIECE_RESERVED_CHARACTERS
    )


def _required_characters_rejected(characters: Sequence[str], probe_corpus: Path) -> bool:
    """Does SentencePiece assert on this set? One throwaway training run."""

    if not characters:
        return False
    with tempfile.TemporaryDirectory() as workspace:
        try:
            spm.SentencePieceTrainer.train(
                input=str(probe_corpus),
                model_prefix=str(Path(workspace) / "probe"),
                vocab_size=len(characters) + 512,
                model_type="unigram",
                character_coverage=0.9999,
                byte_fallback=True,
                normalization_rule_name="identity",
                required_chars="".join(characters),
                hard_vocab_limit=False,
                minloglevel=3,
            )
        except RuntimeError as error:
            if "kUNKChar" in str(error):
                return True
            raise
    return False


def acceptable_required_characters(
    required: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Ask SentencePiece which required characters it accepts.

    One rejected required character makes the trainer assert only after reading
    the full corpus. Instead of guessing undocumented rules, this function trains
    on a small synthetic corpus and bisects any failing character set. A few
    seconds of probes prevent wasting an hours-long production corpus scan.
    """

    candidate = list(required)
    rejected: list[str] = []
    with tempfile.TemporaryDirectory() as workspace:
        probe_corpus = Path(workspace) / "probe.txt"
        # Keep the probe language-neutral and free of required characters. It
        # measures only whether SentencePiece accepts the required set.
        probe_corpus.write_text(
            "".join(f"probe line {index}\n" for index in range(500)),
            encoding="utf-8",
        )

        while _required_characters_rejected(candidate, probe_corpus):
            low = candidate
            while len(low) > 1:
                middle = len(low) // 2
                left, right = low[:middle], low[middle:]
                if _required_characters_rejected(left, probe_corpus):
                    low = left
                elif _required_characters_rejected(right, probe_corpus):
                    low = right
                else:
                    # If neither half fails alone, the failure depends on a
                    # combination. Stop rather than silently dropping a guess.
                    raise RuntimeError(
                        "SentencePiece rejects this required-character set, but no "
                        f"single character explains it ({len(low)} remain). Inspect "
                        "them by hand rather than trusting an automatic drop."
                    )
            offender = low[0]
            rejected.append(offender)
            candidate = [character for character in candidate if character != offender]
    return candidate, rejected


def _source_identity_record(
    path: Path,
    *,
    role: str,
    identity: str,
    language: str | None = None,
) -> dict[str, object]:
    """Hash one regular source and reject mutation during the hash pass."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"tokenizer source must be a regular non-symlink file: {path}")
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"tokenizer source changed while it was hashed: {path}")
    record: dict[str, object] = {
        "role": role,
        "path": identity,
        "size": after.st_size,
        "sha256": digest,
    }
    if language is not None:
        record["language"] = language
    return record


def _tokenizer_source_records(
    paths: Sequence[Path],
    monolingual: MonolingualDiscovery | None,
) -> list[dict[str, object]]:
    resolved_parallel = [path.resolve() for path in paths]
    try:
        parallel_root = Path(os.path.commonpath([str(path.parent) for path in resolved_parallel]))
    except ValueError:
        parallel_root = None

    def parallel_identity(path: Path) -> str:
        resolved = path.resolve()
        if parallel_root is None:
            return resolved.as_posix()
        return resolved.relative_to(parallel_root).as_posix()

    # Preserve the exact traversal order. The list is authenticated as part of
    # the training contract, so any future stream-order change invalidates reuse
    # even when every source file still has the same bytes.
    records = [
        _source_identity_record(
            path,
            role="parallel",
            identity=parallel_identity(path),
        )
        for path in paths
    ]
    if monolingual is not None:
        records.extend(
            _source_identity_record(
                source.path,
                role="monolingual",
                identity=source.path.resolve().relative_to(monolingual.root.resolve()).as_posix(),
                language=source.language,
            )
            for source in monolingual.sources
        )
    return records


def _verify_expected_tokenizer_source_identities(
    paths: Sequence[Path],
    monolingual: MonolingualDiscovery | None,
    source_records: Sequence[Mapping[str, object]],
    expected: Mapping[str, tuple[int, str]],
) -> None:
    """Bind the initial tokenizer snapshot to an external immutable contract."""

    source_paths = [*paths]
    if monolingual is not None:
        source_paths.extend(source.path for source in monolingual.sources)
    if len(source_paths) != len(source_records):
        raise RuntimeError("tokenizer source snapshot has an inconsistent file count")

    actual: dict[str, tuple[int, str]] = {}
    for path, record in zip(source_paths, source_records, strict=True):
        identity = str(path.resolve())
        if identity in actual:
            raise RuntimeError(f"tokenizer source was selected more than once: {identity}")
        size = record.get("size")
        sha256 = record.get("sha256")
        if type(size) is not int or not isinstance(sha256, str):
            raise RuntimeError(f"tokenizer source snapshot is invalid: {identity}")
        actual[identity] = (size, sha256)

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        identity
        for identity in set(expected).intersection(actual)
        if expected[identity] != actual[identity]
    )
    if missing or extra or changed:
        raise RuntimeError(
            "tokenizer sources differ from the immutable preparation contract; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _tokenizer_training_contract(
    *,
    source_records: Sequence[Mapping[str, object]],
    vocab_size: int,
    input_sentence_size: int,
    sampling_alpha: float,
    sampling_seed: int,
    seed_sentencepiece_size: int,
    character_coverage: float,
    required_character_min_occurrences: int,
    validation_fraction: float,
    test_fraction: float,
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]],
    denoise_languages: Sequence[str],
    reasoning_languages: Sequence[str],
    approximate_split: bool,
    source_only_languages: Sequence[str],
    train_only_prefixes: Sequence[str],
    split_digits: bool,
    monolingual_sample_ratio: float,
    sentencepiece_version: str,
    num_threads: int,
) -> dict[str, object]:
    """Build the exact contract used to resume or reject an existing output."""

    return {
        "schema": TOKENIZER_TRAINING_SCHEMA,
        "input_traversal_policy": TOKENIZER_INPUT_TRAVERSAL_POLICY,
        "sources": [dict(record) for record in source_records],
        "vocab_size": vocab_size,
        "input_sentence_size": input_sentence_size,
        "sampling_alpha": sampling_alpha,
        "sampling_seed": sampling_seed,
        "seed_sentencepiece_size": seed_sentencepiece_size,
        "character_coverage": character_coverage,
        "required_character_min_occurrences": required_character_min_occurrences,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "language_pairs": [list(pair) for pair in language_pairs],
        "translation_directions": [list(direction) for direction in translation_directions],
        "denoise_languages": list(denoise_languages),
        "reasoning_languages": list(reasoning_languages),
        "approximate_split": approximate_split,
        "source_only_languages": list(source_only_languages),
        "train_only_prefixes": list(train_only_prefixes),
        "split_digits": split_digits,
        "monolingual_sample_ratio": monolingual_sample_ratio,
        "sentencepiece_version": sentencepiece_version,
        "num_threads": num_threads,
    }


def _assert_plain_tokenizer_file(path: Path, *, role: str) -> None:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction() or not path.is_file():
        raise RuntimeError(f"{role} must be a regular non-symlink file: {path}")


def _validate_tokenizer_generation(
    directory: Path,
    expected_contract: Mapping[str, object],
) -> Path:
    """Authenticate a complete tokenizer generation against its training contract."""

    paths = {name: directory / name for name in TOKENIZER_ARTIFACT_FILENAMES}
    for name, path in paths.items():
        _assert_plain_tokenizer_file(path, role=f"tokenizer artifact {name}")
    try:
        raw_metadata = json.loads(paths[TOKENIZER_METADATA_FILENAME].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("tokenizer metadata is unreadable") from error
    if not isinstance(raw_metadata, dict):
        raise RuntimeError("tokenizer metadata must be a JSON object")
    metadata = cast(dict[str, Any], raw_metadata)
    contract = metadata.get("training_contract")
    if contract != dict(expected_contract):
        raise RuntimeError(
            "existing tokenizer was built from a different source or training contract; "
            "use a new output directory instead of replacing its vocabulary"
        )
    if metadata.get("training_contract_sha256") != _canonical_json_sha256(expected_contract):
        raise RuntimeError("tokenizer training contract digest does not match its payload")
    expected_digests = {
        "sion.model": metadata.get("model_sha256"),
        "sion.vocab": metadata.get("vocab_sha256"),
        "token_features.npz": metadata.get("token_features_sha256"),
    }
    for name, expected_digest in expected_digests.items():
        if not isinstance(expected_digest, str) or file_sha256(paths[name]) != expected_digest:
            raise RuntimeError(f"tokenizer artifact identity differs for {name}")
    return paths["sion.model"]


def _remove_tokenizer_staging(path: Path, output_dir: Path) -> None:
    """Remove only a private staging directory created below this output root."""

    if path.parent != output_dir or not path.name.startswith(TOKENIZER_STAGING_PREFIX):
        raise RuntimeError(f"refusing to remove an unexpected tokenizer staging path: {path}")
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction() or not path.is_dir():
        raise RuntimeError(f"refusing to remove an unsafe tokenizer staging path: {path}")
    pending = [path]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            entry_is_junction = getattr(entry, "is_junction", lambda: False)
            if entry.is_symlink() or entry_is_junction():
                raise RuntimeError(
                    f"refusing to remove tokenizer staging containing a reparse point: {entry}"
                )
            if entry.is_dir():
                pending.append(entry)
            elif not entry.is_file():
                raise RuntimeError(
                    f"refusing to remove tokenizer staging containing a special file: {entry}"
                )
    shutil.rmtree(path)


def _publish_tokenizer_generation(staging: Path, output_dir: Path) -> Path:
    """Publish copies of sidecars first and link the model commit marker last.

    Keeping the authenticated originals in ``staging`` makes an ordinary I/O
    failure resumable. A later run can validate and publish the same completed
    build instead of repeating SentencePiece training after disk space is freed.
    """

    model_path = output_dir / "sion.model"
    if model_path.exists() or model_path.is_symlink():
        raise FileExistsError(f"refusing to replace an existing tokenizer model: {model_path}")
    for name in TOKENIZER_ARTIFACT_FILENAMES:
        source = staging / name
        _assert_plain_tokenizer_file(source, role=f"staged tokenizer artifact {name}")
        # Windows maps ``fsync`` to FlushFileBuffers, which requires a handle
        # opened for writing even though the staged payload is already complete.
        with source.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    for name in TOKENIZER_ARTIFACT_FILENAMES:
        source = staging / name
        destination = output_dir / name
        if name == "sion.model":
            # A hard link is an atomic no-replace operation on the same volume.
            # The staging directory is below output_dir, so this also keeps the
            # completed source available if another process wins the race.
            try:
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to replace an existing tokenizer model: {destination}"
                ) from error
            continue

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=TOKENIZER_PUBLISH_PREFIX,
            dir=staging,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return model_path


def _recover_tokenizer_staging(
    output_dir: Path,
    expected_contract: Mapping[str, object],
) -> Path | None:
    """Publish one completed interrupted build and discard incomplete private builds."""

    recovered: Path | None = None
    for candidate in sorted(output_dir.glob(f"{TOKENIZER_STAGING_PREFIX}*")):
        candidate_is_junction = getattr(candidate, "is_junction", lambda: False)
        if candidate.is_symlink() or candidate_is_junction() or not candidate.is_dir():
            raise RuntimeError(f"unsafe tokenizer staging entry blocks recovery: {candidate}")
        try:
            _validate_tokenizer_generation(candidate, expected_contract)
        except RuntimeError:
            _remove_tokenizer_staging(candidate, output_dir)
            continue
        if recovered is not None:
            _remove_tokenizer_staging(candidate, output_dir)
            continue
        recovered = _publish_tokenizer_generation(candidate, output_dir)
        _remove_tokenizer_staging(candidate, output_dir)
    return recovered


def train_tokenizer(
    input_patterns: Sequence[str],
    output_dir: str | Path,
    *,
    vocab_size: int = 48000,
    # A one-million-sentence language-stratified sample remains practical on a
    # 16 GiB workstation. Set this to 0 only when an explicitly provisioned host
    # should expose every accepted sentence to SentencePiece.
    input_sentence_size: int = DEFAULT_TOKENIZER_INPUT_SENTENCE_SIZE,
    sampling_alpha: float = DEFAULT_TOKENIZER_SAMPLING_ALPHA,
    sampling_seed: int = 0,
    seed_sentencepiece_size: int = 1_000_000,
    # Not 1.0. Full coverage puts *every* character observed in the corpus into
    # the vocabulary, which makes three separate mechanisms fight over the same
    # job: `required_chars` becomes a subset of what coverage already admits,
    # byte fallback's 256 pieces can never fire, and the preflight that gates GPU
    # time on a byte-fallback rate becomes impossible to fail. Measured on the
    # 8.98M-record corpus: 10,760 distinct characters at coverage 1.0, of which
    # 4,275 occur fewer than 25 times in 18M sentences. See the module tests for
    # the division of labour this value restores.
    character_coverage: float = 0.9999,
    required_character_min_occurrences: int = 25,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
    num_threads: int | None = None,
    split_digits: bool = True,
    # Optional monolingual data exposes vocabulary needed by foundation
    # pretraining. monolingual_sample_ratio controls its per-language budget.
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
    foundation_languages: Sequence[str] = (),
    reasoning_languages: Sequence[str] = (),
    expected_source_identities: Mapping[str, tuple[int, str]] | None = None,
) -> Path:
    """Train a joint SentencePiece tokenizer transactionally.

    The full corpus is scanned for character coverage and per-language counts.
    SentencePiece then receives an exact, deterministic, language-stratified
    sample bounded by ``input_sentence_size``. This preserves low-resource
    languages without retaining a Python reservoir of sentence strings.

    ``character_coverage`` trims the global frequency tail while
    ``required_character_min_occurrences`` pulls meaningful characters back out
    of that tail. ``split_digits`` keeps numeric values compositional so the
    translation model can preserve amounts, dates, and measurements.

    Artifacts are built in a private directory. Sidecars are published first and
    ``sion.model`` last, so an interrupted run is either resumable or visibly
    incomplete and never appears as a complete tokenizer generation.
    """
    paths = expand_inputs(input_patterns)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {input_patterns}")
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    if type(input_sentence_size) is not int:
        raise TypeError("input_sentence_size must be an integer")
    if input_sentence_size < 0:
        raise ValueError("input_sentence_size must be non-negative")
    if type(sampling_seed) is not int:
        raise TypeError("sampling_seed must be an integer")
    if isinstance(sampling_alpha, bool) or not math.isfinite(sampling_alpha):
        raise ValueError("sampling_alpha must be finite and in (0, 1]")
    if not 0.0 < sampling_alpha <= 1.0:
        raise ValueError("sampling_alpha must be finite and in (0, 1]")
    if type(seed_sentencepiece_size) is not int or seed_sentencepiece_size < 1:
        raise ValueError("seed_sentencepiece_size must be a positive integer")
    if (
        type(required_character_min_occurrences) is not int
        or required_character_min_occurrences < 0
    ):
        raise ValueError("required_character_min_occurrences must be a non-negative integer")
    if isinstance(character_coverage, bool) or not math.isfinite(character_coverage):
        raise ValueError("character_coverage must be in (0, 1]")
    if (
        isinstance(monolingual_sample_ratio, bool)
        or not math.isfinite(monolingual_sample_ratio)
        or monolingual_sample_ratio < 0
    ):
        raise ValueError("monolingual_sample_ratio must be finite and non-negative")
    if num_workers is not None and (type(num_workers) is not int or num_workers < 1):
        raise ValueError("num_workers must be a positive integer when provided")
    if num_threads is not None and (type(num_threads) is not int or num_threads < 1):
        raise ValueError("num_threads must be a positive integer when provided")
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError("character_coverage must be in (0, 1]")
    if character_coverage >= 1.0 and required_character_min_occurrences > 0:
        raise ValueError(
            "character_coverage=1.0 admits every character in the corpus, which makes "
            "required_chars redundant and byte fallback unreachable. Lower "
            "character_coverage (0.9999 is the default) or set "
            "required_character_min_occurrences=0 to opt out of the frequency floor."
        )

    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    normalized_directions = normalize_translation_directions(
        normalized_pairs,
        translation_directions,
        source_only_languages=source_only_languages,
    )
    languages = languages_from_pairs(normalized_pairs)
    denoise_languages = canonicalize_language_tags(
        [*languages, *foundation_languages],
        field="tokenizer denoise_languages",
        reject_duplicates=False,
    )
    reasoning_languages = canonicalize_language_tags(
        list(reasoning_languages),
        field="tokenizer reasoning_languages",
        reject_duplicates=False,
    )
    unknown_reasoning_languages = sorted(set(reasoning_languages) - set(denoise_languages))
    if unknown_reasoning_languages:
        raise ValueError(
            "reasoning_languages must also be foundation/translation languages; "
            f"unknown {unknown_reasoning_languages}"
        )

    symbols = (
        control_symbols(
            languages,
            denoise_languages=denoise_languages,
            reasoning_languages=reasoning_languages,
        )
        + SLOT_SYMBOLS
    )

    plan = build_cpu_plan(input_files=len(paths))
    workers = num_workers or plan.preprocess_workers
    threads = num_threads or plan.sentencepiece_threads
    sentencepiece_version = validate_sentencepiece_training_runtime(threads)
    normalized_source_only = canonicalize_language_tags(
        list(source_only_languages),
        field="tokenizer source_only_languages",
        reject_duplicates=False,
    )
    normalized_synthetic_prefixes = normalize_synthetic_prefixes(train_only_prefixes)
    initial_source_records = _tokenizer_source_records(paths, monolingual)
    if expected_source_identities is not None:
        _verify_expected_tokenizer_source_identities(
            paths,
            monolingual,
            initial_source_records,
            expected_source_identities,
        )
    sampling_description = (
        "all accepted sentences"
        if input_sentence_size == 0
        else f"at most {input_sentence_size:,} stratified sentences"
    )
    print(
        f"[tokenizer] authenticated {len(initial_source_records):,} source files "
        f"({sum(cast(int, record['size']) for record in initial_source_records) / 2**30:.2f} GiB); "
        f"SentencePiece will receive {sampling_description} "
        f"(alpha={sampling_alpha:g}, seed={sampling_seed})",
        flush=True,
    )
    training_contract = _tokenizer_training_contract(
        source_records=initial_source_records,
        vocab_size=vocab_size,
        input_sentence_size=input_sentence_size,
        sampling_alpha=sampling_alpha,
        sampling_seed=sampling_seed,
        seed_sentencepiece_size=seed_sentencepiece_size,
        character_coverage=character_coverage,
        required_character_min_occurrences=required_character_min_occurrences,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        language_pairs=normalized_pairs,
        translation_directions=normalized_directions,
        denoise_languages=denoise_languages,
        reasoning_languages=reasoning_languages,
        approximate_split=approximate_split,
        source_only_languages=normalized_source_only,
        train_only_prefixes=normalized_synthetic_prefixes,
        split_digits=split_digits,
        monolingual_sample_ratio=monolingual_sample_ratio,
        sentencepiece_version=sentencepiece_version,
        num_threads=threads,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_model = output_dir / "sion.model"
    if canonical_model.exists() or canonical_model.is_symlink():
        return _validate_tokenizer_generation(output_dir, training_contract)
    recovered = _recover_tokenizer_staging(output_dir, training_contract)
    if recovered is not None:
        return _validate_tokenizer_generation(output_dir, training_contract)

    # One exhaustive pass provides both the character floor and exact language
    # counts. The second pass can then sample without a text reservoir.
    counts = corpus_character_counts(
        paths,
        language_pairs=normalized_pairs,
        translation_directions=normalized_directions,
        monolingual=monolingual,
        monolingual_sample_ratio=monolingual_sample_ratio,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        approximate_split=approximate_split,
        source_only_languages=normalized_source_only,
        train_only_prefixes=normalized_synthetic_prefixes,
        num_workers=workers,
    )
    if counts.sentences == 0:
        raise ValueError("no accepted training-partition sentences remain for tokenizer training")
    quotas = stratified_sentence_quotas(
        counts.sentences_per_language,
        input_sentence_size,
        alpha=sampling_alpha,
    )
    print(
        f"[tokenizer] corpus: {counts.sentences:,} sentences, "
        f"{counts.character_total:,} characters; selected {sum(quotas.values()):,} "
        f"sentences across {len(quotas):,} languages",
        flush=True,
    )

    # Reserve characters that carry content. Coverage cuts the global frequency
    # tail; this floor restores characters that are meaningful within any shard.
    required_characters: list[str] = []
    if required_character_min_occurrences > 0:
        required_characters = required_characters_from_counts(
            counts.characters,
            min_occurrences=required_character_min_occurrences,
        )
        # SentencePiece refuses when required_chars plus its meta pieces exceed
        # vocab_size, and it only says so after the corpus scan. Say it here, with
        # the number to change, rather than after a long wait. This costs nothing,
        # so it runs before the probe below, which costs seconds.
        reserved = len(required_characters) + len(symbols) + 256 + SENTENCEPIECE_META_PIECE_COUNT
        if reserved >= vocab_size:
            raise ValueError(
                f"required characters ({len(required_characters):,}) plus control symbols "
                f"({len(symbols):,}), byte fallback (256), and SentencePiece meta pieces "
                f"({SENTENCEPIECE_META_PIECE_COUNT}) consume {reserved:,} slots, but "
                f"vocab_size is {vocab_size:,}. Raise vocab_size or raise "
                f"required_character_min_occurrences (currently "
                f"{required_character_min_occurrences})."
            )
        # Ask SentencePiece whether it will take this set before handing it the
        # corpus. It asserts on a character it dislikes only after reading every
        # sentence, so without this the failure arrives at the end of the run.
        required_characters, refused = acceptable_required_characters(required_characters)
        if refused:
            print(
                "[tokenizer] excluded characters rejected by SentencePiece from "
                f"required_chars: {' '.join(f'U+{ord(c):04X}' for c in refused)}",
                flush=True,
            )

    staging = Path(tempfile.mkdtemp(prefix=TOKENIZER_STAGING_PREFIX, dir=output_dir))
    published = False
    recoverable = False
    try:
        model_prefix = staging / "sion"
        sampled_counts: Counter[str] = Counter()
        sampled_monolingual_counts: Counter[str] = Counter()
        records = iter_tokenizer_records(
            paths,
            monolingual=monolingual,
            monolingual_sample_ratio=monolingual_sample_ratio,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            language_pairs=normalized_pairs,
            translation_directions=normalized_directions,
            approximate_split=approximate_split,
            source_only_languages=normalized_source_only,
            train_only_prefixes=normalized_synthetic_prefixes,
            num_workers=workers,
        )
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter_stratified_tokenizer_sentences(
                records,
                counts.sentences_per_language,
                quotas,
                seed=sampling_seed,
                sampled_counts=sampled_counts,
                sampled_monolingual_counts=sampled_monolingual_counts,
            ),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=character_coverage,
            byte_fallback=True,
            split_digits=split_digits,
            normalization_rule_name="identity",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=symbols,
            required_chars="".join(required_characters),
            # Sampling already happened in our deterministic constant-memory
            # iterator. SentencePiece must not shuffle or sample it again.
            input_sentence_size=0,
            seed_sentencepiece_size=min(seed_sentencepiece_size, sum(quotas.values())),
            shuffle_input_sentence=False,
            train_extremely_large_corpus=True,
            hard_vocab_limit=False,
            num_threads=threads,
        )
        staged_model = model_prefix.with_suffix(".model")
        write_token_features(staged_model, staging / "token_features.npz")
        write_tokenizer_metadata(
            staged_model,
            split_digits=split_digits,
            language_pairs=normalized_pairs,
            translation_directions=normalized_directions,
            denoise_languages=denoise_languages,
            reasoning_languages=reasoning_languages,
            monolingual_sentences=dict(sampled_monolingual_counts) or None,
            monolingual_sample_ratio=monolingual_sample_ratio,
            required_characters=required_characters,
            corpus_sentences=counts.sentences,
            corpus_sentences_per_language=counts.sentences_per_language,
            sampled_sentences=sum(sampled_counts.values()),
            sampled_sentences_per_language=sampled_counts,
            training_contract=training_contract,
        )
        _validate_tokenizer_generation(staging, training_contract)
        if _tokenizer_source_records(paths, monolingual) != initial_source_records:
            raise RuntimeError(
                "tokenizer sources changed during training; refusing to publish mixed inputs"
            )
        recoverable = True
        _publish_tokenizer_generation(staging, output_dir)
        published = True
        return _validate_tokenizer_generation(output_dir, training_contract)
    finally:
        if staging.exists() and (published or not recoverable):
            _remove_tokenizer_staging(staging, output_dir)
        if not published and canonical_model.exists():
            # The model is the commit marker. If publication reached it, the
            # generation is complete even if the caller was interrupted later.
            _validate_tokenizer_generation(output_dir, training_contract)


def _char_script(char: str) -> int:
    codepoint = ord(char)
    if 0xAC00 <= codepoint <= 0xD7A3 or 0x1100 <= codepoint <= 0x11FF:
        return SCRIPT_HANGUL
    if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
        return SCRIPT_HAN
    if 0x3040 <= codepoint <= 0x309F:
        return SCRIPT_HIRAGANA
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return SCRIPT_KATAKANA
    if char.isdigit():
        return SCRIPT_DIGIT
    if "LATIN" in unicodedata.name(char, ""):
        return SCRIPT_LATIN
    return SCRIPT_OTHER


def classify_piece(piece: str) -> int:
    if piece.startswith("<") and piece.endswith(">"):
        return SCRIPT_SPECIAL
    surface = piece.replace("▁", "")
    if not surface:
        return SCRIPT_SPECIAL
    scripts = {_char_script(char) for char in surface if not char.isspace()}
    if not scripts:
        return SCRIPT_SPECIAL
    return next(iter(scripts)) if len(scripts) == 1 else SCRIPT_MIXED


def hangul_features(piece: str) -> tuple[int, int, int]:
    for char in reversed(piece.replace("▁", "")):
        codepoint = ord(char)
        if 0xAC00 <= codepoint <= 0xD7A3:
            syllable = codepoint - 0xAC00
            onset = syllable // 588
            vowel = (syllable % 588) // 28
            coda = syllable % 28
            return onset + 1, vowel + 1, coda + 1
    return 0, 0, 0


def write_token_features(model_path: str | Path, output_path: str | Path) -> None:
    tokenizer = SionTokenizer(model_path)
    scripts = np.zeros(len(tokenizer), dtype=np.uint8)
    onsets = np.zeros(len(tokenizer), dtype=np.uint8)
    vowels = np.zeros(len(tokenizer), dtype=np.uint8)
    codas = np.zeros(len(tokenizer), dtype=np.uint8)
    for token_id in range(len(tokenizer)):
        piece = tokenizer.processor.id_to_piece(token_id)
        scripts[token_id] = classify_piece(piece)
        onsets[token_id], vowels[token_id], codas[token_id] = hangul_features(piece)
    np.savez_compressed(
        output_path,
        script=scripts,
        onset=onsets,
        vowel=vowels,
        coda=codas,
    )
