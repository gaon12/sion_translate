# SentencePiece metadata is returned as an untyped JSON mapping.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import glob
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import sentencepiece as spm

from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
)
from sion_translate.data.monolingual import (
    MonolingualDiscovery,
    monolingual_budgets,
    sample_monolingual_sentences,
)
from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.fingerprint import file_sha256
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


DEFAULT_LANGUAGE_PAIR = ("ko", "ja")
TOKENIZER_METADATA_FILENAME = "tokenizer_metadata.json"
TOKENIZER_METADATA_VERSION = 2

# 언어쌍에 따라 달라지는 제어 토큰: <2xx> = "xx 언어로 번역하라",
# <denoise_xx> = "xx 언어 원문을 복원하라(denoising)".
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

# 나중에 추가된 제어 토큰. 새로 학습하는 토크나이저에는 예약하지만, 이 토큰이 없는
# 기존 토크나이저도 계속 불러올 수 있어야 하므로 **필수 목록에 넣지 않습니다**.
# 없으면 관련 기능(초안 수정)만 사용할 수 없고 번역은 그대로 동작합니다.
#
# <draft> = "이 뒤는 같은 문장의 초벌 번역이다. 원문과 대조해 고쳐라."
OPTIONAL_CONTROL_SYMBOLS = [
    "<draft>",
]


def control_symbols(languages: Sequence[str] = DEFAULT_LANGUAGE_PAIR) -> list[str]:
    """언어 목록에 맞는 전체 제어 토큰 목록 (토크나이저 학습 시 예약)."""

    unique_languages = tuple(dict.fromkeys(languages))
    return (
        [f"<2{language}>" for language in unique_languages]
        + [f"<denoise_{language}>" for language in unique_languages]
        + SHARED_CONTROL_SYMBOLS
        + OPTIONAL_CONTROL_SYMBOLS
    )


# 하위 호환용 별칭 (기존 ko-ja 토크나이저 검증 경로에서 사용)
BASE_CONTROL_SYMBOLS = control_symbols(DEFAULT_LANGUAGE_PAIR)

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


def write_tokenizer_metadata(
    model_path: str | Path,
    *,
    split_digits: bool,
    language_pairs: Sequence[Sequence[str]],
    monolingual_sentences: dict[str, int] | None = None,
    monolingual_sample_ratio: float = 0.0,
) -> Path:
    """Write the reproducibility and identity contract for a trained tokenizer."""

    model_path = Path(model_path)
    vocab_path = model_path.with_suffix(".vocab")
    features_path = model_path.parent / "token_features.npz"
    normalized_pairs = normalize_language_pairs(language_pairs=language_pairs)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    metadata = {
        "version": TOKENIZER_METADATA_VERSION,
        "split_digits": bool(split_digits),
        "language_pair": list(normalized_pairs[0]),
        "language_pairs": [list(pair) for pair in normalized_pairs],
        "vocab_size": int(processor.vocab_size()),
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "vocab_file": vocab_path.name,
        "vocab_sha256": file_sha256(vocab_path),
        # foundation 단계가 이 토크나이저와 같은 어휘를 보는지 확인할 수 있게
        # 단일어 표본 규모를 남깁니다. 0 이면 병렬 코퍼스만으로 학습한 것입니다.
        "monolingual_sample_ratio": float(monolingual_sample_ratio),
        "monolingual_sentences": dict(monolingual_sentences or {}),
    }
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
    """문자열 계열 값을 SentencePiece용 기본 Python ``str``로 변환한다."""
    if isinstance(value, str):
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise TypeError(f"{value_name}이 UTF-8로 해석할 수 없는 bytes입니다.") from error

    # numpy.str_, pandas 문자열 스칼라 등은 item()으로 기본 스칼라를 얻는다.
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
                    f"{value_name}의 스칼라 값이 UTF-8로 해석할 수 없는 bytes입니다."
                ) from error

    raise TypeError(
        f"{value_name}은 문자열이어야 합니다. 현재 타입={type(value).__name__}, 값={value!r}"
    )


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
    return sorted(path.resolve() for path in paths if path.exists())


def _filter_text_batch(
    batch: tuple[
        list[bytes],
        tuple[tuple[str, str], ...],
        float,
        float,
        bool,
        frozenset[str],
        bool,
    ],
) -> list[tuple[str, str, str, str, str, bytes, bytes]]:
    (
        lines,
        language_pairs,
        validation_fraction,
        test_fraction,
        approximate_split,
        source_only_languages,
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
            if language_b in source_only_languages:
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
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
    language_pairs: Sequence[Sequence[str]] | None = None,
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
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
    language_pairs: Sequence[Sequence[str]] | None = None,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
) -> Iterator[tuple[str, str]]:
    """``(language, text)`` 쌍을 낸다.

    언어별 상한을 걸려면 어느 문장이 어느 언어인지 알아야 합니다. 라벨 없는
    ``iter_parallel_text`` 는 이 함수를 감싼 것이라 두 경로가 갈라지지 않습니다.
    """

    policy = QualityPolicy()
    policy.validate()
    estimated_pairs = max(1, sum(path.stat().st_size for path in paths) // 200)
    target_split_guard = TargetSplitGuard(estimated_pairs, validation_fraction, test_fraction)
    workers = num_workers or build_cpu_plan(input_files=len(paths)).preprocess_workers
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    languages = frozenset(languages_from_pairs(normalized_pairs))
    source_only = frozenset(str(language) for language in source_only_languages)
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
            source_only,
            synthetic_path(path, synthetic_prefixes),
        )
        for path, batch in _raw_batches(paths)
    )
    if workers <= 1:
        results = map(_filter_text_batch, inputs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
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

        # 언어쌍 자동 감지: vocab 에 예약된 <2xx> 토큰을 찾아 어떤 언어쌍으로
        # 학습된 토크나이저인지 알아냅니다. 덕분에 설정을 따로 전달하지 않아도
        # ko-ja 든 en-de 든 같은 코드로 동작합니다.
        self.language_tags: dict[str, int] = {}  # {"ja": <2ja> 토큰 id, ...}
        self.denoise_tags: dict[str, int] = {}  # {"ko": <denoise_ko> 토큰 id, ...}
        lang_pattern = re.compile(r"^<2([A-Za-z0-9]+)>$")
        denoise_pattern = re.compile(r"^<denoise_([A-Za-z0-9]+)>$")
        byte_pattern = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
        # 예약 구간의 끝은 첫 byte fallback 조각입니다. SentencePiece 는
        # pad/unk/bos/eos → user_defined_symbols → byte 조각 → 학습된 조각
        # 순으로 배치하므로, 여기서 멈추면 제어 토큰은 전부 보면서 학습된
        # 조각을 <2xx> 로 오인할 일도 없습니다.
        #
        # 고정 상한(예전 256)을 쓰면 언어 수가 늘 때 스캔이 예약 구간 중간에서
        # 끊기고, 증상이 예외가 아니라 "일부 언어만 감지됨" 이라 조용합니다.
        for token_id in range(self.processor.vocab_size()):
            piece = self.processor.id_to_piece(token_id)
            if byte_pattern.match(piece):
                break
            if match := lang_pattern.match(piece):
                self.language_tags[match.group(1)] = token_id
            elif match := denoise_pattern.match(piece):
                self.denoise_tags[match.group(1)] = token_id
        if len(self.language_tags) < 2 or set(self.language_tags) != set(self.denoise_tags):
            raise ValueError(
                "Tokenizer must reserve at least two <2xx> tags with matching "
                f"<denoise_xx> tags; found {sorted(self.language_tags)} / {sorted(self.denoise_tags)}"
            )
        self.languages = tuple(sorted(self.language_tags))

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

        # 선택적 제어 토큰: 없으면 None. 기존 토크나이저를 거부하지 않습니다.
        self.optional_ids: dict[str, int] = {}
        for symbol in OPTIONAL_CONTROL_SYMBOLS:
            token_id = int(self.processor.piece_to_id(symbol))
            if token_id >= 0 and self.processor.id_to_piece(token_id) == symbol:
                self.optional_ids[symbol] = token_id
        self.draft_id: int | None = self.optional_ids.get("<draft>")
        # 하위 호환 별칭 (ko-ja 토크나이저일 때만 존재)
        if {"ko", "ja"} == set(self.language_tags):
            self.ko_to_ja_id = self.language_tags["ja"]
            self.ja_to_ko_id = self.language_tags["ko"]
            self.denoise_ko_id = self.denoise_tags["ko"]
            self.denoise_ja_id = self.denoise_tags["ja"]

    def __len__(self) -> int:
        return self.processor.vocab_size()

    @property
    def splits_digits(self) -> bool:
        """숫자가 한 자리씩 분리되는 토크나이저인지 확인한다.

        SentencePiece 모델 파일에는 학습 플래그가 그대로 남지 않으므로,
        여러 자리 숫자를 실제로 인코딩해 조각을 확인합니다. 거짓이면 금액·
        용량 같은 값이 그럴듯한 다른 값으로 바뀌는 오역 위험이 있습니다
        (자세한 배경은 ``train_tokenizer`` 의 ``split_digits`` 설명 참고).
        """
        pieces = self.processor.encode("38720", out_type=str)
        return all(len(piece.replace("▁", "")) <= 1 for piece in pieces)

    def piece_id(self, piece: str) -> int:
        return int(self.processor.piece_to_id(piece))

    def encode(self, text: str) -> list[int]:
        """문장을 정규화한 뒤 SentencePiece 토큰 ID 목록으로 변환한다."""
        source_text = _to_python_string(
            text,
            value_name="tokenizer.encode() 입력",
        )

        normalized_text = normalize_text(source_text)
        normalized_text = _to_python_string(
            normalized_text,
            value_name="normalize_text() 반환값",
        )

        return list(
            self.processor.encode(
                normalized_text,
                out_type=int,
            )
        )

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode([int(token_id) for token_id in ids])


def iter_tokenizer_sentences(
    paths: Sequence[Path],
    *,
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
    language_pairs: Sequence[Sequence[str]],
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
    monolingual_counts: dict[str, int] | None = None,
) -> Iterator[str]:
    """토크나이저가 볼 문장 전부: 병렬 코퍼스 + 상한을 건 단일어 표본.

    단일어를 넣는 이유는 foundation 단계가 자기 코퍼스에 없는 어휘로 학습하는
    것을 막기 위해서이고, 상한을 거는 이유는 분량이 큰 언어가 vocab 을
    독식하는 것을 막기 위해서입니다. 상한은 병렬 코퍼스를 흘려보내며 언어별로
    센 문장 수에서 나오므로, 추가 pass 없이 결정됩니다 — 병렬을 먼저 전부
    내보낸 뒤에 단일어를 내보내는 순서가 그래서 중요합니다.

    ``monolingual_counts`` 를 주면 언어별로 실제 내보낸 단일어 문장 수가
    기록됩니다(호출자 보고용).
    """

    parallel_counts: Counter[str] = Counter()
    for language, text in iter_parallel_text_with_languages(
        paths,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        language_pairs=language_pairs,
        approximate_split=approximate_split,
        source_only_languages=source_only_languages,
        train_only_prefixes=train_only_prefixes,
        num_workers=num_workers,
    ):
        parallel_counts[language] += 1
        yield text

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
            yield canonical_text(text)
        if monolingual_counts is not None:
            monolingual_counts[language] = emitted


def corpus_character_counts(
    paths: Sequence[Path],
    *,
    language_pairs: Sequence[Sequence[str]],
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
) -> Counter[str]:
    """Count every character in the training partition of ``paths``."""

    counts: Counter[str] = Counter()
    for text in iter_tokenizer_sentences(
        paths,
        monolingual=monolingual,
        monolingual_sample_ratio=monolingual_sample_ratio,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        language_pairs=language_pairs,
        approximate_split=approximate_split,
        source_only_languages=source_only_languages,
        train_only_prefixes=train_only_prefixes,
        num_workers=num_workers,
    ):
        counts.update(text)
    return counts


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
SENTENCEPIECE_RESERVED_CHARACTERS = frozenset("⁇▅")


def required_characters_from_counts(
    counts: Counter[str],
    *,
    min_occurrences: int = 25,
) -> list[str]:
    """Characters frequent enough that byte fallback would be a regression.

    A character seen this often carries content, so splitting it into raw bytes
    costs the model three tokens where one would do. The 한본어 corpus produces
    fused syllables (``네`` + ``ㅋ`` -> ``넼``) that appear over a thousand times
    yet are absent from any tokenizer trained before that corpus existed; the
    shipped one renders ``넼`` as three ``<0x..>`` pieces.

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
    """``(수용된 문자, 거부된 문자)`` — SentencePiece 에게 직접 물어서 가른다.

    ``required_chars`` 에 SentencePiece 가 받지 않는 문자가 하나라도 있으면
    학습은 **코퍼스를 다 읽은 뒤에** assert 로 죽습니다. 3,500만 문장에서는 그
    지점까지 가는 데만 몇 시간이 걸리고, 빌린 CPU 라면 그 시간이 곧 비용입니다.

    어떤 문자가 거부되는지 규칙으로 예측하지 않습니다. U+2047 은 ``kUNKChar``
    라 설명이 되지만 U+2585 는 되지 않습니다 — 이웃한 블록 문자는 통과하고,
    공백 기호 U+2581 도 아닙니다. 규칙을 추측해 목록을 손으로 관리하면 다음
    코퍼스에서 또 같은 방식으로 무너집니다. 그래서 작은 합성 코퍼스로 실제
    학습을 시켜 보고, 실패하면 이분 탐색으로 범인을 찾아 빼기를 반복합니다.

    비용은 몇 초짜리 학습 수십 번이고, 막는 것은 코퍼스 전량 스캔입니다.
    """

    candidate = list(required)
    rejected: list[str] = []
    with tempfile.TemporaryDirectory() as workspace:
        probe_corpus = Path(workspace) / "probe.txt"
        # 어떤 언어에도 치우치지 않게, 그리고 required 문자를 담지 않게 씁니다.
        # 여기서 재는 것은 코퍼스가 아니라 required 집합의 수용 여부입니다.
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
                    # 어느 절반도 단독으로는 실패하지 않으면 조합 문제입니다.
                    # 그런 사례는 아직 관측되지 않았고, 조용히 추측해서 지우는
                    # 것보다 멈추고 말하는 편이 낫습니다.
                    raise RuntimeError(
                        "SentencePiece rejects this required-character set, but no "
                        f"single character explains it ({len(low)} remain). Inspect "
                        "them by hand rather than trusting an automatic drop."
                    )
            offender = low[0]
            rejected.append(offender)
            candidate = [character for character in candidate if character != offender]
    return candidate, rejected


def train_tokenizer(
    input_patterns: Sequence[str],
    output_dir: str | Path,
    *,
    vocab_size: int = 48000,
    # 0 means "use every sentence". The corpus is now large enough that the old
    # 4,000,000 cap covered only 22.2% of it, and uniform sampling shrinks a
    # small shard in proportion: the 한본어 corpus is 0.11% of all sentences, so
    # the fused syllables it exists to teach were sampled a few hundred times and
    # could be pruned out of the vocabulary by chance.
    input_sentence_size: int = 0,
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
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
    language_pairs: Sequence[Sequence[str]] | None = None,
    approximate_split: bool = False,
    source_only_languages: Sequence[str] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    num_workers: int | None = None,
    num_threads: int | None = None,
    split_digits: bool = True,
    # foundation 사전학습용 단일어 코퍼스. 넣으면 그 어휘가 vocab 에 들어가고,
    # 넣지 않으면 foundation 단계가 자기 코퍼스에 없는 어휘로 학습합니다.
    # 언어별 상한은 `monolingual_sample_ratio` 가 정합니다.
    monolingual: MonolingualDiscovery | None = None,
    monolingual_sample_ratio: float = 0.0,
) -> Path:
    """병렬 코퍼스로 joint SentencePiece 토크나이저를 학습한다.

    ``character_coverage`` 와 ``required_character_min_occurrences`` 는 역할이
    다릅니다. 전자는 "빈도 꼬리를 어디서 자를지"를 정하고, 후자는 "그 아래라도
    이건 반드시 넣어라"를 정합니다. 전자가 1.0 이면 자를 꼬리가 없어서 후자도,
    byte fallback 도, byte fallback 비율 관문도 전부 무의미해집니다.

    ``split_digits`` 는 기본으로 켭니다. 끄면 SentencePiece 가 자주 등장하는
    숫자열을 하나의 토큰으로 병합하므로 (예: ``62.5kg`` → ``▁6`` + ``2.5`` + ``kg``,
    ``1,286,400`` → ``▁1,2`` + ``86`` + ``,`` + ``400``) 모델이 숫자를 자릿수로
    다루지 못하고 통째로 암기한 덩어리로만 볼 수 있습니다. 그 결과 금액·용량·
    날짜가 그럴듯한 다른 값으로 바뀌는 오역이 생기며, 사후학습의 숫자 보존
    보상(``reward_number_weight``)도 최적화할 신호를 얻지 못합니다.
    """
    paths = expand_inputs(input_patterns)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {input_patterns}")
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError("character_coverage must be in (0, 1]")
    if character_coverage >= 1.0 and required_character_min_occurrences > 0:
        raise ValueError(
            "character_coverage=1.0 admits every character in the corpus, which makes "
            "required_chars redundant and byte fallback unreachable. Lower "
            "character_coverage (0.9999 is the default) or set "
            "required_character_min_occurrences=0 to opt out of the frequency floor."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "sion"
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    languages = languages_from_pairs(normalized_pairs)
    symbols = control_symbols(languages) + SLOT_SYMBOLS

    plan = build_cpu_plan(input_files=len(paths))
    workers = num_workers or plan.preprocess_workers
    threads = num_threads or plan.sentencepiece_threads

    # Reserve the characters that carry content. This costs one pass over the
    # corpus and is worth it precisely because `character_coverage` is below 1.0:
    # coverage cuts the frequency tail globally, so a character that is rare
    # overall but common in a shard that matters (the 한본어 fused syllables are
    # 0.11% of all sentences) would land in the tail. This floor pulls it back.
    required_characters: list[str] = []
    if required_character_min_occurrences > 0:
        counts = corpus_character_counts(
            paths,
            language_pairs=normalized_pairs,
            monolingual=monolingual,
            monolingual_sample_ratio=monolingual_sample_ratio,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            approximate_split=approximate_split,
            source_only_languages=source_only_languages,
            train_only_prefixes=train_only_prefixes,
            num_workers=workers,
        )
        required_characters = required_characters_from_counts(
            counts,
            min_occurrences=required_character_min_occurrences,
        )
        # SentencePiece refuses when required_chars plus its meta pieces exceed
        # vocab_size, and it only says so after the corpus scan. Say it here, with
        # the number to change, rather than after a long wait. This costs nothing,
        # so it runs before the probe below, which costs seconds.
        reserved = len(required_characters) + len(symbols) + 256
        if reserved >= vocab_size:
            raise ValueError(
                f"required characters ({len(required_characters):,}) plus control symbols "
                f"({len(symbols):,}) and byte fallback (256) need {reserved:,} slots, but "
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
                "[tokenizer] SentencePiece 가 받지 않는 문자를 required_chars 에서 "
                f"제외했습니다: {' '.join(f'U+{ord(c):04X}' for c in refused)}",
                flush=True,
            )

    monolingual_counts: dict[str, int] = {}
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_tokenizer_sentences(
            paths,
            monolingual=monolingual,
            monolingual_sample_ratio=monolingual_sample_ratio,
            monolingual_counts=monolingual_counts,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            language_pairs=normalized_pairs,
            approximate_split=approximate_split,
            source_only_languages=source_only_languages,
            train_only_prefixes=train_only_prefixes,
            num_workers=workers,
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
        input_sentence_size=input_sentence_size,
        seed_sentencepiece_size=seed_sentencepiece_size,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        hard_vocab_limit=False,
        num_threads=threads,
    )
    model_path = model_prefix.with_suffix(".model")
    write_token_features(model_path, output_dir / "token_features.npz")
    write_tokenizer_metadata(
        model_path,
        split_digits=split_digits,
        language_pairs=normalized_pairs,
        monolingual_sentences=monolingual_counts or None,
        monolingual_sample_ratio=monolingual_sample_ratio,
    )
    return model_path


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
