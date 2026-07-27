from __future__ import annotations

import glob
import hashlib
import json
import re
import unicodedata
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
from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.fingerprint import file_sha256
from sion_translate.performance import bounded_ordered_map, build_cpu_plan
from sion_translate.splitting import TargetSplitGuard, choose_split_for_key, normalized_split_key


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
) -> Path:
    """Write the reproducibility and identity contract for a trained tokenizer."""

    model_path = Path(model_path)
    vocab_path = model_path.with_suffix(".vocab")
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
    }
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
    batch: tuple[list[bytes], tuple[tuple[str, str], ...], float, float],
) -> list[tuple[str, str, str, bytes]]:
    lines, language_pairs, validation_fraction, test_fraction = batch
    policy = QualityPolicy()
    accepted: list[tuple[str, str, str, bytes]] = []
    for raw_line in lines:
        try:
            row = json.loads(raw_line.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        expansion = expand_parallel_record(row, language_pairs)
        for pair in expansion.pairs:
            text_a, text_b = canonical_text(pair.text_a), canonical_text(pair.text_b)
            languages = (pair.language_a, pair.language_b)
            if not assess_pair(text_a, text_b, policy, languages=languages).accepted:
                continue
            source_key = normalized_split_key(text_a)
            target_key = normalized_split_key(text_b)
            if len(language_pairs) > 1:
                source_key = f"{pair.language_a}\0{source_key}"
                target_key = f"{pair.language_b}\0{target_key}"
            split = choose_split_for_key(
                source_key,
                validation_fraction,
                test_fraction,
            )
            target_digest = hashlib.sha256(target_key.encode("utf-8")).digest()
            accepted.append((text_a, text_b, split, target_digest))
    return accepted


def _raw_batches(paths: Sequence[Path], batch_size: int = 512):
    for path in paths:
        with path.open("rb") as handle:
            batch: list[bytes] = []
            for raw_line in handle:
                batch.append(raw_line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch


def iter_parallel_text(
    paths: Sequence[Path],
    *,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
    language_pairs: Sequence[Sequence[str]] | None = None,
    num_workers: int | None = None,
) -> Iterator[str]:
    """Yield train-partition text without first materializing a temporary corpus."""

    policy = QualityPolicy()
    policy.validate()
    estimated_pairs = max(1, sum(path.stat().st_size for path in paths) // 200)
    target_split_guard = TargetSplitGuard(estimated_pairs, validation_fraction, test_fraction)
    workers = num_workers or build_cpu_plan(input_files=len(paths)).preprocess_workers
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    inputs = (
        (batch, normalized_pairs, validation_fraction, test_fraction)
        for batch in _raw_batches(paths)
    )
    if workers <= 1:
        results = map(_filter_text_batch, inputs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = bounded_ordered_map(executor, _filter_text_batch, inputs, max_pending=workers * 2)
    try:
        for candidates in results:
            for text_a, text_b, split, target_digest in candidates:
                if not target_split_guard.accept(split, target_digest):
                    continue
                if split != "train":
                    continue
                yield text_a
                yield text_b
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
        # 제어 토큰은 vocab 앞부분에 예약되므로 앞쪽 일부만 훑으면 충분합니다.
        scan_limit = min(self.processor.vocab_size(), 256)
        for token_id in range(scan_limit):
            piece = self.processor.id_to_piece(token_id)
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


def train_tokenizer(
    input_patterns: Sequence[str],
    output_dir: str | Path,
    *,
    vocab_size: int = 48000,
    input_sentence_size: int = 4_000_000,
    seed_sentencepiece_size: int = 1_000_000,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
    language_pairs: Sequence[Sequence[str]] | None = None,
    num_workers: int | None = None,
    num_threads: int | None = None,
    split_digits: bool = True,
) -> Path:
    """병렬 코퍼스로 joint SentencePiece 토크나이저를 학습한다.

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "sion"
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    languages = languages_from_pairs(normalized_pairs)
    symbols = control_symbols(languages) + SLOT_SYMBOLS

    plan = build_cpu_plan(input_files=len(paths))
    workers = num_workers or plan.preprocess_workers
    threads = num_threads or plan.sentencepiece_threads
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_parallel_text(
            paths,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            language_pairs=normalized_pairs,
            num_workers=workers,
        ),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        split_digits=split_digits,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=symbols,
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
