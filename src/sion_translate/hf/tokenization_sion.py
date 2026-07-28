"""SentencePiece tokenizer exposed through the Transformers tokenizer API."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
from transformers import PreTrainedTokenizer


_FEATURE_NAMES = ("script", "onset", "vowel", "coda")
_FEATURE_MAXIMUM_IDS = {
    "script": 9,
    "onset": 20,
    "vowel": 22,
    "coda": 29,
}
_MODEL_FEATURE_NAMES = {
    "script": "src_script_ids",
    "onset": "src_onset_ids",
    "vowel": "src_vowel_ids",
    "coda": "src_coda_ids",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SionTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = [
        "input_ids",
        "attention_mask",
        "src_script_ids",
        "src_onset_ids",
        "src_vowel_ids",
        "src_coda_ids",
        "memory_token_ids",
        "memory_mask",
        "memory_type_ids",
        "memory_mode_ids",
    ]

    def __init__(
        self,
        vocab_file: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        token_features_file: str | None = None,
        token_features_sha256: str | None = None,
        tokenizer_sha256: str | None = None,
        slot_token_ids: list[int] | tuple[int, ...] | None = None,
        tetm_type_id: int = 8,
        tetm_mode_id: int = 4,
        **kwargs: Any,
    ):
        self.vocab_file = str(vocab_file)
        vocab_path = Path(self.vocab_file)
        actual_tokenizer_sha256 = _file_sha256(vocab_path)
        if tokenizer_sha256 is not None and actual_tokenizer_sha256 != tokenizer_sha256:
            raise ValueError(
                "tokenizer.model SHA-256 mismatch: "
                f"expected {tokenizer_sha256}, got {actual_tokenizer_sha256}"
            )
        self.sp_model = spm.SentencePieceProcessor(model_file=self.vocab_file)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.language_tags: dict[str, int] = {}
        self.denoise_tags: dict[str, int] = {}
        language_pattern = re.compile(r"^<2([A-Za-z0-9]+)>$")
        denoise_pattern = re.compile(r"^<denoise_([A-Za-z0-9]+)>$")
        for token_id in range(min(self.sp_model.vocab_size(), 256)):
            piece = self.sp_model.id_to_piece(token_id)
            if match := language_pattern.fullmatch(piece):
                self.language_tags[match.group(1)] = token_id
            elif match := denoise_pattern.fullmatch(piece):
                self.denoise_tags[match.group(1)] = token_id
        discovered_slot_ids: list[int] = []
        for index in range(64):
            symbol = f"<slot_{index}>"
            token_id = int(self.sp_model.piece_to_id(symbol))
            if token_id < 0 or self.sp_model.id_to_piece(token_id) != symbol:
                break
            discovered_slot_ids.append(token_id)
        if slot_token_ids is not None:
            expected_slot_ids = [int(token_id) for token_id in slot_token_ids]
            if expected_slot_ids != discovered_slot_ids:
                raise ValueError(
                    "tokenizer protected slot IDs do not match checkpoint config: "
                    f"expected {expected_slot_ids}, got {discovered_slot_ids}"
                )
        self.slot_token_ids = discovered_slot_ids
        self._slot_token_id_set = frozenset(discovered_slot_ids)
        self.tetm_type_id = int(tetm_type_id)
        self.tetm_mode_id = int(tetm_mode_id)
        if self.tetm_type_id < 0 or self.tetm_mode_id < 0:
            raise ValueError("TETM type and mode IDs must be non-negative")

        feature_path: Path | None
        if token_features_file is None:
            sibling = vocab_path.parent / "token_features.npz"
            feature_path = sibling if sibling.is_file() else None
        else:
            candidate = Path(token_features_file)
            feature_path = candidate if candidate.is_absolute() else vocab_path.parent / candidate
        if feature_path is None:
            if token_features_sha256 is not None:
                raise FileNotFoundError(
                    f"token_features.npz required by tokenizer metadata: {vocab_path.parent}"
                )
            self.token_features: dict[str, np.ndarray] | None = None
            actual_features_sha256 = None
        else:
            if not feature_path.is_file():
                raise FileNotFoundError(f"token feature file does not exist: {feature_path}")
            actual_features_sha256 = _file_sha256(feature_path)
            if (
                token_features_sha256 is not None
                and actual_features_sha256 != token_features_sha256
            ):
                raise ValueError(
                    "token_features.npz SHA-256 mismatch: "
                    f"expected {token_features_sha256}, got {actual_features_sha256}"
                )
            self.token_features = self._load_token_features(feature_path)
        self._token_features_path = feature_path

        base_special_tokens = {"<pad>", "<unk>", "<s>", "</s>"}
        byte_fallback_pattern = re.compile(r"^<0x[0-9A-F]{2}>$")
        additional = [
            piece
            for token_id in range(self.sp_model.vocab_size())
            if (piece := self.sp_model.id_to_piece(token_id)).startswith("<")
            and piece.endswith(">")
            and piece not in base_special_tokens
            and not byte_fallback_pattern.fullmatch(piece)
        ]
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("bos_token", "<s>")
        kwargs.setdefault("eos_token", "</s>")
        kwargs.setdefault("additional_special_tokens", additional)
        kwargs.setdefault(
            "token_features_file",
            feature_path.name if feature_path is not None else None,
        )
        kwargs.setdefault("token_features_sha256", actual_features_sha256)
        kwargs.setdefault("tokenizer_sha256", actual_tokenizer_sha256)
        kwargs.setdefault("slot_token_ids", self.slot_token_ids)
        kwargs.setdefault("tetm_type_id", self.tetm_type_id)
        kwargs.setdefault("tetm_mode_id", self.tetm_mode_id)
        super().__init__(**kwargs)

    def _load_token_features(self, path: Path) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(_FEATURE_NAMES):
                raise ValueError(
                    "token feature file must contain exactly "
                    f"{', '.join(_FEATURE_NAMES)}; got {sorted(loaded.files)}"
                )
            for name in _FEATURE_NAMES:
                values = np.asarray(loaded[name])
                expected_shape = (self.sp_model.vocab_size(),)
                if values.shape != expected_shape:
                    raise ValueError(
                        f"token feature {name} has shape {values.shape}; expected {expected_shape}"
                    )
                if not np.issubdtype(values.dtype, np.integer):
                    raise ValueError(f"token feature {name} must use an integer dtype")
                values = values.astype(np.int64, copy=True)
                if values.size and (
                    int(values.min()) < 0 or int(values.max()) >= _FEATURE_MAXIMUM_IDS[name]
                ):
                    raise ValueError(
                        f"token feature {name} contains IDs outside "
                        f"[0, {_FEATURE_MAXIMUM_IDS[name]})"
                    )
                values.setflags(write=False)
                features[name] = values
        return features

    @staticmethod
    def _rows(input_ids: Any) -> tuple[list[list[int]], bool]:
        if hasattr(input_ids, "tolist"):
            values = input_ids.tolist()
        else:
            values = input_ids
        single = not values or isinstance(values[0], int)
        rows = [values] if single else values
        return [[int(token_id) for token_id in row] for row in rows], single

    @staticmethod
    def _tensor_type(return_tensors: Any) -> str | None:
        if return_tensors is None:
            return None
        value = getattr(return_tensors, "value", return_tensors)
        return str(value)

    @staticmethod
    def _convert_feature_value(
        value: list[Any],
        *,
        tensor_type: str | None,
        dtype: str,
        single: bool,
    ) -> Any:
        if tensor_type == "pt":
            torch_dtype = torch.bool if dtype == "bool" else torch.long
            return torch.tensor(value, dtype=torch_dtype)
        if tensor_type == "np":
            numpy_dtype = np.bool_ if dtype == "bool" else np.int64
            return np.asarray(value, dtype=numpy_dtype)
        if tensor_type == "tf":
            try:
                import tensorflow as tf
            except ImportError as error:
                raise ImportError(
                    "return_tensors='tf' requires TensorFlow to be installed"
                ) from error

            tensorflow_dtype = tf.bool if dtype == "bool" else tf.int64
            return tf.convert_to_tensor(value, dtype=tensorflow_dtype)
        if tensor_type == "jax":
            try:
                import jax.numpy as jnp
            except ImportError as error:
                raise ImportError("return_tensors='jax' requires JAX to be installed") from error

            jax_dtype = jnp.bool_ if dtype == "bool" else jnp.int64
            return jnp.asarray(value, dtype=jax_dtype)
        return value[0] if single else value

    def _add_native_features(
        self,
        encoding: Any,
        *,
        return_tensors: Any,
    ) -> Any:
        rows, single = self._rows(encoding["input_ids"])
        tensor_type = self._tensor_type(return_tensors)
        if self.token_features is not None:
            for feature_name, model_name in _MODEL_FEATURE_NAMES.items():
                table = self.token_features[feature_name]
                values = [[int(table[token_id]) for token_id in row] for row in rows]
                encoding[model_name] = self._convert_feature_value(
                    values,
                    tensor_type=tensor_type,
                    dtype="long",
                    single=single,
                )

        memory_rows = [
            [token_id for token_id in row if token_id in self._slot_token_id_set][:64]
            for row in rows
        ]
        memory_length = max(1, max((len(row) for row in memory_rows), default=0))
        memory_token_ids: list[list[list[int]]] = []
        memory_mask: list[list[bool]] = []
        memory_type_ids: list[list[int]] = []
        memory_mode_ids: list[list[int]] = []
        for row in memory_rows:
            padding = memory_length - len(row)
            memory_token_ids.append(
                [[token_id] for token_id in row]
                + [[int(self.pad_token_id)] for _ in range(padding)]
            )
            memory_mask.append([True] * len(row) + [False] * padding)
            memory_type_ids.append([self.tetm_type_id] * len(row) + [0] * padding)
            memory_mode_ids.append([self.tetm_mode_id] * len(row) + [0] * padding)
        for name, value, dtype in (
            ("memory_token_ids", memory_token_ids, "long"),
            ("memory_mask", memory_mask, "bool"),
            ("memory_type_ids", memory_type_ids, "long"),
            ("memory_mode_ids", memory_mode_ids, "long"),
        ):
            encoding[name] = self._convert_feature_value(
                value,
                tensor_type=tensor_type,
                dtype=dtype,
                single=single,
            )
        return encoding

    def __call__(self, *args: Any, **kwargs: Any):
        return_tensors = kwargs.get("return_tensors")
        encoding = super().__call__(*args, **kwargs)
        return self._add_native_features(encoding, return_tensors=return_tensors)

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.vocab_size())

    def get_vocab(self) -> dict[str, int]:
        vocab = {
            self.sp_model.id_to_piece(index): index for index in range(self.sp_model.vocab_size())
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        del kwargs
        return list(self.sp_model.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self.sp_model.piece_to_id(token))

    def _convert_id_to_token(self, index: int) -> str:
        return str(self.sp_model.id_to_piece(int(index)))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return str(self.sp_model.decode(tokens))

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("Sion tokenizer does not accept paired sentence inputs")
        prefix: list[int] = []
        if self.tgt_lang is not None:
            if self.tgt_lang not in self.language_tags:
                raise ValueError(
                    f"unsupported tgt_lang={self.tgt_lang!r}; "
                    f"available={sorted(self.language_tags)}"
                )
            prefix.append(self.language_tags[self.tgt_lang])
        return [*prefix, *token_ids_0, int(self.eos_token_id)]

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            special = set(self.all_special_ids)
            return [int(token_id in special) for token_id in token_ids_0]
        if token_ids_1 is not None:
            raise ValueError("Sion tokenizer does not accept paired sentence inputs")
        prefix = [1] if self.tgt_lang is not None else []
        return [*prefix, *([0] * len(token_ids_0)), 1]

    def _build_translation_inputs(
        self,
        raw_inputs: str | list[str],
        return_tensors: str,
        src_lang: str | None,
        tgt_lang: str | None,
        **kwargs: Any,
    ):
        if src_lang is None or tgt_lang is None:
            raise ValueError("src_lang and tgt_lang are required for translation")
        if src_lang not in self.language_tags or tgt_lang not in self.language_tags:
            raise ValueError(
                f"unsupported translation direction {src_lang}-{tgt_lang}; "
                f"available={sorted(self.language_tags)}"
            )
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **kwargs)

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str]:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_prefix}-tokenizer.model" if filename_prefix else "tokenizer.model"
        destination = directory / filename
        if Path(self.vocab_file).resolve() != destination.resolve():
            shutil.copyfile(self.vocab_file, destination)
        return (str(destination),)
