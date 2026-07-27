"""SentencePiece tokenizer exposed through the Transformers tokenizer API."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import sentencepiece as spm
from transformers import PreTrainedTokenizer


class SionTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        **kwargs: Any,
    ):
        self.vocab_file = str(vocab_file)
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
        base_special_tokens = {"<pad>", "<unk>", "<s>", "</s>"}
        additional = [
            piece
            for token_id in range(self.sp_model.vocab_size())
            if (piece := self.sp_model.id_to_piece(token_id)).startswith("<")
            and piece.endswith(">")
            and piece not in base_special_tokens
        ]
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("bos_token", "<s>")
        kwargs.setdefault("eos_token", "</s>")
        kwargs.setdefault("additional_special_tokens", additional)
        super().__init__(**kwargs)

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
