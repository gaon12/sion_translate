"""Compatibility typing for the SentencePiece 0.2.1 runtime.

SentencePiece first ships an official ``pyi`` in 0.2.2, but that release has a
native multithreaded trainer regression.  Keep the public surface used by Sion
typed while the runtime remains on the safe 0.2.1 release.
"""

from collections.abc import Sequence
from typing import Any, overload

__version__: str

class SentencePieceProcessor:
    def __init__(
        self,
        model_file: str | None = ...,
        model_proto: bytes | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def pad_id(self) -> int: ...
    def unk_id(self) -> int: ...
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def vocab_size(self) -> int: ...
    def piece_size(self) -> int: ...
    def id_to_piece(self, token_id: int) -> str: ...
    def piece_to_id(self, piece: str) -> int: ...
    @overload
    def encode(
        self,
        input: str | bytes,
        out_type: type[int] = ...,
        **kwargs: Any,
    ) -> list[int]: ...
    @overload
    def encode(
        self,
        input: str | bytes,
        out_type: type[str],
        **kwargs: Any,
    ) -> list[str]: ...
    @overload
    def encode(
        self,
        input: Sequence[str] | Sequence[bytes],
        out_type: type[int] = ...,
        **kwargs: Any,
    ) -> list[list[int]]: ...
    @overload
    def encode(
        self,
        input: Sequence[str] | Sequence[bytes],
        out_type: type[str],
        **kwargs: Any,
    ) -> list[list[str]]: ...
    @overload
    def decode(
        self,
        input: int | str | bytes | Sequence[int] | Sequence[str] | Sequence[bytes],
        **kwargs: Any,
    ) -> str: ...
    @overload
    def decode(self, input: Sequence[Sequence[int]], **kwargs: Any) -> list[str]: ...

class SentencePieceTrainer:
    @staticmethod
    def train(arg: str | None = ..., **kwargs: Any) -> None: ...
    @staticmethod
    def Train(arg: str | None = ..., **kwargs: Any) -> None: ...
