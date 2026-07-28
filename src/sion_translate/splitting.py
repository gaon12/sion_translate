from __future__ import annotations

import hashlib
import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")


def normalized_split_key(text: str) -> str:
    """Return the shared compatibility-normalized key used for split grouping."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).strip())


def choose_split_for_key(
    key: str,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
) -> str:
    """Assign a normalized source to a deterministic shuffled split.

    SHA-256 makes the assignment independent of input-file and row order while
    keeping every occurrence of the same source text in one split.
    """
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Validation and test fractions must be non-negative")
    if validation_fraction + test_fraction >= 0.5:
        raise ValueError("Validation and test fractions are unexpectedly large")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def choose_split_for_text(
    text: str,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
) -> str:
    return choose_split_for_key(normalized_split_key(text), validation_fraction, test_fraction)


class BloomFilter:
    """Compact deterministic membership filter for split leakage guards."""

    def __init__(self, bit_count: int, hash_count: int = 5):
        if bit_count <= 0 or bit_count & (bit_count - 1):
            raise ValueError("Bloom filter bit_count must be a power of two")
        self.bits = bytearray(bit_count // 8)
        self.mask = bit_count - 1
        self.hash_count = hash_count

    def _positions(self, digest: bytes):
        first = int.from_bytes(digest[:8], "little")
        second = int.from_bytes(digest[8:16], "little") | 1
        for index in range(self.hash_count):
            yield (first + index * second) & self.mask

    def add(self, digest: bytes) -> None:
        for position in self._positions(digest):
            self.bits[position >> 3] |= 1 << (position & 7)

    def contains(self, digest: bytes) -> bool:
        return all(
            self.bits[position >> 3] & (1 << (position & 7)) for position in self._positions(digest)
        )


class TargetSplitGuard:
    """Prevent one normalized target surface from crossing data splits."""

    def __init__(
        self,
        estimated_pairs: int,
        validation_fraction: float,
        test_fraction: float,
    ):
        def bit_count(capacity: int, bits_per_item: int, minimum: int) -> int:
            requested = max(minimum, capacity * bits_per_item)
            return 1 << (requested - 1).bit_length()

        train_capacity = max(
            1, round(estimated_pairs * (1.0 - validation_fraction - test_fraction))
        )
        validation_capacity = max(1, round(estimated_pairs * validation_fraction))
        test_capacity = max(1, round(estimated_pairs * test_fraction))
        self.filters = {
            "train": BloomFilter(bit_count(train_capacity, 24, 1 << 16)),
            "validation": BloomFilter(bit_count(validation_capacity, 32, 1 << 14)),
            "test": BloomFilter(bit_count(test_capacity, 32, 1 << 14)),
        }

    def accept(self, split: str, digest: bytes) -> bool:
        if any(
            other != split and membership.contains(digest)
            for other, membership in self.filters.items()
        ):
            return False
        self.filters[split].add(digest)
        return True


__all__ = [
    "choose_split_for_key",
    "choose_split_for_text",
    "normalized_split_key",
    "BloomFilter",
    "TargetSplitGuard",
]
