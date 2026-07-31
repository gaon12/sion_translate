from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable


_WHITESPACE = re.compile(r"\s+")


SHINGLE_SIZE = 5
# One min-wise hash, chosen by measurement rather than by the textbook default.
# See ``minhash_signature`` for the recall table.
SIGNATURE_LENGTH = 1
_MINHASH_SEED = b"sion-minhash-v1"
_MINHASH_MASK = (1 << 64) - 1


def normalized_split_key(text: str) -> str:
    """Return the shared compatibility-normalized key used for split grouping."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).strip())


def character_shingles(text: str, size: int = SHINGLE_SIZE) -> list[str]:
    """Character n-grams of the normalized text, whitespace removed.

    Whitespace is dropped so that a reflowed or differently spaced copy of the
    same sentence produces the same shingles.
    """

    if size < 1:
        raise ValueError("shingle size must be positive")
    compact = normalized_split_key(text).replace(" ", "")
    if len(compact) <= size:
        return [compact] if compact else []
    return [compact[index : index + size] for index in range(len(compact) - size + 1)]


def minhash_signature(text: str, *, num_perm: int = SIGNATURE_LENGTH, size: int = SHINGLE_SIZE):
    """``num_perm`` independent min-wise hashes over the text's shingles.

    Banded LSH with a union-find would be the textbook construction, but the
    preparation pipeline assigns a split from one key per row with no cross-row
    state, so the key has to be self-contained. Concatenating ``num_perm``
    min-wise hashes gives exactly that: two texts share a key with probability
    ``J ** num_perm`` for Jaccard similarity ``J``.

    ``num_perm`` therefore trades recall against how often unrelated rows are
    grouped. The default is 1, which was picked from measurement on this corpus
    rather than from the usual "more permutations are better" instinct. Recall
    over real near-duplicate pairs mined from data29 + data33 (351,422 sources),
    against the rate at which pairs with ``J < 0.2`` are grouped:

        num_perm  J>=0.95  J.85-.95  J.70-.85  J.50-.70  unrelated
               1    96.1%     91.5%     79.5%     48.0%      0.00%
               2    91.3%     81.2%     67.0%     25.2%      0.00%
               4    88.4%     66.2%     41.5%      6.2%      0.00%
               8    76.1%     43.5%     15.0%      0.8%      0.00%

    Unrelated Korean sentences essentially never share their minimum shingle
    hash, so a single permutation costs nothing in precision, and the split
    proportions stay on target: 300,000 data29 rows land 99.02 / 0.45 / 0.53
    against a requested 99.0 / 0.5 / 0.5.

    Note what this does not do. Template families differ by a whole quoted span,
    which puts them near ``J = 0.5`` where even one permutation groups them only
    half the time. Capping frame and quoted-span reuse
    (``scripts/data/resample_generated_shards.py``) is the tool for those; this
    is the tool for genuine near-duplicates.
    """

    if num_perm < 1:
        raise ValueError("num_perm must be positive")
    shingles = character_shingles(text, size)
    if not shingles:
        return (0,) * num_perm
    encoded = [shingle.encode("utf-8") for shingle in shingles]
    signature = []
    for index in range(num_perm):
        key = hashlib.blake2b(index.to_bytes(4, "big"), digest_size=16, key=_MINHASH_SEED).digest()
        signature.append(
            min(
                int.from_bytes(hashlib.blake2b(shingle, digest_size=8, key=key).digest(), "big")
                & _MINHASH_MASK
                for shingle in encoded
            )
        )
    return tuple(signature)


def approximate_split_key(text: str, *, num_perm: int = SIGNATURE_LENGTH) -> str:
    """A split key that groups near-duplicates, not only exact duplicates.

    Very short texts have no reliable shingle signature, so they keep the exact
    normalized key.
    """

    normalized = normalized_split_key(text)
    if len(normalized.replace(" ", "")) <= SHINGLE_SIZE:
        return f"exact\0{normalized}"
    signature = minhash_signature(text, num_perm=num_perm)
    return "minhash\0" + "".join(f"{value:016x}" for value in signature)


def endpoint_split_key(
    language: str,
    text: str,
    *,
    approximate: bool = False,
) -> str:
    """Return a language-scoped key for one parallel-text endpoint.

    A surface may appear as either the source or target of different language
    pairs.  Prefixing every endpoint, including two-language corpora, makes
    those appearances comparable without conflating identical spelling in
    different languages.
    """

    text_key = approximate_split_key(text) if approximate else normalized_split_key(text)
    return f"{language}\0{text_key}"


def endpoint_split_digest(
    language: str,
    text: str,
    *,
    approximate: bool = False,
) -> bytes:
    """Return the stable SHA-256 digest used by endpoint leakage guards."""

    key = endpoint_split_key(language, text, approximate=approximate)
    return hashlib.sha256(key.encode("utf-8")).digest()


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
    """Prevent one normalized language endpoint from crossing data splits.

    The historical class name remains public for compatibility.  Callers now
    register both source and target endpoint digests.
    """

    def __init__(
        self,
        estimated_pairs: int,
        validation_fraction: float,
        test_fraction: float,
    ):
        def bit_count(capacity: int, bits_per_item: int, minimum: int) -> int:
            requested = max(minimum, capacity * bits_per_item)
            return 1 << (requested - 1).bit_length()

        # Every accepted pair now registers both language-scoped endpoints.
        # Size the filters for endpoint count so the v2 guard retains the same
        # false-positive budget as the historical target-only guard.
        estimated_endpoints = estimated_pairs * 2
        train_capacity = max(
            1,
            round(estimated_endpoints * (1.0 - validation_fraction - test_fraction)),
        )
        validation_capacity = max(1, round(estimated_endpoints * validation_fraction))
        test_capacity = max(1, round(estimated_endpoints * test_fraction))
        self.filters = {
            "train": BloomFilter(bit_count(train_capacity, 24, 1 << 16)),
            "validation": BloomFilter(bit_count(validation_capacity, 32, 1 << 14)),
            "test": BloomFilter(bit_count(test_capacity, 32, 1 << 14)),
        }

    def accept_many(self, split: str, digests: Iterable[bytes]) -> bool:
        """Atomically register all ``digests`` in ``split`` when conflict-free."""

        destination = self.filters[split]
        unique_digests = tuple(dict.fromkeys(digests))
        if any(
            membership.contains(digest)
            for other, membership in self.filters.items()
            if other != split
            for digest in unique_digests
        ):
            return False
        for digest in unique_digests:
            destination.add(digest)
        return True

    def accept(self, split: str, digest: bytes) -> bool:
        return self.accept_many(split, (digest,))


__all__ = [
    "SHINGLE_SIZE",
    "approximate_split_key",
    "character_shingles",
    "choose_split_for_key",
    "choose_split_for_text",
    "endpoint_split_digest",
    "endpoint_split_key",
    "minhash_signature",
    "normalized_split_key",
    "BloomFilter",
    "TargetSplitGuard",
]
