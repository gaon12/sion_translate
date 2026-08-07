from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, Sequence, TypedDict, cast

import numpy as np
import torch

from sion_translate.tokenizer import SionTokenizer


_TOKEN_FEATURE_NAMES = ("script", "onset", "vowel", "coda")
_TOKEN_FEATURE_CLASS_COUNTS = {
    "onset": 20,
    "vowel": 22,
    "coda": 29,
}


class _CollatorItem(TypedDict):
    src: Sequence[int]
    tgt: Sequence[int]
    src_language: str
    target_language: str
    src_register: int
    target_register: int
    pair_index: NotRequired[int]
    augmentation_key: NotRequired[int]
    reverse_direction_trained: NotRequired[bool]


class _CollatedExample(TypedDict):
    input_ids: list[int]
    decoder_input_ids: list[int]
    labels: list[int]
    register_label: int
    memory_tokens: list[int]
    source_language_tag_id: int
    reverse_direction_trained: bool


def load_morphoscript_token_features(
    path: str | Path,
    *,
    vocab_size: int,
    script_classes: int,
) -> dict[str, torch.Tensor]:
    """Load and validate token-indexed MorphoScript feature tables."""

    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"MorphoScript is enabled, but its token feature file does not exist: {feature_path}"
        )

    maximum_ids = {
        "script": script_classes,
        **_TOKEN_FEATURE_CLASS_COUNTS,
    }
    features: dict[str, torch.Tensor] = {}
    with np.load(feature_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(_TOKEN_FEATURE_NAMES):
            raise ValueError(
                "token feature file must contain exactly "
                f"{', '.join(_TOKEN_FEATURE_NAMES)}; got {sorted(loaded.files)}"
            )
        for name in _TOKEN_FEATURE_NAMES:
            values = np.asarray(loaded[name])
            expected_shape = (vocab_size,)
            if values.shape != expected_shape:
                raise ValueError(
                    f"token feature {name} has shape {values.shape}; expected {expected_shape}"
                )
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"token feature {name} must use an integer dtype")
            values = values.astype(np.int64, copy=True)
            maximum_id = maximum_ids[name]
            if values.size and (int(values.min()) < 0 or int(values.max()) >= maximum_id):
                raise ValueError(f"token feature {name} contains IDs outside [0, {maximum_id})")
            features[name] = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                values
            )
    return features


def corrupt_spans(
    token_ids: Sequence[int],
    mask_id: int,
    *,
    noise_density: float,
    mean_span: float,
    rng: random.Random | None = None,
) -> list[int]:
    ids = list(map(int, token_ids))
    if len(ids) < 2 or noise_density <= 0:
        return ids
    generator = rng if rng is not None else random.Random()
    target = max(1, min(len(ids) - 1, round(len(ids) * noise_density)))
    masked: set[int] = set()
    attempts = 0
    while len(masked) < target and attempts < target * 10:
        attempts += 1
        start = generator.randrange(len(ids))
        span = max(1, int(generator.expovariate(1.0 / max(mean_span, 1e-3))))
        for position in range(start, min(len(ids), start + span)):
            masked.add(position)
            if len(masked) >= target:
                break

    output: list[int] = []
    previous_masked = False
    for position, token_id in enumerate(ids):
        is_masked = position in masked
        if is_masked and not previous_masked:
            output.append(mask_id)
        elif not is_masked:
            output.append(token_id)
        previous_masked = is_masked
    return output


class SionBatchCollator:
    def __init__(
        self,
        tokenizer: SionTokenizer,
        *,
        max_source_length: int,
        max_target_length: int,
        pad_to_multiple_of: int = 1,
        denoise_probability: float = 0.0,
        denoise_noise_density: float = 0.15,
        denoise_mean_span: float = 3.0,
        source_only_languages: Sequence[str] = (),
        source_token_dropout: float = 0.0,
        decoder_input_noise: float = 0.0,
        augmentation_seed: int = 0,
        augmentation_key: int = 0,
        token_features: str | Path | None = None,
        script_classes: int = 9,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        if pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be at least 1")
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.denoise_probability = denoise_probability
        self.denoise_noise_density = denoise_noise_density
        self.denoise_mean_span = denoise_mean_span
        # Source-only varieties (for example mixed-language ``kj`` or dialect
        # ``kd``/``jd``) are defined only as inputs to a translation edge.  A
        # denoising sample would instead put that variety on the label side and
        # teach the decoder to generate exactly the output the data contract
        # forbids.  Keep the empty default for ordinary bilingual callers.
        self.source_only_languages = frozenset(map(str, source_only_languages))
        # 온라인 증강: 원문 토큰을 이 확률로 무작위 탈락시켜 모델이 일부
        # 단어가 빠진 입력에도 견고해지게 합니다. 학습 collator 에만 적용하고
        # 검증 collator 에는 0 을 넣어야 합니다.
        self.source_token_dropout = source_token_dropout
        # Exposure bias mitigation for the supervised stage. Teacher forcing only
        # ever shows the decoder gold prefixes, so at inference the first error
        # puts it in a state it never trained on and the errors compound. Post
        # training (MRT) samples from the model and does address this, but it runs
        # after the fact and for a few thousand steps; the main run has no such
        # mechanism at all.
        #
        # Replacing a fraction of the decoder *input* tokens makes the model
        # recover from a wrong prefix. The labels stay clean - the target it must
        # produce never changes - so this is teacher forcing over a noisy prefix,
        # not a change of objective. Unlike scheduled sampling it needs no second
        # forward pass.
        #
        # Defaults to 0. It is off until someone measures it: this changes what
        # the decoder conditions on, and turning it on untested before a
        # from-scratch run would be a guess.
        self.decoder_input_noise = decoder_input_noise
        self.augmentation_seed = int(augmentation_seed)
        # shared-memory scalar라 persistent DataLoader worker에도 epoch 변경이
        # 전달됩니다. 각 샘플은 이 값과 pair identity로 독립 RNG를 만듭니다.
        self._augmentation_key: torch.Tensor = torch.tensor(
            int(augmentation_key), dtype=torch.int64
        ).share_memory_()
        self.slot_ids: set[int] = set(tokenizer.slot_ids)
        self.features: dict[str, torch.Tensor] | None = None
        if token_features is not None:
            self.features = load_morphoscript_token_features(
                token_features,
                vocab_size=len(tokenizer),
                script_classes=script_classes,
            )

    @property
    def augmentation_key(self) -> int:
        return int(self._augmentation_key.item())

    def set_augmentation_key(self, augmentation_key: int) -> None:
        self._augmentation_key.fill_(int(augmentation_key))

    def set_epoch(self, epoch: int) -> None:
        """Use the epoch as the deterministic augmentation stream key."""

        self.set_augmentation_key(epoch)

    def _sample_rng(self, item: _CollatorItem) -> random.Random:
        pair_index = int(item.get("pair_index", 0))
        augmentation_key = item.get("augmentation_key", self.augmentation_key)
        identity = "|".join(
            (
                str(self.augmentation_seed),
                str(pair_index),
                str(item["src_language"]),
                str(item["target_language"]),
                str(augmentation_key),
            )
        )
        digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=16).digest()
        return random.Random(int.from_bytes(digest, byteorder="big"))

    def _make_example(self, item: _CollatorItem) -> _CollatedExample:
        src = list(map(int, item["src"]))
        tgt = list(map(int, item["tgt"]))
        target_register = int(item["target_register"])
        rng = self._sample_rng(item)
        denoise = (
            item["src_language"] not in self.source_only_languages
            and self.denoise_probability > 0
            and rng.random() < self.denoise_probability
        )
        if denoise:
            # Truncate before deriving the target. The two sides are cut to
            # different limits further down, so restoring an un-truncated
            # source would ask the model to reconstruct tokens the (truncated)
            # input never contained.
            src = src[: self.max_source_length - 2]
            original = src
            src = corrupt_spans(
                original,
                self.tokenizer.mask_id,
                noise_density=self.denoise_noise_density,
                mean_span=self.denoise_mean_span,
                rng=rng,
            )
            tgt = original
            # <denoise_xx>: 원문 언어에 맞는 복원 과제 태그
            source_language = item["src_language"]
            task_id = self.tokenizer.denoise_tags[source_language]
            target_register = int(item["src_register"])
            # 단일언어 복원 과제에는 번역 방향이 없으므로 순환 번역 보상을
            # 적용하지 않습니다.
            source_language_tag_id = -1
            reverse_direction_trained = False
        else:
            # <2xx>: 목표 언어를 지정하는 방향 태그 (양방향 학습의 핵심)
            target_language = item["target_language"]
            source_language = item["src_language"]
            task_id = self.tokenizer.language_tags[target_language]
            source_language_tag_id = self.tokenizer.language_tags[source_language]
            # Missing graph metadata is treated conservatively: an unknown
            # reverse edge must not receive a roundtrip reward.
            reverse_direction_trained = bool(item.get("reverse_direction_trained", False))
            if self.source_token_dropout > 0:
                # 온라인 증강: 보호 슬롯(<slot_n>)은 남기고, 일반 토큰만
                # 낮은 확률로 탈락. 최소 1개 토큰은 반드시 남깁니다.
                kept = [
                    token_id
                    for token_id in src
                    if token_id in self.slot_ids or rng.random() >= self.source_token_dropout
                ]
                if kept:
                    src = kept

        src = src[: self.max_source_length - 2]
        tgt = tgt[: self.max_target_length - 1]
        input_ids = [task_id, *src, self.tokenizer.eos_id]
        decoder_input_ids = [self.tokenizer.bos_id, *self._noise_decoder_input(tgt, rng)]
        labels = [*tgt, self.tokenizer.eos_id]
        memory_tokens = [token_id for token_id in src if token_id in self.slot_ids]
        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "register_label": target_register,
            "memory_tokens": memory_tokens[:64],
            "source_language_tag_id": source_language_tag_id,
            "reverse_direction_trained": reverse_direction_trained,
        }

    def _noise_decoder_input(self, target: Sequence[int], rng: random.Random) -> list[int]:
        """Corrupt a fraction of the decoder input, leaving the labels alone.

        Protected slots keep their surface, because the point of a slot is that
        it survives verbatim; corrupting one would train the model to break the
        guarantee the slot exists to make.
        """

        if self.decoder_input_noise <= 0:
            return list(target)
        vocab_size_value = cast(object, getattr(self.tokenizer, "vocab_size", None))
        vocab_size = vocab_size_value if isinstance(vocab_size_value, int) else len(self.tokenizer)
        noised: list[int] = []
        for token_id in target:
            if token_id in self.slot_ids or rng.random() >= self.decoder_input_noise:
                noised.append(token_id)
                continue
            # Half the time show <mask>, which says "something was here";
            # half the time show a wrong token, which is what a real decoding
            # error looks like. Only the second teaches recovery from a
            # confident mistake, and only the first is a clean signal.
            if rng.random() < 0.5:
                noised.append(self.tokenizer.mask_id)
            else:
                noised.append(rng.randrange(vocab_size))
        return noised

    def __call__(self, items: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        examples = [self._make_example(cast(_CollatorItem, item)) for item in items]
        batch_size = len(examples)
        src_len = max(len(example["input_ids"]) for example in examples)
        tgt_len = max(len(example["decoder_input_ids"]) for example in examples)
        if self.pad_to_multiple_of > 1:
            src_len = min(
                self.max_source_length,
                max(
                    src_len,
                    ((src_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of)
                    * self.pad_to_multiple_of,
                ),
            )
            tgt_len = min(
                self.max_target_length,
                max(
                    tgt_len,
                    ((tgt_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of)
                    * self.pad_to_multiple_of,
                ),
            )
        memory_len = max(1, max(len(example["memory_tokens"]) for example in examples))

        input_ids = torch.full((batch_size, src_len), self.tokenizer.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, src_len), dtype=torch.bool)
        decoder_input_ids = torch.full(
            (batch_size, tgt_len), self.tokenizer.pad_id, dtype=torch.long
        )
        labels = torch.full((batch_size, tgt_len), -100, dtype=torch.long)
        register_labels = torch.zeros(batch_size, dtype=torch.long)
        source_language_tag_ids = torch.full((batch_size,), -1, dtype=torch.long)
        reverse_direction_trained = torch.zeros(batch_size, dtype=torch.bool)
        memory_token_ids = torch.full(
            (batch_size, memory_len, 1), self.tokenizer.pad_id, dtype=torch.long
        )
        memory_mask = torch.zeros((batch_size, memory_len), dtype=torch.bool)
        memory_type_ids = torch.zeros((batch_size, memory_len), dtype=torch.long)
        memory_mode_ids = torch.zeros((batch_size, memory_len), dtype=torch.long)

        for row, example in enumerate(examples):
            source = torch.tensor(example["input_ids"], dtype=torch.long)
            target_input = torch.tensor(example["decoder_input_ids"], dtype=torch.long)
            target_label = torch.tensor(example["labels"], dtype=torch.long)
            input_ids[row, : len(source)] = source
            attention_mask[row, : len(source)] = True
            decoder_input_ids[row, : len(target_input)] = target_input
            labels[row, : len(target_label)] = target_label
            register_labels[row] = example["register_label"]
            source_language_tag_ids[row] = example["source_language_tag_id"]
            reverse_direction_trained[row] = example["reverse_direction_trained"]
            for column, token_id in enumerate(example["memory_tokens"]):
                memory_token_ids[row, column, 0] = token_id
                memory_mask[row, column] = True
                memory_type_ids[row, column] = 8  # CODE/URL/protected slot bucket
                memory_mode_ids[row, column] = 4  # PROTECT

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "register_labels": register_labels,
            # 학습 목적함수 전용 메타데이터입니다. 모델 forward는 이 값을
            # 사용하지 않고, MRT 순환 번역 rollout만 역방향 태그로 씁니다.
            "source_language_tag_ids": source_language_tag_ids,
            "reverse_direction_trained": reverse_direction_trained,
            "memory_token_ids": memory_token_ids,
            "memory_mask": memory_mask,
            "memory_type_ids": memory_type_ids,
            "memory_mode_ids": memory_mode_ids,
        }
        if self.features is not None:
            for source_name, target_name in (
                ("script", "src_script_ids"),
                ("onset", "src_onset_ids"),
                ("vowel", "src_vowel_ids"),
                ("coda", "src_coda_ids"),
            ):
                if source_name in self.features:
                    batch[target_name] = self.features[source_name][input_ids]
        return batch
