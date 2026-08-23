"""초안 수정(draft revision) 학습 데이터 생성.

``원문 + 초벌 번역 → 고친 번역`` 을 배우게 하면, 한 번에 맞히지 못한 문장을 두 번째
패스에서 고칠 기회가 생깁니다. 인코더-디코더 구조를 그대로 쓸 수 있고, 학습 데이터도
지금 있는 ``원문/번역`` 쌍만으로 만들 수 있습니다.

입력을 ``원문 <draft> 초안`` 한 줄로 직렬화하므로 기존 데이터 파이프라인이 이것을
평범한 번역쌍으로 취급합니다. 인덱싱·shard·collator 를 고칠 필요가 없습니다.

**초안은 어디서 오는가.** 학습된 모델로 뽑는 것이 이상적이지만, 그러면 모델이 있어야
데이터를 만들 수 있는 순환이 생깁니다. 여기서는 정답 번역을 일부러 망가뜨려 초안을
만듭니다. 손상 유형은 이 프로젝트 모델에서 실제로 관측된 오류를 모사합니다 —
그래야 수정 모델이 배우는 것이 실제로 고칠 필요가 있는 오류가 됩니다.

- ``number``: 숫자를 다른 값으로 바꿈 (250mg → 1200mg). 최대 결함이므로 기본
  가중치가 가장 높습니다. 원문에 정답이 남아 있으므로 원문을 봐야만 고칠 수 있습니다.
- ``drop_clause``: 절이나 문장 하나를 통째로 누락
- ``truncate``: 뒷부분을 잘라 조기 종료를 모사
- ``repeat``: 짧은 구절 반복(생성 붕괴)
- ``copy_source``: 번역하지 않고 원문을 그대로 둠
- ``swap``: 인접한 두 조각의 순서를 뒤바꿈 (정렬 붕괴)
- ``identity``: 손상 없음. 이미 맞은 초안을 **그대로 두는** 것도 배워야 하며,
  이것이 없으면 수정 모델이 멀쩡한 문장을 헛되게 고칩니다.
"""

# Revision rows are loaded from JSON and normalized immediately.
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeAlias

from sion_translate.data.record_metadata import RECORD_METADATA_FIELDS
from sion_translate.language_tags import canonicalize_language_pair

DRAFT_SEPARATOR = "<draft>"

# 손상 유형별 기본 비중. identity 를 넉넉히 두는 이유는 위 docstring 참고.
DEFAULT_CORRUPTIONS: dict[str, float] = {
    "number": 0.26,
    "drop_clause": 0.16,
    "truncate": 0.12,
    "repeat": 0.10,
    "copy_source": 0.06,
    "swap": 0.10,
    "identity": 0.20,
}

_NUMBER_RUN = re.compile(r"\d+")
# 절 경계로 쓸 구두점. 한국어·일본어 문장부호를 함께 봅니다.
_CLAUSE_SPLIT = re.compile(r"(?<=[。．.!?！？、,，])\s*")


@dataclass
class RevisionStats:
    """생성 결과 요약. 손상 유형별 개수를 기록합니다."""

    written: int
    by_corruption: dict[str, int]
    unchanged: int

    def as_dict(self) -> dict:
        return {
            "written": self.written,
            "unchanged_drafts": self.unchanged,
            "by_corruption": dict(sorted(self.by_corruption.items())),
        }


@dataclass(frozen=True, slots=True)
class RevisionExample:
    """A revision target with authenticated source-row annotations."""

    serialized_source: str
    target: str
    metadata: Mapping[str, object]
    source_identifier: str | None = None


RevisionOutput: TypeAlias = tuple[str, str] | RevisionExample


def serialize_revision_input(source: str, draft: str) -> str:
    """``원문 <draft> 초안`` 형태로 직렬화합니다."""
    return f"{source.strip()} {DRAFT_SEPARATOR} {draft.strip()}"


def parse_revision_input(text: str) -> tuple[str, str]:
    """직렬화된 입력을 (원문, 초안) 으로 되돌립니다."""
    if DRAFT_SEPARATOR not in text:
        raise ValueError(f"{DRAFT_SEPARATOR} 가 없습니다: {text[:60]}")
    source, _, draft = text.partition(DRAFT_SEPARATOR)
    return source.strip(), draft.strip()


def _clauses(text: str) -> list[str]:
    parts = [part for part in _CLAUSE_SPLIT.split(text) if part.strip()]
    return parts if len(parts) > 1 else []


def _corrupt_number(target: str, rng: random.Random) -> str:
    """숫자 하나를 그럴듯한 다른 값으로 바꿉니다."""
    normalized = unicodedata.normalize("NFKC", target)
    matches = list(_NUMBER_RUN.finditer(normalized))
    if not matches:
        return normalized
    match = rng.choice(matches)
    digits = match.group(0)
    # 자릿수를 유지하거나 한 자리 늘려, 실제 관측된 오류(250 -> 1200)와 비슷하게.
    if rng.random() < 0.5 and len(digits) > 1:
        replacement = list(digits)
        position = rng.randrange(len(digits))
        choices = [d for d in "0123456789" if d != digits[position]]
        replacement[position] = rng.choice(choices)
        new_digits = "".join(replacement)
    else:
        new_digits = str(rng.randint(1, 9)) + digits[: max(1, len(digits) - 1)]
    return normalized[: match.start()] + new_digits + normalized[match.end() :]


def _corrupt_drop_clause(target: str, rng: random.Random) -> str:
    parts = _clauses(target)
    if not parts:
        return target
    removed = rng.randrange(len(parts))
    return "".join(part for index, part in enumerate(parts) if index != removed).strip()


def _corrupt_truncate(target: str, rng: random.Random) -> str:
    if len(target) < 8:
        return target
    keep = rng.randint(len(target) // 4, max(len(target) // 4 + 1, (len(target) * 3) // 4))
    return target[:keep].strip()


def _corrupt_repeat(target: str, rng: random.Random) -> str:
    if len(target) < 6:
        return target
    span = min(len(target), rng.randint(2, 6))
    start = rng.randrange(max(1, len(target) - span))
    fragment = target[start : start + span]
    return target[: start + span] + fragment * rng.randint(4, 6) + target[start + span :]


def _corrupt_swap(target: str, rng: random.Random) -> str:
    parts = _clauses(target)
    if len(parts) < 2:
        return target
    position = rng.randrange(len(parts) - 1)
    parts[position], parts[position + 1] = parts[position + 1], parts[position]
    return "".join(parts).strip()


_CORRUPTIONS = {
    "number": _corrupt_number,
    "drop_clause": _corrupt_drop_clause,
    "truncate": _corrupt_truncate,
    "repeat": _corrupt_repeat,
    "swap": _corrupt_swap,
}


def corrupt_target(
    source: str,
    target: str,
    kind: str,
    rng: random.Random,
) -> str:
    """정답 번역을 ``kind`` 방식으로 망가뜨려 초안을 만듭니다."""
    if kind == "identity":
        return target
    if kind == "copy_source":
        return source
    if kind not in _CORRUPTIONS:
        raise ValueError(f"알 수 없는 손상 유형: {kind} (가능: {sorted(DEFAULT_CORRUPTIONS)})")
    return _CORRUPTIONS[kind](target, rng)


def build_revision_examples(
    pairs: Sequence[tuple[str, str]],
    *,
    weights: dict[str, float] | None = None,
    seed: int = 20260726,
) -> tuple[list[tuple[str, str]], RevisionStats]:
    """각 쌍에서 ``(원문 <draft> 초안, 정답 번역)`` 예제를 하나씩 만듭니다.

    손상이 실제로 아무 효과가 없었던 경우(숫자 없는 문장에 number 손상 등)는
    ``identity`` 와 같아지므로 ``unchanged`` 로 따로 셉니다. 버리지는 않습니다 —
    맞은 초안을 그대로 두는 것도 배워야 하는 동작이기 때문입니다.
    """
    weights = weights or DEFAULT_CORRUPTIONS
    unknown = set(weights) - set(DEFAULT_CORRUPTIONS)
    if unknown:
        raise ValueError(f"알 수 없는 손상 유형: {sorted(unknown)}")
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("손상 가중치의 합이 0보다 커야 합니다")

    kinds = list(weights)
    probabilities = [weights[kind] for kind in kinds]
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    histogram: dict[str, int] = {}
    unchanged = 0
    for source, target in pairs:
        kind = rng.choices(kinds, weights=probabilities, k=1)[0]
        draft = corrupt_target(source, target, kind, rng)
        if not draft.strip():
            # 전부 잘려 나간 초안은 수정할 단서가 없어 학습 신호가 되지 않습니다.
            draft = target
            kind = "identity"
        if draft.strip() == target.strip():
            unchanged += 1
        histogram[kind] = histogram.get(kind, 0) + 1
        examples.append((serialize_revision_input(source, draft), target))
    return examples, RevisionStats(len(examples), histogram, unchanged)


def write_revision_examples(
    output_path: str | Path,
    examples: Iterable[RevisionOutput],
    language_pair: Sequence[str],
) -> int:
    """``prepare_dataset`` 가 읽는 형식으로 씁니다.

    ``원문 <draft> 초안`` 이 그대로 원문 자리에 들어가므로, 데이터 파이프라인은
    이것을 평범한 번역쌍으로 처리합니다.
    """
    key_a, key_b = canonicalize_language_pair(
        language_pair,
        field="revision language_pair",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for example in examples:
                if isinstance(example, RevisionExample):
                    serialized, target = example.serialized_source, example.target
                    metadata = dict(example.metadata)
                    raw_direction = metadata.get("training_direction")
                    if raw_direction is not None:
                        input_direction = canonicalize_language_pair(
                            raw_direction,
                            field="revision input training_direction",
                        )
                        if input_direction != (key_a, key_b):
                            location = (
                                f" at {example.source_identifier}"
                                if example.source_identifier is not None
                                else ""
                            )
                            raise ValueError(
                                "revision input training_direction does not match the requested "
                                f"revision direction{location}: input={input_direction!r}, "
                                f"requested={(key_a, key_b)!r}"
                            )
                else:
                    serialized, target = example
                    metadata = {}
                row: dict[str, object] = {
                    key_a: serialized,
                    key_b: target,
                    "synthetic": True,
                    "training_direction": [key_a, key_b],
                }
                if isinstance(example, RevisionExample):
                    for field in RECORD_METADATA_FIELDS:
                        if field not in {"provenance", "training_direction"} and field in metadata:
                            row[field] = deepcopy(metadata[field])
                    provenance_input: dict[str, object] = {}
                    if example.source_identifier is not None:
                        provenance_input["source"] = example.source_identifier
                    if "provenance" in metadata:
                        provenance_input["provenance"] = deepcopy(metadata["provenance"])
                    if provenance_input:
                        row["provenance"] = {
                            "transformation": "revision",
                            "input": provenance_input,
                        }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return written
