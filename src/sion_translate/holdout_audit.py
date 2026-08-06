"""challenge 문장이 학습 코퍼스에 이미 있는지 감사한다.

기존 누출 방지는 **seed 내부**에서만 동작합니다. 검토한 30쌍을 train 18 /
challenge 12 로 나눌 때 그 안에서 완전일치 중복을 막을 뿐, 12개 challenge
문장이 897만 행짜리 원천 코퍼스에 이미 있는지는 아무도 확인하지 않았습니다.
`세 살 버릇 여든까지 간다` 같은 표현은 말뭉치에 있을 가능성이 높습니다.

## 왜 Jaccard 도 MinHash 도 쓰지 않는가

둘 다 시도했고 둘 다 이 문제에 맞지 않았습니다. 실측입니다.

    사례                        J5     J3     C5     C3
    누출(글자 삽입)             0.32   0.53   0.60   0.83
    누출(더 긴 문장에 포함)      0.39   0.45   0.88   0.90
    누출(완전일치)              1.00   1.00   1.00   1.00
    무관한 문장                 0.00   0.00   0.00   0.00
    주제만 비슷한 문장           0.00   0.00   0.00   0.00

**Jaccard 는 짧은 문장에서 무너집니다.** challenge 문장은 관용구라 10~15자인데,
글자 하나가 삽입되면 그 자리를 지나는 5-gram 다섯 개가 한꺼번에 깨집니다.
그리고 누출의 전형적인 모습은 "같은 길이의 비슷한 문장"이 아니라 **더 긴
문장 안에 관용구가 들어 있는 것**이라, 분모에 상대방 길이가 들어가는 Jaccard
자체가 잘못된 질문입니다.

그래서 **포함도(containment)** 를 씁니다: challenge 의 3-gram 중 몇 %가 코퍼스
행에 들어 있는가. 위 표에서 누출은 0.83 이상, 무관은 0.00 으로 갈립니다.

MinHash 버킷도 버렸습니다. ``num_perm=1`` 의 충돌 확률은 Jaccard 와 같은데,
위 사례의 J5 가 0.32 라 셋 중 둘은 후보에도 못 듭니다. 대신 challenge 쪽
3-gram 역색인을 만듭니다 — challenge 는 수십 개뿐이라 색인이 작고, 코퍼스
행마다 자기 3-gram 을 조회하기만 하면 겹치는 항목이 **빠짐없이** 나옵니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from sion_translate.data.quality import canonical_text
from sion_translate.splitting import character_shingles, normalized_split_key

# challenge 문장의 3-gram 중 코퍼스 행에 들어 있는 비율의 하한. 위 표에서
# 누출은 0.83 이상, 무관·주제유사는 0.00 이라 그 사이 어디든 되지만, 부분
# 인용도 잡도록 0.6 으로 둡니다.
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# 짧은 관용구를 다루므로 5 가 아니라 3 입니다. 5 는 10~15자 문장에서 조각이
# 너무 적게 나와 글자 하나에도 크게 흔들립니다.
SHINGLE_SIZE = 3


@dataclass(frozen=True)
class HoldoutItem:
    """감사 대상 challenge 문장 하나."""

    identifier: str
    language: str
    text: str
    category: str = ""


@dataclass
class LeakMatch:
    """학습 코퍼스에서 발견된 유사 행."""

    file: str
    line: int
    text: str
    similarity: float
    exact: bool


@dataclass
class HoldoutFinding:
    item: HoldoutItem
    matches: list[LeakMatch] = field(default_factory=list)

    @property
    def leaked(self) -> bool:
        return bool(self.matches)

    @property
    def worst(self) -> LeakMatch | None:
        return max(self.matches, key=lambda match: match.similarity, default=None)


def load_holdout_items(
    paths: Sequence[str | Path],
    *,
    languages: Sequence[str] = ("ko", "ja"),
) -> list[HoldoutItem]:
    """challenge JSONL 에서 원문과 정답을 **양쪽 다** 감사 대상으로 삼는다.

    한쪽만 보면 안 됩니다. 정답 쪽이 코퍼스에 있으면 모델은 그 문장을 생성해
    본 적이 있는 것이고, 그것도 누출입니다.
    """

    items: list[HoldoutItem] = []
    allowed = set(languages)
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            identifier = str(row.get("id") or f"{path.name}:{line_number}")
            category = str(row.get("category", ""))
            for field_name, language_field in (
                ("source", "source_language"),
                ("reference", "target_language"),
            ):
                text = row.get(field_name)
                language = str(row.get(language_field, ""))
                if not isinstance(text, str) or not text.strip() or language not in allowed:
                    continue
                items.append(
                    HoldoutItem(
                        identifier=f"{identifier}#{field_name}",
                        language=language,
                        text=canonical_text(text),
                        category=category,
                    )
                )
    return items


def shingles(text: str) -> set[str]:
    return set(character_shingles(text, size=SHINGLE_SIZE))


def containment(holdout_text: str, corpus_text: str) -> float:
    """challenge 의 3-gram 중 코퍼스 행이 담고 있는 비율.

    Jaccard 가 아닌 이유는 분모입니다. 누출의 전형은 긴 문장 안에 관용구가
    들어 있는 것이라, 상대 문장이 길다는 이유로 점수가 떨어지면 안 됩니다.
    """

    holdout_shingles = shingles(holdout_text)
    if not holdout_shingles:
        return 0.0
    return len(holdout_shingles & shingles(corpus_text)) / len(holdout_shingles)


def iter_corpus_texts(
    paths: Iterable[Path],
    *,
    languages: Sequence[str] = ("ko", "ja"),
) -> Iterator[tuple[Path, int, str, str]]:
    """``(파일, 행 번호, 언어, 텍스트)`` 를 낸다. 언어별 필드만 봅니다."""

    allowed = tuple(languages)
    for path in paths:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                try:
                    row = json.loads(raw_line.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                for language in allowed:
                    value = row.get(language)
                    if isinstance(value, str) and value.strip():
                        yield path, line_number, language, value


def audit_holdout_leakage(
    items: Sequence[HoldoutItem],
    corpus_paths: Sequence[Path],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    maximum_matches_per_item: int = 5,
    languages: Sequence[str] = ("ko", "ja"),
) -> list[HoldoutFinding]:
    """challenge 문장이 학습 코퍼스에 있는지 전량 스캔으로 확인한다."""

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if not items:
        raise ValueError("감사할 challenge 문장이 없습니다")

    findings = {item.identifier: HoldoutFinding(item=item) for item in items}
    # challenge 3-gram 역색인. challenge 는 수십 개뿐이라 색인이 작고, 코퍼스
    # 행마다 자기 3-gram 을 조회하면 겹치는 항목이 빠짐없이 나옵니다.
    index: dict[tuple[str, str], set[str]] = {}
    for item in items:
        for shingle in shingles(item.text):
            index.setdefault((item.language, shingle), set()).add(item.identifier)

    for path, line_number, language, raw_text in iter_corpus_texts(
        corpus_paths, languages=languages
    ):
        text = canonical_text(raw_text)
        candidate_ids: set[str] = set()
        for shingle in shingles(text):
            candidate_ids |= index.get((language, shingle), frozenset())
        for identifier in candidate_ids:
            finding = findings[identifier]
            candidate = finding.item
            similarity = containment(candidate.text, text)
            if similarity < similarity_threshold:
                continue
            finding.matches.append(
                LeakMatch(
                    file=str(path),
                    line=line_number,
                    text=text,
                    similarity=similarity,
                    exact=normalized_split_key(text) == normalized_split_key(candidate.text),
                )
            )
            # 상한은 **가장 나쁜** N 개를 남깁니다. 스캔 순서대로 앞의 N 개를
            # 남기면 더 심한 누출이 뒤 파일에 있을 때 그것을 버립니다 —
            # 실제로 `호랑이도 제 말 하면 온다더니.` 는 data12 에서 0.91,
            # data9 에서 1.00 인데 문자열 정렬상 data12 가 먼저라 1.00 이
            # 상한에 걸려 사라졌습니다. 안전 관문이 누출을 과소보고하는
            # 방향이라 허용할 수 없습니다.
            if len(finding.matches) > maximum_matches_per_item:
                finding.matches.sort(key=lambda match: -match.similarity)
                del finding.matches[maximum_matches_per_item:]
    for finding in findings.values():
        finding.matches.sort(key=lambda match: -match.similarity)
    return list(findings.values())


def summarize(findings: Sequence[HoldoutFinding]) -> dict[str, object]:
    leaked = [finding for finding in findings if finding.leaked]
    exact_hits = [finding for finding in leaked if any(match.exact for match in finding.matches)]
    by_category: dict[str, int] = {}
    for finding in leaked:
        by_category[finding.item.category] = by_category.get(finding.item.category, 0) + 1
    return {
        "audited_items": len(findings),
        "leaked_items": len(leaked),
        "exact_leaked_items": len(exact_hits),
        "leak_rate": (len(leaked) / len(findings)) if findings else 0.0,
        "by_category": by_category,
        "note": (
            "누출된 항목은 독립 holdout 이 아닙니다. 회귀 smoke set 으로는 "
            "쓸 수 있지만 품질 benchmark 로 인용하면 안 됩니다."
        ),
    }


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "HoldoutFinding",
    "HoldoutItem",
    "LeakMatch",
    "audit_holdout_leakage",
    "iter_corpus_texts",
    "containment",
    "shingles",
    "load_holdout_items",
    "summarize",
]
