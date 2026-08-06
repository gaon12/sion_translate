"""단일어 코퍼스 탐색과 읽기 (foundation 사전학습 입력).

번역 학습 이전 단계인 foundation 사전학습은 병렬 코퍼스가 아니라 언어별
단일어 텍스트를 씁니다. 배치는 언어 코드 폴더입니다::

    data/corpus/
      ko/
        kowiki_corpus.txt          # 한 줄에 문장/문단 하나
        news.jsonl                 # {"text": "..."} 한 줄에 하나
      ja/
        wiki.txt

허용 형식은 ``.txt`` 와 ``.jsonl`` 둘뿐입니다. 형식을 좁게 잡는 이유는 이
단계가 GPU 시간을 가장 많이 쓰는데 입력 오류는 조용하기 때문입니다 —
잘못된 키 이름 하나가 "그 파일만 0문장"으로 끝나고, 그 사실은 학습이 끝난
뒤에야 드러납니다. 여기서는 건너뛴 것을 전부 이유와 함께 돌려주고,
호출자가 학습 시작 전에 보고합니다.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

# 설정의 언어 키 규칙과 같습니다 (`AppConfig.validate` 참고): ASCII 영숫자,
# 첫 글자는 알파벳. 폴더 이름이 곧 언어 키이므로 규칙이 갈라지면 안 됩니다.
LANGUAGE_DIRECTORY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,15}$")

TEXT_SUFFIX = ".txt"
JSONL_SUFFIX = ".jsonl"
ALLOWED_SUFFIXES = (TEXT_SUFFIX, JSONL_SUFFIX)

JSONL_TEXT_KEY = "text"

DEFAULT_CORPUS_DIRECTORY = "data/corpus"

# 언어 간 온도 샘플링 지수. 1.0 이면 문장 수에 정비례하고, 낮출수록 균등에
# 가까워집니다. 언어별 단일어 코퍼스 규모는 병렬 코퍼스보다 훨씬 크게
# 어긋나므로(한국어 5.3 GB 대 일본어 0) shard 단위 기본값 0.9 보다 더 세게
# 눕힙니다.
DEFAULT_LANGUAGE_SAMPLING_ALPHA = 0.7


@dataclass(frozen=True)
class MonolingualSource:
    """foundation 학습에 실제로 들어가는 파일 하나."""

    language: str
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class SkippedEntry:
    """건너뛴 경로와 그 이유. 조용히 사라지는 입력을 없애기 위한 것입니다."""

    path: Path
    reason: str


@dataclass(frozen=True)
class MonolingualDiscovery:
    root: Path
    sources: tuple[MonolingualSource, ...] = ()
    skipped: tuple[SkippedEntry, ...] = ()
    # 설정에는 있는데 폴더가 없거나 비어 있는 언어.
    languages_without_data: tuple[str, ...] = ()
    # 설정에 없어서 아예 보지 않은 언어 폴더.
    unconfigured_languages: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.sources)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(source.language for source in self.sources))

    def paths_for(self, language: str) -> tuple[Path, ...]:
        return tuple(source.path for source in self.sources if source.language == language)

    def bytes_per_language(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for source in self.sources:
            totals[source.language] = totals.get(source.language, 0) + source.size_bytes
        return totals


def foundation_languages(
    languages: Sequence[str],
    source_only_languages: Sequence[str] = (),
) -> tuple[str, ...]:
    """foundation 학습 대상 언어.

    source-only 언어(혼용문 ``kj``, 방언 ``kd``/``jd``)는 제외합니다. 단일어
    복원 과제는 그 언어를 **디코더 출력**으로 만드는 학습인데, source-only
    는 "번역 결과로 나오면 안 되는 언어"라는 뜻입니다. 여기서 걸러 두지
    않으면 foundation 단계가 이후 번역 단계의 데이터 계약을 정면으로
    거스르는 것을 먼저 가르치게 됩니다.
    """

    excluded = frozenset(str(language) for language in source_only_languages)
    return tuple(
        language
        for language in dict.fromkeys(str(x) for x in languages)
        if language not in excluded
    )


def discover_monolingual_sources(
    root: str | Path,
    languages: Sequence[str],
) -> MonolingualDiscovery:
    """언어 코드 폴더를 훑어 학습 가능한 파일과 건너뛴 것을 함께 돌려준다."""

    root = Path(root)
    configured = tuple(dict.fromkeys(str(language) for language in languages))
    if not configured:
        raise ValueError("at least one language is required to discover monolingual sources")

    if not root.is_dir():
        return MonolingualDiscovery(
            root=root,
            skipped=(SkippedEntry(root, "코퍼스 폴더가 없습니다"),),
            languages_without_data=configured,
        )

    configured_set = set(configured)
    sources: list[MonolingualSource] = []
    skipped: list[SkippedEntry] = []
    unconfigured: list[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            skipped.append(SkippedEntry(entry, "언어 폴더가 아닌 최상위 파일"))
            continue
        if entry.name not in configured_set:
            reason = (
                f"설정에 없는 언어 폴더 (설정된 언어: {', '.join(configured)})"
                if LANGUAGE_DIRECTORY_PATTERN.match(entry.name)
                else "언어 코드 형식이 아닌 폴더 이름"
            )
            skipped.append(SkippedEntry(entry, reason))
            if LANGUAGE_DIRECTORY_PATTERN.match(entry.name):
                unconfigured.append(entry.name)
            continue
        found = False
        for candidate in sorted(entry.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
                skipped.append(
                    SkippedEntry(
                        candidate,
                        f"허용되지 않는 확장자 (허용: {', '.join(ALLOWED_SUFFIXES)})",
                    )
                )
                continue
            size = candidate.stat().st_size
            if size == 0:
                skipped.append(SkippedEntry(candidate, "빈 파일"))
                continue
            sources.append(MonolingualSource(entry.name, candidate, size))
            found = True
        if not found:
            skipped.append(SkippedEntry(entry, "읽을 수 있는 .txt/.jsonl 이 없습니다"))

    present = {source.language for source in sources}
    return MonolingualDiscovery(
        root=root,
        sources=tuple(sources),
        skipped=tuple(skipped),
        languages_without_data=tuple(x for x in configured if x not in present),
        unconfigured_languages=tuple(dict.fromkeys(unconfigured)),
    )


@dataclass
class ReadStats:
    """파일 하나를 읽으며 생긴 손실. 조용한 0문장을 막기 위한 집계입니다."""

    accepted: int = 0
    blank: int = 0
    malformed_json: int = 0
    missing_text_key: int = 0
    non_string_text: int = 0

    @property
    def rejected(self) -> int:
        return self.blank + self.malformed_json + self.missing_text_key + self.non_string_text

    def reasons(self) -> dict[str, int]:
        return {
            name: value
            for name, value in (
                ("blank", self.blank),
                ("malformed_json", self.malformed_json),
                ("missing_text_key", self.missing_text_key),
                ("non_string_text", self.non_string_text),
            )
            if value
        }


def iter_monolingual_lines(
    path: str | Path,
    *,
    stats: ReadStats | None = None,
) -> Iterator[str]:
    """``.txt`` 는 한 줄에 하나, ``.jsonl`` 은 ``text`` 키에서 텍스트를 낸다.

    양쪽 다 빈 줄은 건너뜁니다. ``stats`` 를 주면 건너뛴 이유가 집계됩니다.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"단일어 코퍼스는 {' 또는 '.join(ALLOWED_SUFFIXES)} 만 허용합니다: {path}")
    stats = stats if stats is not None else ReadStats()

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        if suffix == TEXT_SUFFIX:
            for line in handle:
                text = line.strip()
                if not text:
                    stats.blank += 1
                    continue
                stats.accepted += 1
                yield text
            return
        for line in handle:
            raw = line.strip()
            if not raw:
                stats.blank += 1
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                stats.malformed_json += 1
                continue
            if not isinstance(row, dict) or JSONL_TEXT_KEY not in row:
                stats.missing_text_key += 1
                continue
            value = row[JSONL_TEXT_KEY]
            if not isinstance(value, str):
                stats.non_string_text += 1
                continue
            text = value.strip()
            if not text:
                stats.blank += 1
                continue
            stats.accepted += 1
            yield text


def language_sampling_weights(
    counts: dict[str, int],
    *,
    alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
) -> dict[str, float]:
    """언어별 온도 샘플링 가중치 (합이 1).

    ``alpha=1`` 이면 분량에 정비례하고, 0 에 가까울수록 균등해집니다. 분량이
    0 인 언어는 가중치도 0 입니다 — 없는 데이터를 만들어 낼 수는 없으므로,
    그 경우는 가중치가 아니라 경고로 다뤄야 합니다.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("language sampling alpha must be in (0, 1]")
    scaled = {
        language: math.pow(float(count), alpha) for language, count in counts.items() if count > 0
    }
    total = sum(scaled.values())
    if total <= 0:
        return {language: 0.0 for language in counts}
    weights = {language: 0.0 for language in counts}
    weights.update({language: value / total for language, value in scaled.items()})
    return weights


@dataclass
class BalanceReport:
    """언어 균형 실측과 그로부터 나온 경고."""

    counts: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def is_balanced(self) -> bool:
        return not self.warnings


def assess_language_balance(
    counts: dict[str, int],
    *,
    alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    minimum_share: float = 0.05,
) -> BalanceReport:
    """온도 샘플링 뒤에도 남는 불균형을 경고로 만든다.

    온도 샘플링은 격차를 눕힐 뿐 없앨 수 없고, 분량이 0 인 언어에는 아무것도
    하지 못합니다. foundation 단계는 인코더·디코더의 언어별 표현을 만드는
    단계라, 한쪽 언어만 학습하면 그 방향 번역만 강해집니다 — 이 저장소는
    이미 ko→ja 59.81 대 ja→ko 49.87 로 그 증상을 갖고 있습니다.
    """

    weights = language_sampling_weights(counts, alpha=alpha)
    warnings: list[str] = []
    empty = sorted(language for language, count in counts.items() if count <= 0)
    if empty:
        warnings.append(
            f"단일어 데이터가 전혀 없는 언어: {', '.join(empty)}. "
            "이 언어들은 foundation 단계에서 전혀 학습되지 않으므로, "
            "해당 언어를 목표로 하는 번역 방향이 상대적으로 뒤처집니다."
        )
    thin = sorted(
        language
        for language, weight in weights.items()
        if 0.0 < weight < minimum_share and len(counts) > 1
    )
    if thin:
        rendered = ", ".join(f"{language} {weights[language]:.1%}" for language in thin)
        warnings.append(
            f"온도 샘플링(alpha={alpha}) 뒤에도 배치 비중이 {minimum_share:.0%} 미만인 "
            f"언어: {rendered}. alpha 를 더 낮추거나 데이터를 늘리십시오."
        )
    return BalanceReport(counts=dict(counts), weights=weights, warnings=tuple(warnings))


def render_discovery_report(
    discovery: MonolingualDiscovery,
    *,
    maximum_skipped: int = 20,
) -> list[str]:
    """사람이 읽는 요약. 학습 시작 전에 그대로 출력할 용도입니다."""

    lines = [f"단일어 코퍼스 루트: {discovery.root}"]
    totals = discovery.bytes_per_language()
    if totals:
        for language in sorted(totals):
            files = len(discovery.paths_for(language))
            lines.append(f"  {language}: 파일 {files}개, {totals[language] / 1e9:.2f} GB")
    else:
        lines.append("  학습 가능한 파일이 없습니다")
    if discovery.languages_without_data:
        lines.append(f"  데이터 없는 언어: {', '.join(discovery.languages_without_data)}")
    if discovery.unconfigured_languages:
        lines.append(
            f"  설정에 없어 건너뛴 언어 폴더: {', '.join(discovery.unconfigured_languages)}"
        )
    if discovery.skipped:
        lines.append(f"  건너뛴 항목 {len(discovery.skipped)}개:")
        for entry in discovery.skipped[:maximum_skipped]:
            lines.append(f"    - {entry.path}: {entry.reason}")
        remaining = len(discovery.skipped) - maximum_skipped
        if remaining > 0:
            lines.append(f"    ... 외 {remaining}개")
    return lines


__all__ = [
    "ALLOWED_SUFFIXES",
    "BalanceReport",
    "DEFAULT_CORPUS_DIRECTORY",
    "DEFAULT_LANGUAGE_SAMPLING_ALPHA",
    "JSONL_TEXT_KEY",
    "LANGUAGE_DIRECTORY_PATTERN",
    "MonolingualDiscovery",
    "MonolingualSource",
    "ReadStats",
    "SkippedEntry",
    "assess_language_balance",
    "discover_monolingual_sources",
    "foundation_languages",
    "iter_monolingual_lines",
    "language_sampling_weights",
    "render_discovery_report",
]
