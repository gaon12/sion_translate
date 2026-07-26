"""글로서리(용어집) 강제 번역.

고유명사·전문용어를 지정한 대응어로 반드시 번역하게 만드는 기능입니다.
DeepL/Google 의 glossary 기능과 같은 목적이며, 이 프로젝트에 이미 있는
``<slot_n>`` 보호 토큰 인프라를 재사용하는 placeholder 방식으로 구현합니다.

동작 원리 (추론 시, 모델 재학습 불필요):
    1. 원문에서 글로서리의 '원문 언어 표면형'을 찾아 ``<slot_k>`` 로 치환하고
       slot_k → '목표 언어 표면형' 매핑을 기록합니다.
    2. 치환된 문장을 번역합니다. 모델은 학습 때 ``<slot_n>`` 을 그대로
       보존하도록 배웠으므로(protect span/TETM), 출력에도 slot 이 남습니다.
    3. 출력의 ``<slot_k>`` 를 목표 언어 표면형으로 되돌립니다.

이 방식의 장점은 '확실함'입니다. 모델이 용어를 어떻게 번역할지에 기대지
않고, 원문에서 떼어내 두었다가 정해진 대응어로 되돌리기 때문입니다.

글로서리 파일 형식 (JSON) — 언어별 표면형을 한 항목에 담습니다:
    [
      {"ko": "인공지능", "ja": "人工知能"},
      {"ko": "심층학습", "ja": "深層学習"}
    ]
언어 키가 언어쌍과 일치하면 방향에 상관없이(ko→ja, ja→ko) 모두 강제됩니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# 단어 경계가 없는(띄어쓰기로 단어를 나누지 않는) 언어.
# 이 언어들은 부분 문자열 매칭을, 그 외(라틴 문자 등)는 단어 경계 매칭을 씁니다.
NON_WORD_BOUNDARY_LANGUAGES = {"ko", "ja", "zh"}


@dataclass(frozen=True)
class Glossary:
    """언어별 표면형을 담은 용어 목록.

    entries: [{"ko": "인공지능", "ja": "人工知能"}, ...]
    """

    entries: tuple[dict[str, str], ...]

    def __len__(self) -> int:
        return len(self.entries)

    def for_direction(self, source_language: str, target_language: str) -> list[tuple[str, str]]:
        """(원문 표면형, 목표 표면형) 목록. 원문이 긴 것부터 정렬합니다.

        긴 용어를 먼저 치환해야 짧은 용어가 긴 용어의 일부를 먼저 가로채는
        일을 막을 수 있습니다 (예: "인공지능학회" vs "인공지능").
        """
        pairs: list[tuple[str, str]] = []
        for entry in self.entries:
            source = entry.get(source_language)
            target = entry.get(target_language)
            if source and target:
                pairs.append((source.strip(), target.strip()))
        # 중복 원문은 첫 항목만 유지하고, 길이 내림차순으로 정렬합니다.
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for source, target in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
            if source and source not in seen:
                seen.add(source)
                unique.append((source, target))
        return unique


def load_glossary(path: str | Path) -> Glossary:
    """JSON 글로서리 파일을 읽습니다.

    허용 형식:
    - 리스트: [{"ko": "...", "ja": "..."}, ...]
    - 단순 매핑: {"인공지능": "人工知能", ...} — 이 경우 언어 키가 없으므로
      ``as_pair`` 로 어떤 언어쌍인지 알려줘야 하지만, 여기서는 리스트 형식만
      표준으로 삼고 단순 매핑은 호출부에서 변환하도록 둡니다.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(
            "글로서리는 [{\"ko\": ..., \"ja\": ...}, ...] 형식의 리스트여야 합니다."
        )
    entries: list[dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("글로서리의 각 항목은 언어별 표면형 dict 여야 합니다.")
        cleaned = {
            str(key): str(value)
            for key, value in row.items()
            if isinstance(value, str) and value.strip()
        }
        if len(cleaned) >= 2:
            entries.append(cleaned)
    return Glossary(tuple(entries))


def _match_positions(text: str, term: str, language: str) -> list[tuple[int, int]]:
    """text 안에서 term 이 나타나는 (시작, 끝) 위치들을 찾습니다.

    CJK 등 단어 경계가 없는 언어는 부분 문자열로, 그 외 언어는 단어 경계를
    존중해 찾습니다 (예: 영어 "cat" 이 "category" 에 걸리지 않도록).
    """
    if not term:
        return []
    if language in NON_WORD_BOUNDARY_LANGUAGES:
        pattern = re.escape(term)
    else:
        # 앞뒤가 글자/숫자가 아닌 경계에서만 매칭 (대소문자 무시).
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    flags = 0 if language in NON_WORD_BOUNDARY_LANGUAGES else re.IGNORECASE
    return [(match.start(), match.end()) for match in re.finditer(pattern, text, flags)]


def apply_source_placeholders(
    text: str,
    glossary: Glossary,
    *,
    source_language: str,
    target_language: str,
    slot_symbols: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """원문의 글로서리 용어를 slot 토큰으로 치환합니다.

    반환: (치환된 문장, {slot 토큰 문자열: 목표 표면형}).
    같은 용어가 여러 번 나오면 같은 slot 을 재사용하고, slot 개수(기본 64)를
    넘는 용어는 강제하지 않고 원문 그대로 둡니다.
    """
    directional = glossary.for_direction(source_language, target_language)
    if not directional:
        return text, {}

    # 어떤 구간이 이미 다른(더 긴) 용어에 의해 점유됐는지 추적해 겹침을 막습니다.
    claimed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < c_end and c_start < end for c_start, c_end in claimed)

    # 용어별로 slot 을 배정합니다. 같은 원문 용어는 같은 slot 을 공유합니다.
    term_to_slot: dict[str, str] = {}
    slot_to_target: dict[str, str] = {}
    replacements: list[tuple[int, int, str]] = []  # (시작, 끝, slot 토큰)

    for source_term, target_term in directional:
        for start, end in _match_positions(text, source_term, source_language):
            if overlaps(start, end):
                continue
            slot = term_to_slot.get(source_term)
            if slot is None:
                if len(term_to_slot) >= len(slot_symbols):
                    # slot 이 모자라면 이 용어는 강제하지 않습니다.
                    break
                slot = slot_symbols[len(term_to_slot)]
                term_to_slot[source_term] = slot
                slot_to_target[slot] = target_term
            claimed.append((start, end))
            replacements.append((start, end, slot))

    if not replacements:
        return text, {}

    # 뒤에서부터 치환해 인덱스가 밀리지 않게 합니다.
    replacements.sort(key=lambda item: item[0], reverse=True)
    result = text
    for start, end, slot in replacements:
        # slot 이 주변 글자와 붙어 토큰화가 깨지지 않도록 공백으로 감쌉니다.
        result = f"{result[:start]} {slot} {result[end:]}"
    return result, slot_to_target


def restore_targets(translated: str, slot_to_target: dict[str, str]) -> tuple[str, list[str]]:
    """번역문의 slot 토큰을 목표 표면형으로 되돌립니다.

    반환: (복원된 문장, 출력에 나타나지 않은 slot 의 목표 용어 목록).
    누락 용어는 모델이 slot 을 보존하지 못한 경우이며, 호출부에서 경고하거나
    문장 끝에 덧붙이는 등의 보정에 쓸 수 있습니다.
    """
    if not slot_to_target:
        return translated, []
    result = translated
    missing: list[str] = []
    for slot, target in slot_to_target.items():
        if slot in result:
            result = result.replace(slot, target)
        else:
            missing.append(target)
    # slot 을 공백으로 감쌌던 흔적(이중 공백)을 정리합니다.
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result, missing
