"""FLORES-200 등 표준 평가셋을 sion-evaluate 형식(JSONL)으로 변환.

FLORES-200 은 200개 언어에 대해 같은 3,001개 문장을 번역해 둔 공개
평가셋입니다. 공개 논문·상용 서비스 벤치마크가 대부분 이걸 쓰므로,
여기에 맞추면 우리 모델 점수를 외부 숫자와 '절대 비교'할 수 있습니다.

이 모듈은 두 가지 입력을 지원합니다.
1. 로컬 FLORES 배포 파일 (오프라인, 권장):
   언어별 개별 텍스트 파일이며 한 줄이 한 문장입니다.
   파일 이름은 ``<언어_문자>.<split>`` 형식 (예: ``kor_Hang.dev``).
   같은 split 의 두 언어 파일은 줄 순서가 정렬되어 있어, 줄끼리 짝지으면
   그대로 병렬쌍이 됩니다.
2. Hugging Face ``datasets`` (설치돼 있을 때):
   ``facebook/flores`` 의 ``all`` config 에서 ``sentence_<언어>`` 필드 추출.

출력은 학습/평가 데이터와 같은 형식입니다: 한 줄에 {"ko": ..., "ja": ...}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

# 흔한 언어의 FLORES-200 코드 (ISO 639-3 + 문자). 필요하면 --flores-code 로
# 직접 지정할 수 있으므로 모든 언어를 나열할 필요는 없습니다.
FLORES_CODES: dict[str, str] = {
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
}


def flores_code(language: str, override: str | None = None) -> str:
    """언어 키(ko)를 FLORES 코드(kor_Hang)로. override 가 있으면 그걸 씁니다."""
    if override:
        return override
    code = FLORES_CODES.get(language)
    if code is None:
        raise ValueError(
            f"'{language}' 의 FLORES 코드를 모릅니다. --flores-code {language}=<코드> 로 "
            f"직접 지정하세요 (예: {language}=xxx_Yyyy)."
        )
    return code


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def find_flores_file(root: Path, code: str, split: str) -> Path:
    """FLORES 배포 폴더에서 ``<code>.<split>`` 파일을 찾습니다.

    배포판마다 파일이 루트 바로 아래, dev/ 하위, 또는 확장자 없이 있는 등
    조금씩 다르므로 흔한 위치를 모두 시도합니다.
    """
    candidates = [
        root / f"{code}.{split}",
        root / split / f"{code}.{split}",
        root / split / code,
        root / f"{code}.{split}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"FLORES 파일을 찾지 못했습니다: {code}.{split} (확인한 위치: "
        + ", ".join(str(c) for c in candidates)
        + ")"
    )


def pairs_from_local_flores(
    root: str | Path,
    language_pair: Sequence[str],
    *,
    split: str,
    code_overrides: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """로컬 FLORES 파일 두 개를 줄 단위로 짝지어 병렬쌍 목록을 만듭니다."""
    root = Path(root)
    code_overrides = code_overrides or {}
    key_a, key_b = language_pair
    code_a = flores_code(key_a, code_overrides.get(key_a))
    code_b = flores_code(key_b, code_overrides.get(key_b))
    lines_a = _read_lines(find_flores_file(root, code_a, split))
    lines_b = _read_lines(find_flores_file(root, code_b, split))
    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"두 언어 파일의 문장 수가 다릅니다: {code_a}={len(lines_a)}, "
            f"{code_b}={len(lines_b)} — 같은 FLORES split 인지 확인하세요."
        )
    return [
        {key_a: text_a, key_b: text_b}
        for text_a, text_b in zip(lines_a, lines_b, strict=True)
        if text_a.strip() and text_b.strip()
    ]


def pairs_from_hf_datasets(
    language_pair: Sequence[str],
    *,
    split: str,
    code_overrides: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Hugging Face ``datasets`` 로 FLORES 를 내려받아 병렬쌍을 만듭니다."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets 가 설치돼 있지 않습니다. "
            "'pip install datasets' 하거나 --flores-dir 로 로컬 파일을 쓰세요."
        ) from exc
    code_overrides = code_overrides or {}
    key_a, key_b = language_pair
    code_a = flores_code(key_a, code_overrides.get(key_a))
    code_b = flores_code(key_b, code_overrides.get(key_b))
    # FLORES 의 dev/devtest 는 datasets 의 dev/devtest split 에 대응합니다.
    dataset = load_dataset("facebook/flores", "all", split=split)
    field_a = f"sentence_{code_a}"
    field_b = f"sentence_{code_b}"
    if field_a not in dataset.column_names or field_b not in dataset.column_names:
        raise ValueError(
            f"FLORES 데이터에 {field_a} 또는 {field_b} 필드가 없습니다. 언어 코드를 확인하세요."
        )
    pairs: list[dict[str, str]] = []
    for row in dataset:
        text_a, text_b = row[field_a], row[field_b]
        if text_a and text_b:
            pairs.append({key_a: text_a, key_b: text_b})
    return pairs


def write_jsonl(pairs: Sequence[dict[str, str]], output_path: str | Path) -> int:
    """병렬쌍을 JSONL 로 저장하고 저장한 쌍 수를 반환합니다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)
