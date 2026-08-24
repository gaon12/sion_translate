"""Convert standard evaluation sets such as FLORES-200 to evaluation JSONL.

FLORES-200 contains aligned translations of the same 3,001 sentences in 200
languages. It is widely used in published research and commercial translation
benchmarks, so evaluating on it makes external score comparisons meaningful.

This module supports two input sources:

1. A local FLORES distribution, which is the recommended offline option. Each
   language is stored in a separate text file with one sentence per line. File
   names commonly use ``<language_script>.<split>``, such as
   ``kor_Hang.dev``. Matching line numbers across language files form parallel
   examples.
2. Hugging Face ``datasets``, when installed. The loader reads
   ``sentence_<language>`` fields from the ``facebook/flores`` ``all`` config.

The output uses the same JSONL record shape as the training and evaluation
loaders, with one language-keyed object per line.
"""

# Hugging Face datasets is an optional plugin with a dynamic row schema.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

# Common FLORES-200 codes combine an ISO 639-3 language and a script. Users can
# provide any unlisted language with --flores-code, so this table stays small.
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

_SAFE_FLORES_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_WINDOWS_RESERVED_DEVICE_BASENAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


def validate_flores_path_component(value: object, *, field: str) -> str:
    """Validate an untrusted FLORES code or split before using it in a path."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string: {value!r}")
    basename, _, _extension = value.partition(".")
    if basename.casefold() in _WINDOWS_RESERVED_DEVICE_BASENAMES:
        raise ValueError(f"{field} uses a reserved Windows device name: {value!r}")
    if _SAFE_FLORES_PATH_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{field} is not a safe path component: {value!r}")
    return value


def _validated_code_overrides(code_overrides: dict[str, str] | None) -> dict[str, str]:
    overrides = code_overrides or {}
    return {
        language: validate_flores_path_component(
            code,
            field=f"FLORES code for {language!r}",
        )
        for language, code in overrides.items()
    }


def flores_code(language: str, override: str | None = None) -> str:
    """Resolve a language key such as ``ko`` to a FLORES code such as ``kor_Hang``."""
    if override:
        return override
    code = FLORES_CODES.get(language)
    if code is None:
        raise ValueError(
            f"No default FLORES code is known for {language!r}. Provide one with "
            f"--flores-code {language}=<code>, for example {language}=xxx_Yyyy."
        )
    return code


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def find_flores_file(root: str | Path, code: str, split: str) -> Path:
    """Find a ``<code>.<split>`` file inside a FLORES distribution.

    FLORES distributions use several common layouts. Try files at the root,
    below the split directory, and with the optional text extension.
    """
    safe_code = validate_flores_path_component(code, field="FLORES code")
    safe_split = validate_flores_path_component(split, field="FLORES split")
    resolved_root = Path(root).resolve()
    candidates = [
        resolved_root / f"{safe_code}.{safe_split}",
        resolved_root / safe_split / f"{safe_code}.{safe_split}",
        resolved_root / safe_split / safe_code,
        resolved_root / f"{safe_code}.{safe_split}.txt",
    ]
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"FLORES candidate resolves outside the configured root: {candidate}"
            ) from error
        if resolved_candidate.is_file():
            return resolved_candidate
    raise FileNotFoundError(
        f"Could not find FLORES file {safe_code}.{safe_split}. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def pairs_from_local_flores(
    root: str | Path,
    language_pair: Sequence[str],
    *,
    split: str,
    code_overrides: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Pair aligned lines from two local FLORES language files."""
    root = Path(root)
    safe_split = validate_flores_path_component(split, field="FLORES split")
    code_overrides = _validated_code_overrides(code_overrides)
    key_a, key_b = language_pair
    code_a = validate_flores_path_component(
        flores_code(key_a, code_overrides.get(key_a)),
        field=f"FLORES code for {key_a!r}",
    )
    code_b = validate_flores_path_component(
        flores_code(key_b, code_overrides.get(key_b)),
        field=f"FLORES code for {key_b!r}",
    )
    path_a = find_flores_file(root, code_a, safe_split)
    path_b = find_flores_file(root, code_b, safe_split)
    if path_a.samefile(path_b):
        raise ValueError(
            "The two FLORES language codes resolve to the same physical file: "
            f"{key_a}={path_a}, {key_b}={path_b}"
        )
    lines_a = _read_lines(path_a)
    lines_b = _read_lines(path_b)
    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"The language files contain different sentence counts: {code_a}={len(lines_a)}, "
            f"{code_b}={len(lines_b)}. Confirm that both files use the same FLORES split."
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
    """Download FLORES through Hugging Face ``datasets`` and return aligned pairs."""
    safe_split = validate_flores_path_component(split, field="FLORES split")
    code_overrides = _validated_code_overrides(code_overrides)
    key_a, key_b = language_pair
    code_a = validate_flores_path_component(
        flores_code(key_a, code_overrides.get(key_a)),
        field=f"FLORES code for {key_a!r}",
    )
    code_b = validate_flores_path_component(
        flores_code(key_b, code_overrides.get(key_b)),
        field=f"FLORES code for {key_b!r}",
    )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is not installed. Install it with "
            "'pip install datasets' or use local files through --flores-dir."
        ) from exc
    # FLORES dev and devtest map directly to the datasets splits of the same names.
    dataset = load_dataset("facebook/flores", "all", split=safe_split)
    field_a = f"sentence_{code_a}"
    field_b = f"sentence_{code_b}"
    if field_a not in dataset.column_names or field_b not in dataset.column_names:
        raise ValueError(
            f"The FLORES dataset does not contain {field_a} or {field_b}. Check the language codes."
        )
    pairs: list[dict[str, str]] = []
    for row in dataset:
        text_a, text_b = row[field_a], row[field_b]
        if text_a and text_b:
            pairs.append({key_a: text_a, key_b: text_b})
    return pairs


def write_jsonl(pairs: Sequence[dict[str, str]], output_path: str | Path) -> int:
    """Write parallel pairs as JSONL and return the number of written records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)
