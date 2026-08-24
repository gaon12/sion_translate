"""Run optional public translation-model baselines."""

# AutoModel/AutoTokenizer return model-specific dynamic objects.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from sion_translate.comparison import ComparisonCase


HF_BASELINES = {
    "m2m100-418m": {
        "model_id": "facebook/m2m100_418M",
        "language_codes": {"ko": "ko", "ja": "ja"},
        "kind": "m2m100",
    },
    "nllb-200-distilled-600m": {
        "model_id": "facebook/nllb-200-distilled-600M",
        "language_codes": {"ko": "kor_Hang", "ja": "jpn_Jpan"},
        "kind": "nllb",
    },
}


def translate_with_hf_baseline(
    cases: Sequence[ComparisonCase],
    *,
    backend: str,
    device: str = "auto",
    batch_size: int = 8,
    num_beams: int = 4,
    max_new_tokens: int = 256,
) -> dict[str, str]:
    """Download one Hugging Face model and translate comparison cases.

    Weights remain in the local Hugging Face cache and are not copied into this
    project. Users remain responsible for each upstream model's license.
    """
    if backend not in HF_BASELINES:
        raise ValueError(f"Unsupported baseline: {backend}")
    if batch_size < 1 or num_beams < 1 or max_new_tokens < 1:
        raise ValueError("Batch size, beam count, and maximum token count must be positive")

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Public baselines require 'pip install -e .[baselines]'") from error

    spec = HF_BASELINES[backend]
    language_codes = spec["language_codes"]
    used_languages = {case.source_language for case in cases} | {
        case.target_language for case in cases
    }
    unsupported = sorted(used_languages - language_codes.keys())
    if unsupported:
        raise ValueError(f"{backend} has no language code for: {', '.join(unsupported)}")

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    model_id = str(spec["model_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(resolved_device)
    model.eval()

    grouped: dict[tuple[str, str], list[ComparisonCase]] = defaultdict(list)
    for case in cases:
        grouped[(case.source_language, case.target_language)].append(case)

    translations: dict[str, str] = {}
    for (source_language, target_language), direction_cases in grouped.items():
        source_code = language_codes[source_language]
        target_code = language_codes[target_language]
        tokenizer_kwargs = {"src_lang": source_code} if spec["kind"] == "nllb" else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if spec["kind"] == "m2m100":
            tokenizer.src_lang = source_code
            forced_bos_token_id = tokenizer.get_lang_id(target_code)
        else:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)

        for start in range(0, len(direction_cases), batch_size):
            chunk = direction_cases[start : start + batch_size]
            encoded = tokenizer(
                [case.source for case in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {name: value.to(resolved_device) for name, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.update({case.id: text for case, text in zip(chunk, decoded, strict=True)})
    return translations
