"""Gradio Space for interactive sion_translate Korean↔Japanese translation."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download

MODEL_REPO = "gaon12/sion_translate"
MODEL_REVISION = "2e478ef0e62c613c65c09e628c910fd69797ec91"
MODEL_ARTIFACTS = {
    # This revision's INT8 file is a legacy pickled nn.Module. The EMA state
    # dictionary is larger, but it is compatible with weights_only=True.
    "model_ema.pt": {
        "size": 802_029_835,
        "sha256": "829580d1db1693ec00ee641398ce6ba9549388e266f6259f85e5a19ea28bfc2e",
    },
    "kjx.model": {
        "size": 852_890,
        "sha256": "c1b5447e08692bf7a7d2e485ba7b71b3ff5584c96844ecac07e25c9d1340db50",
    },
    "token_features.npz": {
        "size": 69_121,
        "sha256": "54951b569255bd0a2e1d333860db2294113f1368a2e25e4ae5bc2bcdb6976b4c",
    },
}
DIRECTIONS = {
    "한국어 → 日本語": "ja",
    "日本語 → 한국어": "ko",
}
BEAMS = {
    "빠름 (greedy)": 1,
    "균형 (beam 2)": 2,
    "품질 (beam 4)": 4,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(filename: str) -> Path:
    identity = MODEL_ARTIFACTS[filename]
    path = Path(
        hf_hub_download(
            MODEL_REPO,
            filename,
            revision=MODEL_REVISION,
        )
    )
    if path.stat().st_size != identity["size"] or _sha256_file(path) != identity["sha256"]:
        raise RuntimeError(f"다운로드한 {filename}의 크기 또는 SHA-256이 고정된 값과 다릅니다.")
    return path


def _load_safe_state_dict(
    path: str | Path,
    *,
    return_metadata: bool = False,
    allow_legacy: bool = True,
) -> tuple[Any, ...]:
    """Load the pinned legacy state-dict container without executable pickle."""

    import torch

    from sion_translate.model import SionForConditionalGeneration
    from sion_translate.training.export import EXPORT_SCHEMA, _model_config_from_dict

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping):
        raise ValueError("모델 파일이 state-dict export 형식이 아닙니다.")
    schema = payload.get("schema")
    if schema is None and not allow_legacy:
        raise ValueError(f"모델 파일에 {EXPORT_SCHEMA} schema가 없습니다.")
    if schema is not None and schema != EXPORT_SCHEMA:
        raise ValueError(f"지원하지 않는 모델 schema입니다: {schema!r}")
    raw_config = payload.get("model_config")
    stored = payload.get("model")
    if not isinstance(raw_config, Mapping) or not isinstance(stored, Mapping):
        raise ValueError("모델 파일에 config 또는 state dictionary가 없습니다.")
    if any(not isinstance(value, torch.Tensor) for value in stored.values()):
        raise ValueError("모델 state dictionary에는 tensor만 들어 있어야 합니다.")

    config = _model_config_from_dict(raw_config)
    pad_id = int(payload["pad_id"])
    runtime_config = copy.deepcopy(config)
    runtime_config.gradient_checkpointing = False
    with torch.random.fork_rng(devices=[]):
        with torch.device("meta"):
            model = SionForConditionalGeneration(runtime_config, pad_id=pad_id)
    model.load_state_dict(dict(stored), assign=True)
    model.eval()
    raw_metadata = payload.get("metadata") or {}
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("모델 metadata는 object여야 합니다.")
    metadata = copy.deepcopy(dict(raw_metadata))
    if return_metadata:
        return model, config, pad_id, metadata
    return model, config, pad_id


@lru_cache(maxsize=1)
def load_translator():
    """Download one immutable, verified export and keep it loaded on CPU."""
    import sion_translate.inference as inference_module

    model_path = _download_verified("model_ema.pt")
    tokenizer_path = _download_verified("kjx.model")
    _download_verified("token_features.npz")
    # The pinned dependency predates the safe loader. Replace only the loader
    # symbol used by Translator with this Space's narrow weights-only loader.
    inference_module.load_exported_model = _load_safe_state_dict
    return inference_module.Translator(model_path, tokenizer_path, device="cpu")


def translate(text: str, direction: str, quality: str) -> str:
    """Translate one short text using the direction selected in the UI."""
    text = text.strip()
    if not text:
        raise gr.Error("번역할 문장을 입력하세요.")
    if len(text) > 2_000:
        raise gr.Error("한 번에 2,000자 이하로 입력하세요.")

    translator = load_translator()
    result = translator.translate(
        [text],
        target_language=DIRECTIONS[direction],
        num_beams=BEAMS[quality],
        max_new_tokens=min(256, max(32, len(text) * 2)),
        batch_size=1,
    )
    return result[0]


with gr.Blocks(title="sion_translate Translator") as demo:
    gr.Markdown(
        """
        # sion_translate 한국어 ↔ 日本語

        SHA-256으로 검증한 고정 revision의 sion_translate EMA 모델을 CPU에서 실행합니다.
        첫 번역은 모델 로딩 때문에 오래 걸릴 수 있습니다. 중요한 번역은 반드시 사람이 검토하세요.
        """
    )
    with gr.Row():
        direction = gr.Radio(
            choices=list(DIRECTIONS),
            value="한국어 → 日本語",
            label="번역 방향",
        )
        quality = gr.Radio(
            choices=list(BEAMS),
            value="균형 (beam 2)",
            label="탐색 품질",
        )
    source = gr.Textbox(
        label="원문",
        lines=5,
        placeholder="번역할 문장을 입력하세요.",
    )
    submit = gr.Button("번역", variant="primary")
    output = gr.Textbox(label="번역 결과", lines=5)
    gr.Examples(
        examples=[
            ["회의가 끝나면 수정된 자료를 검토해 주시겠어요?", "한국어 → 日本語"],
            ["恐れ入りますが、こちらにお名前をご記入いただけますか。", "日本語 → 한국어"],
        ],
        inputs=[source, direction],
    )
    submit.click(
        fn=translate,
        inputs=[source, direction, quality],
        outputs=output,
        api_name="translate",
    )
    source.submit(
        fn=translate,
        inputs=[source, direction, quality],
        outputs=output,
        api_name=False,
    )

demo.queue(default_concurrency_limit=1, max_size=16)


if __name__ == "__main__":
    demo.launch()
