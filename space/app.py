"""Gradio Space for interactive KJ-X Korean↔Japanese translation."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr
from huggingface_hub import hf_hub_download

MODEL_REPO = "gaon12/sion_translate"
DIRECTIONS = {
    "한국어 → 日本語": "ja",
    "日本語 → 한국어": "ko",
}
BEAMS = {
    "빠름 (greedy)": 1,
    "균형 (beam 2)": 2,
    "품질 (beam 4)": 4,
}


@lru_cache(maxsize=1)
def load_translator():
    """Download the public INT8 export once and keep it loaded on CPU."""
    from kjx.inference import Translator

    model_path = hf_hub_download(MODEL_REPO, "model_int8.pt")
    tokenizer_path = hf_hub_download(MODEL_REPO, "kjx.model")
    return Translator(model_path, tokenizer_path, device="cpu")


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


with gr.Blocks(title="KJ-X Translator") as demo:
    gr.Markdown(
        """
        # KJ-X 한국어 ↔ 日本語

        KJ-X INT8 모델을 CPU에서 실행합니다. 첫 번역은 모델 로딩 때문에 오래 걸릴 수
        있습니다. 중요한 번역은 반드시 사람이 검토하세요.
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
