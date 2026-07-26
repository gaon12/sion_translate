---
language:
- ko
- ja
license: mit
library_name: pytorch
pipeline_tag: translation
tags:
- translation
- korean
- japanese
- seq2seq
- custom-code
widget:
- text: "회의가 끝나면 수정된 자료를 검토해 주시겠어요?"
  example_title: "한국어 → 일본어"
  output:
    text: "会議が終われば、修正された資料を検討していただけますか。"
- text: "恐れ入りますが、こちらにお名前をご記入いただけますか。"
  example_title: "日本語 → 한국어"
  output:
    text: "죄송하지만 여기 성함을 기입해 주시겠어요?"
---

# sion_translate Korean–Japanese Translation

sion_translate is a custom PyTorch encoder-decoder model for bidirectional Korean↔Japanese
translation. The uploaded release contains inference exports and its SentencePiece tokenizer;
training datasets and training checkpoints are not distributed.

## Files

- `model_ema.pt`: recommended FP32 EMA inference export
- `model_int8.pt`: smaller CPU-only dynamic INT8 export
- `sion.model`, `sion.vocab`, `token_features.npz`: tokenizer files

The model uses custom code from
[`gaon12/sion_translate`](https://github.com/gaon12/sion_translate) and is not a Transformers
`AutoModel` checkpoint.

## Usage

```bash
git clone https://github.com/gaon12/sion_translate.git
cd sion_translate
python -m pip install -e .
hf download gaon12/sion_translate --local-dir models/sion_translate
```

```python
from sion_translate.inference import Translator

translator = Translator(
    "models/sion_translate/model_ema.pt",
    "models/sion_translate/sion.model",
)
print(translator.translate(["안녕하세요."], target_language="ja", num_beams=4)[0])
print(translator.translate(["こんにちは。"], target_language="ko", num_beams=4)[0])
```

Use `model_int8.pt` for CPU inference with lower storage and memory requirements. It reduces file
size and memory, not latency: on the same CPU its generation speed matches the FP32 export. Only
load PyTorch pickle-based artifacts from repositories you trust.

## Evaluation

The GitHub repository includes a small, independently authored diagnostic JSONL covering both
directions and provides a common scorer for sion_translate, LibreTranslate, Papago, Google Translation,
DeepL, M2M100 418M, and NLLB-200. No universal quality claim is made from that small set.

Scores from the repository's own test split are in-domain: the split is drawn from the same
sources as the training data. Leakage is guarded, but domain overlap is not, so quality on
unseen domains is substantially lower. Any single reported figure should state which of the two
it is.

## Limitations

- **Numbers can change value.** This release's tokenizer was trained without digit splitting, so
  the model treats numbers as memorised pieces rather than sequences of digits. It may silently
  substitute a plausible wrong value instead of dropping it: `250mg` → `1200mg`, `0.5mL` →
  `120ml`, `38,720円` → `38,000엔`. Beam width does not change this. Verify every amount, dosage,
  date and identifier against the source. Loading the tokenizer emits a warning to this effect.
- Proper nouns are memorised, not generalised. Names and places present in training are accurate,
  while unseen ones get translated morpheme by morpheme (`五十嵐大輔` → `50폭풍 대장`). Supply a
  glossary for names that matter.
- Machine translations can omit, add, or reverse meaning. Rare vocabulary can be replaced by an
  unrelated similar-looking word.
- Short context-dependent utterances, specialist terminology, and long inputs need human review.
- Do not use unreviewed output for medical, legal, safety-critical, or certified translation.
- The custom architecture requires the repository code; hosted Transformers inference is not
  available out of the box, so the `widget` examples above show pre-generated output rather than
  a live run.

## License

The original sion_translate code and uploaded model release are provided under the MIT License. This does
not grant rights to user-provided inputs, generated service outputs, third-party software, or
third-party models used only as comparison baselines.
