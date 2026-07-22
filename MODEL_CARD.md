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
---

# KJ-X Korean–Japanese Translation

KJ-X is a custom PyTorch encoder-decoder model for bidirectional Korean↔Japanese
translation. The uploaded release contains inference exports and its SentencePiece tokenizer;
training datasets and training checkpoints are not distributed.

## Files

- `model_ema.pt`: recommended FP32 EMA inference export
- `model_int8.pt`: smaller CPU-only dynamic INT8 export
- `kjx.model`, `kjx.vocab`, `token_features.npz`: tokenizer files

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
from kjx.inference import Translator

translator = Translator(
    "models/sion_translate/model_ema.pt",
    "models/sion_translate/kjx.model",
)
print(translator.translate(["안녕하세요."], target_language="ja", num_beams=4)[0])
print(translator.translate(["こんにちは。"], target_language="ko", num_beams=4)[0])
```

Use `model_int8.pt` for CPU inference with lower storage and memory requirements. Only load
PyTorch pickle-based artifacts from repositories you trust.

## Evaluation

The GitHub repository includes a small, independently authored diagnostic JSONL covering both
directions and provides a common scorer for KJ-X, LibreTranslate, Papago, Google Translation,
DeepL, M2M100 418M, and NLLB-200. No universal quality claim is made from that small set.

## Limitations

- Machine translations can omit, add, or reverse meaning.
- Proper nouns, short context-dependent utterances, specialist terminology, and long inputs need
  human review.
- Do not use unreviewed output for medical, legal, safety-critical, or certified translation.
- The custom architecture requires the repository code; hosted Transformers inference is not
  available out of the box.

## License

The original KJ-X code and uploaded model release are provided under the MIT License. This does
not grant rights to user-provided inputs, generated service outputs, third-party software, or
third-party models used only as comparison baselines.
