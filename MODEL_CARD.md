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
sources as the training data, so quality on unseen domains is substantially lower. Any single
reported figure should state which of the two it is.

Exact-duplicate leakage is guarded; near-duplicate leakage was not. The split key used for this
release was an exact normalized string, so two rows differing by one particle were assigned
independently. Measured against a character 5-gram MinHash key, near-duplicate leakage into the
holdout was roughly twice as high (1.12% against 0.48%, averaged over four training shards).
Treat the repository's own ja->ko test-split figures accordingly: the same checkpoint scores
chrF 53.43 on an out-of-domain diagnostic set.

## Limitations

- **Numbers can change value.** This release's tokenizer was trained without digit splitting, so
  the model treats numbers as memorised pieces rather than sequences of digits. It substitutes a
  plausible wrong value rather than dropping it. Eight of ten numeric probe sentences are
  corrupted at beam 4:

  | source | output |
  |---|---|
  | `0.0037mg/L 이하로` | `1.337mg/L以下に` |
  | `250mg씩 하루 두 번` | `1200mgずつ1日2回` |
  | `35%에서 62.5%로` | `０．７％から６．７％に` |
  | `부가세 15%가 포함` | `付加税1,500ウォンが含まれる` |
  | `110-482-937561` | `1、0、482-937561` |
  | `38,720개에서 7,842,913개로` | `3万3千5百個から1万4千9十三個に` |
  | `±0.05mm 이내` | `0.00.05mm以内` |

  Every failure involves a merged multi-digit piece: `35%` is a single token and `62.5kg` splits
  as `▁6 | 2.5 | kg`, so the boundary between the number and its unit falls inside a token. Beam
  width does not change this, and neither does post-training: the representation does not expose
  digits, so `reward_number_weight` has no signal to optimise. Verify every amount, dosage, date
  and identifier against the source. Loading the tokenizer emits a warning, and `sion-train`
  refuses it outright, so this checkpoint cannot be trained further — fixing it requires a new
  tokenizer and therefore a new model, because the tied embedding is 18.4% of the parameters.
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
