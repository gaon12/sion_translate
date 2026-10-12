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

# sion_translate 1.0 Korean–Japanese Translation

sion_translate is a custom PyTorch encoder-decoder model for bidirectional Korean↔Japanese
translation. The uploaded release contains inference exports and its SentencePiece tokenizer;
training datasets and training checkpoints are not distributed.

## Known remaining issues after the 2026-10-12 reevaluation

This is a **partial inference correction, not a complete fix for numerical translation**.
The runtime and model weights have not changed since the numeric patch. More extensive
evaluation found failures that the default protected-slot check does not reject:

- A spaced sign can disappear: `誤差は± 0.05mm以内です。` produced
  `오차는 0.05 mm 이내입니다.`
- A value can survive while its unit changes: `용량은 250mL이며 속도는 3.5km/h입니다.`
  produced `容量は250 mmで、速度は 3.5 km/h です。`
- Some suffixes such as `15mile` and `40mile` are not recognized by the numeric
  masker, so their values can be omitted without a protected-slot error.
- Automatically masking non-numeric acronyms can degrade otherwise ordinary text.
  A focused diagnostic that bypassed this masking restored the original output in
  all five tested cases; this experiment has not been deployed.

`strict_numbers=True` checks **recognized protected values** and supported hour/minute
forms. It does not independently verify every source number, sign, unit, or numerical
association. It can also reject an input instead of returning a translation. The earlier
18-case probe documents that limited cohort, not a guarantee for arbitrary inputs.

**한국어 안내:** 숫자 처리는 부분적으로 개선됐습니다. 부호·단위 손실, 지원하지
않는 단위 뒤의 숫자 누락, 일부 일반 문장 회귀가 남아 있으므로 숫자가 완벽하게
안전한 버전으로 설명해서는 안 됩니다. 숫자값뿐 아니라 부호·단위·문맥을 확인하세요.

## v1 numeric inference update

Existing v1 users should download the updated inference code from
[the pinned maintenance runtime](docs/V1_NUMERIC_HOTFIX.md). The existing public
`model_ema.pt`, `kjx.model`, and `token_features.npz` can be reused. No retraining or
weight replacement is required. This preserves numeric literals and checks
Korean/Japanese hour/minute associations; it does not change the model's learned
numerical abilities or general translation quality.

## Files

- `model_ema.pt`: recommended FP32 EMA inference export
- `model_int8.pt`: smaller CPU-only dynamic INT8 export
- `kjx.model`, `kjx.vocab`, `token_features.npz`: tokenizer files

The model uses custom code from
[`gaon12/sion_translate`](https://github.com/gaon12/sion_translate) and is not a Transformers
`AutoModel` checkpoint.

## Usage

```bash
python -m pip install --upgrade "sion-translate @ git+https://github.com/gaon12/sion_translate.git@c2ccc5210f2436017e2adaf067d02a7c83ceec98"
python -m pip install huggingface_hub
hf download gaon12/sion_translate --local-dir models/sion_translate
```

```python
from sion_translate.inference import Translator

translator = Translator(
    "models/sion_translate/model_ema.pt",
    "models/sion_translate/kjx.model",
)
print(translator.translate(["안녕하세요."], target_language="ja", num_beams=4)[0])
print(translator.translate(["こんにちは。"], target_language="ko", num_beams=4)[0])
```

The numeric update was verified with `model_ema.pt` on CPU. The public legacy pickled
`model_int8.pt` is not covered by this release's safe weights-only loading validation.

## Evaluation

The GitHub repository includes a small, independently authored diagnostic JSONL covering both
directions and provides a common scorer for sion_translate, LibreTranslate, Papago, Google Translation,
DeepL, M2M100 418M, and NLLB-200. No universal quality claim is made from that small set.

On the repository's 20-sentence-per-direction beam-4 probe, the uploaded release scored chrF
59.81 for ko→ja and 49.87 for ja→ko. The roughly ten-point gap is material: Japanese→Korean is
the weaker direction and should not be described as equivalent to Korean→Japanese. The current
repository contains direction-balanced checkpoint selection and a compatibility-checked retraining
pipeline, but those code changes do not retroactively improve the weights in this release.

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

- **Historical unprotected v1 output can change number values.** This release's tokenizer was
  trained without digit splitting, so
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
  refuses it outright, so this checkpoint cannot be trained further — changing that learned
  representation requires a new
  tokenizer and model. The numeric inference update instead uses existing protected slots
  to preserve literals without changing token IDs or weights.
- Proper nouns are memorised, not generalised. Names and places present in training are accurate,
  while unseen ones get translated morpheme by morpheme (`五十嵐大輔` → `50폭풍 대장`). Supply a
  glossary for names that matter.
- Machine translations can omit, add, or reverse meaning. Rare vocabulary can be replaced by an
  unrelated similar-looking word.
- Profanity, internet slang, non-lexical vocalizations, and culture-bound idioms are
  style-sensitive. This release was not accepted against the repository's newer isolated
  expressive/cultural challenge set; literal, weakened, or over-strong renderings require review.
- Short context-dependent utterances, specialist terminology, and long inputs need human review.
- Do not use unreviewed output for medical, legal, safety-critical, or certified translation.
- The custom architecture requires the repository code; hosted Transformers inference is not
  available out of the box, so the `widget` examples above show pre-generated output rather than
  a live run.

## License

The original sion_translate code and uploaded model release are provided under the MIT License. This does
not grant rights to user-provided inputs, generated service outputs, third-party software, or
third-party models used only as comparison baselines.
