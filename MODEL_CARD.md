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

## v1 numeric inference update — runtime 0.1.2

**Existing v1 users: reinstall the updated inference code.** Existing verified v1
weights and tokenizer can be reused; no retraining or weight replacement is needed.
The follow-up preserves spaced signs, repairs recognized measurement-unit suffixes
(including compound units and powers), protects previously unrecognized numeric
suffixes, and keeps ordinary uppercase acronyms visible during translation.

Actual v1 EMA execution now preserves ± 0.05, 250mL, 3,850km², 15mile, and 40mile
in the previously failing cases. The five identified acronym regressions are also
corrected. This remains a **partial inference correction**: arbitrary units,
word-written numbers, arithmetic, and full numerical semantics are not guaranteed.
Some numerical omissions still cause an error instead of a translation.

The focused follow-up attempted 238 inputs, returned
236 outputs, and blocked 2.
It includes 72 authored numeric stress inputs, 16 existing diagnostics, and every
affected FLORES input. Average chrF++ on the rerun no-digit subset increased in both
directions; individual regressions are still possible. This was not a fresh full
FLORES rerun. Related regression tests: 98 passed.

**한국어 안내:** 부호·단위·숫자 인식 누락·약어 회귀를 재학습 없이 보완했습니다.
기존 사용자는 추론 코드를 다시 받으세요. 기존 가중치는 재사용 가능합니다.
숫자 처리가 완벽하다는 의미는 아니며, 일부 값 누락은 오류로 차단됩니다.

Details: [update guide](docs/V1_NUMERIC_HOTFIX.md).

Runtime commit: [45254b8cb84e](https://github.com/gaon12/sion_translate/commit/45254b8cb84e7725aa7b58709f060317ff4e2ab6).

## Files

- `model_ema.pt`: recommended FP32 EMA inference export
- `model_int8.pt`: smaller CPU-only dynamic INT8 export
- `kjx.model`, `kjx.vocab`, `token_features.npz`: tokenizer files

The model uses custom code from
[`gaon12/sion_translate`](https://github.com/gaon12/sion_translate) and is not a Transformers
`AutoModel` checkpoint.

## Usage

```bash
python -m pip install --upgrade "sion-translate @ git+https://github.com/gaon12/sion_translate.git@45254b8cb84e7725aa7b58709f060317ff4e2ab6"
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

## AI-Hub dataset attribution for public v1

The AI-Hub portion of the public v1 (KJ-X) training data includes the datasets
below. We acknowledge AI-Hub ([aihub.or.kr](https://aihub.or.kr/)) and the dataset
creators. This table credits those AI-Hub sources; it is not a complete inventory
of all training sources.

| AI-Hub dataset ID | Official dataset name and source |
|---|---|
| 127 | [한국어-일본어 번역 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=127) |
| 546 | [일상생활 및 구어체 한-중, 한-일 번역 병렬 말뭉치 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=546) |
| 71493 | [한국어-다국어(영어 제외) 번역 말뭉치(기술과학)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71493) |
| 71496 | [한국어-다국어 번역 말뭉치(기초과학)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71496) |
| 71498 | [한국어-다국어 번역 말뭉치 (인문학)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71498) |
| 71646 | [생성형AI 다국어 번역 품질 평가 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71646) |
| 71735 | [생성형AI 한-중/한-일 특허 및 기술과학 분야 병렬 말뭉치 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71735) |
| 71782 | [생성형AI 금융 분야 다국어 병렬 말뭉치 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71782) |
| 71591 | [방송콘텐츠 한국어-아시아어 번역 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71591) |
| 71671 | [생성형AI 라이브 스트리밍 영상 일본어 통번역 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71671) |

[AI-Hub's official FAQ](https://aihub.or.kr/aihubnews/faq/list.do) permits commercial
use and distribution of models developed using its data, with attribution to the
dataset names and AI-Hub. Redistribution of original or merely edited source data
is restricted. These terms apply to the AI-Hub sources listed here.

## License

The original sion_translate code and uploaded model release are provided under the MIT License. This does
not grant rights to user-provided inputs, generated service outputs, third-party software, or
third-party models used only as comparison baselines.
