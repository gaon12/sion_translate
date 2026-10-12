# v1 numeric inference update — runtime 0.1.2

Existing v1 users should download the updated inference code. No retraining or
weight replacement is required. Verified model_ema.pt, kjx.model, and
token_features.npz files can be reused. The main training stack is unchanged.

~~~bash
python -m pip install --upgrade "sion-translate @ git+https://github.com/gaon12/sion_translate.git@45254b8cb84e7725aa7b58709f060317ff4e2ab6"
python -m pip install huggingface_hub
hf download gaon12/sion_translate model_ema.pt kjx.model token_features.npz inference_v1.py v1_numeric_hotfix.json NUMERIC_HOTFIX.md --local-dir models/sion_translate
python models/sion_translate/inference_v1.py --text "회의는 17시 30분에 시작합니다." --target-language ja --device cpu
~~~

## Corrections verified with the original v1 EMA

- Spaced signs are protected together with their value: ± 0.05 is retained.
- Recognized measurement units remain visible to the decoder and are restored
  at their numeric slot. The previous 250mL → 250 mm error now returns 250mL.
- Compound units and square notation are retained, including mg/L, km/h,
  and km². The previous 3,850 km mL and missing-square errors now return 3,850km².
- Numeric scanning no longer requires a known suffix. The tested 15mile and
  40mile values both survive. Decimal scientific notation is protected as a whole.
- Pure uppercase alphabetic acronyms such as AI and USAF stay visible. The five
  previously identified acronym regressions are corrected in direct execution.

The fix uses existing tokenizer slots; it does not change vocabulary or weights.
Units use a bounded allowlist of literal measurement symbols and Korean/Japanese
aliases. Recognized units are copied in their source notation, so output can retain
Latin symbols such as mile instead of using a localized unit name.

## Validation and limits

On 2026-10-12, the original FP32 EMA was executed on CPU with beam 4, length
penalty 1.0, and max_new_tokens 256. The final follow-up attempted
**238 inputs: 236 outputs and
2 blocked inputs**. This includes all 72 authored
numeric stress cases, 16 existing authored diagnostics, known failures, and every
FLORES input affected by the changed preprocessing or recognized-unit restoration.
It is a focused follow-up, **not a new full FLORES evaluation**.

All 72 stress inputs returned output. The 68 inputs
containing digit literals preserved all source digit literals under the independent
literal check. Four word-written-number inputs are diagnostic only. This does not
establish semantic accuracy for every numerical statement.

The rerun FLORES subset had 148
accepted outputs and 2
blocked inputs. Among the paired accepted inputs without digit literals, average
chrF++ increased in both directions. This is an average for a selected subset and
does not guarantee that every individual translation improves. Related tests:
**98 passed**, plus Ruff and whitespace checks.

The earlier full 2,024-input FLORES evaluation belongs to runtime 0.1.1
(c2ccc5210f24), not this follow-up. Its paired chrF++ was 21.252 → 23.270 for
Korean→Japanese and 25.860 → 26.766 for Japanese→Korean compared with the original
runtime. It returned 2,013 outputs and blocked 11 inputs. Its 99.3% literal
preservation result excluded blocked inputs and was not a semantic accuracy score.

With the non-digit-splitting v1 tokenizer, strict_numbers=True rejects missing or
duplicated protected values and unsupported hour/minute associations. It restores
a recognized adjacent unit and rejects a unit whose position cannot be located.
The follow-up still blocks two numerical-omission cases involving parenthesized
ages and currency equivalence. No guessed values are appended to make these pass.
strict_numbers=False is for inspecting unverified output and disables these checks.

This is a partial inference correction. It does not guarantee arbitrary units,
word-written numbers, arithmetic, numeric-to-entity associations, or complete
translation semantics. The verified path uses the safe FP32 EMA state dictionary;
the legacy pickled INT8 artifact is outside this validation.

**한국어 안내:** 확인된 부호·단위·숫자 인식 누락·약어 회귀를 재학습 없이
보완했습니다. 기존 사용자는 추론 코드를 다시 설치하세요. 기존 가중치는
그대로 사용할 수 있습니다. 숫자가 완벽하게 안전한 버전이라는 뜻은 아니며,
값을 생략하는 일부 문장은 여전히 오류로 차단됩니다.

Machine-readable evidence: [validation report](V1_NUMERIC_HOTFIX_VALIDATION.json).
