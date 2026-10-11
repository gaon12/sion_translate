# v1 numeric inference update

The public v1 EMA weights can preserve numeric literals without retraining by using
the tokenizer's existing protected slots. This maintenance branch keeps the original
v1 model and tokenizer vocabulary and updates inference only.

**Existing v1 users: download the updated inference code.** Downloading the same
model weights alone will not activate this fix. Existing verified `model_ema.pt`,
`kjx.model`, and `token_features.npz` files can be reused.

```bash
python -m pip install --upgrade "sion-translate @ git+https://github.com/gaon12/sion_translate.git@c2ccc5210f2436017e2adaf067d02a7c83ceec98"
python -m pip install huggingface_hub
hf download gaon12/sion_translate model_ema.pt kjx.model token_features.npz inference_v1.py v1_numeric_hotfix.json --local-dir models/sion_translate
python models/sion_translate/inference_v1.py --text "회의는 17시 30분에 시작합니다." --target-language ja --device cpu
```

```python
from sion_translate.inference import Translator

translator = Translator("models/sion_translate/model_ema.pt", "models/sion_translate/kjx.model")
print(translator.translate(["17시 30분"], target_language="ja")[0])
# 17時30分
```

The decoder sees units such as `시`, `분`, and `mg` while numeric values are masked.
This preserves unit context and avoids the extra counters v1 can invent when a
complete value/unit expression is hidden. Multi-part phone/account identifiers and
`±` signs are restored together with their values.

With a non-digit-splitting tokenizer, `strict_numbers=True` is the default. Dropped
or duplicated protected values and incorrect Korean/Japanese hour/minute associations
raise `NumericPreservationError`. Invalid sampled candidates are excluded from reranking.
Numeric values that cannot fit in the available protected slots also raise an error. The old behavior
of appending a missing number in parentheses is available only through the diagnostic
opt-out `strict_numbers=False`; it must not be presented as a verified translation.

The update also rebuilds non-persistent RoPE buffers after safely loading the state
dictionary, so CPU/CUDA device transfers do not encounter unfilled meta tensors.

## Evidence and scope

The focused probe used the public FP32 EMA (SHA-256
`829580d1db1693ec00ee641398ce6ba9549388e266f6259f85e5a19ea28bfc2e`),
CPU, beam 4, length penalty 1.0, and a 100-token output cap. It includes 18 authored
inputs across both directions: times, decimals, percentages, money, identifiers,
signed tolerances, counters, and two greetings.

Examples from direct execution:

| Source | Original v1 runtime | Updated runtime |
|---|---|---|
| `17시 30분` | `１７時半` | `17時30分` |
| `농도는 0.0037mg/L 이하로 유지하세요.` | `濃度は1.337mg/L以下に保ってください。` | `濃度は0.0037 mg/L以下に保ってください。` |
| `250mg씩 하루 두 번 복용하세요.` | `1200mgずつ1日2回服用してください。` | `250mgずつ1日2回服用してください。` |

The specific `1730분` output was not reproduced with the original runtime at beam 4;
the regression tests explicitly reject that malformed hour/minute representation.
`１７時半` is a valid equivalent time, while the changed amounts are value corruption.

This is an inference correction, not new model training or a general quality benchmark.
It does not improve the model's learned arithmetic, word-written numbers, arbitrary
unit semantics, or general translation accuracy. The v1 EMA path is verified; the
legacy pickled INT8 download is not part of this release's safe loading probe.

Machine-readable diagnostic outputs: [validation report](V1_NUMERIC_HOTFIX_VALIDATION.json).
