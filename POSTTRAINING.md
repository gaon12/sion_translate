# Posttraining design and experiment guide

## Implemented objective

The posttraining loss combines reference supervision, expected sequence risk, and
candidate preferences:

```text
L = L_reference_CE
  + L_reference_auxiliary
  + risk_weight * L_composite_MRT
  + preference_weight * (L_candidate_pairs + L_reference_anchor)
```

- `L_reference_CE` preserves the likelihood of the gold translation while sequence-level
  rewards change the candidate distribution.
- `L_reference_auxiliary` keeps enabled supervised heads, such as semantic alignment and
  evidence budgeting, active during MRT.
- `L_composite_MRT` minimizes expected risk over candidates sampled from the current
  model.
- `L_candidate_pairs` trains every candidate pair whose reward gap exceeds
  `preference_min_gap`.
- `L_reference_anchor` compares sampled candidates with the gold sequence. A rewarded
  sample cannot become preferred merely by outranking other weak samples; its score must
  remain anchored to the reference.

All candidate and reference scores use mean token log-probability, which avoids a
systematic preference for short sequences. Non-finite rewards or sequence scores stop the
step instead of silently contaminating optimizer state.

## Composite reward

The local reward combines:

- chrF reference similarity;
- token multiset F1;
- number preservation;
- structured token and placeholder preservation;
- protected glossary-slot preservation;
- target-language character evidence when a profile is available;
- output/reference length agreement;
- optional round-trip reconstruction.

It separately penalizes excessive repetition, inappropriate source copying, invented or
dropped numeric values, and low round-trip quality. Numeric corruption is a fixed penalty
in addition to the weighted number component because a small ratio loss can otherwise be
overcome by a modest chrF gain.

Tune number, structured, and slot weights for domains where exact values, code, or
localization placeholders are critical. Every change requires an independent holdout;
the training reward is not sufficient evidence of improvement.

## Deployment-aligned generation

Candidate sampling and posttraining validation use the same reference-free constraints as
native deployment:

- training-only control tokens are forbidden;
- a minimum output length is enforced;
- repeated n-grams are blocked;
- per-row output limits are derived from source length plus a bounded margin;
- validation uses the deployment beam count and length penalty;
- candidate-refinement reasoning uses the checkpoint's configured endpoint.

Reference target length is never used to choose a validation or candidate generation
limit. These decoding controls are part of checkpoint objective identity, so changing one
cannot silently resume against an incomparable historical best metric.

## Best-checkpoint selection

Posttraining selects and early-stops on generated translation reward, not teacher-forced
cross-entropy. CE remains in telemetry as a drift diagnostic.

`posttraining.selection_metric` supports:

- `worst_direction_reward` (default): optimize the weakest authenticated direction;
- `macro_direction_reward`: average every observed direction equally;
- `reward`: global row-weighted mean, retained for compatibility.

Per-direction metrics use the form
`validation_direction_<source>_to_<target>_reward`. If a custom caller does not provide
direction metadata, the trainer falls back to global reward and reports the fallback.

This logic operates on arbitrary BCP 47 direction graphs. It does not assume a particular
pair or a bidirectional topology.

## Why these components exist

- [Minimum Risk Training for Neural Machine Translation (ACL 2016)](https://aclanthology.org/P16-1159/)
  provides the expected sequence-risk foundation.
- [M2PO: Multi-Perspective Multi-Pair Preference Optimization (ACL 2026)](https://aclanthology.org/2026.acl-long.469/)
  motivates multi-perspective rewards and more than one preference pair.
- [Direct Quality Optimization for Neural Machine Translation (WMT 2025)](https://aclanthology.org/2025.wmt-1.2/)
  demonstrates direct preference optimization for encoder-decoder NMT.
- [Word Alignment as Preference for Machine Translation (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.188/)
  motivates explicit omission and hallucination signals.
- [xCOMET (TACL 2024)](https://aclanthology.org/2024.tacl-1.54/) and
  [Fine-Grained Reward Optimization (TACL 2026)](https://aclanthology.org/2026.tacl-1.33/)
  motivate error-location and severity rewards. They require a separate large model and
  are not mandatory runtime dependencies here.
- [Metric Bias in Minimum Bayes Risk Decoding (WMT 2024)](https://aclanthology.org/2024.wmt-1.109/)
  explains why evaluation with only the optimized metric can hide reward hacking.
- [Unlikelihood Training for Neural Machine Translation (COLING 2020)](https://aclanthology.org/2020.coling-main.462/)
  motivates explicit repetition suppression.

## Recommended ablation order

Use the same SFT checkpoint, tokenizer, data fingerprint, validation/test splits,
direction graph, and decoding policy for every comparison.

| Experiment | Settings | Question |
|---|---|---|
| A | `risk_weight=0`, `preference_weight=0` | What does reference CE alone preserve? |
| B | chrF-only MRT | What does the historical sequence-risk baseline change? |
| C | Composite MRT | Do preservation, repetition, and copy failures improve? |
| D | Composite MRT plus all candidate pairs | Does candidate ordering improve beyond expected risk? |
| E | D plus gold/reference preference anchoring | Do rewarded samples remain faithful to the reference? |
| F | E plus offline external severity rewards | Do independent fine-grained errors improve enough to justify the cost? |

Record at least:

- chrF and BLEU globally and for every authenticated direction;
- worst-direction and macro-direction values;
- exact number, URL, email, code, placeholder, and glossary-slot preservation;
- repetition, empty-output, source-copy, and output/source length distributions;
- results by general, technical, conversational, and other relevant domains;
- a blind human evaluation large enough to support the claimed conclusion.

Do not select experiment F with the same external metric used to create its preferences.
Score candidates offline when possible, store exact model and dataset provenance, and
compare with independent metrics and human review.

## Tuning rules

- If candidate rewards are nearly tied, increase `samples_per_source` gradually or lower
  `preference_min_gap` toward 0.02.
- If language quality becomes unstable, reduce `preference_weight` first, then
  `risk_weight`. Keep the reference CE and gold preference anchors enabled.
- If repetition and copying improve but chrF falls, halve the relevant penalty and repeat
  the complete evaluation.
- If validation reward rises while independent test metrics fall, treat it as reward
  hacking. Rebalance components or add an independent signal.
- If one direction regresses while the global mean rises, retain
  `worst_direction_reward` or use a documented macro-direction policy.
- If MRT runs out of memory, lower batch size and candidate micro-batch before reducing
  the number of distinct candidates.
