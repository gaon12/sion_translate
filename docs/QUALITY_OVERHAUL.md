# Bidirectional and Expressive-Language Quality Overhaul

This document records the audit of the public checkpoint and local training pipeline as of 2026-08-05, the defects addressed by the code overhaul, and the procedure for validating a newly trained model. The most important conclusion is simple: **correcting the code does not improve existing weights**. The old tokenizer and indexed dataset are incompatible with the revised configuration. Isolate them in a recoverable backup, then train again from the beginning using the standard `artifacts/` paths.

## Confirmed causes and corrective actions

| Area | Confirmed problem | Corrective action |
|---|---|---|
| Per-direction quality | Published measurements differed by approximately 10 chrF: 59.81 for ko→ja and 49.87 for ja→ko. Checkpoint selection used aggregate mean loss and could hide the weaker direction. | Aggregate unsmoothed NLL/PPL per direction and use `macro_direction_nll` as the default selection metric. Also record the worst-direction metric. |
| Direction graph | Source-only languages could become denoising targets, and evaluation and post-training did not consistently use the directions that the bidirectional graph could actually train. | Exclude `kj`, `kd`, and `jd` from target/denoising use and record reverse-edge training eligibility explicitly in each batch. |
| Hugging Face path | Native and Hugging Face paths disagreed on normalization, direction tags, and generation defaults. | Align NFC normalization and trimming, bidirectional BOS/tags, beam and length settings, repetition suppression, and control-token suppression. |
| Post-training | One path used TETM memory to score candidates but did not provide that memory during candidate generation. Supervised auxiliary heads were also disconnected between MRT candidate and reference forwards. | Pass the same memory through sampling, scoring, and validation, and preserve CoRe, BATS, evidence, and parity supervision during the reference pass. |
| Speaker register | During training, CoRe gave the decoder the gold register embedding, while inference used the predicted register. | Use the gold register only for classification loss. Condition the decoder on the predicted distribution in both training and inference. |
| Short expressive text | Valid short or repetitive expressions such as “Ah!”, prolonged cries, laughter strings, and moans could be removed as `too_short` or `excessive_repetition`. | Relax only those two reasons for reviewed `expressive_v1` rows. Keep safety checks for control characters, language mismatch, and structural corruption. |
| MRT reward | Repetition and copy penalties were applied even when the reference itself was a repeated exclamation or correctly preserved source text. | Exempt repetition and copying supported by the reference from those penalties. |
| Data leakage | Putting expressive examples into both training and evaluation would make improvement impossible to measure. | Fix a human-reviewed split of 18 training pairs and 12 challenge pairs, then expand only the challenge set to 24 bidirectional cases. `easy_run.py` generates only the training shard. |
| Artifact reuse | The audited local `artifacts/` contained an old two-language tokenizer and v2 dataset, while the audited training configuration used five languages and a source-only policy. | Keep `artifacts/` as the public path, but validate tokenizer SHA-256, digit splitting, language graph, control tokens, dataset schema, and source fingerprints. Because several runs may share this path, stop on mismatches rather than moving or overwriting artifacts automatically. An operator must inspect all related checkpoints before making a manual backup. |

Release 1.5 generalizes artifact identity to an explicit, configuration-driven BCP 47 language graph. The five-language graph above describes the 2026-08-05 audit configuration, not a hard-coded requirement for every user.

## Audit of legacy token exposure

The audit scanned `.bin` and `.idx.npy` files directly instead of tokenizing every training shard again. The complete report is in [`legacy-token-exposure-2026-08-05.json`](audits/legacy-token-exposure-2026-08-05.json).

```text
physical pairs                  11,129,222
virtual directions              22,258,444
ja→ko target content tokens    194,729,731
ko→ja target content tokens    185,729,708
ordinary vocabulary pieces          47,663
unused ordinary pieces                    11
seen exactly once                        194
seen 1–9 times                         1,376
seen 1–24 times                        2,295
median observed count                  2,503
```

The findings were:

- Byte fallback was low: 0.002762% for Korean and 0.006657% for Japanese. Complete inability to represent strings was therefore not the primary coverage problem.
- However, 2,295 ordinary pieces, or 4.8%, appeared fewer than 25 times as decoder targets, and 11 pieces never received a target update. **A sparse or untrained vocabulary tail was real.** A newly trained model must pass both a raw audit immediately after tokenizer training and an indexed audit immediately after dataset construction.
- The ja→ko target-token count was approximately 4.8% higher than ko→ja, so the ja→ko deficit cannot be explained by lower total token volume alone. More direct candidates include per-direction data quality, Korean surface-form diversity, global-mean checkpoint selection, and training/inference mismatch.
- The legacy tokenizer used `split_digits=false` and contained only the `ja` and `ko` language tags. It could not continue training with the audited five-language configuration. The changed embedding vocabulary requires full retraining.

Reproduce the legacy audit with:

```bash
sion-audit-tokens \
  --dataset artifacts/dataset \
  --tokenizer artifacts/tokenizer/sion.model \
  --split train \
  --rare-threshold 25 \
  --output legacy-token-audit.json
```

After preparing a new dataset, audit the new content fingerprint at the same public paths.

```bash
sion-audit-tokens \
  --dataset artifacts/dataset \
  --tokenizer artifacts/tokenizer/sion.model \
  --split train \
  --rare-threshold 25 \
  --fail-byte-rate 0.001 \
  --output current-token-audit.json
```

The correct value for `--fail-rare-pieces` depends on corpus and vocabulary size. Save the first full scan as the baseline before setting a CI ceiling. The indexed audit counts stored content tokens exactly, but does not include sampler reweighting, repeated epochs, BOS/EOS or language tags, denoising, or collator truncation.

## Contract for profanity, moans, and idiomatic expressions

The seed schema requires `category`, `subcategory`, `intensity`, `register`, `localization_strategy`, and `split`. The three top-level categories are `profanity_slang`, `interjection_moan`, and `idiom_culture`.

```bash
python scripts/data/build_expressive_cultural_corpus.py \
  --training-output data/synthetic_expressive_cultural.jsonl \
  --challenge-output examples/expressive_cultural_cases.jsonl \
  --report reports/expressive-cultural-build.json
```

`synthetic_expressive_cultural.jsonl` follows the training-only synthetic-sampling policy, and challenge sentences never enter that file. The 18 pairs are regression anchors that fix meaning and intensity; they are not sufficient training volume. Use existing natural dialogue and large shards such as `synthetic_netspeak.jsonl` for practical quality, but do not inflate counts through excessive copies of one template.

Evaluate both the overall challenge average and all three categories.

```bash
sion-translate-cases \
  --backend sion \
  --cases examples/expressive_cultural_cases.jsonl \
  --model runs/auto/posttrain/exports/best/model_ema.pt \
  --tokenizer artifacts/tokenizer/sion.model \
  --output comparison_outputs/sion-expressive.jsonl

sion-compare \
  --cases examples/expressive_cultural_cases.jsonl \
  --system sion=comparison_outputs/sion-expressive.jsonl
```

## Scope implemented from the attached research design

Interpreting results requires a clear boundary between the research proposal and current code.

| Proposed idea | Current implementation | Not implemented |
|---|---|---|
| Decoder evidence re-query | A separate GQA source reread follows the decoder stack and applies bounded residual repair through a token-level uncertainty gate. Training tracks pre-repair argmax error, NLL gain before and after repair, penalties for invalid requests, and a request-rate ceiling. Generation projects evidence K/V once. | There is no selection and re-encoding of a source span, repeated query/response cycle, or regeneration after masking an output span. The current reread uses dense attention over every source token and therefore does not provide actual sparse-compute savings. |
| Semantic parity/checksum | Bidirectional contrastive loss and positive cosine align pooled source and teacher-forced decoder representations. Batch size one and an empty target remain finite. | There is no independently encoded checksum of generated text, structured parity for relations, negation, or numbers, or automatic repair triggered by an inference syndrome. The feature is therefore a training-time representation-parity ablation. |
| Adaptive budget | Budget loss is applied only when the evidence-request rate exceeds its configured ceiling. Requests without useful NLL gain receive an additional penalty. | The system does not change latent-token count, numeric precision, or encoder depth per input. |
| Multi-channel latent representation | Existing modules divide roles: CoRe handles register/style, TETM protects entities, and BATS handles alignment. | There is no channel-orthogonality or information-separation loss, so the project does not claim an explicitly disentangled latent channel. |
| Counterfactual pairs | Provenance/category sidecars and a separated challenge set provide a basis for later data experiments. | There is no counterfactual encoder or loss that separates change vectors. The pipeline does not mass-generate unverified automatic counterfactual pairs. |

`sion_translate.yaml` disables evidence and parity by default. These modules cannot be enabled only at inference on existing weights; they must be **trained from initialization and validated through ablation experiments**.

Recommended experiment order:

1. Train a baseline with the same tokenizer, data, and seed.
2. Enable only `evidence_repair_enabled: true`.
3. Start from a fresh initialization and enable only `semantic_parity_enabled: true`.
4. Enable both only after each individual gain is reproduced.

For every run, record `macro_direction_nll`, `worst_direction_nll`, ko→ja and ja→ko chrF, number and proper-name preservation, all three expressive categories, language purity, repetition, and copying. For evidence runs, also compare `evidence_request_rate`, `evidence_repair_gain`, and latency. Do not adopt a run when the average improves but ja→ko or a specific category regresses.

## Retraining procedure

In a trusted editable checkout on the GPU server, the following command connects
expressive-seed generation, tokenizer and dataset construction, SFT, and MRT. An
authenticated extracted GPU bundle must omit the local-checkout option.

```bash
python3 easy_run.py --allow-local-checkout
```

If legacy `artifacts/tokenizer` or `artifacts/dataset` contents fail identity checks, training stops instead of replacing them automatically. Because multiple runs may share the vocabulary, an operator must inspect related checkpoints and manually move both directories to a separate recoverable location. A valid run uses these stable public paths:

```text
artifacts/tokenizer/
artifacts/dataset/
runs/auto/pretrain/
runs/auto/posttrain/
```

Passing code tests does not establish that training is complete. Completion requires fixed-challenge results from newly trained weights. Until GPU retraining has run and those measurements have been reviewed, this work is a **training pipeline with known quality defects corrected**, not a **new model with measured quality improvement**.
