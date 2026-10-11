# GPU training audit — 2026-09-03

The short production-compute probes passed on Modal A100 SXM4 80GB and exact H100
80GB. They do **not** prove that a complete epoch, validation/export pipeline,
convergence, or a weeks-long Modal deployment will finish. In particular, CPU
stage-configuration checks found a real foundation startup failure that the old
microbenchmark missed; that failure is now fixed and regression tested.

The default `sion_translate.yaml` now matches the measured 80GB profile, including
MRT's 100,000-update cap and its effective batch of 16 (`8f2ff35`). Plain
`easy_run.py` therefore selects this budget. Historical uncapped schedules below
remain comparison scenarios; they are no longer the default launch settings.

## How to interpret the long durations

**The hundreds-of-days MRT extrapolation is not a usable training schedule.** It
multiplies random-initialization generation throughput across the complete corpus.
The measured probes establish short-run throughput and memory, not how much
training achieves useful translation quality. The 100,000-update MRT cap below
was chosen as a cost scenario; no validation result establishes it as necessary.

The current setup trains a 273M-parameter model from scratch, adds a separate
foundation stage, and processes 56.1 million directed translation examples per
epoch. MRT additionally samples four candidates per source and performs reverse
translation. These are distinct workloads. At the measured H100 throughput:

| Workload | Projected compute time, excluding unmeasured overhead |
|---|---:|
| Foundation, one epoch | 61.6 hours |
| SFT translation, one epoch | 46.1 hours |
| SFT translation, three epochs | 138.4 hours (5.8 days) |
| MRT, 1,000 updates / 16,000 sources | 1.41 hours at random-weight probe speed |

These smaller units make the workload visible; they are not new convergence
predictions. Decide training length from held-out translation quality and loss
trends during a bounded pilot. Measure MRT again using an actual SFT checkpoint
and its generated lengths before projecting its schedule. Keep candidate
refinement and quality checks in that measurement. The existing 3+3-epoch / 100k
profile is a measured memory configuration and a budget scenario, not a validated
quality recipe. No shorter quality target or full training run was established.

Follow-up performance fix `3914aa9` removes unused draft-loss reductions from
unlabelled decoding. With 48,000 vocabulary entries and 2,048-entry chunks, each
refinement step previously called log-sum-exp, log-add-exp and a loss-only sum 24
times apiece per decode step. The full-vocabulary expectation is unchanged. This
also applies to MRT candidate and round-trip generation. FP32/BF16 tests with tied
and untied output embeddings verify identical expectations and gradients; the
56-test refinement/posttraining suite passes. Formatting/lint and the configured
`src` Pyright check pass. Explicitly checking the legacy test file with strict
Pyright is outside that configured scope and reports errors; it is not a passing
test-file type-check gate. No new GPU probe was run, so the timing table below
predates this optimization and does not quantify its speedup.

## What was fixed

- Foundation derivation cleared source-only languages but retained their synthetic
  evidence files. `train()` then rejected the configuration before its first update.
  Derivation now clears both, while leaving the translation configuration intact.
  The probe now validates every derived stage before constructing its GPU model.
- Recovery saves previously depended only on update count. Slow MRT could leave
  hours of work between checkpoints. `training.save_every_seconds: 900` now also
  saves at the next completed optimizer update, with one rank-zero decision for
  distributed jobs. Partial accumulation is never saved; timed saves skip exports.
- Masked native GQA selected PyTorch's math attention. Expanding K/V only for the
  masked SDPA call enables efficient attention, while the returned KV cache stays
  compact. Profiling confirmed 27 efficient-attention and 9 Flash calls per CE
  forward, replacing the previous 27 math and 9 Flash calls.
- Modal scalar CPU/RAM requests allowed paid bursts beyond the resource estimate.
  Probes now use matching requests/limits: 8 physical cores and 48 GiB RAM.
- Reading status while an image was building raised `KeyError: call_id`.
  Pending status is now readable without overwriting the submitter's receipt.

## Measured scope and results

The authenticated snapshot contains 28,148,216 physical parallel pairs and
1,550,725,850 unique parallel tokens. Automatic sizing selected **273,484,225
parameters**, vocabulary 48,000, width 832, 18 encoder / 9 decoder layers, 8 query /
4 KV heads, FFN 2304. This is not `configs/sion_1_3b.yaml`.

Foundation has 56,593,480 sampled examples per epoch; SFT/MRT has 56,109,623 directed
examples per epoch. The probes used eager BF16, FP32 matmul precision `high`, AdamW,
EMA, production candidate refinement and MRT round-trip rewards, and lengths up to
512. All measurements use random initialization and a resident 4,096-row cohort.

| GPU / stage | Batch / accumulation | Candidate microbatch | Examples/s | Long-row reserved VRAM |
|---|---:|---:|---:|---:|
| A100 / foundation | 32 / 1 | — | 101.86 | 54.61 GiB |
| A100 / SFT | 32 / 8 | — | 116.52 | 49.65 GiB |
| A100 / MRT | 8 / 2 | 4 | 1.23 | 44.69 GiB |
| H100 / foundation | 32 / 1 | — | 255.29 | 54.62 GiB |
| H100 / SFT | 32 / 8 | — | 337.87 | 49.68 GiB |
| H100 / MRT | 8 / 2 | 4 | 3.14 | 44.70 GiB |

The additional A100 batch-24 probe measured foundation/SFT memory falling from
54.04/50.24 GiB in the prior version to 39.82/37.93 GiB. Its MRT batch 8 with candidate
microbatch 8 passed at 1.50 examples/s and 44.69 GiB, faster than candidate microbatch
4 on the later A100 allocation. This is an additional A100 tuning option, not a
matched H100 comparison. CPU limits and host variance changed between the historical
and current runs; the memory improvement is established, but an isolated throughput
speedup from the attention patch is not established.

**MRT source batch 16 failed the longest-row stress on both GPUs**, even with
candidate microbatch 4. Its faster short-row timing is excluded from estimates.
Successful candidates also saved and restored a roughly 4.08 GiB optimizer/EMA
checkpoint and completed a resumed update. Recovery does not cover every full-run
failure mode, and the probe does not run production validation or final exports.

The native binding/PyTorch stack was Python 3.11.12, torch 2.10.0+cu128,
cuDNN 9.10.2.21, NCCL 2.27.5, SentencePiece 0.2.1 with the verified native wrapper,
and transformers 5.16.1. The source hashes, runtime reports, checkpoint timings,
excluded OOM cases and estimates are in
[`audits/modal-training-2026-09-03.json`](audits/modal-training-2026-09-03.json).
Full receipts and stderr remain under `artifacts/modal-probes/` and the Modal
`sion-training-probe-results` Volume. GPU compute probes precede the final foundation
configuration-only fix (`c00799e`); CPU tests cover that fix. The tensor computation
used in the measured probes is unchanged by it.

## Time and money scenarios

These are single-GPU **extrapolations**, not finish-time promises. The formula is
stage examples × epochs / measured examples per second, with measured accumulation
used for an explicit step limit. The range adds an assumed 20–50% for unmeasured
work. It is not a statistical confidence interval. Trained MRT termination lengths,
full-corpus I/O, validation, exports, restarts and sample variance can change the
result substantially.

At [Modal's GPU Tasks list prices](https://modal.com/pricing), GPU-only rates are
$2.4984/h for A100 80GB and $3.9492/h for H100. Including the assumed 8 CPU cores and
48 GiB RAM gives **$3.259296/h** and **$4.710096/h**. Storage, image builds, networking,
region/non-preemptible premiums, discounts and credits are excluded.

| Schedule using the table's batches | A100 time + 20–50% | A100 compute cost | H100 time + 20–50% | H100 compute cost |
|---|---:|---:|---:|---:|
| Foundation 3 epochs + SFT 3 epochs + MRT 2 epochs | 1,313–1,642 days | $102,727–128,409 | 512–640 days | $57,876–72,345 |
| Foundation 3 epochs + SFT 3 epochs + MRT at most 100,000 updates | 61–77 days | $4,797–5,996 | 23–29 days | $2,626–3,282 |

Without the added overhead, foundation + SFT alone is about 36.0 days on A100 or
13.5 days on H100. Full-corpus MRT dominates the original 3+3+2 epoch budget.
The 100,000-update scenario processes at most 1.6 million MRT source rows, rather
than 112.2 million. It is a budget cap for evaluation, not a measured quality target.
The reported runtime cannot justify multiplying speed linearly for multiple GPUs.

The three new bounded probes reported **$0.6556 in compute charges** when reconciled.
All three apps were stopped with zero tasks. Billing can arrive late; image/storage
charges are outside that figure. Modal emitted its known container-bootstrap
`ResourceWarning` and the client timed out waiting for final app logs. Those messages
were retained; completion was established from successful FunctionCall results and
matching durable journals, not the client's “App completed” text.

## Reviewed execution candidate

[`configs/modal_80gb.yaml`](../configs/modal_80gb.yaml) pins the measured architecture
and uses SFT 32×8 and MRT 8×2 with candidate microbatch 4. It caps MRT at 100,000 updates
and sets timed recovery saves. Foundation changes effective batch from 16 to 32;
SFT and MRT retain effective batch 256 and 16 respectively. A new run directory
avoids incompatible resume identities. This profile has now also been selected in
`sion_translate.yaml`; it retains foundation/SFT epochs and caps MRT updates.

Revalidate the profile after changing vocabulary, model, lengths, objectives or the
corpus distribution. The currently edited raw corpora are not the prepared snapshot
used for these measurements. This profile has headroom; filling the final free GiB
does not establish better throughput or safety. No GPU-utilization percentage was
measured, and these results do not establish globally optimal batch sizes.

Before a full Modal run, use persistent Volume checkpoints and resumable function
windows. [Modal functions have a 24-hour maximum timeout](https://modal.com/docs/examples/long-training);
the repository's Modal scripts measured here are bounded probes, not a deployed
weeks-long training scheduler. The 900-second save interval is checked after an
update and is not a hard deadline for a blocked kernel, validation, or I/O.

Reproduce the estimates without renting a GPU:

```powershell
python scripts/estimate_training_cost.py artifacts/modal-probes/probe-20260903t111853z-c5cefa0e8573/result.json
python scripts/estimate_training_cost.py artifacts/modal-probes/probe-20260903t111853z-c5cefa0e8573/result.json --mrt-max-steps 100000
```

## Local verification

- Repository formatting and lint passed; Pyright reported zero errors/warnings.
- Full local suite: 3,073 passed, 14 skipped in 19m23s. The later foundation/probe,
  estimator and preset-registration changes were covered by a final targeted run:
  221 passed. These counts overlap and must not be added as distinct tests.
- The timed-save regression injects a DataLoader failure after partial accumulation
  and verifies that the surviving checkpoint records only the completed update.
  Attention tests compare output/parameter gradients and preserve compact KV caches.
- Local tests used Windows/Python 3.11 and CPU torch 2.13; the separate Modal probes
  exercised the locked Linux CUDA runtime above. Environment-dependent skips do not
  establish unsupported hardware paths as passing.
- JUnit evidence is under `artifacts/training-audit-tests.xml` and
  `artifacts/training-audit-final-tests.xml`. The new preset and both derived stage
  configurations were also parsed and validated locally.
