# Budget-gated Modal GPU smoke tests

This smoke test exercises the CUDA paths that are most likely to fail after a
paid training job starts. It does not train the corpus and does not claim a
translation-quality score. It verifies the reviewed environment, the selected
accelerator, one production-sized train/inference/checkpoint cycle, and the
two-rank distributed path where applicable.

## Cost and failure boundaries

Every command starts exactly one hardware target. There is deliberately no
`all` mode. Each Modal function uses:

- application retries set to zero;
- a 300-second function-execution timeout;
- a separate 180-second container-startup timeout;
- at most one container, with no warm or buffered containers;
- a two-second scale-down window and a single input per container;
- 4 CPU cores, 32 GiB of host memory, and 16 GiB of temporary disk;
- a 150-second maximum for the nested two-rank job; and
- a 20-second parent margin reserved for killing and reaping every `torchrun`
  process if the nested job stalls.

`--max-dollars` is an explicit authorization check, not a Modal account-level
spending cap. The check reserves two complete startup and function windows
because Modal can reschedule an ephemeral container after an infrastructure
crash or OOM even when application retries are zero. Modal may run a function a
few seconds beyond its configured timeout. A workspace budget is therefore the
real external limit. Run one target, inspect Modal billing, and only then run
the next target.

The table uses the public per-second GPU, CPU, and memory prices recorded on
2026-08-31. The authorization is a two-attempt compute contingency for the
configured 4 CPU cores, 32 GiB of memory, and requested GPU resources. It still
excludes image-build, storage, network, and timeout-overrun charges.

| Target | Modal request | Required compute contingency |
| --- | --- | ---: |
| A100 40 GB | `A100-40GB` | $0.68100816 |
| A100 80 GB | `A100-80GB` | $0.78801216 |
| Exact H100 | `H100!` | $1.17650416 |
| Two A100 40 GB GPUs | `A100-40GB:2` | $1.24302016 |

The JSON field `estimated_function_compute_charge_usd` covers the configured
GPU, CPU, and memory during the observed function body plus the two-second idle
window. Observed elapsed time is never truncated to the configured timeout, so
an overrun raises the estimate instead of hiding it. It is not presented as a
billing total. Use
`modal billing report --for today --resolution h --show-resources` for an hourly
workspace record. The current hour can still be omitted while its interval is
open, and billing ingestion can lag. Before authorizing another target, also
inspect the Modal dashboard's remaining workspace credit; do not treat the CLI
report alone as an immediate hard balance check.

## Recommended sequence

Use the repository virtual environment so the authenticated Modal profile and
Modal 1.5.3 client are selected. The entrypoint rejects any other local client
version before it invokes a paid function. With a small balance, the minimum useful set is
the A100 40 GB target, the exact H100 target, and the two-A100 target. Together
they cover Ampere, Hopper, activation checkpointing, non-checkpointed execution,
and FSDP2. The A100 80 GB target is an additional single-GPU capacity check and
can wait for the next credit reset.

Run each command separately:

```powershell
.venv\Scripts\python.exe -m modal run scripts/modal_gpu_smoke.py `
  --target a100-40gb --max-dollars 0.69

.venv\Scripts\python.exe -m modal run scripts/modal_gpu_smoke.py `
  --target h100 --max-dollars 1.19

.venv\Scripts\python.exe -m modal run scripts/modal_gpu_smoke.py `
  --target a100-40gb-x2 --max-dollars 1.25

# Optional after the balance is replenished.
.venv\Scripts\python.exe -m modal run scripts/modal_gpu_smoke.py `
  --target a100-80gb --max-dollars 0.80
```

Do not start the next command until the previous command has returned a strictly
validated JSON result and the billing report has been inspected. `H100!` is
intentional: Modal documents that plain `H100` may be upgraded to H200, while
the exclamation mark requests an exact H100.

## Authenticated environment

The image uses CPython 3.11 on Linux x86-64 with glibc 2.28 or newer. The uv
0.12.3 bootstrap wheel is hash-pinned in `requirements/modal-bootstrap.txt`.
uv then synchronizes the complete PEP 751 GPU lock with required hashes, strict
validation, and binary wheels only. Before CUDA work starts, the remote process
checks the lock digest and the exact versions of Torch, CUDA, torchao, NumPy,
SentencePiece, Transformers, cuDNN, and NCCL.

The runtime rejects the wrong GPU count, family, compute capability, memory
class, missing native BF16, an H200 substituted for the exact H100, and any
non-finite or malformed result field.

## Single-GPU evidence

Every target performs these operations:

1. Run native-BF16 matrix multiplication.
2. Run the production 12-query-head/6-KV-head masked causal attention path,
   backward propagation, and fused AdamW.
3. Construct the exact reviewed 287,127,073-parameter model with dropout 0.1.
   A100 40 GB enables the same activation-checkpointing policy selected by the
   production hardware configuration; A100 80 GB and H100 keep it disabled.
4. Execute a teacher-forced T1-to-T2 full-distribution refinement step and
   require finite refinement loss, aggregate gain, the label-shaped per-token
   gain tensor, and every candidate-refinement gradient. The reported token gain
   is the validated mean over target tokens; padding must remain zero. The refinement
   scale must receive a nonzero
   gradient and change after the optimizer step.
5. Use the production training defaults for AdamW learning rate, betas, epsilon,
   weight decay, gradient clipping, BF16, and EMA decay.
6. Run inference with refinement disabled and enabled. A forward hook proves
   that cached autoregressive generation also invokes the trained refinement
   module, rather than merely returning a plausible output shape.
7. Save and reload a transactional checkpoint after deliberate corruption of
   a live parameter, an EMA shadow, and a non-scalar Adam moment tensor. Model,
   optimizer, scheduler, step, training metadata, RNG state, and EMA must
   round-trip.

## Two-GPU evidence

`a100-40gb-x2` additionally launches a real two-rank `torchrun` job with NCCL.
The small sharded model uses the production A100 interaction: BF16 FSDP2,
automatic BF16 reduction, dropout 0.1, activation checkpointing, fused AdamW,
EMA, and reshard-after-forward.

Both ranks must independently report:

- NCCL world size and rank;
- finite loss and materialized global DTensor gradient norm;
- one T1-to-T2 refinement step, every expected refinement gradient, and a real
  refinement-parameter update;
- a real optimizer update and finite optimizer state;
- distributed checkpoint and EMA save/reload success; and
- device name plus that rank's peak allocated and reserved CUDA memory.

The parent accepts exactly one report from each rank, aggregates the maximum
memory across both GPUs, binds each report to the authenticated runtime device,
and rejects non-finite JSON. The parent starts `torchrun` and both workers through
the Linux parent-death guard with automatic restarts disabled. A parent crash,
timeout, interruption, nonzero exit, or validation failure therefore cannot leave
an unguarded NCCL worker consuming the two paid GPUs. Reported sequential phase
durations must also fit inside the observed function duration used for the cost estimate.

## What this cannot prove

A short smoke test cannot prove long-run convergence, dataset throughput,
production-batch memory headroom, final BLEU/chrF/COMET quality, or freedom from
every possible hardware and filesystem failure. It reduces early paid-run risk
by testing the most failure-prone code paths under real CUDA. A separate
timeout-configured training probe and post-training evaluation remain required before publishing
the foundation and translation repositories.

Current Modal behavior and prices should be rechecked before running:

- [GPU pricing](https://modal.com/pricing)
- [Function timeouts](https://modal.com/docs/guide/timeouts)
- [Failures and container rescheduling](https://modal.com/docs/guide/retries)
- [GPU types and exact H100 selection](https://modal.com/docs/guide/gpu)
- [Billing CLI](https://modal.com/docs/cli/latest/billing)
