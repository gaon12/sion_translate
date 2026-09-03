# Budget-gated Modal GPU smoke tests

This smoke test exercises the CUDA paths that are most likely to fail after a
paid training job starts. It does not train the corpus and does not claim a
translation-quality score. It verifies the reviewed environment, the selected
accelerator, one production-sized train/inference/checkpoint cycle, and the
two-rank distributed path where applicable.

This is separate from prepared-data transfer. Stage and recover the tokenizer and
dataset archive with [MODAL_BUNDLE_STAGE.md](MODAL_BUNDLE_STAGE.md); that workflow
uses only CPU resources and does not run any accelerator test.

## Cost and failure boundaries

Every command starts exactly one hardware target. There is deliberately no
`all` mode. Each Modal function uses:

- application retries set to zero;
- a 300-second function-execution timeout;
- a separate 180-second container-startup timeout;
- at most one container, with no warm or buffered containers;
- a two-second scale-down window and a single input per container;
- 4 CPU cores, 32 GiB of host memory, and Modal's default container disk quota;
- a 150-second maximum for the nested two-rank job; and
- a 20-second parent margin reserved for killing and reaping every `torchrun`
  process if the nested job stalls.

`--max-dollars` is an explicit authorization check, not a Modal account-level
spending cap. The check reserves two complete startup and function windows
because Modal can reschedule an ephemeral container after an infrastructure
crash or OOM even when application retries are zero. Modal may run a function a
few seconds beyond its configured timeout. A Workspace usage budget is therefore
the real external limit.

Do not deploy this GPU App. Modal documents that container crashes in a deployed
App are retried indefinitely, independently of `retries=0`. The durable controller
uses `app.run(detach=True)` instead, so a Codex or terminal disconnect does not
stop the input while the ephemeral App still retains a finite crash-rate cutoff.
Before submission, the controller also requires the user to record the current
Workspace usage and hard budget. It accepts only enough remaining headroom for
the selected target and refuses more than $5. These values are a recorded manual
attestation because Modal 1.5.3 does not expose the configured hard budget through
the public client API. Confirm them in **Usage & Billing** immediately before the
command.

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

First set the Modal Workspace usage budget to the displayed current-cycle usage
plus no more than $5. The budget caps usage before credits are applied, so a $30
credit balance does not require a $30 smoke-test allowance. Then submit one target
with the durable controller. Replace the example budget and usage values with the
numbers shown in the dashboard:

```powershell
.venv\Scripts\python.exe scripts/modal_gpu_smoke_control.py submit `
  --target a100-40gb --max-dollars 0.69 `
  --workspace-budget 4.80 --workspace-usage 0.80
```

The command atomically creates
`artifacts/modal-runs/<run-id>/receipt.json` before it calls `spawn`. It records
the FunctionCall ID before leaving the detached App context. The remote worker
commits numbered progress events, `status.json`, and exactly one terminal
`result.json` or `failure.json` under `runs/<run-id>` in the persistent
`sion-gpu-smoke-results` Volume. Each status records the actual remote
FunctionCall ID and the SHA-256 of every reviewed source and dependency-lock byte
copied into the image. Before any CUDA canary, the worker also requires Modal's
mounted executable entrypoint to be byte-identical to the reviewed copy inside
the image.

A fixed local `.submission-lock` directory serializes the unresolved-run scan,
intent receipt creation, remote spawn, and FunctionCall ID write. The lock is
removed only after that critical section exits. If the controller process is
forcibly terminated, the lock remains deliberately fail-closed. First verify that
no controller process is still running, inspect every receipt and the Modal
dashboard, and recover any accepted call before removing a stale lock. Never
delete the receipt directory to bypass this guard.

Codex rate limits, terminal closure, or a local restart therefore do not lose the
run identity. Recover without waiting by using the receipt printed by `submit`:

```powershell
.venv\Scripts\python.exe scripts/modal_gpu_smoke_control.py status `
  --receipt artifacts/modal-runs/<run-id>/receipt.json
```

The status command polls the FunctionCall with a zero-second timeout, downloads
the current Volume journal, saves `status-latest.json` beside the receipt, and
fetches only a bounded log tail. It distinguishes a pending call from an expired
output and from a terminal remote timeout. A confirmed Modal function timeout or
terminal server output overrides a stale `running` journal because a forced worker
termination cannot write its own final Volume event. A terminal Volume journal
alone is insufficient: the FunctionCall must also be terminal because Modal can
reschedule an input after the journal commit but before its result returns. Client,
authentication, transport, and identity uncertainties remain
`status-unavailable` and cannot authorize another paid input. If the initial
`spawn` response is
ambiguous, the receipt is marked `submission-unknown`; the controller never
automatically retries it. Once the remote journal appears, its validated
FunctionCall ID is used to recover that ambiguous submission. A new target is
blocked until the previous receipt has a locally recovered `passed` or `failed`
state backed by terminal FunctionCall evidence.

If Modal rejects the Function definition before the detached App context opens,
the receipt is instead terminal `submission-failed`, because the controller
body and `spawn` could not have run. Fix the rejected resource or image contract
before creating a new smoke receipt. Failures after the App context opens remain
ambiguous unless a valid FunctionCall ID was persisted.

After a passed result and an updated billing check, the remaining targets use the
same pattern:

```powershell
.venv\Scripts\python.exe scripts/modal_gpu_smoke_control.py submit `
  --target h100 --max-dollars 1.19 `
  --workspace-budget <hard-budget> --workspace-usage <current-usage>

.venv\Scripts\python.exe scripts/modal_gpu_smoke_control.py submit `
  --target a100-40gb-x2 --max-dollars 1.25 `
  --workspace-budget <hard-budget> --workspace-usage <current-usage>

# Optional capacity check.
.venv\Scripts\python.exe scripts/modal_gpu_smoke_control.py submit `
  --target a100-80gb --max-dollars 0.80 `
  --workspace-budget <hard-budget> --workspace-usage <current-usage>
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

## Recovering a CPU runtime-metadata failure

Use `scripts/recover_modal_bundle_runtime_metadata.py` only for a failed CPU
finalizer whose durable journal reports that the injected Modal package has no
distribution metadata. Other failures need their own diagnosis. Commit and push
the replacement runtime, then require CI to pass before submitting a recovery.
The controller rejects runtime files that differ from the pushed commit,
including extra source files that Git ignores.

Start with a read-only preflight:

```text
python scripts/recover_modal_bundle_runtime_metadata.py --receipt <failed-receipt.json> --failed-app-id <stopped-app-id> --max-dollars <authorized-ceiling> --workspace-budget <actual-hard-budget> --workspace-usage <fresh-observed-usage> --preflight-only
```

Supply current budget observations, not example values from an earlier run.
Preflight checks the exact stopped App, terminal FunctionCall, preserved failure
journal, original submission claim, local archive hash, and remote archive entry.
It does not upload or start a Function. Remove `--preflight-only` only after
reviewing that evidence and confirming the paid CPU attempt is authorized.

Recovery uses a fresh attempt ID and a version-2 receipt that points truthfully
to the original incoming archive. It never reuploads, copies, or moves the
archive before verification. The worker authenticates the recovery claim and
preserves the original failed journal; new status and result records belong to
the new attempt. The result also binds the source ID and recovery-claim digest.

A no-overwrite source claim and a separate permanent submission marker prevent
a copied local receipt from silently starting another attempt. A lost submission
response is ambiguous, not permission to retry. Inspect the saved intent,
receipt, exact FunctionCall, and Volume journal instead. Do not delete claims or
use the ordinary `resume` command on a source-bound receipt.

If the worker publishes prepared files but stops before recording its result,
the controller stops because the original incoming archive may no longer exist.
This requires explicit reconciliation of the published files and journal; do
not manufacture a successful result or automatically submit another worker.
CPU recovery is not evidence that either A100 or H100 execution has passed.

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
- [Workspace hard budgets](https://modal.com/docs/guide/budgets)
- [Detached ephemeral Apps](https://modal.com/docs/guide/apps)
- [Durable FunctionCall IDs](https://modal.com/docs/guide/function-invocation-methods)
- [Persistent Volume commits](https://modal.com/docs/guide/volumes)
- [GPU types and exact H100 selection](https://modal.com/docs/guide/gpu)
- [Billing CLI](https://modal.com/docs/cli/latest/billing)
