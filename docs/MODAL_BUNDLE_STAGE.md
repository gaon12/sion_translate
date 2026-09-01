# Durable Modal staging for a prepared GPU bundle

This workflow transfers one locally prepared training archive to a persistent
Modal Volume. It then starts a detached, CPU-only finalizer that verifies the
uploaded bytes, extracts the archive, verifies the extracted tree, and publishes
the result under a content-addressed path.

It does **not** start training, request a GPU, deploy a persistent App, publish a
model, or update a model card. Run the separate GPU smoke tests only after this
staging operation has a recovered `passed` state.

## What the prepared archive contains

Build the archive only after local tokenizer and dataset preparation succeeds:

```powershell
.venv\Scripts\python.exe scripts\package_gpu_bundle.py build `
  --output artifacts\gpu-bundles\sion_translate-prepared.zip `
  --prepared-only

.venv\Scripts\python.exe scripts\package_gpu_bundle.py verify-archive `
  artifacts\gpu-bundles\sion_translate-prepared.zip
```

The prepared-only archive contains the authenticated tokenizer, translation
dataset, foundation dataset when foundation training is enabled, configuration,
and training code. It omits the raw parallel and monolingual preparation inputs.
The GPU server can therefore verify the prepared tree and start training without
repeating CPU-heavy tokenization and indexing.

Do not package a stale prepared tree. Run the current configuration's local
`--prepare-only` command first, then build from a clean tracked Git tree. The
package manifest binds the exact Git commit, Git tree, configuration, artifact
digests, dataset statistics, and candidate-refinement evidence.

## Cost boundary

The remote finalizer is fixed to the following resources:

- no GPU;
- 2 CPU cores and 8 GiB of memory;
- 2 GiB of ephemeral disk;
- one container, no warm or buffered containers;
- zero application retries; and
- a four-hour timeout with a two-second scale-down window.

The controller reserves two complete CPU-and-memory attempts because an
infrastructure failure can still cause platform-level rescheduling. With the
prices recorded on 2026-09-01, that contingency is `1.26622384` USD. The command
requires all three of these conditions before it uploads anything:

1. `--max-dollars` is at least the calculated contingency.
2. The reported hard Workspace budget minus current usage covers the same
   contingency.
3. That remaining Workspace headroom is no more than 5 USD.

`--max-dollars` is a local authorization check. It is not a provider-enforced
per-call spending cap. Set the real hard Workspace budget in Modal immediately
before running the command, read the current-cycle usage from the same dashboard,
and make sure no other Modal job is consuming that Workspace budget.

The estimate excludes image builds, upload traffic, network traffic, Volume
storage, and timeout overruns. Staging temporarily stores both the ZIP and its
extracted tree, so allow approximately the archive size plus the extracted size.
Storage accounting and deletion reclamation can lag. Recheck the current
[Modal pricing](https://modal.com/pricing),
[resource](https://modal.com/docs/guide/resources), and
[Volume](https://modal.com/docs/guide/volumes) documentation before authorizing
the operation.

## Run one durable stage operation

Use the repository virtual environment. The controller requires the reviewed
Modal client version and rejects a different version.

Run the command in an independent PowerShell terminal, not inside a Codex command
whose lifetime depends on the current Codex session. Save the console output to a
local log so a rate limit, UI restart, or terminal scrollback loss cannot hide the
receipt path:

```powershell
New-Item -ItemType Directory -Force artifacts\modal-bundle-logs | Out-Null

.venv\Scripts\python.exe scripts\modal_stage_gpu_bundle.py stage `
  --bundle artifacts\gpu-bundles\sion_translate-prepared.zip `
  --volume sion-prepared-bundles `
  --max-dollars 1.27 `
  --workspace-budget <hard-budget> `
  --workspace-usage <current-usage> 2>&1 | `
  Tee-Object -FilePath artifacts\modal-bundle-logs\stage-latest.log
```

Replace both budget placeholders with values observed immediately before the
command. Do not use a 30 USD credit balance as the hard budget merely because the
credit is available. A small, explicit headroom is enough for this CPU operation.

The local process performs these steps in order:

1. Acquire a local submission lock and reject any unresolved earlier receipt.
2. Verify that the archive is a stable, prepared-only regular file.
3. Atomically write `receipt.json` before the first remote mutation.
4. Upload the large ZIP exactly once with overwrite disabled.
5. Write one small, no-overwrite remote submission claim.
6. Start one detached CPU finalizer and record its FunctionCall ID.

The finalizer writes durable state under `/operations/<upload-id>` on the Volume.
It publishes a verified artifact only at
`/bundles/sha256/<prefix>/<bundle-sha256>`. The `READY` marker is written last,
and publication refuses to replace an existing destination. An exact existing
artifact may be verified and reused; conflicting content is preserved and causes
a failure.

Once the detached FunctionCall ID is recorded, closing the local terminal does
not cancel the remote finalizer. The local upload itself is not detached, so keep
the independent terminal open until `stage` prints its final JSON or an error.

## Recover status without waiting

Every operation has one canonical receipt such as:

```text
artifacts/modal-bundle-uploads/<upload-id>/receipt.json
```

Poll it without waiting:

```powershell
.venv\Scripts\python.exe scripts\modal_stage_gpu_bundle.py status `
  --receipt artifacts\modal-bundle-uploads\<upload-id>\receipt.json
```

The command reconciles the receipt, the durable Volume journal, and the detached
FunctionCall with a zero-second result timeout. It atomically writes
`status-latest.json` beside the receipt. A later Codex session can read those two
files and continue from the same operation.

Do not start another stage operation until `recovered_state` is exactly `passed`
or `failed`. A terminal Volume journal is not accepted while the FunctionCall is
still pending, because the platform may still be rescheduling the input.

## Resume an upload whose finalizer was never submitted

`resume-finalizer` never uploads the archive again. It is only for a receipt whose
`finalizer_state` is still `not-submitted`, such as an interrupted or ambiguous
large upload. It reuses the same upload ID, incoming path, Volume, archive digest,
and remote submission-claim identity.

Read the dashboard again and supply fresh budget evidence:

```powershell
.venv\Scripts\python.exe scripts\modal_stage_gpu_bundle.py resume-finalizer `
  --receipt artifacts\modal-bundle-uploads\<upload-id>\receipt.json `
  --max-dollars 1.27 `
  --workspace-budget <hard-budget> `
  --workspace-usage <current-usage>
```

The new observation is appended to the receipt's budget history. The command
refuses to recreate a missing Volume. It also refuses any existing finalizer
journal, a missing or conflicting attempted claim, a changed local finalizer
runtime, or any receipt that may already have submitted a FunctionCall.

The remote finalizer is the authoritative check for the uploaded ZIP. If an
interrupted upload did not commit a complete archive, finalization fails quickly
and records a durable failure instead of retransmitting unknown bytes.

## Recovery decision table

| Receipt or recovered state | Safe action |
| --- | --- |
| `not-submitted` with `intent`, `uploaded`, or `upload-unknown` | After the original local process has ended, use `resume-finalizer` once with fresh budget values. It never reuploads the ZIP. |
| Claim `creating` or `creation-unknown` | Use `resume-finalizer`. It proceeds only if the exact original claim is durably visible. Missing or conflicting evidence remains fail-closed. |
| Claim `created` and finalizer `not-submitted` | Use `resume-finalizer`; it reuses the claim and checks that no operation journal exists. |
| `submitting` or `submission-unknown` | Do not resubmit. Run `status` and inspect the Modal dashboard. If the remote function writes its journal, the FunctionCall ID can be recovered from it. |
| `submitted` or recovered `pending` | Run `status` later. Do not start another operation. |
| `terminal-journal-pending-call` | The durable result exists, but the call is not terminal. Wait and run `status` again. |
| `status-unavailable`, `claim-creation-unknown`, or `upload-unknown` | Treat the operation as unresolved. Restore connectivity or inspect provider state; never delete evidence to bypass the guard. |
| `output-expired` | Use the consistent durable Volume result or failure when available; otherwise investigate manually. |
| `passed` or `failed` | The receipt is terminal and no longer blocks a new stage operation. Preserve it for audit. |

There is one unavoidable fail-closed edge: Modal 1.5.3 provides no custom
idempotency key or lookup by upload ID for a spawn whose response is lost before
the worker writes its journal. A `submission-unknown` receipt with no remote
journal must not be automatically resubmitted. Inspect the provider dashboard and
retain the receipt.

## Recover a stale local lock

If the local controller is killed, `.submission-lock` remains deliberately. First
prove that the recorded process has ended and that no other terminal or machine is
running the controller. Then use:

```powershell
.venv\Scripts\python.exe scripts\modal_stage_gpu_bundle.py recover-lock
```

The command removes only a same-host lock whose exact recorded process instance is
proven stale. It refuses a live process, an unknown process identity, or a lock
owned by another host. Never remove the directory manually.

## Evidence and access-control rules

- Keep one canonical receipt root. Do not copy, edit, rename, or delete receipts
  while they are writable. Copied receipts can defeat the local single-writer
  assumption.
- Keep the local ZIP at least until remote `passed` recovery. The receipt records
  its original path for audit, although `resume-finalizer` relies on the remote
  archive rather than retransmitting the local file.
- Restrict write access to the receipt root and Modal Volume. The controller
  validates identities and paths but does not cryptographically authenticate
  state against another principal who already has write access.
- Do not run concurrent staging controllers against the same receipt root or
  Volume. The local lock and remote claim prevent ordinary duplicates, not a
  hostile writer racing filesystem path checks.
- Preserve `receipt.json`, `status-latest.json`, the terminal log, and the final
  package manifest with the training record.

After staging passes, run the separate budget-gated accelerator checks described
in [MODAL_GPU_SMOKE.md](MODAL_GPU_SMOKE.md). Validate one hardware target at a
time, inspect billing between targets, and do not begin full H100 or A100 training
until both local tests and the selected real-CUDA smoke checks pass.
