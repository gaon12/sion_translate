# Real-data GPU performance probe

The probe measures the current prepared dataset and production configuration on
one A100 80GB or exact H100 80GB. It does not train or publish a usable model.
Run targets sequentially and inspect each result before authorizing another call.

## Prepare on the CPU

```powershell
python scripts/prepare_training_probe.py --output C:/probe-data
```

Use a new directory outside OneDrive. The command verifies complete indexed
inventories and tokenizer hashes, then extracts representative and longest-row
cohorts with the production sampling policy. Model sizing uses the complete
translation token count, never the small sample. The resulting `plan.json`
records source hashes, full stage sizes, configuration, and sampling limitations.
Raw JSONL sources are not rehashed: this tests the authenticated prepared snapshot.

## Submit one bounded call

Commit reviewed runtime files first. With the user-confirmed $30 Workspace hard
budget and sufficient remaining credits:

```powershell
python scripts/modal_training_probe.py submit --data C:/probe-data --target a100-80gb --max-dollars 4 --workspace-hard-budget 30
python scripts/modal_training_probe.py status --receipt ABSOLUTE_RECEIPT_PATH
```

For the second GPU, use `--target h100` only after the first receipt is reconciled
and its app has stopped. The controller reads fresh billing, rejects overlapping
apps and unresolved receipts, and records submission intent before a paid call.
Its authorization estimate covers two complete startup/function windows because
provider recovery is separate from application retries. Storage, image builds,
network, and billing-report delays remain outside that estimate; the provider's
hard budget is the independent spending backstop.

Each call is limited to 1,200 seconds, with zero application retries. Each batch
candidate runs in a separate process with a shorter timeout. Progress, results,
and raw child stdout/stderr persist in the `sion-training-probe-results` Volume.
Local receipts and recovered results live under `artifacts/modal-probes`.

## Interpret the result carefully

- Foundation denoising, translation SFT, and the actual MRT objective are separate.
  The MRT configuration retains candidate generation and round-trip rewards.
- Timings include CPU collation, transfer, forward/backward, and measured
  optimizer/EMA overhead. Four length strata are measured after warm-up. The
  projected accumulation preserves the original effective batch where possible;
  `effective_batch_changed` flags candidates that change optimization semantics.
- FP32 master parameters, BF16 autocast, AdamW states, and EMA are present in the
  VRAM measurement. Longest-row stress and checkpoint/resumed-update peaks are
  reported separately. An explicit CUDA OOM is a measured capacity boundary,
  not a passing batch. Other exceptions fail the call without being filtered.
- Filling 80GB is not the objective. Compare useful examples/second and retain
  headroom for variable lengths and validation. The 85% search threshold is a
  heuristic, not a guarantee against every possible future allocation.
- These are randomly initialized models. MRT termination and generation length
  can change substantially after SFT. Its speed is an untrained-rollout scenario,
  not a precise prediction for a trained model.
- Full-training compute time can be approximated by stage examples times epochs
  divided by measured examples/second. A `max_steps` override instead limits the
  update count. Whole-corpus disk/DataLoader stalls, validation, checkpoint
  cadence, final exports, and interruptions must be added separately. A short
  resident-cohort probe cannot prove successful full-epoch or full-pipeline completion.

Warnings are not filtered. The known Modal 1.5.3 container-bootstrap unclosed-file
warning occurs before project code starts and must remain visible in the report,
even if every project computation passes.
