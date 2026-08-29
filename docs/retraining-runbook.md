# Retraining from a clean start

This runbook covers the complete `sion_translate` 1.5 workflow: local tokenizer and
dataset preparation, verified transfer to a GPU server, foundation pretraining,
translation SFT, gold-anchored MRT posttraining, evaluation, export, and recovery.

The language graph comes from the configuration. Nothing in this procedure assumes
Korean, English, Japanese, a fixed number of languages, or symmetric translation
directions.

## 1. Understand the two model roles

One training run can produce two distinct models:

- The **foundation model** learns from configured monolingual corpora. It is a base model
  and is deliberately marked `translation_capable: false`.
- The **translation model** starts from the selected foundation weights, learns the exact
  configured directed translation graph, and can then run MRT posttraining.

After real training and evaluation finish, publish the foundation model in a new Hugging
Face repository and publish the translation model in `gaon12/sion_translate`. Do not
create or update model cards before measured artifacts exist.

When MRT is enabled, deploy the best posttraining export. The SFT export remains useful
as a comparison point, but it is not the final T2/M2 result.

## 2. Start from an authenticated source tree

Use Python 3.11 or 3.12. Install the project and run the local quality gates before
preparing large artifacts:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,export,hangul]"
ruff format --check .
ruff check .
pyright
python -m pytest -p no:cacheprovider
```

The `hangul` extra is optional for languages that do not need its morphology helper. The
core pipeline has a built-in fallback, but every production run should record the exact
installed extras and dependency versions.

Raw corpora under `data/` are not ordinary Git payloads. A fresh clone therefore needs
the separately authorized training snapshot. Never substitute evaluation-only data or
an unreviewed local file merely to satisfy a missing path.

## 3. Define a language graph, not a language list

The configuration distinguishes three concepts:

1. **Physical language pairs** describe fields that coexist in a source record.
2. **Translation directions** describe edges the model is allowed to learn and serve.
3. **Source-only languages** may appear as inputs but must not be requested as targets.

A physical pair does not imply both directions. For example, a record with `de` and `fr`
can train only `de -> fr` when that is the configured edge. The tokenizer, indexed
dataset, checkpoints, exports, and inference runtime all authenticate the same exact
graph.

Use short ASCII language identifiers accepted by the schema. The identifiers are data,
not special cases in Python code. Adding a language or changing an edge invalidates
artifacts whose authenticated graph no longer matches.

## 4. Inspect input shards before expensive work

Run the structural preflight:

```bash
python scripts/data/check_shard_keys.py
```

The check samples each shard without repeating the full quality and split pipeline. A
nonzero exit code means that at least one configured pair may produce no usable record.
Fix the source JSONL transactionally, run the check again, and keep only one copy in the
training discovery path. Leaving both the original and repaired file can train duplicate
content.

Also inspect the effective corpus report produced by the preparation command. File counts
alone are not sufficient: filtering, deduplication, direction policy, source-only rules,
and sampling weights determine the effective training mass.

Do not use a historical row count as an integrity check. The authenticated inventory and
source fingerprints are authoritative for the current snapshot.

## 5. Prepare tokenizer and datasets locally

Tokenizer training and indexing are CPU, RAM, and storage work. Run them on the local
machine before renting the GPU server:

```bash
sion-train --allow-local-checkout --config sion_translate.yaml --prepare-only
```

This command stops before allocating a model. It prepares and authenticates:

```text
artifacts/
|-- tokenizer/
|-- dataset/
`-- foundation_dataset/   # only when eligible monolingual input exists
```

Preparation uses the language graph and corpus inventory from the configuration. The
tokenizer sampler is deterministic, bounded, and language-aware, so a large corpus cannot
silently crowd every smaller configured language out of the sample.

Run the same command a second time:

```bash
sion-train --allow-local-checkout --config sion_translate.yaml --prepare-only
```

The second run must authenticate and reuse the complete artifacts. It must not silently
retrain a tokenizer, partially append a dataset, or accept an incompatible directory.
Treat a graph, tokenizer, token-feature, source-fingerprint, or inventory mismatch as a
hard failure. Investigate it instead of deleting metadata and forcing reuse.

Inspect `artifacts/dataset/manifest.json` after preparation. New artifacts must declare
`stats_schema` as `sion-prepare-stats-src-tgt-v1`. The `src_tokens` and `tgt_tokens`
values are physical accepted content-token counts for the two stored shard sides; they
are not Korean/Japanese totals and they do not include virtual reverse directions,
padding, runtime controls, epoch repetition, or online augmentation. Their values must
match the sums of `src_length` and `tgt_length` in the authenticated index shards. A
markerless v1.5 manifest may retain the exact legacy `ko_tokens`/`ja_tokens` names for
compatibility, but do not copy those misleading names into a new manifest.

An interrupted translation build keeps deterministic gzip worker chunks in a hidden
content-addressed progress directory next to `artifacts/dataset/`. The progress contract
binds the exact source bytes, tokenizer, language graph, preprocessing options, worker
algorithm, Python version, SentencePiece version, and Unicode database version. Re-run the
same command to reuse compatible chunks. Do not copy chunks between environments or edit
their JSON payloads. A generation fence prevents workers orphaned by a terminated parent
from writing after a new invocation begins.

Worker checkpointing and final shard construction are separate phases. After all chunks
are complete, preparation calculates a conservative staging requirement from their actual
candidate rows, token IDs, and encoded metadata. An `ENOSPC` result at this boundary leaves
the worker chunks intact, does not create a partial published dataset, and can be retried
after freeing space. The fixed per-record safety contract is language-neutral: 16 MiB per
raw line, 64 MiB of selected raw data per batch, 1,024 expanded physical pairs per line,
256 KiB of supported metadata per pair, and 256 MiB maximum uncompressed chunk size.

### Manual diagnostic commands

The unified preparation entry point is preferred. For a small manual diagnostic, pass
the same graph to both commands:

```bash
sion-train-tokenizer \
  --input "data/*.jsonl" \
  --output-dir artifacts/tokenizer \
  --language-pair de fr \
  --translation-direction de fr

sion-prepare-data \
  --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer/sion.model \
  --output-dir artifacts/dataset \
  --language-pair de fr \
  --translation-direction de fr
```

Repeat the graph options for every physical pair and every directed edge. Do not infer a
reverse direction. A parallel-only diagnostic does not replace the production tokenizer
sample when the configured foundation corpora are also part of tokenizer training.

### Tokenizer compatibility

Changing any of these inputs requires a new tokenizer and newly indexed datasets:

- vocabulary or normalization policy;
- digit splitting or byte fallback policy;
- required characters;
- language control symbols or graph identity;
- effective tokenizer sample inventory.

Never pair old indexed token IDs with a new tokenizer merely because the vocabulary size
is equal. The shapes can match while every embedding row means something different.

The verifier measures byte-fallback use on corpus-derived probes. Rare characters may
legitimately use byte fallback; the ratio and configured limit determine acceptance.
See [`sentencepiece-sigsegv.md`](sentencepiece-sigsegv.md) for the crash investigation
and tokenizer version policy.

## 6. Review automatic model sizing

Automatic sizing uses the authenticated prepared token inventory, not a hard-coded
language combination or the number of graph edges. Sum `src_length + tgt_length` once for
each physical training row. Do not multiply by reverse-direction expansion or planned
epochs: those operations reuse existing content rather than adding unique observations.

The continuous log-log scaling curve has five reference points:

| Unique physical tokens | Anchor | Approximate parameters at 48k vocab with refinement |
|---:|---|---:|
| 6.4 million | `small` | 66.8M |
| 96 million | `medium` | 124.2M |
| 960 million | `base` | 209.7M |
| 3.2 billion | `large` | 439.8M |
| 9.6 billion | `xlarge` | 801.1M |

The runtime interpolates a continuous parameter target and selects the nearest valid point
from a deterministic 61-architecture ladder. Ladder points change one primary dimension
at a time; head counts are derived again when width changes. Every point preserves
query/KV-head divisibility, and every adjacent parameter increase is below 12%; there is
no direct 210M-to-440M promotion. A legacy API caller that has no prepared lengths uses
the explicit 32-tokens-per-pair reference constant, but the production CLI always supplies
the exact indexed total.

Automatic attention uses two query heads per KV head. The KV projection width is exactly
half of `d_model`, so it is nondecreasing across the ladder; head dimensions stay
8-aligned in the 64--160 range. This keeps representational KV capacity from collapsing
at an intermediate width. It also costs more attention work and inference KV cache than
the older, more aggressively grouped anchors, so do not copy old memory estimates.

The continuous total budget is defined at a fixed 48,000-piece tied-embedding reference.
Candidates are counted with the actual vocabulary, embedding-sharing choice, and enabled
modules. This makes a larger vocabulary consume more of the same budget and prevents the
discrete tokenizer vocabulary tiers from adding a second model-capacity cliff.

Do not treat the current 27,602,231 rows as an exact capacity measurement before local
tokenizer and dataset preparation finish. The row-only compatibility preview is about
203.2M parameters with a 48k vocabulary and candidate refinement; the production log will
replace that preview with the authoritative token count, smooth target, selected shape,
and estimated parameter count. Record those values with the graph, data fingerprint,
sampling policy, and validation curves for every run.

Do not override the capacity preflight simply to start training. The runner constructs the
final configured model on the meta device, counts its actual parameters, and then estimates
parameters, gradients, optimizer state, optional EMA, activations, communication buffers,
and runtime headroom before CUDA allocation.

## 7. Build and verify the GPU upload bundle

After both local preparation runs succeed, build a bundle containing the prepared
artifacts:

```bash
python scripts/package_gpu_bundle.py build \
  --output sion_translate-prepared.zip \
  --prepared-only

python scripts/package_gpu_bundle.py verify-archive sion_translate-prepared.zip
```

Prepared-only mode includes the complete tokenizer and every applicable indexed dataset,
but omits raw parallel and monolingual training corpora even when they are tracked. Use
the individual `--with-*` switches only for a deliberate server-side rebuild.
`--with-monolingual-corpus` is required when the GPU server must rebuild the foundation
dataset and intentionally conflicts with `--prepared-only`.

Record the archive path, byte size, SHA-256, source commit, Git tree, configuration
fingerprint, and artifact inventory. Do not upload an archive that fails verification.

## 8. Verify the bundle on the GPU server

Compare the archive digest with the value recorded locally, test the ZIP container, and
verify the extracted tree:

```bash
sha256sum sion_translate-prepared.zip
python3 -m zipfile -t sion_translate-prepared.zip
unzip sion_translate-prepared.zip
cd sion_translate
python3 scripts/package_gpu_bundle.py verify-tree .
```

Stop on any mismatch. Re-upload the verified archive instead of manually repairing a
failed extraction.

Install the GPU environment and confirm CUDA:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,export,hangul]"
python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("BF16:", torch.cuda.is_bf16_supported())
print("NCCL:", torch.distributed.is_nccl_available())
PY
```

Authenticate the prepared inputs once more without starting model training:

```bash
sion-train --config sion_translate.yaml --prepare-only
```

In a prepared-only tree this is an offline verification pass. A missing tokenizer or
dataset is an error because the raw source corpus was intentionally not shipped. The
runner completes this check before allocating model or optimizer storage. Do not pass
`--allow-local-checkout` in this extracted bundle; the authenticated bundle metadata must
remain authoritative.

A complete bundle should reuse every included artifact.

## 9. Start or resume training

The recommended entry point is:

```bash
python3 easy_run.py
```

It validates CUDA and NCCL, authenticates inputs, applies the buffered sizing policy,
selects a distributed strategy from the smallest visible GPU, and resumes the furthest
complete compatible stage.

For an explicit multi-GPU launch:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m sion_translate.cli.train \
  --config sion_translate.yaml
```

Use the actual visible GPU count. One GPU uses the single-device path. Multiple GPUs use
DDP when complete persistent state fits per rank and FSDP2 when sharding is required.
Automatic BF16 is enabled only when every rank supports it.

If a process stops, run the same command again. Checkpoint generations are published
transactionally, authenticated, and separated into `current` and `previous` recovery
states. The loader resumes only a complete compatible generation.

## 10. Training stages and accuracy controls

### Foundation stage

When eligible monolingual data has positive sampling mass, the foundation stage learns a
denoising/reconstruction objective first. The effective language list is derived from
the prepared data, not merely from names present in the configuration.

Weight handoff to translation SFT is not an optimizer resume. The runner verifies the
tokenizer and model configuration, then transfers model weights without inheriting the
foundation optimizer moments, scheduler, or step counter.

### Translation SFT

SFT learns only the authenticated directed edges. Critical corruption in numbers,
placeholders, markup, URLs, code-like spans, and other structured content is rejected
during preparation instead of being treated as ordinary translation noise.

Monitor training and validation loss per direction. A falling training loss with a
matching validation improvement can support a larger-capacity experiment. A widening
gap indicates overfitting, leakage, or a distribution problem rather than an automatic
reason to increase parameters.

### Gold-anchored MRT and preference posttraining

Posttraining generates multiple first-pass candidates, scores the full candidate
distribution, and trains a second-stage refinement path anchored to the gold target.
Candidate generation uses the same safety policy as deployment:

- training-only controls are forbidden;
- minimum output length and repeated n-gram limits are enforced;
- row-specific output limits are derived from the source, not the reference target;
- validation uses the same deployment-aligned defaults;
- round-trip reward is available only when the required authenticated reverse edge
  exists.

This is not candidate reranking. At every decoder position, the deployed T2/M2 path
feeds the first full-vocabulary distribution back into the decoder state and selects the
token from a newly computed second distribution. A separate sequence-level revision
path can feed a completed draft back through a trained revision edge. Export metadata
binds the distribution-refinement feature and the exact translation and revision graphs,
so inference fails closed instead of advertising an untrained capability.

Held-out validation also measures the per-token difference
`NLL(provisional) - NLL(final)`. Positive values mean refinement helped. The trainer records
global, target-language, directed-edge, macro-direction, and worst-direction values without
running a second decoder pass. It checks the raw or EMA family that will actually deploy.

For translation-capable candidate-refinement runs, checkpoint selection is fail-closed:

- run a step-zero validation before the first update so a neutral or useful incoming model
  can remain the safe fallback;
- require finite evidence for every exact configured directed edge;
- reject a checkpoint when its worst edge is below `-1e-6`;
- keep optimizer progress resumable even when no updated checkpoint is release-safe; and
- publish only guard-approved `exports/best`, never an unvalidated `exports/latest`.

A versioned attestation stores the deployed family, graph fingerprint, edge count, and
worst gain with the best checkpoint record. A legacy run may resume its optimizer and data
cursor, but it must establish new release evidence. `RELEASE_INELIGIBLE.json` blocks any
stale inference directory until a successful safe-best export replaces it.

MRT can use more memory than teacher-forced SFT. If it runs out of memory, reduce
`posttraining.batch_size_per_gpu`, then candidate micro-batch size, candidates per source,
and generation limits in that order. Keep at least two candidates per source.

## 11. Output locations

```text
runs/auto/
|-- foundation/
|   |-- checkpoints/
|   |-- exports/best/       # base model; translation is refused
|   `-- stage_complete.json
|-- pretrain/
|   |-- checkpoints/
|   `-- exports/best/       # translation SFT comparison artifact
`-- posttrain/
    |-- checkpoints/
    `-- exports/best/       # final translation artifact when MRT is enabled
```

Checkpoint aliases have distinct purposes:

- `best` is selected by authenticated validation metrics;
- `latest` is the normal restart point;
- `final` records the last completed step.

Use `best` for release evaluation. Always move the matching tokenizer and token-feature
sidecar with model weights.

## 12. Evaluate before publishing

```bash
sion-evaluate --help
sion-translate --help
```

Report results separately for every trained direction and for both first-pass and refined
outputs. Include quality, structured-content preservation, language correctness,
repetition, latency, and memory. Use an external evaluation set that is isolated from all
training discovery paths; an internal approximate split alone is not a release claim.

Do not assume that a beam setting measured on an older checkpoint remains optimal. Sweep
the supported decoding policy on the new best checkpoint and record the complete command,
seed, graph, dataset digest, and metrics.

## 13. Verify and publish exports

The final export manifest authenticates the weight identity, checkpoint step, release
role, pipeline lineage, exact translation and revision edges, feature flags, tokenizer,
token features, and every output digest.

```bash
python - <<'PY'
import json
from sion_translate.training.export import validate_export_directory

path = "runs/auto/posttrain/exports/best"
result = validate_export_directory(path)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["valid"] else 1)
PY
```

Do not upload an invalid directory. Confirm that native, Transformers, and GGUF metadata
describe the same weight set, graph, role, tokenizer, pipeline, and feature flags.

Only after evaluation and export verification should maintainers create the new
foundation-model repository, update `gaon12/sion_translate`, and write evidence-based
model cards.

## 14. Failure report checklist

Keep enough evidence to reproduce a failure:

- complete traceback, not only the final line;
- source commit and Git tree;
- bundle path, size, and SHA-256;
- configuration and authenticated data fingerprints;
- tokenizer and dataset manifest digests;
- Python, PyTorch, CUDA runtime, driver, and dependency versions;
- GPU model, count, node/rank topology, and NCCL status;
- last telemetry records and available disk, RAM, and VRAM;
- checkpoint generation selected for resume.

Do not delete a failed artifact or checkpoint until its manifest and consumers have been
identified. Preserve it under a separate recovery path, fix the cause in code or data,
rerun formatting and linting, rerun tests, and then restart the affected preparation or
training stage.
