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
sion-train --config sion_translate.yaml --prepare-only
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
sion-train --config sion_translate.yaml --prepare-only
```

The second run must authenticate and reuse the complete artifacts. It must not silently
retrain a tokenizer, partially append a dataset, or accept an incompatible directory.
Treat a graph, tokenizer, token-feature, source-fingerprint, or inventory mismatch as a
hard failure. Investigate it instead of deleting metadata and forcing reuse.

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

Automatic sizing uses effective training data, not a hard-coded language combination.
Release 1.5 adds a five-percent promotion buffer at each preset boundary. The current
promotion points are:

| Larger preset | Raw boundary | Promotion point |
|---|---:|---:|
| 200M class | 200,000 | 210,000 |
| 450M class | 3,000,000 | 3,150,000 |
| 1.3B class | 30,000,000 | 31,500,000 |
| 3B class | 100,000,000 | 105,000,000 |

The buffer prevents a modest addition near a boundary from abruptly selecting a much
larger model. It is a stability policy, not a claim that one preset is universally best.
Record the selected preset, effective example count, graph, token budget, and validation
curves for each run.

Do not override the capacity preflight simply to start training. The runner estimates
parameters, gradients, optimizer state, optional EMA, activations, communication buffers,
and runtime headroom before CUDA allocation.

## 7. Build and verify the GPU upload bundle

After both local preparation runs succeed, build a bundle containing the prepared
artifacts:

```bash
python scripts/package_gpu_bundle.py build \
  --output sion_translate.zip \
  --with-tokenizer \
  --with-dataset \
  --with-foundation-dataset

python scripts/package_gpu_bundle.py verify-archive sion_translate.zip
```

Use `--with-monolingual-corpus` only when the GPU server must rebuild the foundation
dataset. If `artifacts/foundation_dataset/` is already complete and authenticated,
omitting raw monolingual corpora makes the upload smaller and prevents accidental
server-side re-preparation.

Record the archive path, byte size, SHA-256, source commit, Git tree, configuration
fingerprint, and artifact inventory. Do not upload an archive that fails verification.

## 8. Verify the bundle on the GPU server

Compare the archive digest with the value recorded locally, test the ZIP container, and
verify the extracted tree:

```bash
sha256sum sion_translate.zip
python3 -m zipfile -t sion_translate.zip
unzip sion_translate.zip
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

This is not only candidate reranking. At inference time, the deployed T2/M2 path feeds
the first prediction back through the learned revision edge and returns the refined
second prediction. Export metadata binds both translation and revision direction graphs,
so inference fails closed when a requested refinement edge was not trained.

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
