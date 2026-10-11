# Start the verified GPU training bundle

Use this checklist after receiving `sion_translate-prepared.zip`. The recommended bundle
already contains the tokenizer and indexed datasets prepared on the local machine, so the
GPU server can begin model training without repeating CPU-heavy work.

## 1. Verify the archive before extraction

Compare the first command with the SHA-256 supplied by the bundle builder:

```bash
sha256sum sion_translate-prepared.zip
python3 -m zipfile -t sion_translate-prepared.zip
unzip sion_translate-prepared.zip
cd sion_translate
python3 scripts/package_gpu_bundle.py verify-tree .
```

Stop if any command fails. `verify-tree` checks the exact file set, sizes, modes, SHA-256
digests, Git commit, and Git tree against `PACKAGE_MANIFEST.json` and `SHA256SUMS`.
Re-upload the archive instead of trying to repair a failed extraction manually.

The exact contents depend on the options used at build time. A prepared training bundle
normally includes:

- all tracked source, configuration, tests, and documentation;
- configured `data/evaluation_only/` files, but no raw parallel training JSONL;
- `artifacts/tokenizer/`, including the authenticated token-feature sidecar;
- `artifacts/dataset/` for translation training;
- `artifacts/foundation_dataset/` when foundation data was prepared locally.

The prepared-only format also excludes the raw monolingual foundation corpus, including
tracked corpus files. Old runs, checkpoints, virtual environments, and caches must not
appear. Use the manifest as the authority instead of relying on a hard-coded file count.
Manifest format 2 binds the exact tracked `sion_translate.yaml` digest, configured graph,
source provenance, tokenizer, datasets, and canonical paths used by the default runner.

## 2. Install a compatible GPU environment

The included installer verifies the extracted tree, installs the locked environment,
and checks the runtime/GPU without starting training:

```bash
python3 setup_gpu.py
```

It requires internet access and Python's `venv`/`ensurepip` support. If the host's
Python lacks them, install its `python3-venv` package first. Use
`.venv/bin/python` for the subsequent preparation and training commands, or run
`source .venv/bin/activate`. The equivalent manual installation is below.

Use the authenticated CPython 3.11/Linux x86_64/CUDA 12.8 dependency lock. The
launcher checks exact critical package versions, including Torch 2.10.0+cu128;
an unpinned install or Python 3.12 will fail its runtime preflight. A100 and H100
machines use the same project entry point.

```bash
nvidia-smi
python3 -m pip install "uv==0.12.3"
uv venv --no-config .venv --python cpython@3.11 --managed-python
uv pip sync --no-config requirements/pylock.gpu-cp311-linux-x86_64-cu128.toml \
  --python .venv/bin/python --require-hashes --strict --only-binary :all:
uv pip install --no-config --python .venv/bin/python --no-deps \
  --no-build-isolation --editable .
source .venv/bin/activate
python - <<'PY'
import torch

assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "This PyTorch installation has no CUDA support"
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("NCCL:", torch.distributed.is_nccl_available())
PY
```

The lock installs the reviewed Torch build. Use the activated environment for all
remaining commands; the launcher also validates the other locked runtime packages.

NCCL is required for multiple GPUs. The runner checks it before any long preparation or
training work begins.

## 3. Confirm that local preparation was included

Run the preparation-only entry point once on the server:

```bash
sion-train --config sion_translate.yaml --prepare-only
```

If local translation indexing is interrupted, run the identical command again. It reuses
content-bound worker chunks, fences workers left by a terminated parent, and waits until
all chunks are complete before creating final shards. A post-worker disk failure preserves
the reusable chunks and never publishes a partial `artifacts/dataset/` directory.

For a fully prepared bundle, this should authenticate and reuse the tokenizer and both
indexed datasets rather than rebuild them. A prepared-only bundle has no raw training
corpus from which to reconstruct a missing artifact, so absence is a preflight error.
This check finishes before model or optimizer allocation.

Do not pass `--allow-local-checkout` for this workflow. That option is only for a trusted
development clone without bundle metadata; it must never replace the manifest and checksum
verification of an extracted GPU bundle.

Do not continue when it reports a tokenizer, graph, source fingerprint, token-feature,
or dataset inventory mismatch. Those errors mean the server files do not represent the
same training contract as the bundle configuration.

## 4. Start training

The launcher reads the authenticated `sion_translate.yaml`; it does not accept
`--config`. The default now matches `configs/modal_80gb.yaml`: the measured 273M
model, foundation batch 32, SFT batch 32 with accumulation 8, and MRT batch 8 with
accumulation 2 and `max_steps: 100000`. The explicit MRT update budget takes
precedence over its epoch field. This profile targets one A100/H100 80GB; its
runtime estimate does not establish multi-GPU scaling. A 100k step cap is a
budget scenario, not a validated quality target. Do not edit an extracted bundle's
authenticated configuration; prepare a new bundle after changing source settings.

```bash
.venv/bin/python easy_run.py
```

The runner performs these operations:

1. Validates CUDA, NCCL, the visible GPU set, and the common BF16 capability.
2. Authenticates the raw inputs and all prepared artifact inventories.
3. Applies the configured model/batch/precision profile and validates capacity
   against the smallest visible GPU; selects the distributed strategy.
4. Resumes the furthest fully authenticated stage.
5. Runs foundation pretraining when a prepared monolingual dataset is available.
6. Runs directed translation SFT from the selected foundation weights.
7. Runs gold-anchored MRT and preference posttraining from the selected SFT weights.
8. Restores the selected best raw or EMA weights and creates verified final exports.

In an interactive terminal with an existing `tmux` installation, `easy_run.py` may create
a checkout-specific session. Slurm, `nohup`, containers, and non-interactive SSH remain
in the current process. Disable tmux explicitly with:

```bash
SION_NO_TMUX=1 .venv/bin/python easy_run.py
```

These commands run on an already provisioned GPU host. The repository's Modal
smoke/performance scripts are bounded diagnostics; they do not deploy a full
training scheduler. A full Modal run additionally needs persistent checkpoint
storage and a launcher that resumes across the service's execution windows.

## 5. Automatic GPU policy

- The runner uses every visible CUDA GPU.
- It uses BF16 only when every rank supports native BF16; otherwise automatic precision
  falls back to FP16.
- The default 80GB profile supplies explicit batch/checkpointing settings; capacity
  checks use the smallest visible VRAM capacity.
- Multiple GPUs use DDP when full persistent state fits, or FSDP2 when sharding is needed.
- MRT starts with a small candidate micro-batch because candidate generation has more
  variable memory use than teacher-forced SFT.
- `torch.compile` is not enabled automatically without an explicit configuration choice.

The preflight capacity gate may reject a model before CUDA allocation. Do not bypass that
result. Reduce the model or use the documented number of GPUs.

## 6. Foundation and translation roles

When `artifacts/foundation_dataset/` has valid sampled languages, foundation training runs
before translation. Its output is a separate `sion` base model under
`runs/*/foundation/`. It is not translation-capable, and `Translator` refuses to load it.

Translation SFT and MRT outputs use the `sion_translate` role. The final deployable model
comes from the posttraining best directory when MRT is enabled.

The intended Hugging Face layout is one new repository for the foundation artifact and
`gaon12/sion_translate` for the translation artifact. Repository creation and model-card
updates happen only after real training and evaluation finish.

## 7. Results and restart locations

```text
runs/modal-80gb/
├── foundation/
│   ├── checkpoints/
│   ├── exports/best/       # foundation model: sion, not translation-capable
│   └── stage_complete.json
├── pretrain/
│   ├── checkpoints/
│   └── exports/best/       # translation SFT
└── posttrain/
    ├── checkpoints/
    └── exports/best/       # final translation model when MRT is enabled
artifacts/
├── tokenizer/
├── dataset/
└── foundation_dataset/
```

If the process stops, run the same command again. The runner validates `current` and
`previous` checkpoint generations and resumes only a complete matching generation. A
valid downstream checkpoint skips earlier stages instead of repeating them.

Before deleting a GPU instance, download at least:

- `runs/`;
- `artifacts/tokenizer/`;
- final export manifests and their validation output;
- the exact configuration and environment report.

## 8. Monitor and report failures

```bash
watch -n 2 nvidia-smi
python3 -m torch.utils.collect_env
```

For a failure report, retain the complete traceback, `nvidia-smi`, the collect-env output,
the Git commit, bundle SHA-256, configuration, dataset fingerprint, node/rank count, and
the last telemetry lines. Do not replace an error with only a screenshot or the final
line of the traceback.

For capacity tuning and export recovery, continue with
[`docs/H100_TRAINING.md`](docs/H100_TRAINING.md).
