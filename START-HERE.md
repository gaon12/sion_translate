# Start the verified GPU training bundle

Use this checklist after receiving `sion_translate.zip`. The recommended bundle already
contains the tokenizer and indexed datasets prepared on the local machine, so the GPU
server can begin model training without repeating CPU-heavy work.

## 1. Verify the archive before extraction

Compare the first command with the SHA-256 supplied by the bundle builder:

```bash
sha256sum sion_translate.zip
python3 -m zipfile -t sion_translate.zip
unzip sion_translate.zip
cd sion_translate
python3 scripts/package_gpu_bundle.py verify-tree .
```

Stop if any command fails. `verify-tree` checks the exact file set, sizes, modes, SHA-256
digests, Git commit, and Git tree against `PACKAGE_MANIFEST.json` and `SHA256SUMS`.
Re-upload the archive instead of trying to repair a failed extraction manually.

The exact contents depend on the options used at build time. A prepared training bundle
normally includes:

- all tracked source, configuration, tests, and documentation;
- approved top-level training JSONL and isolated `data/evaluation_only/` files;
- `artifacts/tokenizer/`, including the authenticated token-feature sidecar;
- `artifacts/dataset/` for translation training;
- `artifacts/foundation_dataset/` when foundation data was prepared locally.

Excluded data, old runs, checkpoints, virtual environments, and caches must not appear.
Use the manifest as the authority instead of relying on a hard-coded file count.

## 2. Install a compatible GPU environment

Use Python 3.11 or 3.12, PyTorch 2.8 or newer, and a CUDA build appropriate for the
server. A100 and H100 machines use the same project entry point.

```bash
nvidia-smi
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,export,hangul]"
python3 - <<'PY'
from importlib.metadata import version
import torch

torch_version = tuple(map(int, version("torch").split("+")[0].split(".")[:2]))
assert torch_version >= (2, 8), f"PyTorch 2.8 or newer is required: {torch.__version__}"
assert torch.cuda.is_available(), "This PyTorch installation has no CUDA support"
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("NCCL:", torch.distributed.is_nccl_available())
PY
```

If PyTorch is not installed, select the correct command with the
[official PyTorch installer](https://pytorch.org/get-started/locally/). Do not guess a
wheel URL from the driver version shown by `nvidia-smi`.

NCCL is required for multiple GPUs. The runner checks it before any long preparation or
training work begins.

## 3. Confirm that local preparation was included

Run the preparation-only entry point once on the server:

```bash
sion-train --config sion_translate.yaml --prepare-only
```

For a fully prepared bundle, this should authenticate and reuse the tokenizer and both
indexed datasets rather than rebuild them. If the bundle intentionally omitted an
artifact, the command prepares only that missing artifact.

Do not continue when it reports a tokenizer, graph, source fingerprint, token-feature,
or dataset inventory mismatch. Those errors mean the server files do not represent the
same training contract as the bundle configuration.

## 4. Start training

```bash
python3 easy_run.py
```

The runner performs these operations:

1. Validates CUDA, NCCL, the visible GPU set, and the common BF16 capability.
2. Authenticates the raw inputs and all prepared artifact inventories.
3. Selects a buffered data-fit model preset, batch size, precision, and distributed
   strategy from the corpus and the smallest visible GPU.
4. Resumes the furthest fully authenticated stage.
5. Runs foundation pretraining when a prepared monolingual dataset is available.
6. Runs directed translation SFT from the selected foundation weights.
7. Runs gold-anchored MRT and preference posttraining from the selected SFT weights.
8. Restores the selected best raw or EMA weights and creates verified final exports.

In an interactive terminal with an existing `tmux` installation, `easy_run.py` may create
a checkout-specific session. Slurm, `nohup`, containers, and non-interactive SSH remain
in the current process. Disable tmux explicitly with:

```bash
SION_NO_TMUX=1 python3 easy_run.py
```

## 5. Automatic GPU policy

- The runner uses every visible CUDA GPU.
- It uses BF16 only when every rank supports native BF16; otherwise automatic precision
  falls back to FP16.
- Batch and activation checkpointing decisions use the smallest visible VRAM capacity.
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
runs/auto/
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
