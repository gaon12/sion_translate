# H100 training and export operations

This guide describes a reproducible H100 workflow for `sion_translate` 1.5. It covers
local preparation, single- and multi-GPU launch commands, capacity checks, telemetry,
out-of-memory recovery, final export, and artifact verification.

Historical runtime or memory measurements are only starting points. Record fresh results
for the exact commit, data fingerprint, graph, configuration, CUDA stack, and hardware
used by each run.

## 1. Create and record the environment

`environment.yml` installs Python 3.11 and the development and export dependencies. It is
a portable environment description, not a platform-specific lock file.

```bash
conda env create -f environment.yml
conda activate sion-translate
```

Update an existing environment with:

```bash
conda env update -n sion-translate -f environment.yml --prune
conda activate sion-translate
```

Verify the runtime before starting an expensive job:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_runtime', torch.version.cuda); print('gpu_count', torch.cuda.device_count()); print('bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported()); print('nccl', torch.distributed.is_nccl_available())"
```

Do not start distributed H100 training when CUDA, BF16, or NCCL support is missing. Save
the output of `conda list --explicit`, the PyTorch version, CUDA runtime, driver version,
and `nvidia-smi -q` with the run metadata.

## 2. Prepare artifacts before the GPU job

Tokenizer training and dataset indexing are CPU and storage work. Run them locally when
the local machine has enough memory and disk space:

```bash
sion-train --allow-local-checkout --config sion_translate.yaml --prepare-only
```

This command stops before model allocation. It authenticates and reuses complete
artifacts on subsequent runs. It also opens the ordinary validation and dedicated
`refinement_evidence` splits and constructs the fixed, graph-complete release cohort.
Missing directions or fewer than the configured distinct examples per direction fail here,
before the GPU job can consume paid time.
The evidence DataLoader uses a separate clean translation collator. Ordinary validation may
enable denoising, but evidence collation always keeps denoising and input noise at zero so every
authenticated row retains its directed translation identity.

For a manual one-way example:

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

For a larger graph, repeat `--language-pairs SOURCE TARGET` and
`--translation-direction SOURCE TARGET` with the exact same graph for both commands.
Never infer a reverse direction merely because a physical pair exists.

A tokenizer is incompatible with older checkpoints when its vocabulary, digit-splitting
policy, language controls, or graph changes. The runtime checks tokenizer hashes, sizes,
vocabulary size, graph metadata, and token features. Do not overwrite a failed artifact
in place. Move it to a separate recovery location, verify which runs use it, and prepare
a new public path.

### Build the upload archive

When local preparation is complete:

```bash
python scripts/package_gpu_bundle.py build \
  --output sion_translate-prepared.zip \
  --prepared-only

python scripts/package_gpu_bundle.py verify-archive sion_translate-prepared.zip
```

Prepared-only bundles include authenticated tokenizer, translation, and enabled
foundation artifacts while omitting raw parallel and monolingual training corpora. This
is the normal path when local preparation has completed. Use the individual `--with-*`
options only for a deliberate server-side rebuild; `--with-monolingual-corpus` is needed
when the server must prepare the foundation dataset and conflicts with `--prepared-only`.
Candidate-refinement GPU jobs must use a prepared dataset-bearing bundle. The builder rejects a
raw-only candidate-refinement archive because it cannot authenticate the fixed per-direction
evidence capacity before paid server work begins.
After extraction on the server:

```bash
python scripts/package_gpu_bundle.py verify-tree sion_translate
sion-train --config sion_translate.yaml --prepare-only
```

The second command performs an offline artifact preflight and stops before model
allocation. Do not train from an archive that fails either verification command or this
preflight. Do not pass `--allow-local-checkout` in the extracted bundle; its manifest and
checksum list are the required trust boundary.

## 3. Run on one H100

```bash
conda activate sion-translate
CUDA_VISIBLE_DEVICES=0 python easy_run.py
```

One GPU always uses the runtime `single` strategy. Keep `training.precision: bf16` on an
H100 unless an experiment specifically requires another precision. The single/DDP path
keeps FP32 master parameters and uses CUDA autocast for BF16 compute. BF16 does not need
an FP16 gradient scaler. `easy_run.py` authenticates the prepared bundle, runs the CUDA
canary, and owns the complete child process tree. It reads the bundle's authenticated
configuration; do not append `--config` to this launcher.

Start with a short capacity probe in a separate output directory. A successful probe
should include the longest realistic sequence buckets and at least one validation pass.

## 4. Run on multiple H100s in one node

Example for eight GPUs:

```bash
conda activate sion-translate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python easy_run.py
```

List exactly the GPUs assigned by the scheduler. The launcher derives the local process
count, checks every device, runs an all-GPU NCCL canary, and gives each process one GPU.

The effective batch is:

```text
effective_batch =
  batch_size_per_gpu * world_size * gradient_accumulation_steps
```

When comparing GPU counts, keep this value constant unless the experiment is explicitly
about batch scaling. Compare steady-state target tokens per second, not steps per second,
because sequence lengths vary.

## 5. Run across multiple nodes

Every node must have the same code commit, environment, configuration, tokenizer, dataset
inventory, and readable checkpoint storage. Example for two nodes with eight GPUs each:

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --max-restarts=0 \
  --node-rank=0 \
  --master-addr=10.0.0.10 \
  --master-port=29500 \
  -m sion_translate.cli.train \
  --config sion_translate.yaml
```

Run the second node with `--node-rank=1`. The master address must be reachable from every
node. Use a 100-500-step probe to validate NCCL, shared storage, checkpoint publication,
and throughput before starting the full run.

The multi-node command is an advanced exception because `easy_run.py` currently owns only
one node. Run it only inside a scheduler allocation whose job teardown kills every process
on every node. Raw `torchrun` does not provide the launcher's local parent-death guard or
pre-training CUDA/NCCL canaries; run the same authenticated preflight and hardware probes
under the scheduler before the paid training allocation.

If global tokens per second barely improves as nodes are added, measure storage reads and
network collectives before changing the model.

## 6. Choose single, DDP, or FSDP2

Use DDP when a complete model, gradients, optimizer state, optional EMA, activations, and
temporary buffers fit on every GPU. Use FSDP2 only when the persistent state cannot fit
per GPU.

| Condition | Strategy | Reason |
|---|---|---|
| One GPU | `single` | No distributed wrapper is needed |
| Full state fits on every GPU | `ddp` | Avoids parameter all-gather overhead |
| Full state does not fit | `fsdp2` | Shards parameters, gradients, and optimizer state |

Before allocating CUDA storage, the runner builds a meta-device model and estimates FP32
master parameters, gradients, AdamW moments, and optional FP32 EMA. The gate reserves
substantial capacity for BF16 all-gathers, activations, kernels, CUDA context, and long
batches. Treat a capacity rejection as a safety result, not as a suggestion to bypass the
check.

Recommended H100 settings are:

```yaml
training:
  precision: bf16
  parallel_strategy: auto
  fsdp_reduce_dtype: bf16
  reshard_after_forward: true
```

Turning `reshard_after_forward` off can improve speed but increases memory. Test it only
after measuring peak allocation with representative sequences.

Generation and sampling are registered as FSDP forward methods. All ranks agree on
generation limits and termination so one rank cannot leave nested decoder collectives
early.

DDP uses gradient bucket views. Configurations with parameters that are intentionally
unused in an objective, such as a label-only auxiliary head during candidate scoring,
must retain the appropriate `find_unused_parameters` behavior.

`torch.compile` has a startup cost. Exclude compilation steps from throughput results. If
compilation fails, save the graph-break diagnostics and compare the same run with
`compile: false`.

## 7. Read telemetry

The trainer writes aggregated JSON and TensorBoard values at `training.log_every`.

| Field | Meaning | Use |
|---|---|---|
| `global_tokens_per_second` | Target/scored tokens processed across all ranks | Primary throughput comparison |
| `seconds_per_step` | Mean wall time per optimizer step | Detect stalls and length-distribution changes |
| `data_wait_fraction` | Fraction of step time waiting for input | Diagnose CPU, worker, storage, or indexing limits |
| `cuda_allocated_gib` | Live tensor memory | Current model and activation footprint |
| `cuda_reserved_gib` | PyTorch allocator reservation | Context for fragmentation; not itself a leak |
| `cuda_peak_allocated_gib` | Peak live memory in the logging interval | Main OOM headroom measure |
| `loss`, `auxiliary_loss` | Main and auxiliary objectives | Confirm that speed changes do not break learning |
| `grad_norm` | Gradient norm around clipping | Detect instability or non-finite updates |
| `reward_cpu_seconds` | CPU text decoding and sequence reward time | MRT CPU cost |
| `reward_wait_seconds` | Time GPU work waits for reward workers | MRT synchronization bottleneck |
| `reward_overlap_fraction` | CPU reward time overlapped with candidate scoring | Higher is better when wait remains low |
| `candidate_scoring_seconds` | Candidate log-probability scoring time | MRT GPU/compute cost |

CUDA peaks reset after each logging interval. Compare p50 and p95 over hundreds of steps,
and record GPU utilization separately with `nvidia-smi dmon` or cluster telemetry.

Diagnosis order:

1. If `data_wait_fraction` stays above roughly 0.10-0.15, check worker count, CPU quota,
   local NVMe, and indexed dataset placement.
2. If data wait, utilization, and memory are all low, increase per-GPU batch size while
   reducing accumulation to preserve the effective batch.
3. If peak memory is high but utilization is low, inspect excessive activation
   checkpointing, collective frequency, and very short sequence buckets.
4. If multi-GPU scaling is poor, separate DDP/FSDP communication, network, and storage
   throughput.

## 8. Find a safe batch size

Use a separate run directory and compare 300-1,000 steady-state steps per candidate:

1. Keep commit, seed, data fingerprint, graph, GPU count, and effective batch fixed.
2. Increase `batch_size_per_gpu` gradually.
3. Reduce `gradient_accumulation_steps` when necessary to preserve effective batch.
4. Record tokens/s, data wait, peak allocated/reserved memory, and GPU utilization.
5. Leave headroom for validation, export, unusual long batches, and allocator variance.

`data.pad_to_multiple_of: 8` pads dynamic batches to Tensor Core-friendly lengths.
Increasing the bucket size can reduce padding, but also changes CPU memory and shuffle
behavior. Compare token throughput and quality, not only batches per second.

## 9. Recover from OOM

### SFT or foundation OOM

Change one control at a time:

1. Lower `training.batch_size_per_gpu`.
2. Raise gradient accumulation if the effective batch must stay constant.
3. Enable `model.gradient_checkpointing` for a large model.
4. Use FSDP2 when persistent state cannot fit under single/DDP.
5. Reduce maximum source and target lengths only when the real task allows it.

### MRT OOM

MRT generates and scores multiple sequences per source, so it can use more memory than
SFT. Reduce these settings in order:

1. `posttraining.batch_size_per_gpu`
2. `posttraining.candidate_micro_batch`
3. `posttraining.samples_per_source` (minimum 2)
4. `posttraining.max_new_tokens`
5. `posttraining.validation_num_beams` and evaluation batch size

Keep `candidate_gradient_checkpointing: true` until measured headroom proves that it is
safe to disable.

At stage transitions, the runner closes persistent workers, drops loader references,
runs garbage collection, and clears the CUDA allocator cache. A high reserved value with
a low allocated value may be harmless allocator caching. A high allocated value means a
live tensor still exists and needs investigation.

Every multi-process DataLoader also has a finite batch wait. The default
`data.dataloader_timeout_seconds: 300` stops the run when a dead worker, unreadable shard,
or stalled filesystem cannot return the next batch. Increase it only after confirming that
healthy batch preparation really needs more than five minutes. A single-process loader uses
PyTorch's required zero timeout internally, but it cannot leave a separate worker process
stuck.

Before preparation, `easy_run.py` verifies the live CPython, Linux, x86-64, glibc,
package-version, PyTorch, and compiled-CUDA target against the dependency contract. The
bundle verifier authenticates the lock file bytes; the live check confirms compatibility
with that lock and does not claim that installed files can be reconstructed from a wheel
hash.

The launcher then starts a separate 60-second canary process for every visible GPU. The
canary uses the production 12-query-head/6-KV-head GQA module with a 72-wide head, one
combined causal-and-padding mask, BF16 autocast, backward, gradient clipping, fused AdamW,
and a one-rank NCCL all-reduce when NCCL is available. Every JSON field is validated,
including finite measurements and the exact Torch/CUDA build. On a multi-GPU host, a
second bounded `torchrun` probe performs one all-reduce through a communicator containing
every visible GPU. Timeout and interruption handling kills the complete process group so
orphaned workers cannot keep a paid server busy. A failed kernel, non-finite value, child
crash, missing rank, or timeout stops the launcher before training.

The real training launch uses the same process-group ownership rule. Ctrl-C, SIGTERM, a
launcher exception, or a non-zero `torchrun` result kills and reaps the launcher and all of
its workers. NCCL monitoring, asynchronous error handling, timeout diagnostics, and a
five-minute watchdog heartbeat are enabled with `setdefault`, so an explicit cluster policy
still wins. Distributed collectives time out after 600 seconds by default instead of waiting
30 minutes. `SION_DISTRIBUTED_TIMEOUT_SECONDS` may select a reviewed value from 30 through
1,800 seconds.

## 10. Create final exports

Intermediate `exports/best` keeps only lightweight native state so large CPU conversions
do not stop the H100 at every evaluation. Runs without candidate refinement may also keep
`exports/latest` for local inspection. Candidate-refinement runs deliberately keep latest
weights only under `checkpoints/latest`: they are restartable, but they are not deployable
until every configured direction passes the held-out positive-improvement guard. After all enabled
stages finish, the CLI restores the selected best raw or EMA weights and generates the
requested final formats once in a transactional directory.

During strict final conversion, rank 0 publishes a heartbeat every 30 seconds. Peer ranks
stop after ten minutes without progress. Every rank also arms a separate clean Python
watchdog before validation or conversion starts. The watchdog terminates its owning process
after the default 30-minute whole-operation deadline even when native code, the filesystem,
or the Python GIL is stuck. A heartbeat thread that cannot stop within five seconds also
terminates the owner instead of continuing to mutate release state in the background.
`SION_FINAL_EXPORT_TIMEOUT_SECONDS` may select a reviewed deadline from 600 through 7,200
seconds for an unusually large format; all ranks must receive the same value. The final
artifact is installed atomically, so termination cannot expose a partly written deployment
directory. Use `easy_run.py` or a scheduler that tears down the complete job when one guarded
rank exits.

The release check uses `NLL(provisional) - NLL(final)`, so positive values are improvements.
It verifies the exact configured graph using the deployed raw or EMA family and rejects
missing, extra, non-finite, or insufficient worst-direction evidence. Every configured edge
must improve by at least
`training.candidate_refinement_min_worst_direction_nll_gain` (default `1e-5`). An exactly
neutral refiner therefore remains release-ineligible.

Release validation reads a dedicated `refinement_evidence` split. Genuine ordinary validation
continues to control absolute NLL, early stopping, and MRT reward; the evidence split contributes
only relative provisional-to-final NLL gains. Exact reviewed synthetic basenames may contribute
only forward edges from source-only languages. They never enter ordinary validation or test and
cannot be treated as absolute translation-quality evidence.
Their target text may overlap training for the relative comparison, but it may not overlap an
ordinary validation or test endpoint. This rule is enforced independently of input file order.

The release sampler selects exactly
`training.candidate_refinement_min_validation_examples_per_direction` distinct logical rows for
every configured edge. It never repeats a rare row; an undersized edge fails during local
`--prepare-only` and again during prepared-bundle construction. Every rank evaluates the same
small cohort, so GPU count and per-rank batch packing do not change membership. The cohort
identity binds the selected indices, graph, seed, per-edge counts, and verified dataset artifact
inventory, while excluding hardware layout.

For translation SFT, step zero is resume-only and supplies the validation comparison floor;
at least one optimizer update must both beat that floor and pass the directional gain rule.
For post-training, step zero is the inherited trained SFT model and may remain the safe reward
fallback when it passes the same directional rule. A later MRT checkpoint replaces it only
after improving the selected reward. These baselines and the v3 release attestation survive
normal and distributed resume. The attestation binds the checkpoint digest, deployed family,
language graph, direction-complete cohort, worst observed gain, and configured release floor.
Final native, Transformers, and GGUF artifacts carry the same evidence, and validation fails if
one representation loses or changes it. Before export, the code authenticates the selected
checkpoint generation, compares its stored guard fields, and hashes the exact live raw or EMA
weights. Native loading requires the matching manifest entry, while Transformers loading hashes
the reconstructed model state. Manual conversion must start from that exact manifested FP32
artifact; FP16 and BF16 outputs cannot become new evidence-bearing parents after lossy rounding.
The run writes `RELEASE_INELIGIBLE.json` into stale
best/latest inference directories before training; all normal discovery, native loading, and
directory validation paths reject the marker. Only a successful guard-approved best export
removes it.

To recover formats manually, provide the exact graph policy:

```bash
sion-export \
  runs/auto/posttrain/exports/best/model_ema.pt \
  --output runs/auto/recovered-export \
  --tokenizer artifacts/tokenizer/sion.model \
  --token-features artifacts/tokenizer/token_features.npz \
  --language-pair de fr \
  --translation-direction de fr
```

For an intentionally bidirectional pair, use `--bidirectional`. For a list of forward-only
pairs, use `--unidirectional`. For a mixed graph, repeat `--translation-direction` with
every exact edge. Current exports do not guess directionality.

Common formats are:

| Name | Output | Purpose |
|---|---|---|
| `fp32` | `model.pt` or `model_ema.pt` | Reference native weights |
| `fp16` | `model_fp16.pt` | FP16 storage/inference |
| `bf16` | `model_bf16.pt` | H100 BF16 inference or reuse |
| `fp8` | FP8 native artifact | Reduced H100-class storage and compute path |
| `int8` | `model_int8.pt` | TorchAO INT8 |
| `int4` | `model_int4.pt` | TorchAO or portable packed INT4 |
| `gguf_q4_k_m` | `model-q4_k_m.gguf` | Sion mixed K-quant exchange artifact |
| `transformers` | `transformers/` | Safetensors, tokenizer, config, and custom AutoClass code |

GGUF is an exchange container for the custom Sion encoder-decoder architecture. Stock
`llama.cpp` does not provide a Sion execution backend, so do not advertise the file as a
drop-in llama.cpp model.

Transformers loading requires trusted bundled code:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

path = "runs/auto/posttrain/exports/best/transformers"
tokenizer = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True,
    src_lang="de",
    tgt_lang="fr",
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda")
model.eval()
encoded = tokenizer("Text to translate", return_tensors="pt").to("cuda")
generated = model.generate(**encoded, num_beams=4, max_new_tokens=256)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
```

## 11. Verify the final artifacts

`export_manifest.json` binds the state, checkpoint step, release role, pipeline lineage,
trained directions, revision directions, candidate-refinement feature flags and v3 release
attestation, tokenizer, token features, and each file or directory digest.

```bash
python -c "import json,sys; from sion_translate.training.export import validate_export_directory; r=validate_export_directory('runs/auto/posttrain/exports/best'); print(json.dumps(r, indent=2)); sys.exit(0 if r['valid'] else 1)"
```

Do not upload a directory that fails validation. Confirm that:

- every required format has `status: ok` and the directory result is `valid: true`;
- every artifact has the same authenticated weight-set identity;
- tokenizer and token-feature hashes and sizes match all sidecars;
- the native payload step exactly matches metadata and manifest steps;
- language pairs, trained directions, revision directions, release role, and pipeline
  lineage agree across native, GGUF, and Transformers metadata;
- architecture feature flags exactly match the model configuration;
- no `RELEASE_INELIGIBLE.json` marker exists in the directory;
- bundled Transformers code imports and every safetensors key and shape validates.

The final run report should include the Git commit and tree, complete configuration, data
fingerprints, tokenizer digest, GPU and node count, PyTorch/CUDA/driver versions, start and
finish times, completed epochs and steps, throughput p50/p95, data-wait p50/p95, peak VRAM,
best validation metrics by direction, and export verification result.
