# sion_translate 1.5

`sion_translate` is a from-scratch encoder-decoder training stack for multilingual
machine translation. The repository currently uses Korean, English, and Japanese data,
but the implementation does not assume those languages. A run may define any set of
BCP 47 language tags, physical language pairs, and directed translation edges.

Version 1.5 separates two model roles:

| Stage | Purpose | Local output | Hugging Face role |
|---|---|---|---|
| Foundation | Monolingual span reconstruction and optional reasoning data | `runs/*/foundation/` | A new base-model repository, created only after training finishes |
| Translation SFT | Directed translation from the selected foundation weights | `runs/*/pretrain/` | Intermediate `sion_translate` weights |
| Translation MRT | Sequence-level risk and preference optimization | `runs/*/posttrain/` | Final `gaon12/sion_translate` release |

The foundation artifact is not a translation model. It has never learned translation
directions, and the inference runtime rejects it. Model cards are intentionally not
updated before the corresponding training and evaluation runs are complete.

## What version 1.5 adds

- An explicit multilingual graph. Physical pairs, trained directions, source-only
  languages, and revision-trained directions are separate authenticated contracts.
- Full-distribution candidate refinement in both training and inference. The first
  next-token distribution is converted into an expected token embedding, fed back into
  the decoder state, and used to compute the final full-vocabulary distribution.
- A three-stage foundation -> SFT -> MRT pipeline with content-addressed lineage and
  restart-safe checkpoints.
- Gold-anchored minimum-risk preference learning. A sampled candidate cannot receive a
  preference reward without also remaining anchored to the reference sequence.
- Conservative structured-content checks for placeholders, format specifiers, template
  syntax, numbers, and protected tokens.
- Smooth token-based model sizing with intermediate valid architectures, so a small
  corpus increase cannot double model capacity at a preset boundary.
- Bounded-memory preprocessing, exact artifact inventories, transactional exports, and a
  verifiable GPU upload bundle.
- Native FP32, BF16, FP16, FP8, INT8, INT4, GGUF, and Transformers export paths. The exact
  supported set depends on the installed optional dependencies.

The package version is `1.5.0`. Model release metadata uses the generation label `1.5`.

## Requirements

- Python 3.11 or newer
- PyTorch 2.8 or newer
- A CUDA GPU for real training; CPU execution is intended for tests and small preparation
  jobs
- Enough local disk space for the raw data, tokenizer, indexed datasets, and final ZIP
- Git when running directly from an editable source tree

Run editable installs from a real Git clone. A metadata-free source directory is accepted
only when Git resolves a committed project root containing the runtime bundle verifier.
This prevents a damaged GPU bundle from silently becoming an unauthenticated local run.
A prepared GPU ZIP does not need a `.git` directory because its manifest and checksum list
provide the stronger identity. For a downloaded source archive, install a built wheel or
clone the repository before using the editable commands below.

Linux or macOS:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,export]"
.venv/bin/python -m pytest -q
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,export]"
.\.venv\Scripts\python.exe -m pytest -q
```

The `export` extra installs GGUF and TorchAO support. Install it before a long run so the
final export does not fail after training because a requested backend is missing.

## Data contract

Each input file is UTF-8 JSONL. A normal row contains one value for each side of a
configured pair:

```json
{"de":"Guten Morgen.","fr":"Bonjour."}
{"de":"Die Sitzung beginnt um drei Uhr.","fr":"La réunion commence à trois heures."}
```

Language keys are canonical BCP 47 tags. The following graph trains `de -> fr`,
`fr -> de`, and `sw -> ar`, without inventing `ar -> sw`:

```yaml
data:
  language_pair: []
  language_pairs:
    - [de, fr]
    - [sw, ar]
  translation_directions:
    - [de, fr]
    - [fr, de]
    - [sw, ar]
  bidirectional: false
```

For a source-only variety, include it in a physical pair and list it under
`source_only_languages`. The pipeline will not create a reverse edge that emits that
variety. This mechanism is generic and is not tied to a particular language.

Revision rows use the following shape:

```json
{
  "de": "Original source. <draft> Imperfect draft.",
  "fr": "Corrected target.",
  "training_direction": ["de", "fr"],
  "provenance": {"transformation": "revision"}
}
```

The indexed dataset derives the exact revision subgraph from authenticated row
provenance. A broad Boolean flag cannot silently advertise revision ability on every
translation edge.

Only use data that you are permitted to process, modify, and redistribute. Raw training
data, prepared datasets, checkpoints, and weights are intentionally excluded from Git.

## Quick start

Place one or more JSONL files under `data/`, review `sion_translate.yaml`, and run:

```bash
python easy_run.py
```

On a supported CUDA server, the automatic runner discovers the input graph, prepares or
reuses authenticated artifacts, chooses a model and batch plan, resumes the furthest
valid stage, and runs foundation, SFT, and MRT in order.

To perform every safe CPU-side preparation step and stop before allocating the model:

```bash
sion-train --config sion_translate.yaml --prepare-only
```

This is the recommended local workflow before uploading a GPU bundle. It can train the
tokenizer and build indexed translation and foundation datasets locally. Re-running the
command checks fingerprints and reuses complete artifacts.

## Manual tokenizer and dataset preparation

Pass the same language graph to tokenizer training and data preparation. The examples
below are intentionally one-way; repeat both `--language-pair` and
`--translation-direction` for a larger graph.

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

The tokenizer splits digits by default. Do not disable this for a production run: merged
multi-digit pieces make it much easier for a model to replace an amount, date, version,
or measurement with another plausible value.

Preparation uses SQLite-backed exact deduplication by default, prevents approximate
duplicates from crossing split boundaries, records source fingerprints, and writes a
manifest for every generated shard.

The dataset manifest names physical token totals by storage side, not by language. Under
`stats_schema: sion-prepare-stats-src-tgt-v1`, `src_tokens` and `tgt_tokens` count the
accepted content token IDs physically stored in the source and target shard files. They
do not count padding, runtime language controls, virtual reverse examples, repeated
epochs, or later training augmentation. This definition stays valid for any configured
language graph. Authenticated v1.5 manifests without a statistics marker are still read
when they contain the exact legacy `ko_tokens` and `ja_tokens` field set, but every new
manifest writes only the storage-neutral names.

Translation preparation writes deterministic, content-bound worker chunks beside the
requested output. If tokenization is interrupted, run the same command again: compatible
chunks are integrity-checked and reused in source order. A new parent process advances a
generation fence before cleanup, so workers left alive by an abrupt parent termination
cannot publish into the resumed generation. Chunk creation finishes before staging starts;
the runner then derives a conservative staging-space plan from the actual candidate token
and metadata totals. If that gate reports insufficient disk space, completed chunks remain
reusable and no partial dataset directory is published.

The language graph remains configuration-driven. Resource limits apply to record shape,
not language names: one physical JSONL line may be at most 16 MiB, one worker batch may
contain at most 64 MiB of selected raw records, one line may expand to at most 1,024
physical pairs, and supported stored metadata for one pair may be at most 256 KiB. These
limits prevent one nested record from exhausting memory or cleanup space. Split a record
into smaller independent JSONL rows when a legitimate source exceeds one of them.

### Foundation corpus

Store monolingual text below one directory per configured language tag:

```text
data/corpus/
  de/
    news.jsonl
  fr/
    books.txt
  sw/
    web.jsonl
```

The foundation stage publishes only languages that have positive sampling mass in the
prepared training split. A language reserved in configuration but absent from the final
sample cannot appear in the foundation model's capability metadata.

## Smooth automatic model sizing

Automatic sizing reads the exact sum of `src_length + tgt_length` from the authenticated
prepared training indexes. Each physical source/target sequence is counted once. Adding
virtual reverse directions or training for more epochs reuses the same content and does
not inflate model capacity. A legacy caller without prepared lengths uses an explicit
32-tokens-per-physical-pair compatibility reference; production training does not use
that proxy.

The continuous scaling curve uses these reference anchors:

| Unique physical tokens | Anchor architecture | Approximate parameters at 48k vocab with refinement |
|---:|---|---:|
| 6.4 million | `small` | 66.8M |
| 96 million | `medium` | 124.2M |
| 960 million | `base` | 209.7M |
| 3.2 billion | `large` | 439.8M |
| 9.6 billion | `xlarge` | 801.1M |

Targets are interpolated continuously in log token/log parameter space. The selector then
chooses the nearest point from a deterministic 61-architecture ladder. Each intermediate
point changes only one primary width, encoder-depth, decoder-depth, or feed-forward
dimension; head counts are derived again when width changes. Every adjacent option
preserves attention divisibility and differs by less than 12%, instead of jumping directly
from about 210M to 440M.

All automatic candidates use two query heads per KV head. Their KV projection width is
therefore exactly half of `d_model` and never falls when model width grows. Head dimensions
remain 8-aligned and between 64 and 160. This quality-oriented policy uses more KV
parameters, attention work, and inference cache than the older, more aggressively grouped
presets; the trainer's preflight counts the constructed model's actual parameters before
allocating optimizer state.

The total target uses a fixed 48,000-piece tied-embedding reference. The candidate scorer
uses the tokenizer's actual vocabulary and enabled modules, so a discrete vocabulary-size
change consumes a different share of the same budget instead of creating another total
parameter cliff. Explicit complete architecture settings in YAML always take precedence.

The 27,602,231-row inventory has no authoritative final size until tokenizer training and
dataset preparation expose its exact token count. Its row-only compatibility preview is
about 203.2M parameters with the current 48k vocabulary and candidate refinement, close to
the 200M class rather than the 440M class. The training log records the exact token count,
continuous target, selected architecture, vocabulary size, and estimated parameter count.

## Training and restart behavior

Run the configured pipeline with:

```bash
sion-train --config sion_translate.yaml
```

All stages use complete shuffled dataset passes by default. `max_steps` exists for smoke
tests and explicit overrides. Early stopping is evaluated at epoch boundaries after the
configured minimum number of epochs.

Before optimizer allocation, the trainer verifies that:

- every advertised language pair has positive sampling mass;
- every advertised directed edge can emit real training rows;
- revision directions exactly match marked indexed rows;
- a foundation lineage matches the tokenizer, dataset, checkpoint, and release identity;
- the selected parallel strategy fits the available device budget.

Checkpoints and exports use content hashes and generation identities. On restart, the
runner selects the most advanced fully authenticated stage. A valid MRT checkpoint skips
foundation and SFT; a valid SFT checkpoint skips foundation. Invalid or partial state is
kept separate rather than being accepted as a successful run.

For detailed H100 and distributed commands, see
[`docs/H100_TRAINING.md`](docs/H100_TRAINING.md).

## Candidate refinement and deployed output

When `model.experimental.candidate_refinement_enabled` is true, both `forward()` and
generation use the same staged next-token computation:

1. Compute the first full-vocabulary distribution.
2. Form its expected token embedding, update the decoder representation, and recompute
   the full-vocabulary logits.
3. If more than one refinement step is configured, repeat step 2 and use the final trained
   endpoint.

Losses are applied to the refined logits during training, and beam or sampling decisions
use the refined logits during inference. Therefore the final translation deployment uses
the second-stage model/output (`M2`/`T2` in design notes), not the provisional first-stage
distribution. Export metadata authenticates the feature flag and the default reasoning
level against the model configuration.

This is distribution refinement, not candidate reranking. Optional sequence-level MBR
reranking remains a separate inference feature.

Validation reports `candidate_refinement_nll_gain` as
`NLL(provisional distribution) - NLL(final distribution)`. A positive value means the
refined endpoint improved the held-out target token. Reports include token-weighted global
and target-language values, exact directed-edge values, a direction macro mean, and the
worst directed edge.

Translation training treats this measurement as a release rule, not only a diagnostic.
Before the first optimizer update, it records a safe step-zero fallback. A later checkpoint
can become the deployable `best` only when all of these conditions hold:

- the checked metrics come from the same raw or EMA weight family that deployment uses;
- every exact edge in the configured language graph is present with finite token evidence;
- every per-edge gain is finite; and
- the worst edge is non-negative, allowing only `1e-6` of numerical tolerance around zero.

The versioned checkpoint attestation binds the deployed weight family and a fingerprint of
the complete directed graph. Legacy checkpoints can still resume optimizer progress, but
their old best record is re-evaluated before release. Guarded runs keep `checkpoints/latest`
for restart safety but do not create `exports/latest`. A `RELEASE_INELIGIBLE.json` marker
blocks stale inference directories from discovery, direct native loading, and export
validation until a guard-approved `best` export succeeds.

## Inference

After training:

```bash
sion-translate \
  --model runs/auto/posttrain/exports/best/model_ema.pt \
  --to fr \
  "Die Sitzung beginnt um drei Uhr."
```

The runtime accepts only a trained and authenticated direction. It refuses foundation
artifacts, unknown edges, mismatched tokenizer identities, corrupted sidecars, and
capability claims that disagree with the architecture.

Optional candidate generation and MBR/QE reranking:

```bash
sion-translate --to fr --candidates 7 --rerank mbr+qe "..."
```

Treat reranking as an evaluation-controlled option. It adds latency, and improvements on
a tiny diagnostic set are not evidence of a production-quality gain.

For sentences containing amounts, dates, identifiers, versions, or measurements, always
check preservation metrics. The data filter and posttraining reward penalize corruption,
but neither can recover a correct value that the tokenizer and model never produced.

## Evaluation and audits

Useful commands include:

```bash
# Token exposure for the exact directed graph
sion-audit-tokens \
  --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer/sion.model \
  --translation-direction de fr \
  --translation-direction fr de

# Leakage audit against an evaluation-only set
python scripts/data/audit_holdout_leakage.py \
  --holdout examples/diagnostic_cases.jsonl \
  --corpus "data/*.jsonl" \
  --language-pair de fr

# Non-destructive human-review queue
python scripts/data/build_review_queue.py \
  --input "data/*.jsonl" \
  --output reports/review_queue.jsonl \
  --language-pair de fr
```

Do not report a score from a set that overlaps training data. Record the exact checkpoint
digest, tokenizer digest, direction graph, decoding settings, and benchmark source with
every result.

## Prepare and verify a GPU upload bundle

After local tokenizer and dataset preparation succeeds, build a self-contained archive:

```bash
python scripts/package_gpu_bundle.py build \
  --output sion_translate-prepared.zip \
  --prepared-only

python scripts/package_gpu_bundle.py verify-archive sion_translate-prepared.zip
```

`--prepared-only` is the recommended GPU handoff. It includes the complete tokenizer,
the translation dataset, and the foundation dataset when foundation training is enabled.
It omits the raw parallel and monolingual training corpora, even if a corpus file happens
to be tracked, while retaining configured evaluation-only material. The extracted server
tree can therefore authenticate and train from the prepared shards without repeating
CPU-heavy tokenization or indexing.

Use the individual `--with-*` switches only for an intentional rebuild workflow. For
example, add `--with-monolingual-corpus` when the GPU server must build the foundation
dataset itself. That raw-corpus option deliberately conflicts with `--prepared-only`.

The builder authenticates the tokenizer, token features, translation dataset, foundation
dataset, source provenance, configuration-selected graph, Git tree, file sizes, and
SHA-256 digests. Manifest format 2 also binds the exact tracked `sion_translate.yaml`
path and digest used by the default training command. The current format intentionally
requires the canonical artifact and corpus paths so an alternate local layout cannot be
packaged and then interpreted differently on the server. The builder publishes the ZIP
atomically and refuses to overwrite an existing archive unless `--overwrite` is explicit.

After extraction on the GPU server, verify the tree before training:

```bash
python scripts/package_gpu_bundle.py verify-tree sion_translate
```

Run `sion-train --config sion_translate.yaml --prepare-only` once after verification. In
a prepared-only tree this is an offline authentication pass: missing or inconsistent
prepared artifacts are errors instead of a request to rediscover omitted raw corpora.

## Export and repository roles

The final run exports from the restored best weights, not merely the last optimizer step.
Current manifests bind the checkpoint step, state hash, tokenizer, token features,
pipeline lineage, trained directions, revision subgraph, feature flags, and release role.

The intended publication layout after training and evaluation is complete is:

- a new Hugging Face repository for the `sion` foundation model;
- `gaon12/sion_translate` for the final translation model;
- deployment from the refined second-stage translation output.

The project does not guess a foundation repository name or publish a model card before a
real artifact exists.

## Development checks

Use the same order for every change:

```bash
python -m ruff format src tests
python -m ruff check src tests
python -m pyright src
python -m pytest -q
```

If a check fails, return to the code change, fix the cause, and run formatting, linting,
and tests again before committing. Keep commits scoped to one feature, security fix,
documentation update, or file group, and write detailed English commit messages.

## Documentation

- [`docs/README.md`](docs/README.md): documentation index
- [`START-HERE.md`](START-HERE.md): GPU handoff checklist
- [`docs/H100_TRAINING.md`](docs/H100_TRAINING.md): H100 and distributed operation
- [`POSTTRAINING.md`](POSTTRAINING.md): MRT and preference training design
- [`docs/foundation-pretraining.md`](docs/foundation-pretraining.md): foundation stage
- [`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md): corpus expansion plan
- [`docs/QUALITY_OVERHAUL.md`](docs/QUALITY_OVERHAUL.md): quality work and evaluation rules

## License

Code is released under the [MIT License](LICENSE). Dataset and model artifact rights are
separate; verify every source before use or redistribution.
