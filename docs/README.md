# Documentation index

Choose the document that answers the question you have. `START-HERE.md` and
`how_to_run.txt` contain the same GPU handoff procedure in Markdown and plain text; update
both whenever that procedure changes.

## Top-level guides

| Document | Question | Audience |
|---|---|---|
| [`../README.md`](../README.md) | What is the project, and how do I install, prepare, train, and export it? | New contributors and operators |
| [`../START-HERE.md`](../START-HERE.md) | How do I verify and run the prepared GPU ZIP? | GPU training operator |
| [`../how_to_run.txt`](../how_to_run.txt) | What is the same GPU procedure without Markdown formatting? | Plain terminal users |
| [`../MODEL_CARD.md`](../MODEL_CARD.md) | What can the currently published weights do? | Model users |
| [`../POSTTRAINING.md`](../POSTTRAINING.md) | How do MRT, preferences, rewards, and selection work? | Training and evaluation developers |

Model cards describe completed, published artifacts. They are not updated merely because
the code for a future run changed.

## Operational and design documents

| Document | Question |
|---|---|
| [`retraining-runbook.md`](retraining-runbook.md) | How do I run every preparation and training stage manually? |
| [`foundation-pretraining.md`](foundation-pretraining.md) | What is the reconstruction foundation stage, and how is its separate model role authenticated? |
| [`sentencepiece-sigsegv.md`](sentencepiece-sigsegv.md) | What caused the tokenizer-training SIGSEGV, and what operational guard prevents it? |
| [`H100_TRAINING.md`](H100_TRAINING.md) | How do I prepare, size, launch, monitor, recover, and export on H100 hardware? |
| [`QUALITY_OVERHAUL.md`](QUALITY_OVERHAUL.md) | Which quality problems and evaluation controls have been identified? |
| [`corpus-gaps.md`](corpus-gaps.md) | Which domains and direction segments are underrepresented? |
| [`DATA_EXPANSION_PLAN.md`](DATA_EXPANSION_PLAN.md) | How should licensed data expansion address those gaps? |
| [`COMPARISON.md`](COMPARISON.md) | How should this model be compared with other translation systems? |

## Documents intentionally outside Git

- `PROJECT_ROAST.md` is an internal corpus audit that identifies local source rows by file
  and line. It is useful only beside the non-redistributed corpus.
- `SERVER-OPS.md` is included only in an internal GPU operations package.

## Capacity notes

[`../configs/aspirational/README.md`](../configs/aspirational/README.md) explains why some
syntactically valid presets are isolated from runnable configurations: the available data
does not support their capacity.
