# Foundation pretraining

Foundation pretraining runs before translation. It first teaches the encoder-decoder to
reconstruct corrupted monolingual text and may mix a small, explicitly structured
prompt-to-reasoning task.

```text
foundation reconstruction -> translation SFT -> MRT/preference posttraining
runs/*/foundation/           runs/*/pretrain/   runs/*/posttrain/
```

If no valid foundation corpus or prepared foundation dataset exists, a new run starts
translation SFT from fresh initialization. It does not create a fake foundation stage or
publish a `sion` artifact. A compatible authenticated SFT or MRT checkpoint still takes
priority over all earlier-stage work.

## Corpus layout

Create one directory per canonical BCP 47 language tag under `data/corpus/`. Subdirectories
inside a language directory are allowed.

```text
data/corpus/
  de/
    news.txt
    2026/web.jsonl
  fr/
    books.txt
  sw/
    articles.jsonl
```

Supported ordinary inputs are:

- `.txt`: one sentence or paragraph per line;
- `.jsonl`: one JSON object per line with a string `text` field.

Files named `reasoning_*.jsonl` use the separate reasoning schema described below. The
scanner reports unsupported extensions, top-level files, invalid language directories,
and malformed rows. It does not silently convert an empty or misplaced source into a
successful corpus.

To add a reconstruction-only language, set `foundation.languages` without adding a
translation pair:

```yaml
foundation:
  languages: [de, fr, sw]
```

Changing this set changes the required `<denoise_LANGUAGE>` controls. Rebuild the
tokenizer and foundation dataset under new artifact paths.

## Structured reasoning task

Reasoning examples remain a foundation-only encoder-decoder objective. They never pretend
to be translation rows.

Each `reasoning_*.jsonl` row has this form:

```json
{
  "prompt": "What is 2 plus 3?",
  "think": "Add the two integers.",
  "answer": "5",
  "language": "en",
  "source": "example/source",
  "license": "Apache-2.0"
}
```

The encoder receives `<reason_en> + prompt`. The decoder target is:

```text
<think> Add the two integers. </think> <answer> 5 </answer>
```

Reasoning controls differ from translation `<2LANGUAGE>` and reconstruction
`<denoise_LANGUAGE>` controls. Reasoning traces therefore do not become ordinary
translation targets. The foundation collator also bypasses span corruption for these
rows and truncates content without cutting the structural delimiters.

Long traces contain many more target tokens than ordinary reconstruction rows. The
default target row share is deliberately small:

```yaml
foundation:
  reasoning_sample_share: 0.05
```

Use `0` to disable the task. The validated range is at most `0.10`. Adding or removing
reasoning languages changes tokenizer controls and requires new prepared artifacts. Use
licensed, reviewed, self-contained examples; do not add grounded questions that cannot
be answered from the supplied prompt.

## Effective language contract

If `foundation.languages` is non-empty, it defines the reserved foundation language set.
Otherwise, the set is derived from the translation configuration. Source-only languages
are excluded in either case.

This exclusion matters because reconstruction trains a language as a decoder output.
A source-only language is explicitly forbidden as a translation output; foundation
training must not teach the opposite contract.

The published foundation capability is narrower than the reservation. It contains only
languages with positive sampling mass in the finalized train split. A configured but
empty language cannot appear in the model metadata merely because its directory was
reserved.

## Language balance

Temperature sampling uses `foundation.language_sampling_alpha` (default `0.7`) to reduce
large corpus imbalances. It cannot manufacture missing data. A zero-row language receives
zero probability and is reported as a data gap.

By default, the pipeline warns and continues with available languages. Use a fail-closed
policy when the experiment requires every reserved language:

```yaml
foundation:
  require_all_languages: true
```

Record per-language rows, characters, sampled probability, and effective train-split mass
with every run. A global reconstruction loss can hide a language that is never sampled.

## Tokenizer sampling

Monolingual text contributes to tokenizer training so foundation-only vocabulary does not
collapse into byte fallback. It is not included without a bound: each language receives a
sample cap derived from its parallel-corpus exposure and
`foundation.tokenizer_sample_ratio`.

```yaml
foundation:
  tokenizer_sample_ratio: 0.40
```

Sampling is deterministic and spread across files rather than taking a source prefix.
Prefix truncation can select only the first domain in a source collection and encode that
bias into the vocabulary. The same bounded sample contract applies to required-character
frequency analysis.

The project pins SentencePiece `0.2.1`. The reason and reproducer are documented in
[`sentencepiece-sigsegv.md`](sentencepiece-sigsegv.md).

## The output is not a translation model

The foundation release uses the separate name `sion`. The derived translation release is
`sion_translate`.

Foundation and translation checkpoints share an architecture, so tensor shapes alone
cannot prove capability. Foundation export metadata therefore states:

```json
{
  "release_name": "sion",
  "release_version": "1.5",
  "translation_capable": false,
  "languages": ["de", "fr", "sw"]
}
```

It contains no `language_pairs` and no `translation_directions`; those concepts do not
exist for monolingual reconstruction weights. `Translator` rejects the artifact and
directs the user to an SFT or MRT export.

After real training and evaluation finish, the foundation artifact belongs in a new
Hugging Face repository, separate from `gaon12/sion_translate`. The repository name and
model card are not created speculatively before that artifact exists.

## Completion, reuse, and stage transfer

A successful run publishes `runs/*/foundation/stage_complete.json`. Later runs verify the
marker, tokenizer, dataset manifest, checkpoint generation, exact state digest, export,
release identity, and effective language list before reusing the stage.

An authenticated downstream checkpoint takes priority:

- valid MRT resume skips foundation and SFT;
- valid SFT resume skips foundation;
- a completed foundation stage transfers only its selected weights into a new SFT
  optimizer.

Stage transfer is not optimizer resume. SFT receives no foundation optimizer moments,
scheduler position, or step counter because its objective and loss surface differ. The
tokenizer and model configuration must still match exactly; equal tensor shapes do not
protect against vocabulary rows that refer to different pieces.

An interrupted foundation run has no completion marker and resumes through its own
authenticated checkpoint generation. A completed stage is not repeated merely because a
later translation run failed.

## Configuration reference

```yaml
foundation:
  enabled: true
  corpus_dir: data/corpus
  dataset_dir: artifacts/foundation_dataset
  release_name: sion

  languages: [de, fr, sw]
  language_sampling_alpha: 0.7
  minimum_language_share: 0.05
  require_all_languages: false
  minimum_characters: 8
  maximum_characters: 4000
  deduplicate: true
  reasoning_sample_share: 0.05

  noise_density: 0.15
  mean_span: 3.0
  tokenizer_sample_ratio: 0.40

  num_train_epochs: 3
  early_stopping_min_epochs: 2
  batch_size_per_gpu: 16
  learning_rate: 0.0003
  warmup_steps: 2000
  final_export_formats: [fp32, bf16, transformers]
```

`enabled: true` means "run when a valid corpus is available." It does not turn an absent
corpus into a foundation model.
