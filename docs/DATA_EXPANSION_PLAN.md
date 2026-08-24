# Plan for Expanding to 100 Million Parallel Pairs

Snapshot date: 2026-07-27. In this document, a “pair” does not mean one physical JSONL line. It means one `valid_pairs` item produced by the current preprocessor's `expand_parallel_record` function after quality filtering and deduplication. A single JSONL record can produce several valid pairs when it contains multiple language edges or aligned sentence arrays. Conversely, reverse-direction training examples created by `bidirectional: true` do not count again toward the physical parallel-pair target.

## 1. Snapshot totals and exact remaining requirement

The verified training root contained the following totals at the snapshot date.

| Category | Parallel pairs |
|---|---:|
| Real data | 8,414,581 |
| Rule-generated numeric synthetic data | 240,000 |
| Verified total at the snapshot date | **8,654,581** |
| Queue awaiting translation and round-trip validation | 10,000,000 |
| Expected total if the full queue passes | **18,654,581** |
| Additional pairs required to reach 100,000,000 | **81,345,419** |

The calculation is `100,000,000 - 8,654,581 - 10,000,000 = 81,345,419`. The 10,000,000 sentences in `translation_queue/namuwiki_ko_to_ja.jsonl` did not yet have Japanese references or quality decisions and therefore were not included in the verified training total. Every failed or duplicate queue row must be replaced in addition to the stated 81,345,419 pairs.

Use `data/dataset_remediation_20260727.json` and the individual remediation manifests as the single sources of truth for snapshot totals and SHA-256 identities. If a source is rebuilt or corrected, update its hash and exclusion reasons together with its row count.

## 2. Domain allocation for the remaining 81,345,419 pairs

In the table below, “minimum real data” means human translation, independently authored multilingual records joined by a shared ID, verifiable document alignment, or high-confidence parallel mining. “Synthetic ceiling” includes model translation, backtranslation, rule generation, and contrastive or revision variants. The real and synthetic columns add up exactly for every row and for the full target.

| Domain | Additional target | Minimum real data | Synthetic ceiling | Primary coverage goal |
|---|---:|---:|---:|---|
| General web, news, and public information | 20,000,000 | 20,000,000 | 0 | Current events, daily life, regional and cultural text, explanations, and recent terminology |
| Dialogue, customer support, and messaging | 12,000,000 | 10,200,000 | 1,800,000 | Colloquial language, omission, honorifics, multi-turn context, questions, and answers |
| Technology, software, and UI | 8,000,000 | 7,500,000 | 500,000 | Documentation, error messages, CLI text, placeholders, and mixed code |
| Legal, administrative, and government | 6,000,000 | 5,700,000 | 300,000 | Statutes, contracts, civil requests, institution names, articles, and versions |
| Medical, pharmaceutical, and public health | 5,000,000 | 4,700,000 | 300,000 | Symptoms, care instructions, drugs, doses, and safety text |
| Finance, commerce, and e-commerce | 5,000,000 | 4,700,000 | 300,000 | Amounts, payment, shipping, refunds, and product attributes |
| Science, education, and academia | 5,000,000 | 4,700,000 | 300,000 | Textbooks, abstracts, prose around formulas, and units |
| Literature, web fiction, and subculture | 10,000,000 | 9,500,000 | 500,000 | Narrative prose, dialogue, character voice, and genre vocabulary |
| Speech transcripts, lectures, and interviews | 5,000,000 | 5,000,000 | 0 | Spoken language, cleaned hesitation, sentence boundaries, and long context |
| Structured strings, numbers, and mixed code | 2,000,000 | 500,000 | 1,500,000 | Dates, times, units, IDs, URLs, and placeholders |
| Terminology, short queries, and commands | 1,500,000 | 1,250,000 | 250,000 | Short UI strings, search terms, proper nouns, and homonyms |
| Dialect, honorific, and formality registers | 1,000,000 | 1,000,000 | 0 | Casual and polite speech, written and spoken style, and regional and generational expressions |
| Adversarial revision and error contrasts | 845,419 | 250,000 | 595,419 | Altered numbers, omission, copying, repetition, and language mixing |
| **Total** | **81,345,419** | **75,000,000** | **6,345,419** |  |

Assuming all 10,000,000 queued pairs are model-translated and pass validation, the final 100 million-pair composition would be:

| Provenance | Final pairs | Physical share |
|---|---:|---:|
| Real data | 83,414,581 | 83.415% |
| Synthetic data | 16,585,419 | 16.585% |
| Total | **100,000,000** | **100%** |

If the current default `synthetic_sampling_weight: 0.5` remains in use, the simple weighted effective share of synthetic records in this physical composition is approximately 9.043%. Source-temperature sampling and actual per-source sizes can change that value, so recompute it from the indexed-dataset manifest and sampler statistics. Never admit low-quality translations merely to fill the synthetic ceiling. Replace rejected synthetic data with real data first.

## 3. Language-graph principles

If the product target is Korean↔Japanese translation, acquire most added data as direct `ko-ja` pairs. Do not approve auxiliary edges such as English or Russian merely because the parser can read them. Add an edge to `data.language_pairs` only when tokenizer vocabulary, model capacity, an edge-specific validation set, and a sufficient minimum data volume are all ready.

When building a multilingual model, follow these rules.

- The 100 million target is the sum of direct physical pairs after expanding every configured edge. Do not count a pivot-inferred `ko-ru` relation as a real direct pair.
- Track separate provenance, source revision, valid-pair count, synthetic share, and validation/test metrics for every edge.
- Write each undirected storage edge once in `language_pairs`. Do not specify both `ko-ja` and `ja-ko`; use an explicit directed graph or `bidirectional: true` when both decoder directions were trained.
- Keep every edge derived from one multilingual record in the same split, preventing the same content from appearing in both training and validation. The preprocessor uses the multilingual record's raw-record hash as its split key.
- Adding a language graph requires a newly trained compatible tokenizer, dataset, and checkpoint. Build the tokenizer with all required `<2xx>` and `<denoise_xx>` controls rather than reusing artifacts from a different graph.
- Release 1.5 metadata records the exact trained translation directions. A storage pair does not by itself authorize an untrained reverse direction.

Example YAML:

```yaml
data:
  language_pairs:
    - [ko, ja]
    - [en, ru]
  bidirectional: true
  synthetic_sampling_weight: 0.5
```

## 4. Supported JSONL records

Language identities use canonical BCP 47 tags in release 1.5. Simple tags such as `ko`, `ja`, `en`, and `ru` remain valid, while tags such as `pt-BR` and `zh-Hant` allow regional or script-specific graphs. The current preprocessor can mix the following layouts in one file.

### Flat language keys

```json
{"ko":"Korean greeting.","ja":"Equivalent Japanese greeting."}
```

One record may contain several configured edges.

```json
{"ko":"Korean greeting.","ja":"Japanese greeting.","en":"Hello.","ru":"Russian greeting."}
```

With `language_pairs: [[ko, ja], [en, ru]]`, the row above produces two pairs. The preprocessor does not invent language combinations absent from the configured graph.

### Equal-length sentence arrays

```json
{"ko":["First Korean sentence.","Second Korean sentence."],"ja":["First Japanese sentence.","Second Japanese sentence."]}
```

The record is rejected as `unaligned_lists` when the arrays have different lengths.

### Explicit source and target languages

```json
{"source_language":"ko","target_language":"ja","source":"Korean source text","target":"Japanese translation"}
```

`src_language` and `tgt_language` are also accepted. Source-field aliases are `source`, `src`, and `input`; target-field aliases are `target`, `tgt`, `reference`, `translation`, and `output`.

### Multiple items in one record

```json
{"records":[
  {"source_language":"ko","target_language":"ja","source":"Korean source","target":"Japanese target"},
  {"source_language":"en","target_language":"ru","source":"Hello","target":"Russian greeting"}
]}
```

Supported container names are `records`, `items`, `pairs`, and `translations`.

### Pair-named containers

```json
{"pairs":{
  "ko-ja":[{"source":"Korean source","target":"Japanese target"}],
  "en-ru":[{"source":"Hello","target":"Russian greeting"}]
}}
```

For simple tags, the parser recognizes `ko-ja`, `ja-ko`, `ko/ja`, `ja/ko`, `ko_to_ja`, and `ja_to_ko`, then restores reverse labels to the configured canonical storage-edge order. Because hyphens also delimit BCP 47 subtags, pair labels containing compound tags must use `/` or `_to_` to remain unambiguous.

### Synthetic provenance

```json
{"ko":"The meeting starts at 14:30.","ja":"The equivalent Japanese translation.","synthetic":true}
```

The marker may also be nested in metadata.

```json
{"ko":"Korean source","ja":"Japanese target","metadata":{"synthetic":true}}
```

`synthetic` must be the JSON boolean `true`, not the string `"true"`. A record-level marker or a filename beginning with `bt_`, `concat_`, `revise_`, or `synthetic_` makes the data training-only and applies the default 0.5 sampling weight. Also record the source model, prompt, generation time, decoding settings, and source-text hash in a separate source manifest.

## 5. Collection and alignment pipeline

Process every new source independently in this order.

1. **Isolate raw inputs:** record the download URL, source revision or commit, file hash, acquisition date, and extraction-tool version. Never edit the original asset directly.
2. **Align deterministically:** use shared document, paragraph, or sentence IDs; identical resource keys; or verified alignment scores. Do not assume two sentences are translations merely because they share an intent label, as demonstrated by the MASSIVE case.
3. **Apply source-specific normalization:** preserve HTML or markup structure on both sides, and retain domain-specific join keys such as UI placeholders, legal article numbers, Bible verse IDs, or game resource keys in the manifest.
4. **Expand common records:** emit one of the JSONL schemas above and extract only configured language edges.
5. **Run automatic quality gates:** validate UTF-8, JSON, non-empty values, types, length, script, repetition, structured tokens, and duplicates.
6. **Review a semantic sample:** never approve translation equivalence from an automatic score alone.
7. **Build the indexed dataset:** write fingerprints and statistics into a new output directory.
8. **Evaluate fixed sets:** measure a completely separate domain/time benchmark in addition to the source's internal holdout.

When systematic misalignment requires excluding an entire source, do not quietly retain a small subset. Move it to `data/excluded/`, preserve the original data, exclusion reason, and row count, and update both `data/data.txt` and the remediation manifest.

## 6. Automatic quality gates

The current `sion-prepare-data` path provides:

- UTF-8, JSON, mapping-record, and string-type validation
- A minimum of two characters on each side and a maximum character-length ratio of 5.0
- Configuration-driven language and writing-system checks
- Rejection of identical strings, control characters, and excessive repetition
- Warnings for noncritical number, URL, email, and identifier mismatches, plus fail-closed rejection for critical placeable, template, placeholder, `printf`, and entity corruption
- A maximum of 510 tokens on either side after tokenization
- SQLite-backed, bounded-memory exact-pair deduplication
- A leakage guard preventing one target sentence from appearing in different splits
- Deterministic raw-record-level splitting
- Training-only treatment for synthetic records and synthetic-prefix files

Add the following **source-specific rejection conditions**.

| Source type | Required validation |
|---|---|
| Numeric, financial, and medical | Preserve numeric values, signs, decimal points, currency, units, ranges, dates, and times |
| UI and technical | Preserve `{name}`, `%s`, `%1$d`, `${var}`, XML/HTML tags, URLs, commands, and code spans |
| Legal and administrative | Preserve article and paragraph identifiers, institution names, effective dates, document versions, and negation |
| Dialogue | Preserve speaker and turn order, honorific register, and separation between responses and scene metadata |
| Games and localization | Join identical resource keys from the same build or revision and preserve character and proper-name consistency |
| Subtitles and speech | Do not align from time overlap alone; verify sentence merges/splits and speaker transitions |
| Multilingual records | Verify that every edge is actually translation-equivalent; a shared ID is not sufficient evidence |

Latin-script languages cannot be distinguished by script alone, so apply language identification and manual sampling separately. Any warning-only source-specific constraint must be promoted to a hard ingestion rejection when exact preservation is essential.

## 7. Manual semantic-quality sampling

Build a pilot of no more than 100,000 pairs for each new source, then review a human sample with the following minimums.

- At least 1,000 pairs per source
- At least 0.02% of valid pairs for large sources
- Stratified sampling across domain, length decile, quality-score band, and language edge
- Separate oversampling of numbers, placeholders, proper nouns, negation, and honorific cases

Stop full automatic ingestion and repair the aligner or source if any condition below is met.

- Severe misalignment, reversed meaning, or different scenes in at least 0.5% of the sample
- Missing meaning or errors in numbers, units, or negation in at least 1.0%
- Material errors such as language mixing, broken markup, or speaker mismatch in at least 2.0%
- Errors clustered in a particular document, game build, or language edge

After repair, do not reuse the same review sample. Approve a new sample drawn with a new seed. If the error is systematic, exclude the complete file and record the reason in its manifest. A single automatic translation-fluency score cannot replace this process.

## 8. Deduplication and split-leakage prevention

Apply hierarchical deduplication in this order.

1. Use raw-asset hashes to prevent reacquiring the same archive or revision.
2. Apply NFC storage normalization and a compatibility-based deduplication key to remove exact pairs.
3. Remove the same edge with source and target reversed by using a canonical edge identity.
4. Manually review one-to-many candidates sharing only one source side, then retain them or group them by source family.
5. Build template and near-duplicate clusters using character 5-gram MinHash, SimHash, or a comparable method.
6. Assign each cluster or document to exactly one split.
7. Build a denylist from hashes of both sides of evaluation-only examples and apply it before training ingestion.

The indexed preprocessor provides exact-pair deduplication and a target-side split guard. Near-duplicate removal and external benchmark denylists must run earlier during raw ingestion. In particular, do not overcount millions of template rows that differ only in a number or name as independent sentences; apply a per-cluster sampling cap.

Keep multiple translations from one record, adjacent sentences from one document, and turns from one conversation in the same split. Validation cut from a training source measures only in-domain behavior, so use separate fixed benchmarks for public reporting and early-stopping decisions.

## 9. Holdout design

By default, indexed-dataset construction deterministically assigns 0.5% of real data to validation and 0.5% to test. A 100 million-pair corpus does not automatically require 500,000 validation and 500,000 test pairs. Fix the fractions according to final scale and evaluation cost, then keep the hash split unchanged across retraining runs. Never put synthetic data into validation or test.

The minimum evaluation bundle includes:

- Real-data validation and test sets for every domain
- An out-of-source test whose source and documents do not overlap training
- A temporal test that holds out newer documents by date
- A preservation challenge set for numbers, units, dates, IDs, and placeholders
- Long-context, multi-sentence, and conversational-turn challenge sets
- Register sets for polite, casual, written, spoken, and dialectal language
- An independent test for every directed language edge

Keep MKQA, PAWS-X dev/test, WMT24++, NTREX, BOUQuET, and FLORES-200 entries listed in `data/data.txt` excluded from training and reserved for evaluation. Never train on an evaluation set and then report its score as generalization performance.

## 10. Reproducible preprocessing and accounting

Example for a multilingual graph:

```bash
sion-train-tokenizer --input "data/*.jsonl" \
  --output-dir artifacts/tokenizer-v2 \
  --language-pairs ko ja \
  --language-pairs en ru

sion-prepare-data --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer-v2/sion.model \
  --output-dir artifacts/dataset-v2 \
  --language-pairs ko ja \
  --language-pairs en ru \
  --validation-fraction 0.005 \
  --test-fraction 0.005 \
  --dedup-backend sqlite
```

Do not mix new shards into an existing dataset directory. The current code rejects a non-empty output directory. Create a new directory for every remediation and change the configuration path only after approval.

For final accounting, reconcile global `stats.valid_pairs`, `stats.synthetic_pairs`, per-source statistics in `artifacts/dataset-v2/manifest.json`, and the raw remediation manifests. Declare the 100 million target complete only when every equality below holds.

```text
sum(source valid_pairs) = global valid_pairs
real_pairs + synthetic_pairs = 100,000,000
train + validation + test = valid_pairs
excluded/invalid/duplicate/too_long rows are not included in valid_pairs
```

Record the following for every added batch.

- Source name, URL, revision or commit, archive/file SHA-256, and acquisition date
- Physical JSONL row count and pair counts before and after expansion
- Real and synthetic pair counts for every directed language edge
- Counts for `invalid`, `quality_filtered`, `duplicate`, `too_long`, and `split_conflicts`
- Rejection counts for every automatic gate and the reviewed semantic sample and error rate
- Training, validation, and test counts plus benchmark-denylist results
- Final output SHA-256 and exact rebuild command

Do not declare success merely because a file is large or `wc -l` reports 100 million lines. The recorded accounting above is mandatory.
