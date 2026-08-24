# Translation System Comparison Guide

This document explains how to build a reproducible comparison using the same inputs and evaluation criteria. It does not assume that one system is always better than every other system. Service output can change with the request date, product version, subscription plan, and enabled options, so record the generation date and complete settings alongside each result JSONL file.

## What to examine for each system

| System | Strengths to examine | Important cautions |
|---|---|---|
| sion_translate | Specialized Korean↔Japanese translation, fully local execution, control over code and generation settings, and slot-based terminology enforcement | A small specialized model has limited generalization, requires a custom PyTorch loader, and public scores alone cannot establish production quality |
| [LibreTranslate](https://docs.libretranslate.com/) | Open-source API, self-hosting, and operation without sending text to an external provider | Language coverage depends on the installed models. Translation usually pivots through English when no direct pair exists, so record the `/languages` response |
| [Papago](https://api.ncloud-docs.com/docs/en/ai-naver-papagowebsitetranslation-translation) | The official API explicitly supports both Korean-to-Japanese and Japanese-to-Korean translation | It is a closed cloud service, so pinning a model version may be impossible. Authentication, pricing, and terms of use also require review |
| [Google Cloud Translation](https://cloud.google.com/translate/docs) | Broad language coverage and options such as glossaries and adaptive translation | Cloud credentials and usage fees are required. Mixing Basic, Advanced, or adaptive settings would make the comparison unfair |
| [DeepL](https://developers.deepl.com/docs) | Translation controls such as `context`, glossaries, and language-specific style rules | Feature availability varies by language and API version. Authentication and usage fees are required |
| [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | Direct translation among 100 languages, reproducible local execution, and a pinnable model revision | The 418M checkpoint has memory and latency costs, is an older general-purpose checkpoint, and requires a separate review of upstream terms |
| [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Very broad language coverage, reproducible local execution, and usefulness as a low-resource research baseline | CC-BY-NC 4.0 licensing, an explicit research/non-production limitation in the model card, and possible quality degradation beyond 512 tokens |

The strengths in this table are deployment and control characteristics, not claims that a system wins on translation quality. Measure actual quality from the same JSONL inputs and outputs.

## Two diagnostic sets

| File | Sentences | Purpose |
|---|---:|---|
| `examples/comparison_cases.jsonl` | 16 | Language-focused cases covering honorifics, homonyms, numbers, technical strings, colloquial speech, long-range dependencies, proper nouns, and idioms |
| `examples/diagnostic_cases.jsonl` | 40 | Domain-focused cases that add medical, legal, administrative, tourism, academic, and negation examples, with additional proper-noun and number cases |

Both sets contain synthetic sentences written specifically for this project and are not included in any training corpus. The 40-sentence set was designed to expose degradation in domains absent from training data. Compare it with in-domain holdout scores to estimate the generalization gap.

Both files use the same schema, so switch between them with `--cases`.

```bash
sion-translate-cases --backend sion \
  --cases examples/diagnostic_cases.jsonl \
  --model runs/auto/posttrain/exports/best/model_ema.pt \
  --tokenizer artifacts/tokenizer/sion.model \
  --output comparison_outputs/sion-diagnostic.jsonl
```

Forty sentences are still too few to establish a ranking. Each category contains only two to four sentences, so use the set to identify what fails in each category instead of relying on the aggregate score difference between systems.

## Rules for a fair run

1. If `examples/comparison_cases.jsonl` changes, rerun every system.
2. Specify the source and target languages and disable automatic language detection.
3. Translate sentence by sentence, and do not give context or a glossary to only one system.
4. Record the beam count, model revision, API product name, and request date.
5. Do not delete failed sentences. Record the error and retry under the same conditions.
6. Never write API keys, response headers, or account information into the JSONL output.

## Human-review criteria

- Missing or added meaning, including reversed negation
- Honorific level and the relationship between speakers
- Contextual resolution of homonyms
- Preservation of numbers, currency, dates, filenames, and HTTP status codes
- Transliteration and document-wide consistency of proper nouns
- Japanese and Korean particles, word order, and naturalness
- Preservation of subjects, conditional clauses, and causal relationships in long inputs

chrF or BLEU can score a valid translation poorly when it differs from the reference wording. Conversely, a superficially similar output can contain incorrect numbers or negation. Always review the sentence-level table together with aggregate metrics.

## Licensing and data boundaries

The comparison code is MIT-licensed, but each service, model, input sentence, and generated output has separate terms. Confirm redistribution rights before committing any third-party benchmark or API output publicly. This repository ignores `benchmarks/`, `comparison_outputs/`, and `reports/` by default.
