# SentencePiece 0.2.2 SIGSEGV investigation

## Conclusion

The crash was not caused by corpus size or a particular character. It was a
multi-threaded trainer-normalization regression in SentencePiece 0.2.2. Upstream commit
[`de32a1e`](https://github.com/google/sentencepiece/commit/de32a1eb2e8ae1d63380e586685db4863b84a9a2)
stopped building the byte-offset vector in the trainer's
`Normalizer::Normalize(string_view)` path.

This project pins `sentencepiece==0.2.1`. `train_tokenizer()` refuses SentencePiece 0.2.2
with more than one trainer thread before scanning the corpus. If 0.2.2 is unavoidable,
`num_threads=1` is a measured but slower workaround.

## Reproduction and eliminated hypotheses

The preserved failure input is `corpus_balanced_short.txt`. It contains 20,355,467
physical lines. SentencePiece skips 12 rows containing reserved characters, leaving
20,355,455 trainer sentences, 2,774,987,960 UTF-8 bytes, and 1,057,602,757 Unicode scalar
values.

| Condition | Result | Runtime |
|---|---:|---:|
| 0.2.2 wheel, four threads, file input, char model | SIGSEGV (`-11`) | 74.12 s |
| 0.2.2 wheel, one thread, same input | Passed | 259.66 s |
| 0.2.1 wheel, four threads, same input | Passed | 138.40 s |
| 0.2.2 C++ CLI, four threads | SIGSEGV | Reproduced |
| 0.2.2 with only the new pool replaced by direct `std::thread` | SIGSEGV (`-11`) | 78.16 s |
| 0.2.2 with only normalization offsets restored | Passed | 117.27 s |

The last two rows are one-change comparisons with the remaining 0.2.2 source held fixed.
They exclude the contemporaneous thread-pool change
[`c5ed56a`](https://github.com/google/sentencepiece/commit/c5ed56a5501676f38804662893ec345b7fa1570b)
and identify the offset-removal change as necessary for the failure.

A RelWithDebInfo build of the 0.2.2 `spm_train` binary produced this native stack:

```text
glibc sysmalloc
std::string::_M_append
sentencepiece::normalizer::PrefixMatcher::GlobalReplace (normalizer.cc:384)
TrainerInterface::LoadSentences normalization worker
sentencepiece::ThreadPool::Impl::WorkLoop
```

The process failed before `Done! preprocessed` and `Making suffix array`, so it never
reached seed-piece or suffix-array construction. The C++ CLI reproduced the failure
without Python bindings, which also excluded `sentence_iterator` as the cause.

Measurements excluded these additional hypotheses:

- A smaller failing input (1.08 billion scalar values) existed beside a larger passing
  input (1.97 billion), excluding a simple size, sentence, character, or 2^31 boundary.
- The process failed in the same location with 256 GiB available, while a passing run
  peaked below 49.8 GiB, excluding ordinary OOM.
- The failure input is strict UTF-8. Thirty-six individual and combined runs containing
  the eight C0/C1 rows all passed.
- Removing every row longer than 4 KiB produced the exact same trainer stream, excluding
  a long-row or long-token cause.
- Sequential and 16-thread normalization of the full corpus with an existing model
  passed. The failure was specific to the 0.2.2 trainer's offset-free parallel path.

### Minimization boundary

An ordered-prefix binary search used `sentence_iterator`. A prefix of 20,350,000 physical
rows, or 20,349,988 trainer sentences after exclusions, passed. A prefix of 20,350,172
rows, or 20,350,160 trainer sentences, failed twice with `-11`.

The immediately preceding 20,350,171-row prefix (2,790,882,460 bytes and 20,350,159
trainer sentences) passed once and then failed on repetition. There is therefore no
deterministic row boundary. Scheduling and allocator layout influence the native memory
bug near the threshold.

Row 20,350,172 is an ordinary 302-character Korean sentence containing only Unicode
categories `Lo`, `Po`, and `Zs`; its SHA-256 is
`2eab3a8f81b7e209681db5bcc63b1effa6f4052f78d817f2dadf90b61087af35`. Training on that
row alone passed in 0.21 seconds under 0.2.2 with four threads. It is not an independently
bad row.

The stable preserved reproducer is the full `corpus_balanced_short.txt`, which failed in
eight consecutive runs. The one-change offset restoration remains the decisive causal
comparison.

## Restoring monolingual exposure

The failure corpus contains 11,997,047 parallel sentences, 4,251,477 Japanese monolingual
sentences, and 4,251,476 Korean monolingual sentences. Its monolingual-to-parallel ratio is
0.709, a stronger stress condition than the roughly 3.62 million sentences per language
requested by the later 0.40 policy.

A real 48,000-piece unigram model trained successfully on this corpus with
SentencePiece 0.2.1 and 16 threads in 1,308.99 seconds.

`easy_run._verify_tokenizer(model, Path("data"))` observed 670 byte-fallback tokens among
1,312,459 sampled tokens, or 0.0510%, below the 0.2% limit. `넼` became one piece after the
dummy prefix, and `38,720` split as `3 8 , 7 2 0`. The model SHA-256 was
`6149c905411a38ca900ee7d45fd29594ea895a01d140c950ddc1fb1854735454`.

Deterministic foundation samples compared the older 0.13 model with the higher-monolingual
candidate:

| Language | Sentences | Model | Byte fallback | Pieces/character |
|---|---:|---|---:|---:|
| Korean | 18,500 | older 0.13 | 0.0255% | 0.475843 |
| Korean | 18,500 | higher-monolingual candidate | 0.0267% | **0.453025** |
| Japanese | 19,166 | older 0.13 | 0.0038% | 0.539918 |
| Japanese | 19,166 | higher-monolingual candidate | 0.0039% | **0.526961** |

Fallback remained essentially unchanged, while piece count fell by 4.79% for Korean and
2.40% for Japanese. This stress candidate remained in the reproduction workspace and did
not replace the published artifact.

### Production 0.40 verification on 2026-08-08

The production path `scripts/modal_train_tokenizer.py` trained a SentencePiece 0.2.1
unigram model with 16 preprocessing workers, 16 trainer threads, and a 48,000-piece
vocabulary. Its run ID was
`ratio-040-fe9a4799de05-6b2fd43b3111-20260808t081122z-d0d1ac`.

- Parallel: 18,177,344 sentences
- Monolingual: Japanese 3,445,471; Korean 3,308,940
- Total: 24,931,755 sentences in both the character-plan pass and final trainer pass
- Required characters: 9,751, SHA-256
  `bc1e656d90cd109cb8631583c588aeb9b51125dafa31744c9016eba392ed1e86`
- Model SHA-256:
  `082695f2d42314061fe3c5431816ef501cb2257d6af6c334f726816aea1bdc98`

`easy_run._verify_tokenizer()` observed 736 byte-fallback tokens among 1,304,691 sampled
tokens, or 0.0564%, below the 0.2% limit. `넼` remained one content piece and `38,720`
remained digit-split. The parallel indexed dataset was rebuilt with the same model, and
its manifest and freshness fingerprint recorded that model digest. Exact source and
artifact hashes and both pass counts live beside the tokenizer in
`training_manifest.json`.

## Reproduce the diagnostic safely

Use a separate virtual environment so the project pin is not changed. The corpus and
`tokenizer_plan.json` are too large for Git and live under the ignored local path
`artifacts/sentencepiece_repro/`.

```powershell
py -3.11 -m venv C:\tmp\spm-022
C:\tmp\spm-022\Scripts\python.exe -m pip install sentencepiece==0.2.2
C:\tmp\spm-022\Scripts\python.exe scripts\diagnose_sentencepiece_crash.py `
  --corpus artifacts\sentencepiece_repro\corpus_balanced_short.txt `
  --plan artifacts\sentencepiece_repro\tokenizer_plan.json `
  --model-type char --threads 4
```

Preserved input digests:

- corpus:
  `855cdc67378272490691279deb2ce5d4160dc56d4d4e4373e889d683f98ad12a`
- plan:
  `ef286833960dd6ac5cac22852b8a9fdcc0a0762fd14a97edd71fb1785a7858f6`

The parent process launches the trainer as a child, so it can record return code and
runtime in JSON after a native crash. `--maximum-sentences N` tests an ordered prefix via
`sentence_iterator`. Because file and iterator inputs enter the same native normalization
path, perform the final confirmation without that option using direct file input.

## Operational consequence

The former `foundation.tokenizer_sample_ratio: 0.13` was merely a corpus combination that
happened not to trigger the regression; it was not a safe size limit. Pinning 0.2.1 allows
the ratio to return to 0.40.

The ratio applies to both monolingual trainer sentences and the monolingual sentences used
for required-character frequency analysis. It does not count every character outside the
sample, so older claims that all content characters were preserved were incorrect.

Tokenizer sidecars record the SentencePiece version, actual per-language sample counts,
required-character count, and digest. Metadata remains schema version 2 to preserve
compatibility with existing v1/v2 digit-splitting policy readers.
