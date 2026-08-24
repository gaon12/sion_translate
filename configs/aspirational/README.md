# Why these presets are outside `configs/`

The configurations in this directory are syntactically valid but must not be trained on
the currently measured corpus. The available token budget does not support their model
capacity.

## Historical measurement

The recorded audit covered 51 top-level JSONL shards and 8,978,338 parallel records. With
the tokenizer used by that audit, one complete pass contained approximately 0.357 billion
source-plus-target tokens.

| Preset | Parameters | Tokens/parameter per pass | Chinchilla reference tokens | Shortfall |
|---|---:|---:|---:|---:|
| `configs/sion_data_fit.yaml` | 200,486,400 | 1.78 | 4.0 B | 11x |
| `configs/sion_1_3b.yaml` | 1,206,340,608 | 0.296 | 24.1 B | 68x |
| `sion_8b.yaml` | 8,216,715,264 | 0.043 | 164.3 B | 460x |

Chinchilla scaling was derived for decoder-only language models, not translation
encoder-decoders, so the table is a reference rather than a universal law. The 8B ratio
is nevertheless too small to justify a runnable production preset.

A former 32B file was removed. Its roughly 31.9 billion parameters produced a historical
shortfall near 1,790x. A configuration that far beyond the data is misleading even if a
future capacity gate could technically allocate it. The parameter-based preflight remains
capable of checking custom large configurations without keeping an attractive but unsafe
preset in the runnable list.

## What to use

Use `configs/sion_data_fit.yaml` as the measured baseline. Treat
`configs/sion_1_3b.yaml` as a later capacity step only after expanding and auditing the
corpus.

Before using any preset here:

1. Acquire enough licensed, high-quality data for every required direction and domain.
2. Recalculate effective source and target tokens after filtering and deduplication.
3. Verify direction balance and worst-direction holdouts.
4. Run the automatic capacity and VRAM preflight.
5. Start with a smaller controlled baseline rather than assuming a larger model improves
   quality.

See [`../../docs/corpus-gaps.md`](../../docs/corpus-gaps.md) and
[`../../docs/DATA_EXPANSION_PLAN.md`](../../docs/DATA_EXPANSION_PLAN.md) for data planning.

## Architecture warning

`sion_8b.yaml` uses the historical 30-encoder/30-decoder layout. It conflicts with the
project's deep-encoder, shallow-decoder design: autoregressive decoder layers are paid at
every generated token and are often weight-bandwidth bound. The 1.3B preset was adjusted
to 30 encoder and 12 decoder layers while preserving its general capacity class.

If the 8B configuration ever becomes data-supported, redesign and benchmark its layer
allocation before treating it as runnable.
