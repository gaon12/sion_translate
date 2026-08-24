"""Standalone inference CLI for a trained sion_translate model.

Examples:
    python inferences.py --to ja "오늘 날씨가 좋습니다."
    python inferences.py --quality accurate --thinking high --to ko < input.txt
    python inferences.py --quality best --to ja --input input.txt
    python inferences.py --int8 --to ja --input input.txt
    python inferences.py --quality best --batch-size 1 --profile --to ja --input input.txt

``--int8`` reduces file size and memory use; it is not a speed option.
Use ``--quality`` to control the quality/speed tradeoff.

``thinking`` does not expose a hidden reasoning process. This is a
translation-only seq2seq model, so the option specifies how much computation
to allocate to wider beam search.

Output contains only the final translation.

Generation-option precedence:
    --num-beams > --thinking > --quality

For example, ``--thinking high`` in the following command overrides the default
beam count from ``--quality best``:

    python inferences.py --quality best --thinking high ...

Omit ``--thinking`` to retain the recommended beam count of 4 from the ``best``
preset:

    python inferences.py --quality best ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

# Add src to the module search path so this file can run from the project root
# without an editable installation.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from sion_translate.config import config_from_raw, load_raw_config  # noqa: E402
from sion_translate.console import configure_stdio  # noqa: E402
from sion_translate.glossary import Glossary, load_glossary  # noqa: E402
from sion_translate.inference import Translator, find_exported_model  # noqa: E402


# Quality presets.
#
# These values follow the model's holdout evaluation and the recommended core
# generate() settings. Excessive beam counts can reduce quality and increase
# repetition, so ``best`` uses the validated count of 4. Request wider search
# explicitly with ``--thinking`` or ``--num-beams``.
QUALITY_DEFAULTS = {
    "fast": {
        "num_beams": 1,
        "batch_size": 32,
        "length_penalty": 1.0,
    },
    "balanced": {
        "num_beams": 2,
        "batch_size": 16,
        "length_penalty": 1.0,
    },
    "accurate": {
        "num_beams": 4,
        "batch_size": 8,
        "length_penalty": 1.0,
    },
    "best": {
        "num_beams": 4,
        "batch_size": 8,
        "length_penalty": 1.0,
    },
}


# The thinking option controls the beam-search budget; it does not expose an
# internal reasoning process.
THINKING_BEAMS = {
    "off": 1,
    "low": 2,
    "medium": 4,
    "high": 8,
    "max": 16,
}


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate between Korean and Japanese with a trained sion_translate model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "text",
        nargs="*",
        help="Text to translate. If omitted, use --input or standard input.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Input text file. Each line is treated as one sentence.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file. Results are written to standard output when omitted.",
    )

    parser.add_argument(
        "--to",
        dest="target",
        help="Target language. Defaults to the second language in the configured pair.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="Path to model.pt, model_ema.pt, or model_int8.pt.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "sion_translate.yaml",
        help="Path to the sion_translate configuration file.",
    )

    parser.add_argument(
        "--int8",
        action="store_true",
        help=(
            "Use the CPU-only INT8 export. It reduces file size and memory use "
            "but does not make translation faster (it is independent of "
            "--quality fast)."
        ),
    )

    parser.add_argument(
        "--quality",
        choices=tuple(QUALITY_DEFAULTS),
        default="balanced",
        help=("Speed/quality preset. best uses beam 4, which was validated in holdout evaluation."),
    )

    parser.add_argument(
        "--thinking",
        choices=tuple(THINKING_BEAMS),
        help=(
            "Search budget. "
            "off=greedy, low=beam 2, medium=beam 4, "
            "high=beam 8, and max=beam 16. "
            "Only the final translation is shown."
        ),
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        help="Beam count. This overrides --quality and --thinking when provided.",
    )

    parser.add_argument(
        "--length-penalty",
        type=float,
        help="Length penalty used by beam search.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per sentence.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="Forbid repeated n-grams of this size. Set to 0 to disable.",
    )
    parser.add_argument(
        "--max-output-length-ratio",
        type=float,
        default=3.0,
        help=(
            "Maximum output-to-source token ratio. An additional allowance of "
            "16 tokens is applied separately."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size. Lower this value if inference runs out of memory.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device, such as auto, cuda, cuda:0, or cpu.",
    )

    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="Compute precision for regular exports. Ignored for INT8 exports.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile on CUDA. The first inference may be slow.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="Number of PyTorch threads to use for CPU inference.",
    )

    parser.add_argument(
        "--glossary",
        type=Path,
        help="Path to a glossary JSON file.",
    )

    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="Disable glossaries from both the command line and configuration file.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Write one JSONL object per translation.",
    )

    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write inference time, total time, and throughput to stderr. Disable with --no-timing."
        ),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Write a detailed breakdown of configuration, model preparation, "
            "inference, and output time to stderr."
        ),
    )

    parser.add_argument(
        "--degeneration-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retry only repetitive or excessively long outputs with narrower "
            "beams. Disable with --no-degeneration-retry."
        ),
    )

    return parser


def to_python_string(value: object, *, value_name: str) -> str:
    """Convert a string-like value to a built-in Python ``str``.

    argparse and ordinary text files normally provide built-in strings, but
    external callers can pass ``numpy.str_``, pandas string scalars, or bytes.

    Structured data such as lists and dictionaries is intentionally rejected.
    Coercing those values with ``str`` could conceal a data error.
    """
    if isinstance(value, str):
        # Normalize possible string subclasses to the built-in str type.
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SystemExit(f"{value_name} contains bytes that are not valid UTF-8.") from error

    # numpy and pandas scalar objects can expose a built-in Python scalar via
    # item().
    item_method = getattr(value, "item", None)

    if callable(item_method):
        try:
            scalar_value = item_method()
        except (TypeError, ValueError):
            scalar_value = None

        if isinstance(scalar_value, str):
            return str(scalar_value)

        if isinstance(scalar_value, bytes):
            try:
                return scalar_value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise SystemExit(
                    f"The scalar value for {value_name} contains bytes that are not valid UTF-8."
                ) from error

    raise SystemExit(
        f"{value_name} must be a string; got type={type(value).__name__}, value={value!r}"
    )


def read_lines(args: argparse.Namespace) -> list[str]:
    """Read sentences from arguments, an input file, or standard input."""
    if args.text and args.input:
        raise SystemExit("Positional text and --input cannot be used together.")

    raw_lines: Sequence[object]

    if args.text:
        raw_lines = args.text

    elif args.input:
        try:
            raw_lines = args.input.read_text(
                encoding="utf-8",
            ).splitlines()
        except (OSError, UnicodeError) as error:
            raise SystemExit(f"Could not read input file {args.input}: {error}") from error

    else:
        raw_lines = [line.rstrip("\r\n") for line in sys.stdin]

    lines: list[str] = []

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = to_python_string(
            raw_line,
            value_name=f"input line {line_number}",
        ).strip()

        # Exclude blank lines from translation.
        if line:
            lines.append(line)

    if not lines:
        raise SystemExit("No text was provided for translation.")

    return lines


def choose_device(value: str, int8: bool) -> torch.device:
    """Select the inference device from the command-line options."""
    if int8:
        if value not in ("auto", "cpu"):
            print(
                "[sion] INT8 exports are CPU-only; using device=cpu.",
                file=sys.stderr,
            )

        return torch.device("cpu")

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Invalid --device value: {value}") from error

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but no CUDA GPU is available.")

    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise SystemExit(
                f"CUDA device {device.index} was requested, but only "
                f"{torch.cuda.device_count()} GPUs are available."
            )

    return device


def synchronize_device(device: torch.device) -> None:
    """Wait for asynchronous CUDA work to finish for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_config_path(
    value: str | Path,
    *,
    config_path: Path | None,
) -> Path:
    """Resolve a configured relative path from the configuration directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    base = config_path.parent if config_path is not None else ROOT
    return base / path


def require_file(path: Path, *, value_name: str) -> Path:
    """Require an input path to exist as a regular file."""
    if not path.exists():
        raise SystemExit(f"Could not find {value_name}: {path}")
    if not path.is_file():
        raise SystemExit(f"{value_name} must be a file: {path}")
    return path


def apply_runtime_options(
    translator: Translator,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Apply device, precision, and torch.compile settings to the model.

    INT8 modules are CPU-only quantized models, so their device and dtype remain
    unchanged.
    """
    if args.threads is not None:
        if args.threads < 1:
            raise SystemExit("--threads must be at least 1.")

        torch.set_num_threads(args.threads)

    if translator.quantized:
        return

    if args.dtype == "fp16" and device.type != "cuda":
        raise SystemExit("fp16 requires CUDA. Use fp32 or bf16 on CPU.")

    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(args.dtype)

    if dtype is not None:
        translator.model.to(
            device=device,
            dtype=dtype,
        )
    else:
        translator.model.to(device=device)

    translator.device = device

    if args.compile:
        if device.type != "cuda":
            print(
                "[sion] --compile is applied only on CUDA.",
                file=sys.stderr,
            )

        elif not hasattr(torch, "compile"):
            print(
                "[sion] This PyTorch build has no torch.compile; skipping compilation.",
                file=sys.stderr,
            )

        else:
            translator.model = torch.compile(
                translator.model,
                mode="reduce-overhead",
            )


def generation_options(
    args: argparse.Namespace,
) -> tuple[int, int, float]:
    """Combine the quality preset and CLI overrides into generation options."""
    preset = QUALITY_DEFAULTS[args.quality]

    # An explicit --num-beams value has the highest precedence.
    beams = args.num_beams

    if beams is None:
        # An explicit --thinking value overrides the quality preset's beam count.
        if args.thinking is not None:
            beams = THINKING_BEAMS[args.thinking]
        else:
            beams = int(preset["num_beams"])

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = int(preset["batch_size"])

    if args.length_penalty is not None:
        length_penalty = args.length_penalty
    else:
        length_penalty = float(preset["length_penalty"])

    if beams < 1:
        raise SystemExit("--num-beams must be at least 1.")

    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")

    if length_penalty <= 0:
        raise SystemExit("--length-penalty must be greater than 0.")

    return beams, batch_size, length_penalty


def validate_translations(
    sources: Sequence[str],
    translations: Sequence[object],
) -> list[str]:
    """Validate the number and type of translations returned by the model."""
    translation_list = list(translations)

    if len(translation_list) != len(sources):
        raise SystemExit(
            "The number of translations does not match the number of input "
            f"sentences: input={len(sources)}, output={len(translation_list)}"
        )

    validated: list[str] = []

    for index, translation in enumerate(
        translation_list,
        start=1,
    ):
        validated.append(
            to_python_string(
                translation,
                value_name=f"translation {index}",
            )
        )

    return validated


def degeneration_reasons(source: str, translation: str) -> set[str]:
    """Return signals of obvious generation degeneration.

    To avoid flagging short interjections or intentional repetition, this check
    targets only runs of at least five characters, at least four repeated
    phrases, and outputs that are abnormally long relative to the source.
    """
    stripped = translation.strip()
    if not stripped:
        return {"empty"}

    reasons: set[str] = set()
    compact = re.sub(r"\s+", "", stripped)

    if re.search(r"(.)\1{4,}", compact):
        reasons.add("character_repetition")

    if re.search(r"(.{2,12})(?:[\s,.!?~·ㆍ-]*\1){3,}", stripped):
        reasons.add("phrase_repetition")

    length_limit = max(48, len(source.strip()) * 3 + 10)
    if len(stripped) > length_limit:
        reasons.add("excessive_length")

    return reasons


def retry_degenerate_translations(
    *,
    translator: Translator,
    sources: Sequence[str],
    translations: Sequence[str],
    target: str,
    beams: int,
    length_penalty: float,
    max_new_tokens: int,
    batch_size: int,
    glossary: Glossary | None,
    no_repeat_ngram_size: int = 4,
    max_output_length_ratio: float = 3.0,
) -> tuple[list[str], int, int]:
    """Replace only degenerate results with narrower-beam candidates.

    A candidate is accepted only when it reduces the number of degeneration
    reasons, leaving normal translations unchanged. For beam 4, the remaining
    problem sentences are retried with beam 2 and then greedy search.
    """
    resolved = list(translations)
    initial_problem_indices = [
        index
        for index, (source, translation) in enumerate(zip(sources, resolved, strict=True))
        if degeneration_reasons(source, translation)
    ]
    pending = initial_problem_indices

    retry_beams = sorted({1, max(1, beams // 2)}, reverse=True)
    retry_beams = [candidate for candidate in retry_beams if candidate < beams]

    for retry_beam in retry_beams:
        if not pending:
            break

        retry_sources = [sources[index] for index in pending]
        retry_raw = translator.translate(
            retry_sources,
            target_language=target,
            num_beams=retry_beam,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            batch_size=min(batch_size, len(retry_sources)),
            glossary=glossary,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_output_length_ratio=max_output_length_ratio,
        )
        retry_translations = validate_translations(retry_sources, retry_raw)

        for index, candidate in zip(pending, retry_translations, strict=True):
            previous_reasons = degeneration_reasons(sources[index], resolved[index])
            candidate_reasons = degeneration_reasons(sources[index], candidate)
            if len(candidate_reasons) < len(previous_reasons):
                resolved[index] = candidate

        pending = [
            index for index in pending if degeneration_reasons(sources[index], resolved[index])
        ]

    rescued_count = len(initial_problem_indices) - len(pending)
    return resolved, rescued_count, len(pending)


def render_rows(
    sources: Sequence[str],
    translations: Sequence[str],
    as_json: bool,
) -> Iterator[str]:
    """Render translations as plain-text or JSONL rows."""
    for source, translation in zip(
        sources,
        translations,
        strict=True,
    ):
        if as_json:
            yield json.dumps(
                {
                    "source": source,
                    "translation": translation,
                },
                ensure_ascii=False,
            )
        else:
            yield translation


def write_output(
    rows: str,
    output_path: Path | None,
) -> None:
    """Write results to a file or standard output."""
    if output_path is None:
        sys.stdout.write(rows)
        sys.stdout.flush()
        return

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            rows,
            encoding="utf-8",
        )

    except OSError as error:
        raise SystemExit(f"Could not write output file {output_path}: {error}") from error


def print_timing_report(
    *,
    sentence_count: int,
    config_elapsed: float,
    model_elapsed: float,
    inference_elapsed: float,
    output_elapsed: float,
    total_elapsed: float,
    detailed: bool,
) -> None:
    """Write timing measurements to stderr."""
    safe_inference_elapsed = max(
        inference_elapsed,
        1e-9,
    )

    throughput = sentence_count / safe_inference_elapsed
    average_milliseconds = safe_inference_elapsed / max(sentence_count, 1) * 1000.0

    print(
        (
            f"[sion] Inference complete: "
            f"{sentence_count} sentences / "
            f"inference {inference_elapsed:.3f}s / "
            f"total {total_elapsed:.3f}s / "
            f"{throughput:.2f} sentences/s / "
            f"average {average_milliseconds:.2f} ms/sentence"
        ),
        file=sys.stderr,
        flush=True,
    )

    if detailed:
        print(
            "[sion] Detailed timing:",
            file=sys.stderr,
        )

        print(
            f"[sion]   Configuration and input: {config_elapsed:.3f}s",
            file=sys.stderr,
        )

        print(
            f"[sion]   Model loading and preparation: {model_elapsed:.3f}s",
            file=sys.stderr,
        )

        print(
            f"[sion]   Translation inference: {inference_elapsed:.3f}s",
            file=sys.stderr,
        )

        print(
            f"[sion]   Result rendering and output: {output_elapsed:.3f}s",
            file=sys.stderr,
        )

        print(
            f"[sion]   Total execution: {total_elapsed:.3f}s",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    """Run the complete CLI inference workflow."""
    configure_stdio()
    total_started = time.perf_counter()

    args = build_parser().parse_args()

    # Prepare the configuration, input sentences, device, and generation options.
    config_started = time.perf_counter()

    sources = read_lines(args)

    default_config_path = ROOT / "sion_translate.yaml"
    config_path = args.config if args.config.exists() else None

    # Fail immediately when an explicitly selected non-default config is missing.
    if args.config != default_config_path and config_path is None:
        raise SystemExit(f"Could not find configuration file: {args.config}")

    raw_config = load_raw_config(config_path) if config_path is not None else {}

    config = config_from_raw(raw_config)

    device = choose_device(
        args.device,
        args.int8,
    )

    beams, batch_size, length_penalty = generation_options(args)

    if args.model is not None:
        model_path = require_file(
            args.model,
            value_name="model",
        )
    else:
        output_dir = resolve_config_path(
            config.training.output_dir,
            config_path=config_path,
        )
        try:
            model_path = find_exported_model(
                output_dir,
                int8=args.int8,
            )
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error

    tokenizer_path = require_file(
        resolve_config_path(
            config.data.tokenizer_model,
            config_path=config_path,
        ),
        value_name="tokenizer model",
    )

    if args.no_glossary:
        glossary_path = None
    elif args.glossary is not None:
        glossary_path = args.glossary
    else:
        # An empty DataConfig.glossary string disables the glossary. Filter it
        # before constructing a Path because Path("") resolves to ".".
        configured_glossary = to_python_string(
            config.data.glossary,
            value_name="configuration data.glossary",
        ).strip()
        glossary_path = (
            resolve_config_path(
                configured_glossary,
                config_path=config_path,
            )
            if configured_glossary
            else None
        )

    if glossary_path is None:
        glossary = None
    else:
        glossary_path = require_file(
            glossary_path,
            value_name="glossary",
        )
        try:
            glossary = load_glossary(glossary_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"Could not read glossary {glossary_path}: {error}") from error

    config_elapsed = time.perf_counter() - config_started

    print(
        (
            f"[sion] model={model_path} "
            f"device={device} "
            f"quality={args.quality} "
            f"beams={beams} "
            f"batch_size={batch_size} "
            f"length_penalty={length_penalty}"
        ),
        file=sys.stderr,
        flush=True,
    )

    # Load the model and tokenizer, then apply runtime options.
    model_started = time.perf_counter()

    translator = Translator(
        model_path,
        tokenizer_path,
        device=device,
    )

    apply_runtime_options(
        translator,
        args,
        device,
    )

    target = (
        to_python_string(
            args.target,
            value_name="--to",
        ).strip()
        if args.target is not None
        else to_python_string(
            config.data.language_pair[1],
            value_name="configured target language",
        ).strip()
    )

    if target not in translator.languages:
        supported_languages = ", ".join(sorted(translator.languages))

        raise SystemExit(
            f"--to {target} is unsupported. Supported languages: {supported_languages}"
        )

    synchronize_device(device)
    model_elapsed = time.perf_counter() - model_started

    # Measure the translation inference itself.
    #
    # CUDA work is asynchronous, so synchronization is required before and after
    # inference to measure the true completion time.
    synchronize_device(device)
    inference_started = time.perf_counter()

    try:
        raw_translations = translator.translate(
            sources,
            target_language=target,
            num_beams=beams,
            length_penalty=length_penalty,
            max_new_tokens=args.max_new_tokens,
            batch_size=batch_size,
            glossary=glossary,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_output_length_ratio=args.max_output_length_ratio,
        )

    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()

        raise SystemExit(
            "CUDA ran out of memory. Lower --batch-size. For best mode, "
            "--batch-size 1 is recommended."
        ) from error

    except TypeError as error:
        error_message = str(error).lower()

        if "not a string" in error_message:
            raise SystemExit(
                "SentencePiece received a non-string value. Convert both the "
                "input and the normalize_text() result to a built-in Python str "
                "inside src/sion_translate/tokenizer.py encode()."
            ) from error

        raise

    translations = validate_translations(
        sources,
        raw_translations,
    )

    if args.degeneration_retry and beams > 1:
        translations, rescued_count, remaining_count = retry_degenerate_translations(
            translator=translator,
            sources=sources,
            translations=translations,
            target=target,
            beams=beams,
            length_penalty=length_penalty,
            max_new_tokens=args.max_new_tokens,
            batch_size=batch_size,
            glossary=glossary,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_output_length_ratio=args.max_output_length_ratio,
        )
        if rescued_count or remaining_count:
            print(
                f"[sion] Degeneration retry: rescued {rescued_count} sentences; "
                f"{remaining_count} remain.",
                file=sys.stderr,
                flush=True,
            )

    synchronize_device(device)
    inference_elapsed = time.perf_counter() - inference_started

    # Validate returned results and render them in the requested output format.
    output_started = time.perf_counter()

    rows = (
        "\n".join(
            render_rows(
                sources,
                translations,
                args.json,
            )
        )
        + "\n"
    )

    write_output(
        rows,
        args.output,
    )

    output_elapsed = time.perf_counter() - output_started
    total_elapsed = time.perf_counter() - total_started

    # --profile emits detailed timings even when --timing is disabled.
    if args.timing or args.profile:
        print_timing_report(
            sentence_count=len(sources),
            config_elapsed=config_elapsed,
            model_elapsed=model_elapsed,
            inference_elapsed=inference_elapsed,
            output_elapsed=output_elapsed,
            total_elapsed=total_elapsed,
            detailed=args.profile,
        )


if __name__ == "__main__":
    main()
