"""학습한 sion_translate 모델을 위한 독립 실행형 추론 CLI.

사용 예시:
    python inferences.py --to ja "오늘 날씨가 좋습니다."
    python inferences.py --quality accurate --thinking high --to ko < input.txt
    python inferences.py --quality best --to ja --input input.txt
    python inferences.py --int8 --to ja --input input.txt
    python inferences.py --quality best --batch-size 1 --profile --to ja --input input.txt

``--int8``은 용량과 메모리를 줄이는 옵션이며 속도 옵션이 아닙니다.
품질/속도는 ``--quality``로만 조절하십시오.

``thinking``은 번역 모델의 숨은 사고 과정을 출력하는 기능이 아닙니다.
이 모델은 번역 전용 seq2seq 모델이므로, 여기서는 더 넓은 beam 탐색에
할당할 계산량을 뜻합니다.

출력에는 최종 번역만 포함됩니다.

품질 우선순위:
    --num-beams > --thinking > --quality

예를 들어 다음 명령에서 --thinking high가 지정되었으므로
--quality best의 기본 beam 수보다 --thinking high의 beam 수가 우선됩니다.

    python inferences.py --quality best --thinking high ...

best 프리셋의 권장 beam 4를 그대로 사용하려면 --thinking을 생략하십시오.

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

# editable install을 하지 않은 상태에서도 프로젝트 루트에서
# 바로 실행할 수 있도록 src 디렉터리를 모듈 검색 경로에 추가한다.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from sion_translate.config import config_from_raw, load_raw_config  # noqa: E402
from sion_translate.console import configure_stdio  # noqa: E402
from sion_translate.glossary import Glossary, load_glossary  # noqa: E402
from sion_translate.inference import Translator, find_exported_model  # noqa: E402


# 품질 프리셋이다.
#
# 이 모델의 holdout 평가와 코어 generate() 권장값에 맞춘 프리셋이다.
# beam 수를 지나치게 늘리면 품질이 오히려 떨어지고 반복 생성이 늘 수 있으므로,
# best는 검증된 beam 4를 사용한다. 더 넓은 탐색 실험은 --thinking 또는
# --num-beams로 명시할 수 있다.
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


# thinking 옵션은 내부 사고 과정 출력 기능이 아니라
# beam search에 할당할 탐색량을 의미한다.
THINKING_BEAMS = {
    "off": 1,
    "low": 2,
    "medium": 4,
    "high": 8,
    "max": 16,
}


def build_parser() -> argparse.ArgumentParser:
    """명령행 인자 파서를 생성한다."""
    parser = argparse.ArgumentParser(
        description="sion_translate 학습 모델로 한↔일 번역을 수행합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "text",
        nargs="*",
        help="번역할 문장. 없으면 --input 또는 표준 입력을 사용합니다.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="입력 텍스트 파일입니다. 한 줄을 한 문장으로 처리합니다.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="출력 파일입니다. 지정하지 않으면 표준 출력으로 출력합니다.",
    )

    parser.add_argument(
        "--to",
        dest="target",
        help="목표 언어입니다. 기본값은 설정 언어쌍의 두 번째 언어입니다.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="model.pt, model_ema.pt 또는 model_int8.pt 경로입니다.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "sion_translate.yaml",
        help="sion_translate 설정 파일입니다.",
    )

    parser.add_argument(
        "--int8",
        action="store_true",
        help=(
            "CPU용 INT8 export 모델을 사용합니다. 파일과 메모리가 작아지지만 "
            "번역 속도는 빨라지지 않습니다 (--quality fast 와는 무관)."
        ),
    )

    parser.add_argument(
        "--quality",
        choices=tuple(QUALITY_DEFAULTS),
        default="balanced",
        help=("속도와 품질 프리셋입니다. best는 holdout 평가에서 검증된 beam 4를 사용합니다."),
    )

    parser.add_argument(
        "--thinking",
        choices=tuple(THINKING_BEAMS),
        help=(
            "탐색 예산입니다. "
            "off=greedy, low=beam 2, medium=beam 4, "
            "high=beam 8, max=beam 16입니다. "
            "출력에는 최종 번역만 표시됩니다."
        ),
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        help="beam 수입니다. 지정하면 --quality와 --thinking보다 우선합니다.",
    )

    parser.add_argument(
        "--length-penalty",
        type=float,
        help="beam search 길이 보정값입니다.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="문장당 최대 생성 토큰 수입니다.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="이 크기의 n-gram 재생성을 금지합니다. 0이면 끕니다.",
    )
    parser.add_argument(
        "--max-output-length-ratio",
        type=float,
        default=3.0,
        help="원문 토큰 수 대비 출력 상한 비율입니다. 여유 토큰 16개를 별도로 둡니다.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="배치 크기입니다. 메모리 부족이 발생하면 값을 낮추십시오.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="실행 장치입니다. auto, cuda, cuda:0, cpu 등을 사용할 수 있습니다.",
    )

    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="일반 export 모델의 계산 정밀도입니다. INT8에서는 무시됩니다.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="CUDA에서 torch.compile을 적용합니다. 첫 추론은 느릴 수 있습니다.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="CPU 추론에 사용할 PyTorch 스레드 수입니다.",
    )

    parser.add_argument(
        "--glossary",
        type=Path,
        help="용어집 JSON 파일 경로입니다.",
    )

    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="명령행 및 설정 파일의 용어집을 모두 사용하지 않습니다.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="각 번역 결과를 행별 JSONL 형식으로 출력합니다.",
    )

    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "추론 시간, 전체 시간 및 처리량을 stderr에 출력합니다. "
            "--no-timing으로 비활성화할 수 있습니다."
        ),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help=("설정 준비, 모델 준비, 추론, 출력 시간을 구분하여 상세하게 stderr에 출력합니다."),
    )

    parser.add_argument(
        "--degeneration-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "반복·과다 길이 출력만 더 좁은 beam으로 재시도합니다. "
            "--no-degeneration-retry로 비활성화할 수 있습니다."
        ),
    )

    return parser


def to_python_string(value: object, *, value_name: str) -> str:
    """문자열 계열 값을 Python 기본 str로 변환한다.

    argparse와 일반 텍스트 파일에서는 대부분 기본 str이 들어오지만,
    외부 코드에서 main 함수나 번역 함수를 호출하는 경우 numpy.str_,
    pandas 문자열 스칼라 또는 bytes가 들어올 수 있다.

    리스트나 딕셔너리 같은 구조적 데이터는 문자열로 강제 변환하지 않는다.
    해당 값을 강제로 str로 바꾸면 데이터 오류가 숨겨질 수 있기 때문이다.
    """
    if isinstance(value, str):
        # 문자열 하위 클래스일 가능성까지 고려해 기본 str로 변환한다.
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SystemExit(f"{value_name}이 UTF-8로 해석할 수 없는 bytes입니다.") from error

    # numpy 또는 pandas 스칼라 객체는 item()으로
    # 기본 Python 스칼라를 얻을 수 있다.
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
                    f"{value_name}의 스칼라 값이 UTF-8로 해석할 수 없는 bytes입니다."
                ) from error

    raise SystemExit(
        f"{value_name}은 문자열이어야 합니다. 현재 타입={type(value).__name__}, 값={value!r}"
    )


def read_lines(args: argparse.Namespace) -> list[str]:
    """명령행, 입력 파일 또는 표준 입력에서 번역할 문장을 읽는다."""
    if args.text and args.input:
        raise SystemExit("문장 위치 인자와 --input은 함께 사용할 수 없습니다.")

    raw_lines: Sequence[object]

    if args.text:
        raw_lines = args.text

    elif args.input:
        try:
            raw_lines = args.input.read_text(
                encoding="utf-8",
            ).splitlines()
        except (OSError, UnicodeError) as error:
            raise SystemExit(f"입력 파일을 읽을 수 없습니다: {args.input}: {error}") from error

    else:
        raw_lines = [line.rstrip("\r\n") for line in sys.stdin]

    lines: list[str] = []

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = to_python_string(
            raw_line,
            value_name=f"{line_number}번째 입력",
        ).strip()

        # 빈 줄은 번역 대상에서 제외한다.
        if line:
            lines.append(line)

    if not lines:
        raise SystemExit("번역할 문장이 없습니다.")

    return lines


def choose_device(value: str, int8: bool) -> torch.device:
    """명령행 옵션에 따라 추론 장치를 결정한다."""
    if int8:
        if value not in ("auto", "cpu"):
            print(
                "[sion] INT8 export는 CPU 전용이므로 device=cpu를 사용합니다.",
                file=sys.stderr,
            )

        return torch.device("cpu")

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"유효하지 않은 --device 값입니다: {value}") from error

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA를 요청했지만 사용 가능한 CUDA GPU가 없습니다.")

    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise SystemExit(
                f"CUDA 장치 {device.index}번을 요청했지만 "
                f"사용 가능한 GPU는 {torch.cuda.device_count()}개입니다."
            )

    return device


def synchronize_device(device: torch.device) -> None:
    """정확한 CUDA 시간 측정을 위해 비동기 연산 완료를 기다린다."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_config_path(
    value: str | Path,
    *,
    config_path: Path | None,
) -> Path:
    """설정에 적힌 상대 경로를 설정 파일 위치를 기준으로 해석한다."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    base = config_path.parent if config_path is not None else ROOT
    return base / path


def require_file(path: Path, *, value_name: str) -> Path:
    """필수 입력 경로가 읽을 수 있는 일반 파일인지 검사한다."""
    if not path.exists():
        raise SystemExit(f"{value_name}을 찾을 수 없습니다: {path}")
    if not path.is_file():
        raise SystemExit(f"{value_name}은 파일이어야 합니다: {path}")
    return path


def apply_runtime_options(
    translator: Translator,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """장치, 정밀도 및 torch.compile 설정을 모델에 적용한다.

    INT8 모듈은 CPU 전용 양자화 모델이므로 장치나 dtype을 변경하지 않는다.
    """
    if args.threads is not None:
        if args.threads < 1:
            raise SystemExit("--threads는 1 이상이어야 합니다.")

        torch.set_num_threads(args.threads)

    if translator.quantized:
        return

    if args.dtype == "fp16" and device.type != "cuda":
        raise SystemExit("fp16은 CUDA에서만 지원합니다. CPU에서는 fp32 또는 bf16을 사용하십시오.")

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
                "[sion] --compile은 CUDA에서만 적용합니다.",
                file=sys.stderr,
            )

        elif not hasattr(torch, "compile"):
            print(
                "[sion] 현재 PyTorch에는 torch.compile이 없어 컴파일을 건너뜁니다.",
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
    """품질 프리셋과 명령행 옵션을 합쳐 생성 옵션을 결정한다."""
    preset = QUALITY_DEFAULTS[args.quality]

    # 명시적인 --num-beams가 가장 높은 우선순위를 가진다.
    beams = args.num_beams

    if beams is None:
        # --thinking이 지정되었으면 품질 프리셋의 beam 수보다 우선한다.
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
        raise SystemExit("--num-beams는 1 이상이어야 합니다.")

    if batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens는 1 이상이어야 합니다.")

    if length_penalty <= 0:
        raise SystemExit("--length-penalty는 0보다 커야 합니다.")

    return beams, batch_size, length_penalty


def validate_translations(
    sources: Sequence[str],
    translations: Sequence[object],
) -> list[str]:
    """모델이 반환한 번역 결과의 개수와 타입을 검사한다."""
    translation_list = list(translations)

    if len(translation_list) != len(sources):
        raise SystemExit(
            "번역 결과 개수가 입력 문장 개수와 다릅니다. "
            f"입력={len(sources)}, 결과={len(translation_list)}"
        )

    validated: list[str] = []

    for index, translation in enumerate(
        translation_list,
        start=1,
    ):
        validated.append(
            to_python_string(
                translation,
                value_name=f"{index}번째 번역 결과",
            )
        )

    return validated


def degeneration_reasons(source: str, translation: str) -> set[str]:
    """명백한 생성 붕괴 신호를 반환한다.

    짧은 감탄사나 의도적인 반복을 과도하게 잡지 않도록 5회 이상의 문자 반복,
    4회 이상의 구절 반복, 원문에 비해 비정상적으로 긴 출력만 대상으로 한다.
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
    """붕괴한 결과만 좁은 beam 후보로 교체한다.

    후보가 기존 결과보다 붕괴 사유 수를 실제로 줄일 때만 채택하므로 정상 번역은
    건드리지 않는다. beam 4라면 beam 2, greedy 순으로 남은 문제 문장만 재시도한다.
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
    """번역 결과를 일반 텍스트 또는 JSONL 문자열로 변환한다."""
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
    """결과를 파일 또는 표준 출력으로 기록한다."""
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
        raise SystemExit(f"출력 파일을 쓸 수 없습니다: {output_path}: {error}") from error


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
    """시간 측정 결과를 stderr에 출력한다."""
    safe_inference_elapsed = max(
        inference_elapsed,
        1e-9,
    )

    throughput = sentence_count / safe_inference_elapsed
    average_milliseconds = safe_inference_elapsed / max(sentence_count, 1) * 1000.0

    print(
        (
            f"[sion] 추론 완료: "
            f"{sentence_count}문장 / "
            f"추론 {inference_elapsed:.3f}초 / "
            f"전체 {total_elapsed:.3f}초 / "
            f"{throughput:.2f}문장/초 / "
            f"문장당 평균 {average_milliseconds:.2f}ms"
        ),
        file=sys.stderr,
        flush=True,
    )

    if detailed:
        print(
            "[sion] 상세 시간:",
            file=sys.stderr,
        )

        print(
            f"[sion]   설정 및 입력 준비: {config_elapsed:.3f}초",
            file=sys.stderr,
        )

        print(
            f"[sion]   모델 로딩 및 준비: {model_elapsed:.3f}초",
            file=sys.stderr,
        )

        print(
            f"[sion]   실제 번역 추론: {inference_elapsed:.3f}초",
            file=sys.stderr,
        )

        print(
            f"[sion]   결과 변환 및 출력: {output_elapsed:.3f}초",
            file=sys.stderr,
        )

        print(
            f"[sion]   전체 실행: {total_elapsed:.3f}초",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    """CLI의 전체 추론 절차를 실행한다."""
    configure_stdio()
    total_started = time.perf_counter()

    args = build_parser().parse_args()

    # 설정 파일, 입력 문장, 장치 및 생성 옵션을 준비한다.
    config_started = time.perf_counter()

    sources = read_lines(args)

    default_config_path = ROOT / "sion_translate.yaml"
    config_path = args.config if args.config.exists() else None

    # 사용자가 기본 경로가 아닌 별도 설정 파일을 지정했는데
    # 해당 파일이 없으면 즉시 오류를 발생시킨다.
    if args.config != default_config_path and config_path is None:
        raise SystemExit(f"설정 파일을 찾을 수 없습니다: {args.config}")

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
            value_name="모델",
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
        value_name="토크나이저 모델",
    )

    if args.no_glossary:
        glossary_path = None
    elif args.glossary is not None:
        glossary_path = args.glossary
    else:
        # DataConfig.glossary의 빈 문자열은 "용어집 사용 안 함"을 뜻한다.
        # Path("")는 현재 디렉터리(".")가 되므로 Path로 바꾸기 전에 걸러야 한다.
        configured_glossary = to_python_string(
            config.data.glossary,
            value_name="설정 파일의 data.glossary",
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
            value_name="용어집",
        )
        try:
            glossary = load_glossary(glossary_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"용어집을 읽을 수 없습니다: {glossary_path}: {error}") from error

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

    # 모델과 토크나이저를 불러오고 런타임 옵션을 적용한다.
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
            value_name="설정 파일의 목표 언어",
        ).strip()
    )

    if target not in translator.languages:
        supported_languages = ", ".join(sorted(translator.languages))

        raise SystemExit(f"--to {target}는 지원하지 않습니다. 지원 언어: {supported_languages}")

    synchronize_device(device)
    model_elapsed = time.perf_counter() - model_started

    # 실제 번역 추론 시간을 측정한다.
    #
    # CUDA 연산은 기본적으로 비동기이므로 추론 시작 전과 종료 후에
    # synchronize를 호출해야 실제 완료 시간을 정확히 측정할 수 있다.
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
            "CUDA 메모리가 부족합니다. "
            "--batch-size 값을 낮추십시오. "
            "best 모드라면 --batch-size 1을 권장합니다."
        ) from error

    except TypeError as error:
        error_message = str(error).lower()

        if "not a string" in error_message:
            raise SystemExit(
                "SentencePiece에 문자열이 아닌 값이 전달되었습니다. "
                "src/sion_translate/tokenizer.py의 encode()에서 입력값과 "
                "normalize_text() 반환값을 Python 기본 str로 변환해야 합니다."
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
                f"[sion] 반복 붕괴 재시도: 복구 {rescued_count}문장 / 잔여 {remaining_count}문장",
                file=sys.stderr,
                flush=True,
            )

    synchronize_device(device)
    inference_elapsed = time.perf_counter() - inference_started

    # 반환 결과를 검사하고 출력 형식으로 변환한다.
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

    # --profile은 --timing이 꺼져 있어도 상세 시간 측정을 출력한다.
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
