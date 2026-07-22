# KJ-X

KJ-X는 한국어↔일본어 번역 모델을 처음부터 학습하고 평가하는 PyTorch 프로젝트입니다.
joint SentencePiece, GQA, RoPE, pre-RMSNorm, QK-norm, SwiGLU, EMA, 양방향 번역,
용어집 강제, SFT 뒤 최소위험 사후학습을 포함합니다.

학습 데이터, 전처리 산출물, 체크포인트, 모델 가중치와 로컬 평가 결과는 Git 저장소에
포함하지 않습니다. 사용자는 이용·가공·재배포 권한을 직접 확인한 JSONL만 준비해야
합니다. 공개 모델 가중치는 별도
[Hugging Face 저장소](https://huggingface.co/gaon12/sion_translate)에서 제공합니다.

## 빠른 시작

Python 3.11 이상에서 설치합니다.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
```

`data/`에 UTF-8 JSONL 파일을 하나 이상 둡니다. 파일 이름은 자유롭고, 빈 줄 없이
한 줄에 한 번역쌍을 기록합니다.

```json
{"ko":"안녕하세요.","ja":"こんにちは。"}
{"ko":"오늘은 날씨가 맑습니다.","ja":"今日は天気が晴れています。"}
```

형식 예시는 [`examples/training.example.jsonl`](examples/training.example.jsonl)에
있습니다. 실제 데이터는 `.gitignore`로 차단됩니다.

```bash
python easy_run.py
```

`easy_run.py`는 입력 JSONL과 실행 환경을 감지해 토크나이저 준비, 품질 필터링,
중복 제거, split 생성, 모델 크기·배치·정밀도 선택, SFT와 사후학습, 체크포인트 재개를
순서대로 처리합니다. 세부 GPU 서버 실행법은 [`how_to_run.txt`](how_to_run.txt),
사후학습 설계는 [`POSTTRAINING.md`](POSTTRAINING.md)를 참고하세요.

## 번역

학습이 끝나면 EMA export를 우선 사용합니다.

```bash
kjx-translate --to ja "회의는 세 시에 시작합니다."
kjx-translate --to ko "会議は三時に始まります。"
kjx-translate --to ja --int8 "CPU에서 번역합니다."
```

직접 모델 파일을 지정할 수도 있습니다.

```bash
kjx-translate --model runs/auto/posttrain/exports/best/model_ema.pt --to ja "안녕하세요."
```

## 여러 번역 시스템 비교

저장소가 제공하는 16개 문장은 모두 이 프로젝트용으로 새로 작성한 합성 예시입니다.
존댓말, 동음이의어, 숫자, 기술 문자열, 구어체, 장문 의존성, 고유명사와 관용 표현을
한국어→일본어와 일본어→한국어 양쪽에서 확인합니다.

비교 대상은 KJ-X, LibreTranslate, Papago, Google Cloud Translation, DeepL,
M2M100 418M, NLLB-200 distilled 600M입니다. 실제 서비스 출력이나 점수는 저장소에
미리 넣지 않습니다. 모델 버전, API 시점과 옵션이 달라지면 결과도 달라지므로 같은
문장으로 직접 생성한 결과만 비교합니다.

### 1. 비교 문장 확인

[`examples/comparison_cases.jsonl`](examples/comparison_cases.jsonl)의 스키마는 다음과
같습니다.

```json
{"id":"ko-ja-honorific-01","source_language":"ko","target_language":"ja","category":"honorific","source":"...","reference":"..."}
```

### 2. 모델 출력 JSONL 생성

KJ-X와 공개 baseline은 명령으로 바로 생성할 수 있습니다. 공개 baseline 가중치는
Hugging Face 캐시에만 내려받고 프로젝트에는 복사하지 않습니다.

```bash
mkdir -p comparison_outputs

kjx-translate-cases \
  --backend kjx \
  --cases examples/comparison_cases.jsonl \
  --model runs/auto/posttrain/exports/best/model_ema.pt \
  --tokenizer artifacts/tokenizer/kjx.model \
  --output comparison_outputs/kjx.jsonl

python -m pip install -e ".[baselines]"
kjx-translate-cases --backend m2m100-418m \
  --cases examples/comparison_cases.jsonl \
  --output comparison_outputs/m2m100-418m.jsonl
kjx-translate-cases --backend nllb-200-distilled-600m \
  --cases examples/comparison_cases.jsonl \
  --output comparison_outputs/nllb-200.jsonl
```

LibreTranslate, Papago, Google, DeepL의 결과는
[`examples/system_output.example.jsonl`](examples/system_output.example.jsonl)을 복사한
뒤 각 `id`의 `translation`만 채웁니다. API 키나 서비스별 SDK를 저장소에 넣을 필요가
없습니다.

```json
{"id":"ko-ja-honorific-01","translation":"会議が終わったら……"}
```

### 3. 같은 지표로 채점

```bash
kjx-compare \
  --cases examples/comparison_cases.jsonl \
  --system kjx=comparison_outputs/kjx.jsonl \
  --system libretranslate=comparison_outputs/libretranslate.jsonl \
  --system papago=comparison_outputs/papago.jsonl \
  --system google=comparison_outputs/google.jsonl \
  --system deepl=comparison_outputs/deepl.jsonl \
  --system m2m100-418m=comparison_outputs/m2m100-418m.jsonl \
  --system nllb-200=comparison_outputs/nllb-200.jsonl
```

결과는 `reports/comparison-*.json`과 문장별 나란히 보기가 포함된 Markdown으로
저장됩니다. chrF와 문자 단위 BLEU는 보조 지표이며, 16문장 결과를 보편적인 서비스
순위로 해석하면 안 됩니다. 의미 보존, 높임말, 용어 일관성과 숫자 보존을 사람이 함께
검토해야 합니다.

서비스별 비교 관점과 라이선스 주의사항은
[`docs/COMPARISON.md`](docs/COMPARISON.md)에 정리했습니다.

## 설정과 수동 실행

기본 설정은 [`kjx.yaml`](kjx.yaml) 하나로 관리합니다. `data.language_pair`의 두 값이
JSONL 키, 방향 태그와 품질 검사에 사용됩니다.

```bash
kjx-train-tokenizer --input "data/*.jsonl" --output-dir artifacts/tokenizer
kjx-prepare-data --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer/kjx.model \
  --output-dir artifacts/dataset
kjx-train --config configs/kjx_data_fit.yaml
```

다중 GPU에서는 다음처럼 실행합니다.

```bash
torchrun --standalone --nproc-per-node=8 -m kjx.cli.train
```

## 저장소에 포함하지 않는 파일

- `data/`, `data_mono/`, `benchmarks/`: 학습·평가 입력
- `artifacts/`, `runs/`, `checkpoints/`, `exports/`: 토크나이저, indexed data,
  체크포인트와 가중치
- `reports/`, `comparison_outputs/`: 로컬 평가 문장과 서비스 출력
- `tools/`: 로컬 데이터 취득·변환 도구
- `*.pt`, `*.safetensors`, `*.model`, `*.bin`, 압축 아카이브

커밋 전에는 `git status --ignored`와 아래 명령으로 대용량 파일을 확인하는 것을
권장합니다.

```bash
git ls-files -z | xargs -0 -n1 sh -c 'test "$(wc -c < "$0")" -lt 50000000 || echo "$0"'
```

## 라이선스

이 저장소의 원본 코드와 자체 작성 예시는 [MIT License](LICENSE)로 배포됩니다.
MIT는 사용자가 준비한 데이터, 생성한 번역 결과, 외부 API, 제3자 모델 또는 그
가중치에 적용되지 않습니다. 특히 NLLB-200 배포 가중치는 CC-BY-NC 4.0이며 연구용으로
명시되어 있으므로 상업·프로덕션 사용 전에 upstream 조건을 확인해야 합니다.
