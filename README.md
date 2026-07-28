# sion_translate

sion_translate는 한국어↔일본어 번역 모델을 처음부터 학습하고 평가하는 PyTorch 프로젝트입니다.
joint SentencePiece, GQA, RoPE, pre-RMSNorm, QK-norm, SwiGLU, EMA, 양방향 번역,
용어집 강제, SFT 뒤 최소위험 사후학습을 포함합니다.

학습 데이터, 전처리 산출물, 체크포인트, 모델 가중치와 로컬 평가 결과는 Git 저장소에
포함하지 않습니다. 사용자는 이용·가공·재배포 권한을 직접 확인한 JSONL만 준비해야
합니다. 공개 모델 가중치는 별도
[Hugging Face 저장소](https://huggingface.co/gaon12/sion_translate)에서 제공합니다.
모델 페이지의 widget에는 양방향 입력 예시와 그에 대해 미리 생성해 둔 출력이 적혀
있습니다. 이 모델은 Transformers `AutoModel` 체크포인트가 아니므로 hosted inference로
그 자리에서 실행되지는 않습니다. 직접 돌려 보려면 아래 설치 절차를 따르십시오.

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
사후학습 설계는 [`POSTTRAINING.md`](POSTTRAINING.md)를 참고하세요. H100
단일·다중 GPU 용량 점검과 7종 내보내기는
[`docs/H100_TRAINING.md`](docs/H100_TRAINING.md), 데이터 정비 현황과 1억 쌍
분야별 확장량은 [`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md)에
정리되어 있습니다.

## 번역

학습이 끝나면 EMA export를 우선 사용합니다.

```bash
sion-translate --to ja "회의는 세 시에 시작합니다."
sion-translate --to ko "会議は三時に始まります。"
sion-translate --to ja --int8 "CPU에서 번역합니다."
```

`--int8`은 모델 파일과 메모리 사용량을 줄이는 옵션입니다. 같은 CPU에서 측정하면
생성 속도는 FP32와 사실상 같으므로, 속도가 목표라면 `--num-beams`를 낮추십시오.

### 후보 재순위 (재학습 없이 계산량만 늘리기)

`--candidates N`을 주면 beam 결과에 더해 N개 후보를 뽑고 하나를 고릅니다. 모델을
다시 학습하지 않고 추론 계산량만 늘리는 경로입니다.

```bash
sion-translate --to ja --candidates 7 --rerank mbr+qe "..."
```

홀드아웃 40문장 측정 결과 (배포된 export 기준):

| 설정 | ko-ja chrF | ja-ko chrF | 숫자 일치 | 시간 |
|---|---:|---:|---:|---:|
| beam4 (기준) | 59.81 | 49.87 | 19/20, 15/20 | 104s |
| mbr+qe, n=7, T=0.3 | **60.53** | **50.36** | 19/20, 16/20 | 133s |
| mbr+qe, n=15, T=0.5 | 58.69 | 52.06 | 19/20, 16/20 | 166s |
| qe 단독, n=7, T=0.7 | 54.33 | 46.56 | 19/20, 16/20 | 121s |

세 가지를 유의하십시오.

- **`--rerank qe` 단독은 쓰지 마십시오.** ko-ja에서 5.5점을 잃습니다. 참조 없는
  신호만으로는 유창한 오역을 걸러내지 못합니다. MBR의 합의 조건과 함께여야 합니다.
- **이득은 작고 잡음과 구분되지 않습니다.** 방향당 20문장에서 +0.7 / +0.5입니다.
  채택하려면 더 큰 셋에서 확인하십시오.
- **숫자 오역은 이 방법으로 고쳐지지 않습니다.** 숫자 일치가 한 문장 움직였을
  뿐입니다. 토크나이저가 숫자를 자릿수로 못 보면 후보 전체가 같이 틀리므로 고를
  올바른 후보가 없습니다. 아래 항목이 진짜 해결책입니다.

### 초안 수정과 반복 (실험적, 재학습 필요)

`원문 + 초안 → 고친 번역`을 학습해 두 번째 패스를 주는 경로입니다. 학습 데이터는
지금 있는 쌍만으로 만들 수 있습니다.

```bash
sion-revise-data --input "data/*.jsonl" --output data/revise_synthetic.jsonl
# 위 파일을 학습 입력에 포함해 재학습한 뒤:
sion-translate --to ja --revise-rounds 3 --accept-score 0.95 "..."
```

`--revise-rounds`는 쉬운 문장은 한 번만 번역하고 QE 기준에 못 미친 문장만 다시
고칩니다. 세 조건(`accept-score` 도달 / `min-gain` 미만 / 라운드 상한)으로 멈추고,
점수가 떨어진 라운드는 채택하지 않으므로 결과가 후퇴하지 않습니다.

`<draft>` 제어 토큰이 필요하므로 토크나이저를 다시 학습해야 하고, 그 데이터로
학습하지 않은 모델에서는 의미가 없습니다 (CLI가 명확히 거부합니다).

### 공유 블록 반복 (실험적, 재학습 필요)

```yaml
model:
  experimental:
    recurrent_block_layers: 2   # 인코더 마지막 2개 층을
    recurrent_steps: 3          # 같은 가중치로 3번 통과
```

파라미터를 늘리지 않고 유효 깊이만 늘립니다. 가중치를 새로 만들지 않으므로 기존
체크포인트가 그대로 로드되지만, 반복 적용을 학습하지 않은 가중치에 켜도 품질이
오르지는 않습니다. 비용은 해당 층 계산량과 활성 메모리가 `recurrent_steps`배입니다.

### 숫자가 있는 문장 주의

금액·용량·날짜가 들어간 문장은 번역 결과를 반드시 대조하십시오. 토크나이저가 숫자를
자릿수로 분리하지 않고 학습된 경우 (2026-07 이전에 만든 토크나이저가 모두 해당)
모델이 값을 누락하는 대신 **그럴듯한 다른 값으로 바꿔 쓸** 수 있습니다. 실제로
관측된 예: `250mg` → `1200mg`, `0.5mL` → `120ml`, `38,720円` → `38,000엔`.
해당 토크나이저를 불러오면 `Translator`가 경고를 한 번 출력합니다. 재학습할 때는
`sion-train-tokenizer`의 기본값(숫자 분리 켜짐)을 그대로 쓰십시오.

직접 모델 파일을 지정할 수도 있습니다.

```bash
sion-translate --model runs/auto/posttrain/exports/best/model_ema.pt --to ja "안녕하세요."
```

## 여러 번역 시스템 비교

저장소는 두 개의 진단셋을 제공합니다. 모두 이 프로젝트용으로 새로 작성한 합성
문장이며 어떤 학습 코퍼스에도 포함되지 않습니다.

- [`examples/comparison_cases.jsonl`](examples/comparison_cases.jsonl) — 16문장.
  존댓말, 동음이의어, 숫자, 기술 문자열, 구어체, 장문 의존성, 고유명사, 관용 표현.
- [`examples/diagnostic_cases.jsonl`](examples/diagnostic_cases.jsonl) — 40문장.
  위 항목에 의료, 법률, 행정, 관광, 학술, 부정 표현을 더하고 고유명사·숫자 케이스를
  늘렸습니다. 학습 데이터에 없는 도메인에서 품질이 얼마나 떨어지는지 보기 위한
  셋이므로, 자체 holdout 점수와 함께 보면 일반화 격차를 가늠할 수 있습니다.

둘 다 한국어→일본어와 일본어→한국어를 같은 수로 담고 있고 스키마가 같으므로
`--cases`만 바꿔 끼우면 됩니다.

비교 대상은 sion_translate, LibreTranslate, Papago, Google Cloud Translation, DeepL,
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

sion_translate와 공개 baseline은 명령으로 바로 생성할 수 있습니다. 공개 baseline 가중치는
Hugging Face 캐시에만 내려받고 프로젝트에는 복사하지 않습니다.

```bash
mkdir -p comparison_outputs

sion-translate-cases \
  --backend sion \
  --cases examples/comparison_cases.jsonl \
  --model runs/auto/posttrain/exports/best/model_ema.pt \
  --tokenizer artifacts/tokenizer/sion.model \
  --output comparison_outputs/sion.jsonl

python -m pip install -e ".[baselines]"
sion-translate-cases --backend m2m100-418m \
  --cases examples/comparison_cases.jsonl \
  --output comparison_outputs/m2m100-418m.jsonl
sion-translate-cases --backend nllb-200-distilled-600m \
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
sion-compare \
  --cases examples/comparison_cases.jsonl \
  --system sion=comparison_outputs/sion.jsonl \
  --system libretranslate=comparison_outputs/libretranslate.jsonl \
  --system papago=comparison_outputs/papago.jsonl \
  --system google=comparison_outputs/google.jsonl \
  --system deepl=comparison_outputs/deepl.jsonl \
  --system m2m100-418m=comparison_outputs/m2m100-418m.jsonl \
  --system nllb-200=comparison_outputs/nllb-200.jsonl
```

결과는 `reports/comparison-*.json`과 문장별 나란히 보기가 포함된 Markdown으로
저장됩니다. chrF와 문자 단위 BLEU는 보조 지표이며, 16문장 결과를 보편적인 서비스
순위로 해석하면 안 됩니다. 의미 보존, 높임말, 용어 일관성을 사람이 함께 검토해야
합니다.

표에는 `숫자 F1`과 `숫자 일치` 열도 함께 나옵니다. chrF는 문자 n-gram이 대부분
겹치면 높은 점수를 주므로 값 하나만 바뀐 오역을 거의 벌하지 않기 때문에, 금액·용량·
날짜 보존은 따로 집계합니다. chrF가 가장 높은 시스템이 숫자에서는 가장 나쁠 수
있으므로 두 열을 함께 보십시오.

### 자체 holdout 점수를 인용할 때

`sion-evaluate`가 `--dataset-dir`의 test split으로 내는 점수는 학습 데이터와 같은
출처에서 잘라낸 in-domain holdout입니다. 누수는 막혀 있지만 도메인은 겹치므로,
학습에 쓰지 않은 도메인의 문장에서는 점수가 크게 낮아집니다. 이 저장소의 데이터
구성으로 측정했을 때 in-domain chrF와 도메인 밖 chrF의 차이는 20점을 넘었습니다.
대외적으로 숫자를 제시할 때는 어느 쪽인지 함께 밝히고, 도메인별 고정 benchmark를
따로 두십시오 (`sion-prepare-benchmark`).

FLORES-200을 학습에 포함했다면 `sion-prepare-benchmark`로 만든 FLORES 점수는 자기
채점이므로 인용할 수 없습니다. 학습에 넣을지, 평가에 쓸지 중 하나만 선택하십시오.

서비스별 비교 관점과 라이선스 주의사항은
[`docs/COMPARISON.md`](docs/COMPARISON.md)에 정리했습니다.

## 설정과 수동 실행

기본 설정은 [`sion_translate.yaml`](sion_translate.yaml) 하나로 관리합니다. `data.language_pair`의 두 값이
JSONL 키, 방향 태그와 품질 검사에 사용됩니다.

```bash
sion-train-tokenizer --input "data/*.jsonl" --output-dir artifacts/tokenizer
sion-prepare-data --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer/sion.model \
  --output-dir artifacts/dataset
sion-train --config configs/sion_data_fit.yaml
```

토크나이저를 다시 만들면 vocab이 달라지므로 `artifacts/dataset`과 기존 체크포인트는
재사용할 수 없습니다. 위 세 단계를 순서대로 다시 실행해야 합니다.

### 다문장 입력 학습 (문장 이어붙이기)

원문 코퍼스가 전부 한 문장짜리면 모델은 여러 문장을 한 번에 받았을 때 뒤쪽을
빠뜨리거나 중간에서 멈춥니다. 서로 **무관한** 쌍을 이어붙이면 추가 수집 없이
긴 입력·긴 출력·문장 경계·누락 방지를 지도할 수 있습니다.

```bash
sion-concat --input "data/*.jsonl" --output data/concat_multi.jsonl \
  --count 300000 --min-sentences 2 --max-sentences 4 \
  --tokenizer artifacts/tokenizer/sion.model --max-tokens 510
```

무관한 문장을 쓰는 것이 핵심입니다. 문맥이 이어지는 문단을 쓰면 모델이 앞 문장으로
뒤 문장을 추측할 수 있어, "빠뜨리지 않고 다 옮기는" 능력 대신 문맥 예측을 배웁니다.

`--separator seg`를 주면 `<seg>` 제어 토큰으로 경계를 명시적으로 지도합니다. 기본값인
공백은 사용자가 여러 문장을 그냥 붙여 넣는 실제 사용 형태에 가깝습니다.

산출 파일이 `concat_`으로 시작하면 `sion-prepare-data`의 `--train-only-prefix`
기본값(`bt_ concat_`)에 걸려 train split에만 들어갑니다. 합성 예제가 holdout에
들어가면 점수가 번역 품질이 아니라 합성 규칙을 재게 되므로 이 접두어를 유지하십시오.

다중 GPU에서는 다음처럼 실행합니다.

```bash
torchrun --standalone --nproc-per-node=8 -m sion_translate.cli.train
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
