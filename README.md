# sion_translate 1.0

> 어느 문서를 읽어야 할지 모르겠다면 [`docs/README.md`](docs/README.md)가
> 질문별 색인입니다.

sion_translate는 한국어↔일본어 번역 모델을 처음부터 학습하고 평가하는 PyTorch 프로젝트입니다.
joint SentencePiece, GQA, RoPE, pre-RMSNorm, QK-norm, SwiGLU, EMA, 양방향 번역,
용어집 강제, SFT 뒤 최소위험 사후학습을 포함합니다.

학습은 **세 단계**입니다.

| 단계 | 목적 | 산출물 | 배포 이름 |
|---|---|---|---|
| foundation | 언어별 단일어 span-corruption 복원 | `runs/*/foundation/` | **`sion`** |
| SFT | 번역 (foundation 가중치에서 시작) | `runs/*/pretrain/` | `sion_translate` |
| MRT | 복합 보상 사후학습 | `runs/*/posttrain/` | `sion_translate` |

foundation 단계는 `data/corpus/<언어코드>/`에 단일어 텍스트가 있으면 자동으로
먼저 돌고, 없으면 이유를 출력하고 건너뜁니다. 그 산출물은 번역쌍을 한 번도
보지 않았으므로 **번역 모델이 아니며**, `Translator`가 싣기를 거부합니다.
자세한 내용은 [`docs/foundation-pretraining.md`](docs/foundation-pretraining.md).

세 단계 모두 dataset 일부를 임의의 step 수만큼 보는 방식이 아니라, 설정된
`num_train_epochs`만큼 전체 loader를 완주합니다. 조기 종료는 epoch 경계에서만
판단하며 기본적으로 최소 2 epoch를 보장한 뒤 patience를 적용합니다. dropout,
label smoothing, weight decay, 입력 증강, EMA 검증과 best 가중치 복원을 함께 써서
학습 부족과 validation 과적합을 양쪽에서 제어합니다. `max_steps`는 디버그와 구버전
설정 호환을 위한 명시적 override입니다.

학습 데이터, 전처리 산출물, 체크포인트, 모델 가중치와 로컬 평가 결과는 Git 저장소에
포함하지 않습니다. 사용자는 이용·가공·재배포 권한을 직접 확인한 JSONL만 준비해야
합니다. 유지관리자가 별도로 만드는 `sion_translate.zip`은 권한 확인이 끝난 특정
학습 snapshot을 코드와 함께 전달하는 GPU 실행물이며 Git 저장소와는 구분됩니다.
공개 모델 가중치는 별도
[Hugging Face 저장소](https://huggingface.co/gaon12/sion_translate)에서 제공합니다.
모델 페이지의 widget에는 양방향 입력 예시와 그에 대해 미리 생성해 둔 출력이 적혀
있습니다. 이 모델은 Transformers `AutoModel` 체크포인트가 아니므로 hosted inference로
그 자리에서 실행되지는 않습니다. 직접 돌려 보려면 아래 설치 절차를 따르십시오.

## 빠른 시작

Python 3.11 이상에서 설치합니다.

Linux/macOS:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,export]"
.venv/bin/python -m pytest -q
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,export]"
.\.venv\Scripts\python.exe -m pytest -q
```

`export` extra는 기본 최종 산출물인 INT8과 GGUF 변환에 필요합니다. 요청한 포맷의
의존성이 없으면 `sion-train`이 데이터 준비나 학습을 시작하기 전에 중단합니다.

`data/`에 UTF-8 JSONL 파일을 하나 이상 둡니다. 파일 이름은 자유롭고, 빈 줄 없이
한 줄에 한 번역쌍을 기록합니다.

```json
{"ko":"안녕하세요.","ja":"こんにちは。"}
{"ko":"오늘은 날씨가 맑습니다.","ja":"今日は天気が晴れています。"}
```

형식 예시는 [`examples/training.example.jsonl`](examples/training.example.jsonl)에
있습니다. 실제 데이터는 `.gitignore`로 차단됩니다.

```bash
.venv/bin/python easy_run.py
```

`easy_run.py`는 입력 JSONL과 실행 환경을 감지해 토크나이저 준비, 품질 필터링,
표현·문화 seed의 train/challenge 분리, 중복 제거, split 생성, 모델 크기·배치·정밀도
선택, SFT와 사후학습, 체크포인트 재개를 순서대로 처리합니다. tokenizer와 dataset은
안정된 공개 경로인 `artifacts/`에 만들며, 재사용 전 메타데이터·SHA-256·언어 태그·
숫자 분리 정책·dataset 지문을 검사합니다. 이 자동 실행기는 Linux CUDA GPU 서버용이며 CPU나 Windows에서는
수동 CLI 경로를 사용해야 합니다. 세부 GPU 서버 실행법은 [`START-HERE.md`](START-HERE.md)
(평문판은 [`how_to_run.txt`](how_to_run.txt), 같은 절차의 두 표기입니다), 사후학습 설계는
[`POSTTRAINING.md`](POSTTRAINING.md)를 참고하세요. 80GB급 GPU의 수동 용량 점검과
7종 내보내기는
[`docs/H100_TRAINING.md`](docs/H100_TRAINING.md), 데이터 정비 현황과 1억 쌍
분야별 확장량은 [`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md)에
정리되어 있습니다. 방향별 품질 원인, 구형 48k vocabulary의 실제 target 노출,
표현 데이터 계약과 evidence/parity ablation 범위는
[`docs/QUALITY_OVERHAUL.md`](docs/QUALITY_OVERHAUL.md)를 먼저 확인하십시오. 학습·데이터·
평가·운영 전반의 미해결 출고 차단 사유는
[`docs/PROJECT_ROAST.md`](docs/PROJECT_ROAST.md)에 우선순위별로 정리했습니다.

### 자체 포함 GPU ZIP 만들기

모든 코드·문서 변경을 커밋하고 추적 파일이 깨끗한 상태에서 실행합니다.

```bash
python scripts/package_gpu_bundle.py build \
  --output sion_translate.zip \
  --overwrite
python scripts/package_gpu_bundle.py verify-archive sion_translate.zip
```

빌더는 Git stage-0 일반 파일, `data/*.jsonl`, `data/evaluation_only/**`만 선택해
단일 `sion_translate/` 루트의 ZIP64 archive를 만듭니다. `data/excluded`,
`artifacts`, `runs`, 체크포인트, 가상환경과 캐시는 포함될 수 없습니다.
`PACKAGE_MANIFEST.json`에는 Git commit/tree와 각 파일의 크기·mode·SHA-256이,
`SHA256SUMS`에는 추출 후 검증값이 기록됩니다. 완성 ZIP 전체 SHA-256은 빌드
출력에서 별도로 전달합니다.

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
자릿수로 분리하지 않고 학습된 경우 (2026-07 이전에 만든 토크나이저가 모두 해당,
공개된 48k 토크나이저 포함) 모델이 값을 누락하는 대신 **그럴듯한 다른 값으로 바꿔
쓸** 수 있습니다. 배포 체크포인트에 beam 4 로 숫자 문장 10개를 넣으면 8개에서 값이
바뀝니다:

| 입력 | 출력 |
|---|---|
| `0.0037mg/L 이하로` | `1.337mg/L以下に` |
| `250mg씩 하루 두 번` | `1200mgずつ1日2回` |
| `35%에서 62.5%로` | `０．７％から６．７％に` |
| `부가세 15%가 포함` | `付加税1,500ウォンが含まれる` |
| `110-482-937561` | `1、0、482-937561` |
| `38,720개에서 7,842,913개로` | `3万3千5百個から1万4千9十三個に` |
| `±0.05mm 이내` | `0.00.05mm以内` |

실패한 경우는 모두 다자리 병합 토큰을 포함합니다: `35%` 는 토큰 하나이고
`62.5kg` 는 `▁6 | 2.5 | kg` 로 쪼개져 숫자와 단위의 경계가 토큰 내부에 들어갑니다.
값이 보존된 두 건(`-2.5mg`, `62.5kg`)은 병합 덩어리가 그대로 복사된 경우입니다.

이 결함은 사후학습이나 데이터 추가로 고쳐지지 않습니다. 표현이 자릿수를 드러내지
않으므로 `reward_number_weight` 도 최적화할 신호를 얻지 못합니다.

해당 토크나이저를 불러오면 `Translator`가 경고를 한 번 출력하고, **`sion-train`은
학습을 거부합니다.** 따라서 공개된 체크포인트에서 이어서 학습할 수는 없습니다.
`split_digits=True`로 토크나이저를 다시 학습하면 vocabulary가 바뀌고, tie된
`token_embedding`(48000×768 = 전체 파라미터의 18.4%)이 무효가 되므로 처음부터
재학습해야 합니다. 재학습할 때는 `sion-train-tokenizer`의 기본값(숫자 분리 켜짐)을
그대로 쓰십시오.

직접 모델 파일을 지정할 수도 있습니다.

```bash
sion-translate --model runs/auto/posttrain/exports/best/model_ema.pt --to ja "안녕하세요."
```

## 여러 번역 시스템 비교

저장소는 세 개의 진단셋을 제공합니다. 모두 이 프로젝트용으로 새로 작성한 합성
문장이며 어떤 학습 코퍼스에도 포함되지 않습니다.

- [`examples/comparison_cases.jsonl`](examples/comparison_cases.jsonl) — 16문장.
  존댓말, 동음이의어, 숫자, 기술 문자열, 구어체, 장문 의존성, 고유명사, 관용 표현.
- [`examples/diagnostic_cases.jsonl`](examples/diagnostic_cases.jsonl) — 40문장.
  위 항목에 의료, 법률, 행정, 관광, 학술, 부정 표현을 더하고 고유명사·숫자 케이스를
  늘렸습니다. 학습 데이터에 없는 도메인에서 품질이 얼마나 떨어지는지 보기 위한
  셋이므로, 자체 holdout 점수와 함께 보면 일반화 격차를 가늠할 수 있습니다.
- [`examples/expressive_cultural_cases.jsonl`](examples/expressive_cultural_cases.jsonl) —
  24문장. 욕설·인터넷 비속어, 감탄·신음, 관용구·문화 현지화를 각각 양방향으로
  평가합니다. seed의 train 행과 ID·표면형이 분리돼 있습니다.

세 파일 모두 한국어→일본어와 일본어→한국어를 같은 수로 담고 있고 스키마가 같으므로
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
출처에서 잘라낸 in-domain holdout입니다. 도메인이 겹치므로 학습에 쓰지 않은
도메인의 문장에서는 점수가 크게 낮아집니다. 이 저장소의 데이터 구성으로 측정했을
때 in-domain chrF와 도메인 밖 chrF의 차이는 20점을 넘었습니다.

**완전일치 split 만으로는 근사 중복이 막히지 않습니다.** 기본값인
`data.approximate_split: true`는 문자 5-gram MinHash로 split을 배정합니다.
이를 `false`로 끄거나 `sion-prepare-data --exact-split`을 사용하면
NFKC·공백 정규화 후 완전일치 문자열로만 누수를 막으므로, 조사 하나만 다른 문장이
train 과 holdout 에 따로 들어갈 수 있습니다. 실측하면
문자 5-gram MinHash split 대비 근사 중복 유출이 두 배였습니다 (data9/29/33/41
평균 1.12% → 0.48%). 이 저장소가 이전에 보고한 자체 test split ja→ko
BLEU 81.95 는 이 유출을 포함한 값이므로 인용하지 마십시오 — 같은 체크포인트의
외부 diagnostic chrF 는 53.43 입니다. 과거 실험 재현 외에는 기본값을 유지하고,
템플릿성 생성 데이터는
`scripts/data/resample_generated_shards.py` 로 프레임·인용구 재사용을 먼저
제한하십시오. MinHash 는 근사 중복용이고 템플릿용이 아닙니다.

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

`sion-train`은 `data/corpus/`에 단일어 텍스트가 있으면 foundation 단계를 먼저
실행합니다. 끝나면 `runs/*/foundation/stage_complete.json`이 남고, 이후 실행은
학습을 건너뛰고 그 가중치만 물려받습니다 — 번역 학습이 실패해 다시 실행할 때마다
며칠짜리 사전학습을 반복하지 않기 위한 것입니다.

토크나이저를 다시 만들면 vocab이 달라지므로 `artifacts/dataset`과 기존
체크포인트는 재사용할 수 없습니다. 위 세 단계를 순서대로 다시 실행해야 합니다.
해당 경로에 호환되지 않는 과거 산출물이 있다면 먼저 별도 백업 경로로 옮기십시오.
학습기는 호환성을 경로 이름으로 추측하지 않고 실제 내용이 다르면 시작을 거부합니다.

### 언어쌍 바꾸기 / 늘리기

언어쌍은 코드에 고정돼 있지 않습니다. JSONL의 키 이름이 곧 언어 이름이고,
`sion_translate.yaml`만 고치면 토크나이저의 `<2xx>`·`<denoise_xx>` 제어 토큰,
전처리, 방향 태그가 전부 따라갑니다.

한 쌍만 쓸 때는 `data.language_pair`를 바꿉니다.

```yaml
data:
  language_pair: [en, de]    # JSONL 이 {"en": ..., "de": ...} 형태여야 합니다
```

여러 쌍을 한 모델에 넣을 때는 `language_pair` 대신 `language_pairs`를 씁니다.

```yaml
data:
  language_pairs:
    - [ko, ja]
    - [en, ko]
```

#### source 전용 언어

일부 언어는 **입력으로만** 받아야 합니다. 한본어(`kj`)는 한국어와 일본어가
섞인 입력이고 번역 결과는 항상 한쪽 단일어여야 합니다. 지역 방언(`kd`, `jd`)도
목표가 방언을 *이해*하는 것이지 표준 입력에 사투리로 답하는 것이 아닙니다.

`source_only_languages`에 등재하면 `kj->ko`, `kd->ja` 같은 이해 방향만
학습되고 역방향은 만들어지지 않습니다. **등재하지 않으면** bidirectional 학습이
혼용문과 방언을 target 으로도 배워서, 표준 한국어를 요청했는데 가나가 섞이거나
사투리가 나오는 모델이 됩니다.

```yaml
data:
  language_pairs:
    - [kj, ko]
    - [kj, ja]
    - [kd, ko]
    - [kd, ja]
    - [jd, ko]
    - [jd, ja]
    - [ko, ja]
  source_only_languages: [kj, kd, jd]
```

방언 shard 는 지역을 태그가 아니라 행 메타데이터(`dialect_region`)로 담습니다.
특정 지역 방언으로 *생성*하게 하려면 별도 태그 스킴이 필요하고, 그것은 아직
구현돼 있지 않습니다.

### 재학습 전 확인할 설정

처음부터 다시 학습할 때 놓치면 결과가 무의미해지는 항목입니다.

| 설정 | 값 | 안 하면 |
|---|---|---|
| `data.approximate_split` | `true` | 근사 중복이 holdout 으로 새어 점수가 번역 품질을 재지 않습니다 |
| `data.source_only_languages` | `[kj, kd, jd]` | 표준어를 요청해도 혼용문·사투리가 나옵니다 |
| 토크나이저 `split_digits` | `true` (기본) | 숫자가 덩어리로 병합돼 금액·용량이 다른 값으로 바뀝니다 |
| `--input-sentence-size` | `0` (기본) | 코퍼스 일부만 보고 어휘를 정합니다 |

`--input-sentence-size 0` 은 전량을 뜻합니다. 상한을 두면 균등 무작위 추출이라
작은 shard 가 비중만큼만 보이고, 거기서만 흔한 문자가 어휘에서 빠질 수 있습니다.
`--required-character-min-occurrences`(기본 25)는 그 문자가 byte fallback 으로
쪼개지지 않도록 어휘에 못을 박습니다.

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

## 학습 전 감사 도구

전부 GPU 시간을 쓰기 전에 도는 관문입니다. 이 프로젝트의 실패는 대부분
"조용히 잘못된 데이터로 학습이 끝난 뒤에야 드러나는" 종류라 사전 검사를 둡니다.

```bash
# 1. 토큰 노출 — 어휘 조각이 디코더 타깃으로 몇 번 나오는가
sion-audit-tokens --input "data/*.jsonl" --tokenizer artifacts/tokenizer/sion.model   --monolingual-corpus data/corpus

# 2. holdout 누출 — challenge 문장이 학습 코퍼스에 이미 있는가 (누출 시 종료 코드 != 0)
python scripts/data/audit_holdout_leakage.py   --holdout examples/expressive_cultural_cases.jsonl --corpus "data/*.jsonl"

# 3. 오염된 정답쌍 — 사람 검수 queue 생성 (아무것도 지우지 않음)
python scripts/data/build_review_queue.py --input "data/*.jsonl"   --output reports/review_queue.jsonl
```

측정된 결과와 그 해석은 [`docs/PROJECT_ROAST.md`](docs/PROJECT_ROAST.md)에 있습니다.
`examples/expressive_cultural_cases.jsonl`은 실측으로 **48개 중 28개(58.3%)가
학습 코퍼스와 겹치므로** 회귀 smoke set으로만 쓰고 품질 benchmark로 인용하지
마십시오.

### foundation 코퍼스 만들기

```bash
# 국립국어원 모두의 말뭉치 ZIP → 발화 단위 JSONL
python scripts/data/extract_nikl_corpus.py --archive "kli_corpus/NIKL_*.zip"   --output data/corpus/ko/nikl_spoken_dialogue.jsonl

# 일본어 구어 (Apache-2.0)
python scripts/data/fetch_open2ch.py --output data/corpus/ja/open2ch_cleaned.jsonl
```

둘 다 **발화/문단 단위로 한 줄씩** 씁니다. 문서 단위로 넣으면 학습 준비에서
문자의 25.9%가 상한 초과로 폐기됩니다(실측).

## export 형식

`training.final_export_formats`로 정합니다. `fp32`, `fp16`, `bf16`, `int8`,
`int4`, `fp8`, `gguf_q4_k_m`, `transformers`.

`fp8`은 가중치 전용 E4M3 + 블록 스케일입니다. export 파일과 모델의 상주
가중치 메모리를 줄이면서, 활성값까지 내리는 것보다 정확합니다(출력 오차
2.57% 대 3.63%). 현재 런타임은 매 forward에서 가중치를 BF16으로 역양자화하되,
BF16을 지원하지 않는 CUDA 장치에서는 FP16으로 자동 fallback한 뒤 dense GEMM을
합니다. A100처럼 네이티브 FP8 텐서코어가 없는 장치에서도 실행할 수 있습니다.
현재 경로는 네이티브 FP8 텐서코어를 사용하지 않으므로 실행 대역폭이나 연산량
절감은 보장하지 않습니다. 기본 범위는 FFN 뿐입니다 — attention까지 내리면
최종 logits 오차가 6.39%에서 13.11%로 두 배가 됩니다. 어휘 projection은 어떤
설정에서도 제외됩니다(argmax 6.45% 변경). 근거 수치는
`src/sion_translate/fp8.py` 문서에 있습니다.

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
