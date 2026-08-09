# H100 학습·내보내기 운영 가이드

이 문서는 현재 코드와 설정을 기준으로 H100 80GB 단일·다중 GPU 학습을
재현하고, 병목과 OOM을 판단한 뒤 최종 배포 산출물을 검증하는 절차를 정리한다.
예전 커밋에서 관찰된 “단일 H100, 사후학습 포함 약 16시간, GPU 사용률 최대 약
50%, 사전학습 VRAM 약 56GB, 사후학습 OOM”은 비교 기준일 뿐이다. 현재 구현은
BF16, DDP/FSDP2 선택, Tensor Core 친화적 패딩, 단계 전환 시 worker·CUDA cache
정리, 사후학습 후보 micro-batch와 activation checkpointing, 처리량·VRAM
telemetry를 추가했으므로 같은 조건에서 다시 측정해야 한다.

## 1. Conda 환경

저장소 루트의 `environment.yml`은 Python 3.11과 개발·내보내기 의존성을 설치한다.
이 파일은 버전 범위를 가진 환경 명세이지 플랫폼별 lock 파일은 아니다. 실제
H100 run마다 `conda list --explicit` 결과와 PyTorch/CUDA/driver 버전을 run
metadata와 함께 보존한다.

```bash
conda env create -f environment.yml
conda activate sion-translate
```

이미 환경이 있으면 다음처럼 갱신한다.

```bash
conda env update -n sion-translate -f environment.yml --prune
conda activate sion-translate
```

학습 노드에서 CUDA와 BF16 지원을 먼저 확인한다.

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_runtime', torch.version.cuda); print('gpu_count', torch.cuda.device_count()); print('bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported()); print('nccl', torch.distributed.is_nccl_available())"
```

`cuda=False`, `bf16=False`, 또는 다중 GPU에서 `nccl=False`이면 학습을 시작하지
말고, 해당 H100 노드의
드라이버와 CUDA를 지원하는 PyTorch 설치를 먼저 바로잡는다. 저장소의 로컬
Windows/CPU 환경은 H100 성능이나 NCCL 동작을 검증할 수 없으므로 최종 속도와
메모리 수치는 반드시 실제 Linux H100 노드에서 기록한다.

## 2. 새 토크나이저·데이터·run 사용

숫자 분리 정책이나 언어 명칭·언어쌍이 바뀐 토크나이저는 기존 체크포인트와
호환되지 않는다. 현재 코드는 토크나이저 SHA256, 크기, 숫자 분리 정책,
`language_pairs`, vocab 크기가 맞지 않으면 명시적으로 중단한다. 공개 경로는
`artifacts/tokenizer`, `artifacts/dataset`, `runs/*`로 유지한다. 기존 산출물이
내용 검사에 실패하면 자동 교체하지 않는다. 모든 관련 run을 확인한 뒤 운영자가
복구 가능한 별도 위치로 직접 옮기고 같은 공개 경로를 다시 만든다.

대규모 분산 작업 전에 단일 프로세스로 전처리를 끝내 두면 다른 rank가 전처리
barrier에서 오래 기다리는 일을 피할 수 있다.

```bash
sion-train --config configs/sion_data_fit.yaml --prepare-only
```

명시적으로 수행하려면 다음 순서를 쓴다.

```bash
sion-train-tokenizer --input "data/*.jsonl" \
  --output-dir artifacts/tokenizer

sion-prepare-data --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer/sion.model \
  --output-dir artifacts/dataset
```

다국어라면 두 명령 모두 같은 언어쌍을 반복해서 지정해야 한다.

```bash
sion-train-tokenizer --input "data/*.jsonl" \
  --output-dir artifacts/tokenizer-multilingual \
  --language-pairs ko ja \
  --language-pairs en ru

sion-prepare-data --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer-multilingual/sion.model \
  --output-dir artifacts/dataset-multilingual \
  --language-pairs ko ja \
  --language-pairs en ru
```

YAML에서도 `data.language_pair`와 `data.language_pairs` 중 하나만 사용하고,
토크나이저·데이터셋 경로를 새 디렉터리로 맞춘다. `bidirectional: true`이면
한 물리 병렬쌍에서 양방향 학습 예제를 만들기 때문에 `ko-ja`와 `ja-ko`를 둘 다
열거하지 않는다.

## 3. 단일 H100 실행

중간 크기 기준 모델은 현재 H100용 설정을 그대로 사용한다.

```bash
conda activate sion-translate
sion-train --config configs/sion_data_fit.yaml
```

1.3B 프리셋은 다음과 같다.

```bash
sion-train --config configs/sion_1_3b.yaml
```

프로세스를 명시적으로 GPU 하나에 고정하려면 Linux에서 다음처럼 실행할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 sion-train --config configs/sion_data_fit.yaml
```

GPU 하나에서는 설정의 `parallel_strategy`와 관계없이 런타임 전략이 `single`로
해석된다. H100에서는 `training.precision: bf16`을 유지한다. DDP/single 경로는
FP32 파라미터를 유지하면서 CUDA autocast로 BF16 연산을 사용하고, FP16과 달리
GradScaler를 쓰지 않는다.

## 4. 한 노드의 다중 H100 실행

8-GPU 노드 예시는 다음과 같다.

```bash
conda activate sion-translate
torchrun --standalone --nproc-per-node=8 \
  -m sion_translate.cli.train \
  --config configs/sion_data_fit.yaml
```

GPU 수가 2개 또는 4개면 `--nproc-per-node`만 실제 GPU 수에 맞춘다. 각 프로세스는
로컬 GPU 하나를 사용하며 NCCL로 동기화한다. 유효 배치는 다음과 같다.
단, GPU 수만 맞춘다고 모든 모델이 들어가는 것은 아니다. 현재 preflight 기준
8B+EMA는 최소 4×80GB, 32.08B+EMA는 최소 16×80GB가 필요하다.

```text
effective_batch =
  batch_size_per_gpu × GPU 수 × gradient_accumulation_steps
```

예를 들어 `sion_data_fit.yaml`의 GPU당 배치 32, accumulation 1을 H100 8장에
쓰면 update당 256 sequence다. GPU 수를 늘릴 때 학습 동역학을 유지하려면 이 식의
값을 고정하고 accumulation이나 GPU당 배치를 조절한다. 처리량 한계를 찾는
실험에서는 학습률을 바꾸기 전에 먼저 같은 effective batch로 비교한다.

## 5. 여러 노드의 H100 실행

모든 노드에서 저장소, Conda 환경, 원천 데이터 또는 공유 indexed dataset,
토크나이저가 동일해야 한다. 두 노드에 H100 8장씩 있을 때 각 노드에서
`NODE_RANK`만 다르게 실행한다.

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank=0 \
  --master-addr=10.0.0.10 \
  --master-port=29500 \
  -m sion_translate.cli.train \
  --config configs/sion_data_fit.yaml
```

두 번째 노드는 `--node-rank=1`로 실행한다. `master-addr`은 rank 0 노드에서 다른
노드가 접근 가능한 주소여야 한다. 먼저 100~500 step의 별도 output 디렉터리로
NCCL, 저장 장치, 처리량을 검증한 뒤 전체 작업을 시작한다. 여러 노드에서
`global_tokens_per_second`가 거의 늘지 않으면 GPU를 더 붙이는 것보다 데이터
스토리지와 네트워크 병목을 먼저 해결한다.

## 6. DDP와 FSDP2 선택

현재 원칙은 “GPU 한 장에 모델·optimizer·활성값이 들어가면 DDP, 들어가지 않을
때만 FSDP2”다.

| 조건 | 권장 전략 | 현재 프리셋 | 이유 |
|---|---|---|---|
| 단일 GPU | 런타임 `single` | 모든 설정 | 분산 wrapper가 필요 없음 |
| 각 GPU에 전체 모델이 들어감 | `ddp` | `sion_data_fit`, `sion_1_3b` | parameter all-gather가 없어 보통 더 빠름 |
| 전체 모델/optimizer가 한 GPU에 안 들어감 | `fsdp2` | `configs/aspirational/sion_8b` | layer 단위 parameter·gradient·optimizer state sharding |

`configs/aspirational/sion_8b.yaml` 은 현재 코퍼스(0.357B 토큰/epoch)로 학습해서는
안 됩니다. 이유와 필요한 데이터 규모는 `configs/aspirational/README.md` 를 보십시오.

모든 CUDA 전략은 먼저 meta device에 저장 공간 없는 모델을 만들고 FP32 master
parameter, gradient, AdamW 1·2차 moment, 선택적 FP32 EMA를 합산한다. 따라서
single/DDP의 과대 모델도 실제 GPU constructor OOM 전에 명확한 용량 오류로
종료한다. BF16 layer all-gather, activation, 커널 임시공간, CUDA context를 위해
GPU VRAM의 51% 이상을 남기는 보수적 gate다. 이 계산에서 8B는 2×80GB, 32B는
8×80GB를 명시적으로 거부하며 각각 최소 4장과 16장을 안내한다. 32B를 더 적은
GPU로 실행하려면 EMA 비활성화만으로 충분하다고 가정하지 말고 optimizer/offload
정책까지 별도 구현·검증해야 한다.

H100에서는 `precision: bf16`과 `fsdp_reduce_dtype: bf16`을 사용한다. FSDP2의
`reshard_after_forward: true`는 메모리를 줄이는 대신 다음 backward/forward 전에
all-gather 비용을 늘린다. 80GB에서 여유가 확인된 경우에만 `false`를 비교하고,
peak allocated가 안전 범위인지 확인한다.
사후학습이 직접 호출하는 `generate()`와 `sample()`도 FSDP2의 공개
`register_fsdp_forward_method`에 등록되므로 root parameter all-gather를
우회하지 않는다. 토큰별 종료 조건은 rank 전체에 `MIN` 합의되므로 한 rank만
먼저 nested decoder collective를 빠져나가지 않는다. 먼저 끝난 rank는 EOS를
반복하며 다른 rank와 같은 횟수로 디코더를 호출하다가 전 rank가 함께 종료한다.
rank별 batch padding 길이에서 계산된 생성 한도 역시 시작 시 `MAX`로 통일한다.

DDP는 `gradient_as_bucket_view`를 사용한다. BATS는 SFT의 label 기반 보조
손실에서는 사용되지만 MRT candidate scoring의 label-free forward에서는
사용되지 않는다. 따라서 BATS+사후학습 구성은
`find_unused_parameters=true`로 시작하고, parameter 사용 집합이 고정된
구성은 `false`를 사용한다. 현재 모든 DDP 구성은 `static_graph=false`다.
`SionOutput` dataclass가 PyTorch pytree로 등록되어 있지 않아 PyTorch 2.8의
static-graph 첫 backward에서 delayed all-reduce가 누락될 수 있기 때문이다.

`torch.compile`은 현재 `sion_data_fit`과 `sion_1_3b`에서 켜져 있다. 첫 step들의
컴파일 시간은 steady-state 처리량에서 제외한다. graph break나 컴파일 실패가
나면 원인 로그를 보존한 뒤 `compile: false`로 동일 조건 비교를 수행한다.

## 7. Telemetry 읽는 법

학습은 `training.log_every`마다 rank 전체를 집계한 JSON 한 줄을 출력하고 같은
값을 TensorBoard의 `train/*`에 기록한다.

| 필드 | 의미 | 판단 |
|---|---|---|
| `global_tokens_per_second` | 모든 rank가 처리한 target label token/s(사후학습은 objective가 보고한 scored token/s) | 속도 비교의 주 지표. 같은 source·target 길이 분포에서 비교 |
| `seconds_per_step` | rank 평균 wall time/step | 증가하면서 token/s가 감소하면 병목 또는 길이 분포 변화 확인 |
| `data_wait_fraction` | step 시간 중 다음 batch를 기다린 비율 | 지속적으로 0.10~0.15 이상이면 CPU 전처리·스토리지·worker 병목 의심 |
| `cuda_allocated_gib` | 로그 시점에 tensor가 실제 점유한 최대 rank 메모리 | 현재 live tensor 규모 |
| `cuda_reserved_gib` | PyTorch allocator가 확보한 최대 rank 메모리 | allocated보다 큰 것 자체는 leak가 아님 |
| `cuda_peak_allocated_gib` | 직전 로그 구간의 최대 실제 할당 | OOM 여유 판단의 핵심 |
| `cuda_peak_reserved_gib` | 직전 로그 구간의 최대 예약량 | allocator 여유와 fragmentation 참고 |
| `loss`, `auxiliary_loss` | 주 손실과 보조 손실 | 처리량을 높여도 학습이 깨지지 않는지 확인 |
| `grad_norm` | clipping 전후 학습 안정성 관찰값 | 급격한 폭증·비유한 값이면 설정을 되돌림 |
| `reward_cpu_seconds` | 사후학습 문자열 decode·복합 reward CPU 시간 | 길어지면 tokenizer·chrF·구조 검사 CPU 병목 |
| `reward_wait_seconds` | GPU candidate scoring 뒤 reward worker를 기다린 시간 | 0에 가까울수록 CPU 작업이 GPU 작업 뒤에 숨겨짐 |
| `reward_overlap_seconds` / `reward_overlap_fraction` | CPU reward worker와 candidate scoring의 실제 시작·종료 구간 교집합과 CPU 시간 대비 비율 | overlap은 높고 wait는 낮은 상태가 목표 |
| `candidate_scoring_seconds` | 사후학습 후보 점수화 시간(CUDA는 event, CPU는 monotonic wall clock) | reward 시간과 함께 MRT 병목 분리 |
| `reward_input_transfer_seconds` | reward 입력을 CPU로 옮긴 시간 | 비정상적으로 크면 동기화·전송 병목 확인 |

CUDA peak 통계는 로그 구간마다 reset된다. 따라서 한 줄만 보지 말고 최소 수백
step의 p50/p95를 비교한다. GPU 사용률은 코드 telemetry가 아니므로 별도로
`nvidia-smi dmon` 또는 클러스터 모니터링에서 함께 기록한다.

해석 순서는 다음과 같다.

1. `data_wait_fraction`이 높으면 `data.num_workers`, CPU 코어 할당, 로컬 NVMe,
   indexed dataset 위치를 먼저 확인한다. 학습 loader는 worker당 prefetch 4와
   pinned memory를 사용한다.
2. data wait가 낮은데 GPU 사용률과 VRAM이 모두 낮으면 GPU당 배치를 올리고
   accumulation을 내려 effective batch를 유지한다.
3. VRAM peak는 높은데 사용률이 낮으면 지나친 activation checkpointing,
   너무 잦은 분산 통신, 짧은 sequence bucket을 확인한다.
4. 다중 GPU에서 GPU 수만큼 token/s가 늘지 않으면 DDP/FSDP 통신, 노드 간
   네트워크, 공유 스토리지의 read throughput을 분리 측정한다.

## 8. H100 배치·속도 탐색

현재 `sion_data_fit.yaml`은 이전 GPU당 배치 16에서 관찰된 약 56GB 사용량의
headroom을 이용해 GPU당 배치 32, accumulation 1, BF16, compile을 사용한다.
하지만 문장 길이 분포와 실험 모듈에 따라 peak가 달라지므로 이 값은 실측 시작점이지
보장값이 아니다.

다음 순서로 별도 run 디렉터리에서 300~1,000 step씩 비교한다.

1. 설정과 seed, 데이터 지문, GPU 수를 고정한다.
2. GPU당 배치를 16 → 24 → 32처럼 올린다.
3. effective batch를 유지해야 하면 accumulation을 반대로 낮춘다.
4. 각 실험의 steady-state `global_tokens_per_second`, `data_wait_fraction`,
   peak allocated/reserved, GPU 사용률을 기록한다.
5. peak reserved를 H100 전체 VRAM 끝까지 밀지 말고 평가·export·길이가 긴 batch를
   위한 여유를 둔다.

프리셋의 batch 32는 “들어간다고 보장된 값”이 아니라 이전 약 56GB 관측에서
출발한 가설이다. 전체 run 전에 반드시 100~500 step capacity probe를 먼저
통과시킨다.

`data.pad_to_multiple_of: 8`은 동적 batch를 Tensor Core 친화적인 길이로
올림 padding한다. `data.bucket_size`를 크게 하면 비슷한 길이가 더 잘 묶이지만
CPU 메모리와 shuffle 범위가 바뀐다. 단순히 batch 수만 보지 말고 token/s를
비교한다.

## 9. OOM 대응

### 사전학습 OOM

아래 순서로 하나씩 조절한다.

1. `training.batch_size_per_gpu`를 낮춘다.
2. 같은 effective batch가 필요하면 `gradient_accumulation_steps`를 올린다.
3. 큰 모델이면 `model.gradient_checkpointing: true`를 켠다.
4. 한 GPU에 전체 모델/optimizer가 들어가지 않으면 DDP 대신
   `parallel_strategy: fsdp2`를 사용한다.
5. 실제 요구 길이가 512보다 짧다면 `data.max_source_length`와
   `data.max_target_length`를 줄인다.

### 사후학습 OOM

MRT는 source마다 여러 후보를 생성하고 다시 점수화하므로 SFT보다 메모리를 훨씬
많이 쓸 수 있다. 다음 필드를 순서대로 낮춘다.

1. `posttraining.batch_size_per_gpu`
2. `posttraining.candidate_micro_batch` — 이미 1이면 더 낮출 수 없다.
3. `posttraining.samples_per_source` — 최소 2
4. `posttraining.max_new_tokens`
5. `posttraining.validation_num_beams`와 `eval_batch_size_per_gpu`

`candidate_gradient_checkpointing: true`를 유지하면 후보 scoring의 큰 activation을
backward 때 재계산한다. 속도를 위해 끄려면 사후학습 peak VRAM을 먼저 측정한다.

현재 단계 전환 코드는 사전학습 loader의 persistent worker를 종료하고 loader,
sampler, collator 참조를 제거하며, GC와 `torch.cuda.empty_cache()`를 실행한 뒤
allocated/reserved 전후 값을 출력한다. 사후학습 시작 전에 reserved가 줄어들지
않아도 allocated가 줄었다면 PyTorch allocator cache일 수 있다. allocated가
그대로 높게 남으면 출력 로그와 최소 재현 설정을 보존해 실제 live tensor를
조사한다.

## 10. 7개 형식 내보내기

학습 중간의 `exports/best`와 `exports/latest`에는 CPU 양자화와 중복 full-state
수집으로 H100을 오래 세우지 않도록 EMA 사용 시 `model_ema.pt` 하나(EMA를
끄면 `model.pt`)만 저장한다. raw 학습 가중치는 checkpoint에 남는다. 사전학습과
선택적 사후학습이 모두 끝나면 CLI가 선택된 best
가중치에서 아래 일곱 형식을 `final_export_formats` 순서로 한 번 생성하고, 실제
loader 검증이 모두 성공한 디렉터리만 `exports/best`에 원자적으로 교체한다.
EMA가 켜졌다면 이 최종 일곱 형식은 복원된 best EMA 가중치 기준이다.

실패한 형식만 다시 만들거나 다른 출력 폴더로 수동 변환할 때는 다음을 쓴다.

```bash
sion-export \
  runs/sion-data-fit/posttrain/exports/best/model_ema.pt \
  --output runs/sion-data-fit/recovered-export \
  --tokenizer artifacts/tokenizer/sion.model \
  --token-features artifacts/tokenizer/token_features.npz \
  --language-pair ko ja
```

EMA가 꺼진 run은 `model.pt`를 쓴다. strict 최종 변환이 실패했더라도 이전의
중간 best 디렉터리를 원자적으로 보존하므로, EMA 기본 구성의 복구 원본은
대개 `model_ema.pt`다.

`--formats`를 생략하면 다음 일곱 항목을 모두 만든다.

| 이름 | 산출물 | 용도 |
|---|---|---|
| `fp32` | `model_ema.pt`(EMA run) 또는 `model.pt` | 최대 호환·기준 가중치 |
| `fp16` | `model_fp16.pt` | FP16 저장·추론 |
| `bf16` | `model_bf16.pt` | H100 BF16 추론·재사용 |
| `int8` | `model_int8.pt` | TorchAO INT8 |
| `int4` | `model_int4.pt` | TorchAO 또는 portable packed INT4 |
| `gguf_q4_k_m` | `model-q4_k_m.gguf` | Sion mixed K-quant 저장·교환 |
| `transformers` | `transformers/` | config, custom AutoClass 코드, safetensors, tokenizer |

다국어 모델은 언어 방향을 반복한다.

```bash
sion-export \
  runs/multilingual/posttrain/exports/best/model_ema.pt \
  --output runs/multilingual/recovered-export \
  --tokenizer artifacts/tokenizer-multilingual/sion.model \
  --token-features artifacts/tokenizer-multilingual/token_features.npz \
  --language-pair ko ja \
  --language-pair en ru
```

위 명령은 각 edge의 양방향을 기본으로 기록한다. 실제 학습이
`data.bidirectional: false`였다면 `--unidirectional`을 반드시 추가해
나열한 `SOURCE→TARGET` 방향만 허용한다. native manifest, GGUF metadata,
Transformers config/tokenizer가 같은 `translation_directions` 계약을 공유하며
학습하지 않은 역방향 요청은 추론 전에 거부된다.

INT4는 `--int4-backend auto`가 기본이며 TorchAO가 실패하면 packed INT4로
전환한다. 강제하려면 `--int4-backend torchao` 또는 `packed`를 사용한다.

GGUF는 실제 `MOSTLY_Q4_K_M` 메타데이터와 mixed Q4_K/Q5_K/F16 tensor를 가진
컨테이너다. 그러나 Sion은 커스텀 encoder-decoder 구조이고 stock llama.cpp에는
이 architecture 실행 backend가 없다. 따라서 현재 GGUF는 저장·교환 산출물이며,
llama.cpp에서 바로 실행되는 모델이라고 배포하면 안 된다.

Transformers 디렉터리는 표준 `save_pretrained` 구조와 safetensors를 사용하며
custom AutoClass 코드를 포함한다. 로컬 로딩은 다음처럼 검증한다.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

path = "runs/sion-data-fit/posttrain/exports/best/transformers"
tokenizer = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True,
    src_lang="ko",
    tgt_lang="ja",
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda")
model.eval()
encoded = tokenizer("번역할 문장", return_tensors="pt").to("cuda")
generated = model.generate(**encoded, num_beams=4, max_new_tokens=256)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
```

## 11. 산출물 무결성 검증

`export_manifest.json`은 weight state, metadata compatibility, tokenizer,
각 파일 또는 Transformers 디렉터리의 크기와 SHA256을 기록한다. 변환 직후
모든 성공 artifact를 실제 loader로 열어 보는 검증을 실행한다.

```bash
python -c "import json,sys; from sion_translate.training.export import validate_export_directory; r=validate_export_directory('runs/sion-data-fit/posttrain/exports/best'); print(json.dumps(r, ensure_ascii=False, indent=2)); sys.exit(0 if r['valid'] else 1)"
```

검증이 실패한 디렉터리는 업로드하지 않는다. 특히 아래를 확인한다.

- 일곱 format 모두 `status: ok`, 검증 결과 `valid: true`
- 원본 state와 모든 artifact의 `artifact_set_id` 일치
- tokenizer와 `token_features.npz`의 SHA256·크기가 native/HF sidecar와 일치
- Transformers config의 `language_pairs`, `translation_directions`,
  `revision_trained`, vocab, pad ID 일치
- bundled remote-code config/tokenizer/model의 실제 import 성공과 모든
  safetensors key·shape 일치
- 재변환 뒤 파일/디렉터리 hash가 manifest와 일치

최종 H100 보고에는 commit, config 전체, 데이터 fingerprint, GPU·노드 수,
PyTorch/CUDA/driver 버전, 시작·종료 시각, 처리 step, token/s p50/p95,
data-wait p50/p95, peak VRAM, best 검증 지표, export 검증 결과를 함께 남긴다.
