# GPU 서버 시작 안내

`sion_translate.zip` 하나에 실행 코드와 이번 학습용 코퍼스가 함께 들어 있습니다.
압축을 푼 뒤 의존성을 설치하고 `easy_run.py`만 실행하면 토크나이저 학습부터
사전학습과 MRT 사후학습까지 순서대로 진행됩니다.

## 1. 업로드 전에 무결성 확인

배포자가 함께 전달한 ZIP SHA-256과 서버에서 계산한 값이 같아야 합니다.

```bash
sha256sum sion_translate.zip
python3 -m zipfile -t sion_translate.zip
unzip sion_translate.zip
cd sion_translate
python3 scripts/package_gpu_bundle.py verify-tree .
```

`verify-tree`는 모든 파일의 SHA-256, 파일 집합, Git commit/tree를
`PACKAGE_MANIFEST.json`과 `SHA256SUMS`에 대조합니다. 하나라도 다르면 설치하거나
학습하지 말고 ZIP을 다시 업로드하십시오.

ZIP에는 다음이 포함됩니다.

- Git이 추적하는 소스·설정·테스트·문서
- `data/` 바로 아래의 학습 JSONL 51개
- 학습에서 격리된 `data/evaluation_only/`

품질 문제로 제외한 `data/excluded/`, 과거 `artifacts/`, `runs/`, 체크포인트,
가상환경과 캐시는 포함되지 않습니다. 서버에서 현재 코드와 데이터로 새로 만듭니다.

## 2. GPU 환경과 패키지 설치

Python 3.11 또는 3.12와 CUDA 지원 PyTorch 2.8 이상이 설치된 NVIDIA GPU 이미지를
권장합니다. A100·H100 모두 같은 진입점을 사용합니다. PyTorch가 없다면 임의의
CUDA wheel을 추측하지 말고 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
서버 환경에 맞는 명령을 먼저 실행하십시오.

```bash
nvidia-smi
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,export,hangul]"
python3 - <<'PY'
from importlib.metadata import version
import torch
torch_version = tuple(map(int, version("torch").split("+")[0].split(".")[:2]))
assert torch_version >= (2, 8), f"PyTorch 2.8 이상이 필요합니다: {torch.__version__}"
assert torch.cuda.is_available(), "CUDA 지원 PyTorch가 아닙니다"
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("NCCL:", torch.distributed.is_nccl_available())
PY
```

여러 GPU를 쓸 때는 NCCL이 필수입니다. `easy_run.py`도 전처리 전에 이를 검사합니다.

## 3. 실행

```bash
python3 easy_run.py
```

설치가 끝난 뒤 필요한 학습 명령은 이것 하나입니다. 자동 실행기는 다음을 수행합니다.

1. CUDA/NCCL과 보이는 모든 GPU를 확인합니다.
2. JSONL shard 구조를 빠르게 검사합니다.
3. 여유가 충분하면 원천 데이터와 전처리 산출물을 `/dev/shm`에서 처리합니다.
4. 현재 split 정책으로 SentencePiece와 MorphoScript sidecar를 생성·검증합니다.
5. 품질 필터, 중복 제거와 누수 방지 split으로 indexed dataset을 만듭니다.
6. GPU 중 가장 작은 VRAM과 가장 낮은 BF16 능력에 맞춰 공통 설정을 선택합니다.
7. SFT 사전학습을 실행하고 체크포인트를 저장합니다.
8. best SFT 모델에서 MRT 사후학습을 실행하고 별도로 저장합니다.

대화형 터미널이고 `tmux`가 이미 설치돼 있으면 체크아웃별 세션을 만듭니다.
Slurm, `nohup`, 컨테이너 또는 비대화형 SSH에서는 현재 프로세스로 그대로 실행합니다.
명시적으로 끄려면 다음처럼 실행합니다.

```bash
SION_NO_TMUX=1 python3 easy_run.py
```

## 4. GPU별 동작

- 모든 GPU가 native BF16을 지원하면 BF16을 사용합니다.
- 하나라도 BF16을 지원하지 않으면 자동 설정은 FP16으로 통일합니다.
- VRAM이 작을수록 micro-batch를 1까지 낮추고 activation checkpointing을 켭니다.
- 사후학습은 후보 생성 변동폭을 고려해 기본 micro-batch 1을 사용합니다.
- 다중 GPU는 DDP 또는 필요 시 FSDP2를 사용합니다.
- `torch.compile`은 드라이버·컨테이너 조합별 검증 없이 자동으로 켜지 않습니다.

`configs/sion_1_3b.yaml`, `sion_8b.yaml`, `sion_32b.yaml`은 일반 자동 설정이 아니라
주석에 적힌 80GB급 GPU 수를 전제로 한 용량 기준 설정입니다. `easy_run.py`는 기본
`sion_translate.yaml`과 자동 설정을 사용하므로 이 파일들을 자동 선택하지 않습니다.

## 5. 결과와 재개

```text
runs/auto/
├── pretrain/
│   ├── checkpoints/
│   └── exports/best/
└── posttrain/
    ├── checkpoints/
    └── exports/best/
artifacts/
├── tokenizer/
└── dataset/
```

사후학습이 활성화돼 있으므로 최종 추론 모델은
`runs/auto/posttrain/exports/best/`입니다. 중단 뒤 같은 명령을 다시 실행하면 각
단계의 `checkpoints/latest`에서 재개합니다. 인스턴스를 삭제하기 전에 `runs/`와
`artifacts/tokenizer/`를 내려받으십시오.

실제 A100/H100에서의 최종 smoke test는 서버의 드라이버, CUDA 이미지, GPU 수와
VRAM에 의존합니다. 오류가 나면 전체 traceback, `nvidia-smi`, 위 PyTorch 확인
출력을 보존하십시오.
