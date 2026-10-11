# A100/H100 80GB 학습 ZIP 실행 안내

이 번들은 Linux x86_64의 A100 80GB 또는 H100 80GB 한 장을 기준으로 합니다.
모델은 약 2.73억 파라미터이며, 사전학습 3회와 SFT 3회 후 MRT를 최대
100,000 optimizer step 실행합니다. MRT 유효 배치는 16이므로 약 160만 문장을
처리합니다. 사전학습/SFT의 검증 조건에 따라 일찍 종료될 수 있습니다.

ZIP에는 코드, 설정, 토크나이저, 전처리된 번역/사전학습 데이터와 평가 자료가
들어 있습니다. Python/CUDA 패키지는 포함하지 않으며 설치 중 인터넷으로 받습니다.
GPU 서버의 NVIDIA 드라이버는 미리 설치되어 있어야 합니다.

## 1. 업로드와 압축 해제

ZIP과 함께 제공된 `.sha256` 파일을 같은 폴더에 업로드합니다. 학습 출력이 서버
종료 후에도 보존되는 디스크에서 다음 명령을 실행합니다.

```bash
sha256sum -c sion_translate-a100-h100-80gb-mrt100k.zip.sha256
unzip sion_translate-a100-h100-80gb-mrt100k.zip
cd sion_translate
```

준비 데이터만 압축 해제 후 약 20GiB입니다. 이외에 패키지, 설치 캐시,
단계별 체크포인트와 모델 출력 공간이 필요합니다. 진단에서 체크포인트 한 개는
약 4.08GiB였으며 여러 단계·세대가 보관됩니다. 데이터 크기만 보고 디스크를
할당하지 말고 이 출력 공간도 확보하세요.

## 2. 환경 설치

```bash
python3 setup_gpu.py
```

파일 무결성을 검증한 뒤 `.venv`에 CPython 3.11과 저장소가 고정한
PyTorch 2.10.0+cu128 및 의존성을 설치합니다. 시스템 Python 패키지를 바꾸지
않으며, 학습은 시작하지 않습니다. 마지막에 GPU와 주요 패키지 버전을 확인합니다.
Ubuntu/Debian에서 `venv`/`ensurepip`가 없다는 오류가 나면 서버의 패키지 관리자로
`python3-venv`를 먼저 설치해야 합니다.

설치가 실패하면 오류 원인을 해결하고 같은 명령을 다시 실행합니다. 기존 `.venv`는
Python 3.11 환경이어야 합니다. ZIP 안의 설정·소스·데이터를 직접 수정하면 무결성
검증에 실패하므로, 설정 변경은 원본 저장소에서 새 번들을 만드는 방식으로 합니다.

## 3. 학습 시작 또는 재개

```bash
.venv/bin/python easy_run.py
```

기본 `sion_translate.yaml`에 80GB 배치와 MRT 10만 스텝 제한이 이미 반영되어
있습니다. `--config`나 `--allow-local-checkout`을 붙이지 않습니다.
실행기는 CUDA 검사와 데이터 인증을 마친 후 사전학습 → SFT → MRT 순으로 진행합니다.
전처리 데이터가 완전하면 토크나이저와 인덱스를 다시 만들지 않습니다.

SSH 연결을 끊어도 유지해야 하는 일반 서버에서는 설치된 `tmux` 안에서 실행하세요.
실행기도 대화형 터미널에서 `tmux`가 있으면 세션을 사용합니다. `tmux`는 서버 자체의
종료나 클라우드 실행 제한을 연장하지 않습니다.

```bash
tmux new -s sion-training
.venv/bin/python easy_run.py
# 세션에서 빠져나오기: Ctrl+B, D
# 다시 접속: tmux attach -t sion-training
```

## 4. 저장 위치와 복구

```text
runs/modal-80gb/
├── foundation/checkpoints/       사전학습
├── foundation/exports/best/
├── pretrain/checkpoints/         SFT
├── pretrain/exports/best/
├── posttrain/checkpoints/        MRT
└── posttrain/exports/best/
```

각 단계는 `latest`, 검증을 통과한 `best`, 종료 시 `final` 체크포인트를
별도로 저장합니다. 체크포인트에는 가중치, optimizer, scheduler, EMA 및 학습 진행
상태가 포함됩니다. 추론용 모델 내보내기는 검증 조건을 통과해야 합니다.
시간 기반 저장은 마지막 저장 후 900초가 지난 다음 완료된 optimizer update에서
실행되며, 멈춘 커널이나 I/O에 대한 15분 절대 보장은 아닙니다.

중단 후에는 저장된 `runs/modal-80gb`를 유지한 채 같은 명령을 실행합니다.
실행기는 가장 뒤 단계의 호환되고 완전한 체크포인트를 찾아 재개합니다.

## 5. 시간 추정과 Modal의 차이

기존 짧은 진단 속도에 검증·저장 등 20~50%의 여유 시간을 가정하면, GPU 한 장으로
24시간 연속 실행할 때 A100 80GB는 약 61~77일, H100은 약 23~29일입니다.
이는 전체 학습의 실측이나 완료 보장이 아닙니다. 특히 MRT는 학습 전 가중치로
측정했으므로 SFT 후 생성 길이와 속도에 따라 크게 달라집니다. 마지막 생성 연산
최적화의 GPU 가속률은 아직 이 수치에 반영하지 않았습니다.

Modal에서는 ZIP만 올려도 장기 학습 앱이 자동 생성되지는 않습니다. 저장 루트를
영속 Volume에 연결하고, 서비스 실행 시간제한 안에서 종료·재개하는 실행 구성이
별도로 필요합니다. 저장소의 `modal_*` 진단 스크립트는 전체 장기 학습 실행기가
아닙니다. 이 ZIP은 이미 제공된 GPU 실행 환경 안에서 사용할 학습 페이로드입니다.
