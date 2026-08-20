"""Canonical public paths for generated artifacts and training runs.

Compatibility is established by artifact metadata and content hashes, not by a
release-looking directory name.  Keep these paths stable for scripts and users.
"""

from __future__ import annotations

from sion_translate._version import MODEL_RELEASE_VERSION

DEFAULT_ARTIFACT_ROOT = "artifacts"
DEFAULT_TOKENIZER_MODEL = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/sion.model"
DEFAULT_TOKENIZER_FEATURES = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/token_features.npz"
DEFAULT_DATASET_DIRECTORY = f"{DEFAULT_ARTIFACT_ROOT}/dataset"
DEFAULT_RUN_DIRECTORY = "runs/auto"

# 단일어 사전학습(foundation) 입력과 그 tokenized 산출물. 병렬 데이터셋과
# 섞이면 안 되므로 경로를 나눕니다 — 같은 토크나이저를 쓰지만 record 규격도
# split 규칙도 다릅니다.
DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY = "data/corpus"
DEFAULT_FOUNDATION_DATASET_DIRECTORY = f"{DEFAULT_ARTIFACT_ROOT}/foundation_dataset"

# 배포 이름. foundation 단계 산출물은 번역 모델이 아니라 encoder-decoder
# 파운데이션 모델이고, 번역 단계는 거기서 파생된 것입니다. 두 산출물이 같은
# 이름으로 나가면 어느 가중치가 번역을 할 수 있는지 구분되지 않습니다.
FOUNDATION_RELEASE_NAME = "sion"
TRANSLATION_RELEASE_NAME = "sion_translate"
# 공개 모델 세대. Python 패키지는 semantic versioning을 사용하지만 모델 계보는
# 사용자가 보는 major.minor 이름으로 고정합니다. 번역 모델은 같은 세대의
# sion에서 파생되므로 새로 학습한 두 배포가 이 값을 함께 기록합니다.

# 실행 디렉터리 안의 단계별 하위 경로. `pretrain` 은 이미 번역 SFT 를 가리키는
# 이름이라, 그 앞 단계는 다른 이름을 씁니다.
FOUNDATION_STAGE_DIRECTORY = "foundation"
SUPERVISED_STAGE_DIRECTORY = "pretrain"
POSTTRAINING_STAGE_DIRECTORY = "posttrain"


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DATASET_DIRECTORY",
    "DEFAULT_FOUNDATION_DATASET_DIRECTORY",
    "DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY",
    "DEFAULT_RUN_DIRECTORY",
    "DEFAULT_TOKENIZER_FEATURES",
    "DEFAULT_TOKENIZER_MODEL",
    "FOUNDATION_RELEASE_NAME",
    "FOUNDATION_STAGE_DIRECTORY",
    "MODEL_RELEASE_VERSION",
    "POSTTRAINING_STAGE_DIRECTORY",
    "SUPERVISED_STAGE_DIRECTORY",
    "TRANSLATION_RELEASE_NAME",
]
