# GPU 서버 시작 안내

2026-07-31 패키징. 이 압축본은 **git이 추적하는 파일만** 담고 있습니다.
코퍼스와 가중치는 들어 있지 않습니다.

## 세 줄 요약

```bash
pip install -e ".[dev,export,hangul]"
python easy_run.py
```

`easy_run.py` 하나가 전부 합니다. 다른 명령을 칠 필요가 없습니다.

## 함께 올려야 할 것

| 항목 | 크기 | 비고 |
|---|---:|---|
| `data/*.jsonl` | **약 2.26 GB** | 51개 파일, 8,978,338행. 이게 없으면 아무것도 못 합니다 |
| `data/evaluation_only/` | 작음 | FLORES-200. 학습에서 격리된 평가 전용 |

`data/excluded/`(약 0.66 GB)는 **올리지 마십시오.** 품질 문제로 제외한
원본 보관분이라 학습에 쓰이지 않습니다.

`artifacts/`, `runs/`, `checkpoints/`는 서버에서 새로 만들어집니다.

## easy_run.py가 하는 일

1. **tmux 세션 생성** (없으면 자동 설치) — 접속이 끊겨도 학습이 계속됩니다.
   분리는 `Ctrl+B, D`, 재접속은 `tmux attach -t sion`
2. **shard 키 검사** — 설정된 언어쌍으로 읽히지 않는 shard가 있으면 즉시 중단.
   그런 파일은 0문장을 내놓고 아무 말도 하지 않기 때문입니다
3. `/dev/shm` 여유가 있으면 코퍼스를 RAM 디스크로 복사
4. **토크나이저 학습** (없을 때) + 데이터셋 준비
5. **byte fallback 검사** — 코퍼스 표본에서 비율을 재고 상한을 넘으면 중단
6. GPU 개수만큼 분산 학습
7. 학습 뒤 **MRT 사후학습** (`posttraining.enabled` 기본 true)

2번과 5번이 이번에 추가된 관문입니다. 둘 다 조용히 실패하는 종류라서,
GPU 시간을 쓰기 전에 잡는 것이 요점입니다.

## 설정 요약

`sion_translate.yaml`이 이번 run의 설정이고 이미 맞춰져 있습니다.

| 항목 | 값 |
|---|---|
| 언어쌍 | kj→ko, kj→ja, kd→ko, kd→ja, jd→ko, jd→ja, ko→ja |
| `source_only_languages` | `[kj, kd, jd]` |
| `approximate_split` | `true` |
| 토크나이저 vocab | 48,000 (코퍼스 규모로 자동) |
| `input_sentence_size` | 0 (전량) |
| 모델 | `base(~200M)` — d_model 768 / enc16 / dec8 |
| 실험 모듈 | CoRe만 검증 대상. MorphoScript·TETM은 off |

`source_only_languages`를 빼면 표준 한국어를 요청해도 사투리가 나옵니다.
`approximate_split`을 끄면 홀드아웃 점수가 번역 품질을 재지 않습니다.

## 학습 중 볼 것

- **train/val loss.** train이 계속 떨어지는데 val이 따라오면 underfit이라
  모델을 키울 근거가 됩니다. 현재 200M은 양방향 target 토큰
  357,344,643/epoch에 잘 맞는 크기입니다
- **register loss.** 떨어지지 않으면 CoRe가 신호를 못 찾는 것이므로 끄십시오

## 평가할 때

자체 test split 점수는 신뢰하지 마십시오. 과거 근사 중복 유출로 부풀려져
있었습니다(test split chrF 77.50 대 실제 진단 61.79). `approximate_split: true`가
이번에는 완화하지만, 진짜 기준선은 격리해 둔 `data/evaluation_only/`입니다.

beam은 4를 쓰십시오. 실측에서 1→2→4가 77.28→77.36→77.50이고 16에서 반복
붕괴가 일어났습니다.

## 더 읽을 것

- [`docs/retraining-runbook.md`](docs/retraining-runbook.md) — 단계별 수동 절차,
  중간 산출물 확인, 되돌아볼 만한 실패 지점
- [`docs/corpus-gaps.md`](docs/corpus-gaps.md) — 아직 비어 있는 도메인과
  실데이터 확보처
- [`README.md`](README.md) — 언어쌍 설정 방법
