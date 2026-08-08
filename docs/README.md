# 어느 문서를 읽어야 하는가

문서가 여럿이고 여러 개가 스스로를 시작점이라고 부릅니다. 질문별로 갈라
놓았습니다. **하나만 읽어야 한다면 자기 질문에 해당하는 것 하나입니다.**

## 저장소 최상위

| 문서 | 답하는 질문 | 대상 |
|---|---|---|
| [`../README.md`](../README.md) | 이 프로젝트는 무엇이고 어떻게 설치·실행하는가 | 처음 오는 사람 |
| [`../START-HERE.md`](../START-HERE.md) | GPU 서버에서 `sion_translate.zip` 을 어떻게 돌리는가 | 학습을 실행하는 사람 |
| [`../how_to_run.txt`](../how_to_run.txt) | 같은 내용의 평문판 | 마크다운 렌더러가 없는 터미널 |
| [`../MODEL_CARD.md`](../MODEL_CARD.md) | 공개된 가중치는 무엇을 하고 무엇을 못 하는가 | 모델을 쓰려는 사람 |
| [`../POSTTRAINING.md`](../POSTTRAINING.md) | MRT 사후학습은 어떻게 설계됐는가 | 보상을 손볼 사람 |

`START-HERE.md` 와 `how_to_run.txt` 는 **같은 절차의 두 표기**입니다. 둘 중
하나만 보면 되고, 절차를 고칠 때는 둘 다 고쳐야 합니다.

## `docs/`

| 문서 | 답하는 질문 |
|---|---|
| [`retraining-runbook.md`](retraining-runbook.md) | `easy_run.py` 없이 단계별로 직접 돌리려면 어떤 명령을 어떤 순서로 치는가 |
| [`foundation-pretraining.md`](foundation-pretraining.md) | 단일어 span-corruption 단계는 무엇이고 언제 도는가 |
| [`sentencepiece-sigsegv.md`](sentencepiece-sigsegv.md) | 토크나이저 학습 SIGSEGV의 원인·재현·운영 방어는 무엇인가 |
| [`H100_TRAINING.md`](H100_TRAINING.md) | 다중 GPU 에서 어떤 병렬 전략을 고르는가 |
| [`QUALITY_OVERHAUL.md`](QUALITY_OVERHAUL.md) | 방향 격차와 토큰 노출은 어떻게 진단했는가 |
| [`corpus-gaps.md`](corpus-gaps.md) | 어떤 분야 데이터가 비어 있는가 |
| [`DATA_EXPANSION_PLAN.md`](DATA_EXPANSION_PLAN.md) | 그 빈 곳을 어디서 채울 것인가 |
| [`COMPARISON.md`](COMPARISON.md) | 다른 번역기와 어떻게 비교하는가 |

## 저장소에 없는 문서

* `PROJECT_ROAST.md` — 내부 감사. 원천 행을 파일명·행 번호로 지목하는데 그
  shard 는 재배포하지 않으므로 `.gitignore` 에 있습니다. 코퍼스를 가진
  로컬에서만 의미가 있습니다.
* `SERVER-OPS.md` — GPU 패키지에만 들어가는 운영 문서입니다.

## 설정 문서

* [`../configs/aspirational/README.md`](../configs/aspirational/README.md) —
  왜 어떤 preset 은 `configs/` 밖에 있는가. 보유 데이터가 그 용량을 지탱하지
  못한다는 계산이 표로 있습니다.
