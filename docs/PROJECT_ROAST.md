# sion_translate 프로젝트 roast

감사 기준일: 2026-08-05

## 한 줄 판정

이 저장소는 **번역 연구를 수행할 수 있는 꽤 강한 실험 플랫폼**입니다. 하지만 현재
공개 가중치가 욕설·신음·관용구·일→한 품질 문제를 해결했다는 증거는 없습니다.
지금 새 GPU 학습을 시작하면 구조보다 먼저 오염된 정답을 더 비싸게 암기할 위험이
큽니다.

테스트 수와 번역 품질은 같은 지표가 아닙니다. 1,000개가 넘는 테스트는 checkpoint,
export, split, cache 같은 엔지니어링 회귀를 잘 막지만, `씨발 → 種まき`가 의미상
틀렸다는 사실은 잡지 못합니다.

## 이번 감사에서 바로 고친 것

- 릴리스 합의 없이 만든 버전형 경로를 제거했습니다. 공개 기본 경로는 다시
  `artifacts/{tokenizer,dataset}`과 `runs/auto`이며, 호환성은 경로명이 아니라
  tokenizer SHA-256·숫자 분리·언어 태그·dataset fingerprint로 검사합니다.
- 모든 preset이 같은 artifact를 공유하면서 서로 다른 언어쌍을 요구하던 계약을
  루트의 7개 언어쌍과 3개 source-only 언어로 통일했습니다.
- meta-device materialization 뒤 CoRe/TETM/BATS/evidence raw parameter가 쓰레기값으로
  남던 초기화 결함을 수정했습니다. 같은 seed의 baseline/CoRe가 공통 backbone과
  gate-zero logits를 정확히 공유하도록 RNG 계약도 고정했습니다.
- 수학적으로 uniform zero-loss 해가 있는 BATS coverage-only 설정과 SiTU-GLU를 기본
  ablation에서 껐습니다. 루트는 CoRe 하나만 켜며, 용량 preset은 깨끗한 baseline,
  `debug.yaml`만 all-on smoke 용도입니다.
- 숫자 보존을 reference가 아니라 source와 비교하고, `1 → 1, 999`처럼 원래 숫자를
  남긴 채 새 숫자를 추가하는 발명도 집계하도록 고쳤습니다.
- YAML 중복 키와 `trainng:` 같은 최상위 오타를 학습 전에 거부합니다.

## P0: 새 학습을 막아야 하는 문제

### 1. 정답 데이터가 핵심 실패를 직접 가르칩니다

실제 원천 데이터에서 확인된 예시는 다음과 같습니다.

- `data/data10.jsonl:1562254` — `씨발...`을 `種まき...`로 번역
- `data/data10.jsonl:1562912` — 반복된 `씨발`을 반복된 `種まき`로 번역
- `data/data10.jsonl:1496551` — `같은 값이면 다홍치마`를 `同じ値段なら紅スカート`로 직역
- `data/data1.jsonl:802004` — 닮았다는 뜻의 `붕어빵`을 음식 `たい焼き`로 번역
- `data/synthetic_dialect_ko.jsonl:1019` — 위 오역이 합성 방언 데이터로 증식

현 `assess_pair`는 이 대표 오역들을 `accepted=True`, `score=100`, 경고 없음으로
통과시킵니다. 길이·문자 비율·반복 검사는 의미 등가성, 욕설 강도, 관용 현지화를
판단하지 않기 때문입니다. 휴리스틱 감사에서는 주요 한국어 욕설 포함 1,025행 중
85행에 `種まき/種の足`, 65행에 의심스러운 `犬` 직역이 관측됐습니다. 이 수치는
자동 삭제 기준이 아니라 사람 검수 queue를 만들 근거입니다.

**판정:** corpus 정제·재번역·사람 검수 없이 재학습 금지.

### 2. 개선 가중치와 개선 측정이 없습니다

현재 변경은 새 학습 파이프라인입니다. GPU에서 새 가중치를 학습하지 않았고,
표현 challenge 결과도 생성되지 않았습니다. 공개 가중치의 기존 측정은 ko→ja chrF
59.81, ja→ko 49.87로 약 10점 차이입니다. 코드가 바뀌어도 공개 가중치는 바뀌지
않습니다.

**판정:** “해결된 모델”이 아니라 “해결 후보를 검증할 파이프라인”이라고만 표현.

### 3. artifact 생성은 transaction이 아닙니다

SentencePiece는 live 경로에 `.model/.vocab`을 먼저 쓰고 feature와 metadata를 나중에
씁니다. 두 Slurm 작업이 동시에 실행될 때 막는 artifact-root lock도 없습니다.
RAM→disk publish 역시 tokenizer와 dataset을 별도 rename합니다. 중단·동시 실행 시
서로 다른 세대가 섞일 수 있습니다.

여러 `runs/*`가 한 tokenizer를 공유하므로 현재 run만 보고 자동 이동하는 것도
안전하지 않습니다. 그래서 이번 교정은 불일치 artifact를 자동 이동하지 않고
fail-fast합니다.

**필요한 해법:** interprocess lock, staging generation directory, 전체 manifest 검증,
한 번의 atomic current-pointer 전환.

### 4. resume identity가 실험 의미를 다 담지 않습니다

checkpoint identity는 모델·tokenizer·dataset·loader의 큰 축은 묶지만 다음은
빠뜨립니다.

- SFT learning rate, Adam betas/epsilon, weight decay, precision, EMA decay
- MRT reward weights, temperature, sampling, preference, roundtrip, post LR

reward 정의를 바꿔도 과거 `validation_reward` best와 숫자를 비교하고 optimizer state를
이어받을 수 있습니다. 또한 `resolved_config.json`은 resume identity 검증 전에
덮어써져, 실패한 재개 시도도 원래 provenance를 훼손할 수 있습니다.

**필요한 해법:** stage별 objective/optimizer identity, 검증 성공 뒤 provenance의 atomic
publish, 의도적 migration을 위한 별도 명시 옵션.

## P1: 품질 주장을 막는 문제

### 5. expressive 데이터 18쌍은 anchor이지 corpus가 아닙니다

검토 seed 30쌍 중 학습은 18쌍, challenge는 12쌍입니다. 학습 shard는 synthetic로
표시돼 기본 가중치 0.5를 받습니다. 대규모 상충 데이터에 저가중치 18쌍을 넣는다고
욕설·관용구가 교정되지는 않습니다.

`intensity`, `register`, `localization_strategy` metadata와 `quality_score`도 현재 batch와
loss에 들어가지 않습니다. annotation은 보존·감사용이지 학습 신호가 아닙니다.

### 6. challenge는 독립 holdout이 아닙니다

seed 내부 train/challenge exact 중복은 막지만 전체 원천 corpus와 대조하지 않습니다.
`세 살 버릇 여든까지 간다`, `호랑이도 제 말 하면 온다` 같은 challenge 표현은 기존
말뭉치에 동일·유사 대응이 있습니다. 양방향 24 case도 독립 의미쌍은 12개뿐이고
category/direction당 4개입니다.

**판정:** 회귀 smoke set으로는 유용하지만 누출 없는 품질 benchmark라고 부르지 않기.

### 7. 방향 수량 50:50은 방향 품질 50:50이 아닙니다

모든 물리 병렬쌍을 뒤집어 양방향 예제로 만듭니다. 한국어 원문을 번역해 얻은 일본어를
다시 “자연 일본어 원문”처럼 사용하면 번역투와 원래 오역이 ja→ko 정답으로 들어갑니다.
`original_direction` provenance는 기록되지만 sampling/loss 가중치에 사용되지 않습니다.
ja→ko 열세의 유력한 원인인데 단순 token 수 균형만으로는 해결되지 않습니다.

### 8. 최종 MRT가 방향별 SFT 개선을 되돌릴 수 있습니다

SFT는 방향별 NLL을 기록하지만 MRT best는 방향·category를 합친 reward 하나로 고릅니다.
ja→ko 또는 욕설 category가 후퇴해도 평균 reward가 오르면 최종 best가 될 수 있습니다.
chrF와 token F1 중심의 단일-reference reward는 올바른 관용 의역도 표면형이 다르면
벌합니다.

**필요한 해법:** ko→ja/ja→ko 별도 생성 지표, primary-direction no-regression gate,
expressive category gate, 다중 reference와 사람 MQM.

### 9. CoRe가 핵심 구어 범주를 충분히 감독하지 않습니다

register suffix heuristic에 걸리지 않은 fragment·욕설·감탄사는 class 0이 되지만 loss는
`labels > 0`만 학습합니다. class 0은 positive supervision이 없는데 모든 문장에는 4-way
예측 embedding을 조건으로 넣습니다. register/alignment/coverage loss도 현재 trainer의
주요 로그 목록에 빠져 있어 runbook이 “register loss를 보라”고 해도 관측하기 어렵습니다.

## P1: 운영·배포 문제

- 기본 export는 fp32/fp16/bf16/int8/int4/GGUF/Transformers 7종입니다. 32B FP32만 약
  128GB이고 optimizer·EMA·best/latest/final까지 합치면 TB급이 될 수 있지만 host RAM과
  disk preflight가 없습니다. GGUF는 현재 llama.cpp 실행용이 아닌 interchange 산출물입니다.
- dependency는 open lower bound이고 lockfile이 없습니다. run manifest에도 Git dirty state,
  dependency/CUDA/driver/GPU topology가 충분히 남지 않습니다.
- CI는 CPU 중심이며 실제 CUDA/NCCL/BF16/2-rank/Inductor/easy-run E2E를 검증하지 않습니다.
- 현재 Hugging Face 카드는 고정 Git commit과 `sion_translate.inference.Translator`를 사용해
  FP32 quick start를 올바르게 안내합니다. 그러나
  [파일 목록](https://huggingface.co/gaon12/sion_translate/tree/main)의
  `model_int8.pt`는 `kjx.*` 클래스를 포함한 legacy pickle이고, 현 Translator는 executable
  pickle을 기본 거부합니다. 카드가 권장하는 INT8 경로는 clean-environment smoke가 필요합니다.

## 잘한 부분

- checkpoint staging, atomic publish, previous fallback, weights-only safe load가 강합니다.
- tokenizer/model/dataset fingerprint와 portable identity의 기본 방향이 좋습니다.
- exact/approximate split guard, source-only target 차단, synthetic train-only/downweight 정책이
  합리적입니다.
- native cached decode와 full decode 동등성, export clean-environment 검증, bundle SHA-256
  검증이 있습니다.
- 모델 카드가 기존 숫자 손상과 방향 격차를 숨기지 않습니다.
- evidence/parity 모듈이 아직 완전한 active loop/checksum이 아니라는 한계를 문서에
  솔직하게 적었습니다.

## 출고 합격 순서

1. 오염 쌍을 자동 삭제하지 말고 provenance별 사람 검수 queue로 만들고 재번역합니다.
2. 원문 방향·출처·도메인·품질 등급을 sampling과 loss에 반영합니다.
3. 전체 corpus와 격리된 충분한 expressive holdout 및 다중 reference를 만듭니다.
4. 실험 모듈을 모두 끈 baseline을 먼저 학습합니다.
5. CoRe, SiTU, evidence, parity를 하나씩 같은 seed/data로 ablation합니다.
6. SFT와 MRT 모두 primary direction/category no-regression gate를 통과시킵니다.
7. bilingual human MQM으로 의미, 자연스러움, 높임말, 욕설 강도, 문화 현지화를 평가합니다.
8. artifact lock/transaction, resume identity, disk budget, clean Hub smoke를 통과한 뒤에만
   새 가중치를 공개합니다.

그 전까지 가장 정직한 릴리스 문구는 이것입니다: **“코드는 크게 개선됐지만, 번역
품질 향상은 아직 학습·측정되지 않았다.”**
