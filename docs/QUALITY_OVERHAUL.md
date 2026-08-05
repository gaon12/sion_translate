# 양방향·표현력 품질 개편 기록

이 문서는 2026-08-05 기준으로 공개 체크포인트와 로컬 학습 파이프라인을 감사한
결과, 이번 코드 개편이 해결한 문제, 그리고 새 모델을 검증하는 절차를 기록합니다.
가장 중요한 결론은 하나입니다. **코드가 고쳐져도 기존 가중치의 품질은 바뀌지
않습니다.** 과거 토크나이저와 indexed dataset은 새 설정과 호환되지 않으므로
`artifacts/sion-v6`에서 처음부터 다시 학습해야 합니다.

## 확인된 원인과 조치

| 영역 | 확인된 문제 | 이번 조치 |
|---|---|---|
| 방향별 품질 | 공개 측정은 ko→ja chrF 59.81, ja→ko 49.87로 약 10점 차이인데, checkpoint 선택은 전체 평균 loss 중심이어서 약한 방향을 숨길 수 있었습니다. | label smoothing이 섞이지 않은 방향별 NLL/PPL을 집계하고 `macro_direction_nll`을 기본 선택 기준으로 사용합니다. worst-direction 지표도 기록합니다. |
| 방향 구성 | 입력 전용 언어도 denoising target이 될 수 있었고, 양방향 그래프의 실제 학습 가능 방향을 평가·사후학습이 일관되게 보지 못했습니다. | `kj`, `kd`, `jd`를 target/denoising에서 제외하고, reverse edge 학습 여부를 batch에 명시합니다. |
| HF 경로 | native 경로와 HF wrapper의 정규화·방향 태그·생성 기본값이 달랐습니다. | NFC와 trim, 양방향 BOS/tag, beam·길이·반복 억제·제어 토큰 억제를 동일하게 맞췄습니다. |
| 사후학습 | TETM memory를 사용해 후보를 채점하면서 후보 생성에는 전달하지 않는 경로가 있었습니다. supervised auxiliary head도 MRT candidate/reference forward 사이에서 끊겼습니다. | sampling·scoring·validation에 같은 memory를 전달하고, reference pass에서 CoRe/BATS/evidence/parity supervision을 유지합니다. |
| 화자 레지스터 | CoRe가 학습 중에는 정답 register embedding을 decoder에 주고 추론 중에는 예측값을 줬습니다. | 정답 register는 분류 loss에만 쓰고 decoder는 학습·추론 모두 예측 분포로 조건화합니다. |
| 짧고 반복적인 표현 | `아!`, `으아아아`, `ㅋㅋ`, 신음처럼 정상적인 표현이 `too_short`나 `excessive_repetition`으로 제거될 수 있었습니다. | 검토된 `expressive_v1` 행에 한해 두 사유만 완화합니다. 제어 문자, 언어 mismatch, 구조 손상 같은 안전 검사는 그대로 유지합니다. |
| MRT 보상 | 정답 자체가 반복형 감탄사이거나 원문 유지가 정답인 경우에도 repetition/copy penalty가 걸렸습니다. | reference가 뒷받침하는 반복과 복사는 벌점에서 제외합니다. |
| 데이터 누수 | 표현 예제를 학습과 평가에 함께 넣으면 개선을 측정할 수 없습니다. | 사람이 검토한 seed를 train 18쌍과 challenge 12쌍으로 고정 분리하고 challenge만 양방향 24 case로 확장합니다. `easy_run.py`가 train shard만 자동 생성합니다. |
| 아티팩트 재사용 | 로컬 `artifacts/`는 구형 2언어 토크나이저와 v2 dataset이지만 현재 설정은 5언어와 source-only 정책을 사용합니다. | 모든 기본 경로와 자동 실행을 `artifacts/sion-v6`로 옮겨 구형 산출물을 묵시적으로 재사용하지 못하게 했습니다. |

## 실제 구형 토큰 노출 감사

전체 train shard를 다시 tokenize하지 않고 `.bin`/`.idx.npy`를 직접 스캔했습니다.
전체 결과는
[`legacy-token-exposure-2026-08-05.json`](audits/legacy-token-exposure-2026-08-05.json)에
있습니다.

```text
physical pairs                  11,129,222
virtual directions              22,258,444
ja→ko target content tokens    194,729,731
ko→ja target content tokens    185,729,708
ordinary vocabulary pieces          47,663
unused ordinary pieces                    11
seen exactly once                        194
seen 1–9 times                         1,376
seen 1–24 times                        2,295
median observed count                  2,503
```

판정은 다음과 같습니다.

- byte fallback 비율은 ko 0.002762%, ja 0.006657%로 낮습니다. 즉 문자열을 아예
  표현하지 못하는 coverage 문제가 주원인은 아닙니다.
- 반면 ordinary piece의 4.8%인 2,295개는 decoder target으로 25회 미만 노출됐고
  11개는 한 번도 target update를 받지 못했습니다. **희소·미학습 tail은 실제로
  존재합니다.** 새 모델에서는 tokenizer 학습 직후 raw audit, dataset 생성 직후
  indexed audit를 모두 통과시켜야 합니다.
- ja→ko 쪽 target token 수는 오히려 ko→ja보다 약 4.8% 많습니다. 따라서 ja→ko
  열세를 단순한 총 token 수 부족으로 설명할 수 없습니다. 방향별 데이터 품질,
  한국어 표면형 다양성, 전역 평균 checkpoint 선택, 학습/추론 불일치가 더 직접적인
  원인 후보입니다.
- 구형 tokenizer는 `split_digits=false`이고 언어 태그도 `ja`, `ko`뿐입니다.
  현재 5언어 설정에 이어 학습할 수 없으며 embedding vocabulary가 달라지므로
  처음부터 재학습해야 합니다.

재현 명령은 다음과 같습니다.

```bash
sion-audit-tokens \
  --dataset artifacts/dataset \
  --tokenizer artifacts/tokenizer/sion.model \
  --split train \
  --rare-threshold 25 \
  --output legacy-token-audit.json
```

새 dataset을 준비한 뒤에는 경로만 v6로 바꿉니다.

```bash
sion-audit-tokens \
  --dataset artifacts/sion-v6/dataset \
  --tokenizer artifacts/sion-v6/tokenizer/sion.model \
  --split train \
  --rare-threshold 25 \
  --fail-byte-rate 0.001 \
  --output sion-v6-token-audit.json
```

`--fail-rare-pieces`는 corpus와 vocab 크기에 따라 기준이 달라지므로 첫 full scan을
기준선으로 저장한 뒤 CI 상한을 정합니다. indexed audit은 stored content token을
정확히 세지만 sampler 재가중치, epoch 반복, BOS/EOS/언어 태그, denoising과 collator
truncation은 포함하지 않습니다.

## 욕설·신음·관용 표현 데이터 계약

seed schema에는 `category`, `subcategory`, `intensity`, `register`,
`localization_strategy`, `split`이 필수입니다. 현재 세 상위 category는
`profanity_slang`, `interjection_moan`, `idiom_culture`입니다.

```bash
python scripts/data/build_expressive_cultural_corpus.py \
  --training-output data/synthetic_expressive_cultural.jsonl \
  --challenge-output examples/expressive_cultural_cases.jsonl \
  --report reports/expressive-cultural-build.json
```

`synthetic_expressive_cultural.jsonl`은 train-only synthetic sampling 정책을 받고,
challenge 문장은 그 파일에 들어가지 않습니다. 18쌍은 의미와 강도를 고정하는
회귀 anchor이지 충분한 학습량이 아닙니다. 실제 품질에는 기존 자연 대화 데이터와
`synthetic_netspeak.jsonl` 같은 대규모 shard를 함께 사용하되, 동일 template의
과다 복제로 숫자만 높은 모델을 만들지 않아야 합니다.

challenge 평가는 전체 평균과 세 category를 함께 봅니다.

```bash
sion-translate-cases \
  --backend sion \
  --cases examples/expressive_cultural_cases.jsonl \
  --model runs/sion-v6/posttrain/exports/best/model_ema.pt \
  --tokenizer artifacts/sion-v6/tokenizer/sion.model \
  --output comparison_outputs/sion-expressive.jsonl

sion-compare \
  --cases examples/expressive_cultural_cases.jsonl \
  --system sion=comparison_outputs/sion-expressive.jsonl
```

## 첨부 구조에서 구현한 범위

연구 아이디어와 현재 코드 사이의 경계를 명확히 해야 결과를 해석할 수 있습니다.

| 첨부 아이디어 | 현재 구현 | 아직 구현하지 않은 것 |
|---|---|---|
| decoder의 evidence 재질문 | decoder stack 뒤에 별도 GQA source reread를 두고 token별 uncertainty gate로 bounded residual repair를 적용합니다. 수정 전 argmax 오류, 수정 전후 NLL gain, 무효 요청 벌점, 요청률 상한을 학습하며 생성 시 evidence K/V를 한 번만 투영합니다. | 특정 source 구간 선택, 해당 구간 재인코딩, 여러 번의 질의/응답, 출력 구간 mask 후 재생성은 없습니다. 현재 reread는 모든 source token에 대한 dense attention이라 실제 sparse compute 절약도 없습니다. |
| semantic parity/checksum | source와 teacher-forced decoder pooled representation을 양방향 contrastive loss와 positive cosine으로 맞춥니다. batch size 1과 empty target도 유한합니다. | 생성문을 독립적으로 encode한 checksum, 관계·부정·숫자별 구조화 parity, inference syndrome에 의한 자동 repair는 없습니다. 따라서 이름 그대로 학습용 representation parity ablation입니다. |
| adaptive budget | evidence request rate가 설정 상한을 넘을 때만 budget loss를 냅니다. 유용하지 않은 요청은 실제 NLL gain으로 추가 벌점 처리합니다. | 입력별 latent token 수, precision, encoder depth를 바꾸지는 않습니다. |
| multi-channel latent | 기존 CoRe는 register/style, TETM은 보호 entity, BATS는 alignment 채널 역할을 분담합니다. | 채널 직교성·정보 분리 loss가 없으므로 명시적인 disentangled latent channel이라고 주장하지 않습니다. |
| counterfactual pair | provenance/category sidecar와 challenge 분리는 후속 데이터 실험 기반을 제공합니다. | 변화 벡터를 분리하는 counterfactual encoder/loss는 아직 없습니다. 검증되지 않은 자동 반사실 쌍을 대량 생성하지 않았습니다. |

`sion_translate.yaml`에서는 evidence/parity를 기본으로 끕니다. 새 구조는 기존
가중치에 추론 때만 켤 기능이 아니라 **처음부터 학습하고 제거 실험으로 검증할
모듈**이기 때문입니다.

권장 실험 순서는 다음과 같습니다.

1. 같은 v6 tokenizer/data/seed로 baseline을 학습합니다.
2. `evidence_repair_enabled: true`만 켭니다.
3. 새 초기화로 `semantic_parity_enabled: true`만 켭니다.
4. 각각의 이득이 재현된 경우에만 둘을 함께 켭니다.

모든 run에서 `macro_direction_nll`, `worst_direction_nll`, ko→ja/ja→ko chrF,
숫자·고유명사 보존, 세 expressive category, language purity, repetition/copy를
기록합니다. evidence run은 `evidence_request_rate`, `evidence_repair_gain`, latency도
함께 비교해야 합니다. 평균만 좋아지고 ja→ko나 특정 category가 후퇴하면 채택하지
않습니다.

## 새 학습 절차

GPU 서버에서는 다음 한 명령이 표현 seed 생성부터 새 v6 tokenizer/dataset, SFT,
MRT까지 연결합니다.

```bash
python3 easy_run.py
```

구형 `artifacts/tokenizer`, `artifacts/dataset`, 기존 checkpoint는 삭제하지 않고
그대로 남습니다. 새 실행은 오직 아래 경로를 사용합니다.

```text
artifacts/sion-v6/tokenizer/
artifacts/sion-v6/dataset/
runs/sion-v6/pretrain/
runs/sion-v6/posttrain/
```

학습 완료라는 판정은 코드 test 통과가 아니라 새 가중치의 고정 challenge 결과까지
확인했을 때만 내립니다. 이 저장소에서 GPU 재학습을 실행하지 않은 상태라면 이번
변경은 “품질 결함을 수정한 학습 파이프라인”이지 “품질 향상이 실측된 새 모델”은
아닙니다.
