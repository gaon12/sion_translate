# KJ-X 사후학습 설계와 실험 가이드

## 현재 구현

사후학습의 전체 loss는 다음과 같습니다.

```text
L = L_reference_CE + risk_weight * L_composite_MRT
                   + preference_weight * L_multi_pair
```

- `L_reference_CE`: 정답 번역 likelihood를 유지해 보상 최적화 중 문장 품질이
  무너지는 것을 막는 anchor입니다.
- `L_composite_MRT`: 확률적으로 생성한 후보들의 기대 위험을 최소화합니다.
- `L_multi_pair`: reward 차이가 `preference_min_gap` 이상인 모든 후보쌍에 대해
  더 좋은 후보의 평균 log-probability가 더 높아지도록 학습합니다.
- 후보 reward는 `chrF + token F1 + 숫자 + 구조 문자열 + glossary slot + 목표
  언어 문자 + 길이`의 가중 평균이며, 과도한 반복과 원문 복사를 감점합니다.
- 사후학습의 `best`와 early stopping은 CE가 아니라 validation 문장의 beam
  번역에 계산한 복합 reward를 사용합니다. CE도 모델 drift 진단용으로 계속
  기록합니다.

기본 보상 가중치는 일반 문장 품질을 위한 시작점입니다. 숫자/코드가 중요한
업무 데이터라면 `reward_number_weight`, `reward_structured_weight`,
`reward_slot_weight`를 올리고 반드시 별도 holdout으로 확인해야 합니다.

## 설계 근거

- [Minimum Risk Training for Neural Machine Translation (ACL 2016)](https://aclanthology.org/P16-1159/):
  문장 단위 평가 척도의 기대 위험을 직접 최적화하는 기본 구조입니다.
- [M2PO: Multi-Perspective Multi-Pair Preference Optimization (ACL 2026)](https://aclanthology.org/2026.acl-long.469/):
  단일 관점의 품질 추정이 부분 환각/누락을 놓칠 수 있으며, 후보 하나만 고르는
  대신 여러 관점과 모든 후보쌍을 쓰는 방법을 제안합니다. 현재의 복합 reward와
  all-pair preference loss에 반영했습니다.
- [Direct Quality Optimization for Neural Machine Translation (WMT 2025)](https://aclanthology.org/2025.wmt-1.2/):
  encoder-decoder NMT에서도 직접 선호 최적화가 품질을 높일 수 있음을 보입니다.
- [Word Alignment as Preference for Machine Translation (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.188/):
  정렬 기반 선호가 환각과 누락을 줄일 수 있다는 결과를 숫자·구조 문자열·slot
  보존 보상에 반영했습니다.
- [xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection (TACL 2024)](https://aclanthology.org/2024.tacl-1.54/),
  [Fine-Grained Reward Optimization for Machine Translation using Error Severity (TACL 2026)](https://aclanthology.org/2026.tacl-1.33/):
  문장 점수뿐 아니라 오류 위치/심각도를 이용하는 fine-grained reward가 학습을
  안정화할 수 있습니다. xCOMET는 별도 대형 모델과 VRAM이 필요하므로 기본 실행의
  필수 의존성으로 넣지 않았고, 아래 2차 실험으로 분리합니다.
- [Metric Bias in Minimum Bayes Risk Decoding (WMT 2024)](https://aclanthology.org/2024.wmt-1.109/):
  최적화에 사용한 metric 하나만으로 결과를 평가하면 reward hacking을 놓칠 수
  있음을 보여 줍니다. 그래서 chrF 단독 reward를 피하고 최종 평가는 별도 지표로
  수행합니다.
- [Unlikelihood Training for Neural Machine Translation (COLING 2020)](https://aclanthology.org/2020.coling-main.462/):
  반복 생성 억제의 근거입니다. 현재는 반복을 직접 음의 보상으로 적용합니다.

## 권장 실험 순서

동일한 사전학습 checkpoint, validation/test split, 생성 설정으로 아래를 비교합니다.

| 실험 | 설정 | 확인할 효과 |
|---|---|---|
| A | `risk_weight=0`, `preference_weight=0` | CE anchor 기준선 |
| B | chrF MRT만 사용 | 기존 사후학습 기준선 |
| C | 기본 복합 MRT | 숫자/slot 누락, 반복, 원문 복사 개선 |
| D | 복합 MRT + multi-pair | 후보 순위와 전체 번역 품질 개선 |
| E | D + xCOMET 오류 심각도 reward | 더 미세한 환각/누락 개선 가능성 |

최종 비교에는 학습 reward만 쓰지 말고 다음을 함께 기록합니다.

- test split의 chrF/BLEU와 방향별 점수(ko→ja, ja→ko)
- 숫자, URL/이메일/코드, glossary slot의 exact preservation rate
- 반복 출력률, 빈 출력률, 원문 복사율, 길이 비율 분포
- 일반/전문/대화 도메인별 점수와 최소 200문장 blind 사람 평가

E는 실험 옵션입니다. xCOMET 같은 외부 reward 모델을 online으로 함께 돌리면 GPU
메모리와 학습 시간이 크게 증가하므로, 먼저 후보를 offline으로 scoring해 preference
pair를 저장한 뒤 학습하는 방식을 권장합니다. 같은 xCOMET 점수만으로 E를 선택하지
말고 C/D와 독립 metric 및 사람 평가로 비교해야 합니다.

## 조정 기준

- 후보 reward가 거의 같아 pair가 적으면 `samples_per_source`를 4~8로 늘리거나
  `preference_min_gap`을 0.02까지 낮춥니다.
- 문장이 불안정해지면 `preference_weight`를 먼저 0.05로, 그다음
  `risk_weight`를 0.10으로 낮춥니다. CE anchor는 끄지 않는 편이 안전합니다.
- 반복/복사율은 낮지만 chrF가 떨어지면 해당 penalty를 절반으로 낮춥니다.
- validation reward는 오르는데 외부 test 지표가 떨어지면 reward hacking으로 보고
  가중치를 재조정하거나 독립 reward를 추가합니다.
