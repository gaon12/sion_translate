# 왜 이 preset 들은 `configs/` 밖에 있는가

여기 있는 설정은 **현재 코퍼스로 학습하면 안 됩니다.** 문법적으로 유효하고
`load_config` 를 통과하지만, 보유한 데이터가 이 용량을 지탱하지 못합니다.

## 실측

`data/*.jsonl` 51개 shard, 8,978,338 레코드. 배포 토크나이저로 센 결과
한 번 통과에 약 **0.357B 토큰**(원문+정답)입니다.

| preset | 파라미터 | tok/param | Chinchilla 기준 필요량 | 부족 |
|---|---:|---:|---:|---:|
| `configs/sion_data_fit.yaml` | 200,486,400 | 1.78 | 4.0 B | 11× |
| `configs/sion_1_3b.yaml` | 1,206,340,608 | 0.296 | 24.1 B | 68× |
| `sion_8b.yaml` (여기) | 8,216,715,264 | 0.043 | 164.3 B | 460× |
| `sion_32b.yaml` (여기) | 31,955,783,680 | 0.011 | 639.1 B | **1,790×** |

`sion_32b.yaml` 의 자체 step 예산으로 검산해도 같습니다:
400,000 step × batch 1 × accum 4 × 16 GPU = 25.6M 시퀀스 ≈ 코퍼스 1.4회 통과
≈ **0.51B 타깃 토큰**. 320억 파라미터 모델이 0.5B 토큰을 봅니다.

Chinchilla 는 decoder-only LM 에서 유도된 법칙이고, 번역용 encoder-decoder 는
관례적으로 그보다 아래에서 학습합니다. 그래도 참고로 NLLB-200 의 54B MoE 는
180억 문장쌍을 썼습니다. **0.011 tok/param 은 어떤 잣대로도 방어되지 않습니다.**

## 그래서 무엇을 하라는 것인가

`configs/sion_data_fit.yaml` 이 현재 데이터에 맞는 기준선입니다.
`configs/sion_1_3b.yaml` 은 데이터를 늘렸을 때의 다음 단계입니다.

여기 있는 설정을 쓰려면 **먼저 코퍼스를 늘리십시오.** 필요한 규모는 위 표의
"Chinchilla 기준 필요량" 열입니다. 확보처 검토는 `docs/corpus-gaps.md` 와
`docs/DATA_EXPANSION_PLAN.md` 를 보십시오.

## 남아 있는 다른 문제

이 두 파일은 `encoder_layers == decoder_layers` (30/30, 42/42) 입니다.
`SionForConditionalGeneration` 의 docstring 이 설명하는 설계 — 깊은 encoder /
얕은 decoder — 와 반대이고, 자기회귀 디코딩이 가중치 대역폭 바운드라는 측정과도
어긋납니다. `configs/sion_1_3b.yaml` 은 파라미터 수를 유지한 채 30/12 로
재조정했습니다. 이 두 파일은 실행 대상이 아니므로 그대로 두었습니다.
데이터가 갖춰져 실제로 쓸 때가 되면 같은 재조정을 먼저 하십시오.
