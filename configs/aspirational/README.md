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
| `sion_8b.yaml` (여기) | 8,216,715,264 | 0.043 | 164.3 B | **460×** |

Chinchilla 는 decoder-only LM 에서 유도된 법칙이고, 번역용 encoder-decoder 는
관례적으로 그보다 아래에서 학습합니다. 그래도 참고로 NLLB-200 의 54B MoE 는
180억 문장쌍을 썼습니다. **0.043 tok/param 은 어떤 잣대로도 방어되지 않습니다.**

한때 `sion_32b.yaml` 도 여기 있었습니다 (319억 파라미터, 부족 1,790×).
삭제했습니다. 부족분이 세 자릿수를 넘어가면 그것은 "데이터가 갖춰지면 쓸
설정"이 아니라 그냥 쓸 일이 없는 설정이고, 실행 가능한 형태로 저장소에
두는 것 자체가 잘못된 신호입니다. 용량 게이트(`validate_training_capacity`)
는 파라미터 수로 동작하므로 그 규모를 다룰 능력은 preset 파일과 무관하게
남아 있습니다.

## 그래서 무엇을 하라는 것인가

`configs/sion_data_fit.yaml` 이 현재 데이터에 맞는 기준선입니다.
`configs/sion_1_3b.yaml` 은 데이터를 늘렸을 때의 다음 단계입니다.

여기 있는 설정을 쓰려면 **먼저 코퍼스를 늘리십시오.** 필요한 규모는 위 표의
"Chinchilla 기준 필요량" 열입니다. 확보처 검토는 `docs/corpus-gaps.md` 와
`docs/DATA_EXPANSION_PLAN.md` 를 보십시오.

## 남아 있는 다른 문제

`sion_8b.yaml` 은 `encoder_layers == decoder_layers` (30/30) 입니다.
`SionForConditionalGeneration` 의 docstring 이 설명하는 설계 — 깊은 encoder /
얕은 decoder — 와 반대이고, 자기회귀 디코딩이 가중치 대역폭 바운드라는 측정과도
어긋납니다. `configs/sion_1_3b.yaml` 은 파라미터 수를 유지한 채 30/12 로
재조정했습니다. 이 파일은 실행 대상이 아니므로 그대로 두었습니다.
데이터가 갖춰져 실제로 쓸 때가 되면 같은 재조정을 먼저 하십시오.
