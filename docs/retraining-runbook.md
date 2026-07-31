# 처음부터 재학습 실행 안내

GPU 서버에서 토크나이저부터 사후학습까지 순서대로 돌리는 절차입니다.
2026-07-31 기준이며, 이 문서의 수치는 전부 실측입니다.

## 0. 전제

기존 체크포인트와의 호환은 고려하지 않습니다. 배포된 체크포인트는
`split_digits=False` 토크나이저로 학습돼 `sion-train`이 거부하므로, 어차피
토크나이저부터 새로 만들어야 합니다.

`data/`는 Git 저장소에 포함되지 않습니다(`.gitignore`). 일반 clone에서는
권한을 확인한 코퍼스를 별도로 준비해야 합니다. 유지관리자가 만든
`sion_translate.zip`에는 manifest에 기록된 학습 snapshot과
`data/evaluation_only/`가 이미 포함됩니다.

## 1. 가장 짧은 경로

GPU ZIP을 서버에 올린 경우 먼저 압축과 내부 checksum을 검증합니다.

```bash
sha256sum sion_translate.zip  # 배포자가 전달한 값과 비교
python -m zipfile -t sion_translate.zip
unzip sion_translate.zip
cd sion_translate
python scripts/package_gpu_bundle.py verify-tree .

pip install -e ".[dev,export,hangul]"
python easy_run.py
```

설치 뒤 학습 명령은 `easy_run.py` 하나입니다.

1. CUDA와 다중 GPU NCCL 사전검사
2. **shard 구조 검사** — 설정된 언어쌍 레코드가 표본에 없으면 즉시 중단
3. `/dev/shm` 여유가 있으면 코퍼스를 RAM 디스크로 복사
4. 토크나이저 학습 (없을 때) + 데이터셋 준비
5. **byte fallback/MorphoScript 검사** — 잘못된 sidecar나 표현력 부족이면 중단
6. 모든 rank의 최소 VRAM/BF16 능력으로 공통 설정 후 분산 SFT
7. best SFT 모델에서 MRT 사후학습 (`posttraining.enabled` 기본 true)

대화형 셸에 `tmux`가 이미 있으면 체크아웃별 세션을 사용합니다. 비대화형
Slurm/nohup/container 또는 tmux가 없는 서버에서는 설치를 강제하지 않고 현재
프로세스에서 계속합니다.

아래 2절부터는 수동으로 단계를 나눠 돌리거나 중간 산출물을 확인할 때
읽으십시오.

## 2. 환경

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,export]"
# 한국어 조사 처리에 쓰는 선택 의존성 (없으면 내장 fallback 사용)
python -m pip install -e ".[hangul]"
```

CI가 Python 3.11과 3.12에서 돕니다. 그 외 버전은 검증되지 않았습니다.

설치 확인:

```bash
python -m pytest -p no:cacheprovider
ruff format --check . && ruff check .
```

## 3. 코퍼스 확인

학습 전에 실제로 무엇이 들어가는지 세십시오.

```bash
python - <<'PY'
import glob, io
total = 0
for path in sorted(glob.glob("data/*.jsonl")):
    rows = sum(1 for line in io.open(path, "rb") if line.strip())
    total += rows
    print(f"{path:44} {rows:>10,}")
print(f"{'TOTAL':44} {total:>10,}")
PY
```

2026-07-31 기준 **8,978,338행 / 51 shard**입니다. 숫자가 크게 다르면 코퍼스
업로드가 덜 끝난 것입니다.

### 키 이름 확인 — 반드시 하십시오

행 수가 맞아도 학습에 안 들어갈 수 있습니다. JSONL의 키 이름이 설정된 언어와
다르면 그 파일은 **0문장**을 내놓고 로그에 아무 말도 남지 않습니다.

```bash
python scripts/data/check_shard_keys.py
```

종료코드가 0이 아니면 그 파일은 학습에서 빠질 가능성이 있습니다. 검사는 각
shard에서 최대 2,000개 물리 행만 구조적으로 확인하고, 정상 파일은 첫 유효
레코드에서 멈춥니다. 품질 필터와 split을 다시 실행하지 않으므로 대용량 코퍼스를
두 번 전처리하거나 저품질 문장을 키 오류로 오진하지 않습니다. 실제로
`data40.jsonl`이 키를 `한국어`/`일본어`로 써서 빠지던 문제를 이 검사로 찾았습니다.

`easy_run.py`가 이 검사를 학습 전에 자동으로 돌리므로, 수동 실행은 미리
확인하고 싶을 때만 하면 됩니다.

고치는 방법은 **JSONL의 키를 바꾸는 것 하나뿐**입니다. 언어쌍에 추가하는 것은
불가능합니다 — 언어 키는 1~16자 ASCII 영숫자여야 하므로 `한국어`는 언어 키가
될 수 없습니다.

```bash
python - <<'PY'
import io, json
src, dst = "data/data40.jsonl", "data/data40.fixed.jsonl"
with io.open(src, encoding="utf-8-sig") as fin, io.open(dst, "w", encoding="utf-8", newline="\n") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        fout.write(json.dumps({"ko": row["한국어"], "ja": row["일본어"]}, ensure_ascii=False) + "\n")
print("wrote", dst)
PY
```

바꾼 뒤 `check_shard_keys.py`를 다시 돌려 0이 나오는지 보고, 원본은 지우거나
`data/` 밖으로 옮기십시오. 둘 다 남으면 같은 내용이 중복 학습됩니다.

## 4. 토크나이저

```bash
sion-train-tokenizer \
  --input "data/*.jsonl" \
  --output-dir artifacts/tokenizer \
  --vocab-size 48000 \
  --input-sentence-size 0 \
  --required-character-min-occurrences 25 \
  --language-pairs kj ko --language-pairs kj ja \
  --language-pairs kd ko --language-pairs kd ja \
  --language-pairs jd ko --language-pairs jd ja \
  --language-pairs ko ja \
  --approximate-split \
  --source-only-language kj kd jd \
  --train-only-prefix bt_ concat_ revise_ synthetic_
```

`--input-sentence-size 0`이 핵심입니다. 기본 상한을 두면 SentencePiece가
코퍼스의 22.2%만 보고, 균등 무작위 추출이라 작은 shard가 비중만큼만
보입니다 — 한본어는 전체 문장의 0.11%입니다.

`--required-character-min-occurrences 25`는 25회 이상 나오는 문자를 어휘에
못 박습니다. 코퍼스에 고유 문자가 10,761개 있고 이 임계값에서 6,492개가
남습니다(48k 어휘의 13.5%). 5~49회 구간 2,231자는 대부분 희귀 한자·한글
음절이라 byte fallback이 맞는 대상입니다.

### 끝나면 반드시 확인할 것

`easy_run.py`가 자동으로 검사하므로 보통은 따로 할 일이 없습니다. 수동으로
확인하려면:

```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")
from pathlib import Path
import easy_run
easy_run._verify_tokenizer(Path("artifacts/tokenizer/sion.model"), Path("data"))
PY
```

코퍼스에서 표본을 뽑아 byte fallback 비율을 재고, 원인 문자를 코드포인트와
함께 출력합니다. 고정된 프로브 문자열을 쓰지 않으므로 다른 언어쌍에서도
그대로 동작합니다.

판정은 0이 아니라 **비율**입니다. 임계값 25 미만 문자는 의도적으로 byte
fallback 대상이고, 그것이 byte fallback의 용도입니다. 기본 상한은 0.2%입니다.

비율이 높으면 `--required-character-min-occurrences`를 낮추거나 vocab을
키우십시오. 다만 required 문자 수 + 제어 심볼 + byte fallback 256이 vocab을
넘으면 학습이 시작 전에 거부됩니다.

## 5. 데이터셋 준비

```bash
sion-prepare-data \
  --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer/sion.model \
  --output-dir artifacts/dataset \
  --language-pairs kj ko --language-pairs kj ja \
  --language-pairs kd ko --language-pairs kd ja \
  --language-pairs jd ko --language-pairs jd ja \
  --language-pairs ko ja \
  --approximate-split \
  --source-only-language kj kd jd \
  --train-only-prefix bt_ concat_ revise_ synthetic_
```

`sion-train`을 인자 없이 돌리면 이 단계가 자동으로 실행되므로 건너뛰어도
됩니다. 수동으로 돌리면 중간 산출물을 확인할 수 있습니다.

## 6. 학습

```bash
# 단일 GPU
sion-train

# 다중 GPU
torchrun --standalone --nproc-per-node=8 -m sion_translate.cli.train
```

`sion_translate.yaml`을 자동으로 읽습니다. 적지 않은 항목(모델 크기, 배치,
step 예산, 정밀도)은 GPU와 데이터 규모를 보고 자동 결정됩니다.

### 이 run에서 켠 것

| 모듈 | 상태 | 이유 |
|---|---|---|
| SiTU-GLU | on | projection 모양 그대로. 위험 최소 |
| BATS coverage 0.01 | on | 누락·중복 제어에 정렬 신호가 필요 |
| CoRe (register) | **on — 검증 대상** | 반말이 존댓말의 2.1배라 register 혼입이 실측된 실패 |
| TETM | off | 한 번에 하나씩 원칙 |
| MorphoScript | off | script 위반이 현재 0%. 실패하지 않는 문제 |

CoRe의 `inject_gate`는 0에서 시작하고 `tanh(gate)`로 곱해집니다. 즉 step 0에서
forward 기여가 정확히 0이고 보조 loss만 흐릅니다. 모델이 이 신호를 쓸지를
학습으로 정하므로 켜는 위험이 낮습니다.

### 학습 중 볼 것

- **train/val loss 곡선.** train이 계속 떨어지는데 val이 따라오면 underfit이라
  모델을 키울 근거가 됩니다. val이 벌어지면 반대입니다. 현재 형상(200M)은
  양방향 target 토큰 357,344,643/epoch에 대해 잘 맞는 크기입니다.
- **register loss.** 떨어지지 않으면 CoRe가 신호를 못 찾는 것이므로 끄십시오.

## 7. 사후학습 (MRT)

`posttraining.enabled`가 기본 true라 학습 뒤 자동으로 이어집니다.
복합 보상 7종(chrF / token_f1 / number / structured / slot / language / length)에
반복 페널티와 복사 페널티가 붙습니다.

`roundtrip_enabled`는 `sion_translate.yaml`에서 **true**입니다(가중치 0.20).
후보를 원문 언어로 되번역해 원문이 복원되는지 보는 성분입니다.

여기에 대해 이전에 "끄십시오"라고 안내한 적이 있는데, 그건 **다른 맥락의
측정을 옮긴 것이라 부정확했습니다.** 그 측정은 번역 대기열에서 왕복 점수를
**하드 필터**로 썼을 때 32행 파일럿에서 적합성과 역상관이더라는 것이었습니다.
여기서는 8개 성분 복합 보상의 0.20 가중치라 성격이 다릅니다.

지금 판단할 근거가 없으므로 켠 채로 둡니다. 다만 비용은 알고 계십시오:
후보마다 역방향 디코드가 한 번 더 돕니다(`roundtrip_num_beams: 1`이라
greedy). MRT 생성 시간이 늘어납니다.

끄고 싶다면 `roundtrip_enabled: false`로 바꾸고, 남은 7개 성분의 가중치 합이
달라진다는 점만 유의하십시오(정규화되므로 비율은 유지됩니다).

## 8. 산출물 위치

```
runs/auto/
├── pretrain/
│   ├── checkpoints/      best / latest / final
│   └── exports/best/     fp32 fp16 bf16 int8 int4 gguf_q4_k_m transformers
└── posttrain/            MRT 사후학습
    ├── checkpoints/
    └── exports/best/
artifacts/
├── tokenizer/            sion.model, token_features.npz
└── dataset/              전처리된 indexed 데이터
```

`posttraining.enabled`가 true이므로 **최종 산출물은
`runs/auto/posttrain/exports/best/`** 입니다. `pretrain/` 쪽은 MRT 이전
단계이고 비교할 때만 씁니다.

체크포인트는 `best`(validation 기준), `latest`(재시작용), `final`(마지막 step)
입니다. 쓸 것은 `best`입니다.

가중치를 가져갈 때는 **토크나이저를 반드시 함께** 가져가십시오. vocab이
맞지 않으면 가중치만으로는 아무것도 못 합니다.

## 9. 평가

```bash
sion-evaluate --help
sion-translate --help
```

**평가 시 주의.** 저장소 자체 test split 점수는 근사 중복 유출로 부풀려져
있었습니다(test split chrF 77.50 대 실제 진단 61.79). `approximate_split: true`가
켜져 있으면 이번에는 완화되지만, 진짜 기준선은 외부 평가셋입니다.

`data/evaluation_only/data22.jsonl`(FLORES-200)은 학습에서 격리돼 있습니다.
배포판은 이것을 학습에 섞어서 홀드아웃 점수가 부풀었습니다.

beam은 4를 쓰십시오. 실측에서 1→2→4가 chrF 77.28→77.36→77.50이고 16에서
심한 반복 붕괴가 일어났습니다.

## 10. 되돌아볼 만한 실패 지점

- 토크나이저를 다시 만들면 `artifacts/dataset`과 체크포인트를 재사용할 수
  없습니다. 3~5단계를 순서대로 다시 돌려야 합니다.
- `source_only_languages`를 빠뜨리면 표준 한국어를 요청해도 혼용문·사투리가
  나옵니다. yaml 주석 처리된 예시를 실제로 풀어야 합니다.
- `approximate_split`을 끄면 홀드아웃 점수가 번역 품질을 재지 않습니다.

## 11. 선택 사항 — exposure bias

`data.decoder_input_noise`가 0(꺼짐)입니다. teacher forcing이 정답 접두사만
보여 주는 문제의 본학습 단계 대책인데, 디코더 조건부를 바꾸는 개입이라
측정 없이 켜지 않았습니다.

시도한다면 0.1부터 보고 검증 loss로 A/B하십시오. labels는 건드리지 않으므로
목적함수는 그대로입니다.
