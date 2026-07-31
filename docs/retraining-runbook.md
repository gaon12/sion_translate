# 처음부터 재학습 실행 안내

GPU 서버에서 토크나이저부터 사후학습까지 순서대로 돌리는 절차입니다.
2026-07-31 기준이며, 이 문서의 수치는 전부 실측입니다.

## 0. 전제

기존 체크포인트와의 호환은 고려하지 않습니다. 배포된 체크포인트는
`split_digits=False` 토크나이저로 학습돼 `sion-train`이 거부하므로, 어차피
토크나이저부터 새로 만들어야 합니다.

`data/`는 저장소에 포함되지 않습니다(`.gitignore`). 코퍼스를 서버로 따로
올려야 합니다.

## 1. 가장 짧은 경로

GPU 서버에서는 이것 하나면 됩니다.

```bash
pip install -e ".[dev,export,hangul]"
python easy_run.py
```

`easy_run.py`가 전부 합니다.

1. tmux 세션 생성 (없으면 자동 설치). 접속이 끊겨도 학습이 계속됩니다
2. **shard 키 검사** — 읽히지 않는 shard 가 있으면 즉시 중단
3. `/dev/shm` 여유가 있으면 코퍼스를 RAM 디스크로 복사
4. 토크나이저 학습 (없을 때) + 데이터셋 준비
5. **byte fallback 검사** — 비율이 상한을 넘으면 중단
6. GPU 개수만큼 분산 학습
7. 학습 뒤 MRT 사후학습 (`posttraining.enabled` 기본 true)

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
python -m pytest -q          # 911개 통과해야 합니다
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

종료코드가 0이 아니면 그 파일은 학습에서 빠집니다. 실제로 `data40.jsonl`이
키를 `한국어`/`일본어`로 써서 10,075행이 빠지고 있었고, 지금은 고쳐졌습니다.
전량 스캔에서 51개 shard 전부 읽히고 총 18,124,108문장입니다.

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
  --language-pairs ko ja
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
  --output-dir artifacts/dataset
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

`roundtrip_enabled`는 **false로 두십시오.** 32행 파일럿에서 왕복 점수가
적합성과 역상관으로 측정됐습니다. 켜려면 독립 계열 QE로 게이트를 다시
설계해야 합니다.

## 8. 평가

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

## 9. 되돌아볼 만한 실패 지점

- 토크나이저를 다시 만들면 `artifacts/dataset`과 체크포인트를 재사용할 수
  없습니다. 3~5단계를 순서대로 다시 돌려야 합니다.
- `source_only_languages`를 빠뜨리면 표준 한국어를 요청해도 혼용문·사투리가
  나옵니다. yaml 주석 처리된 예시를 실제로 풀어야 합니다.
- `approximate_split`을 끄면 홀드아웃 점수가 번역 품질을 재지 않습니다.

## 10. 선택 사항 — exposure bias

`data.decoder_input_noise`가 0(꺼짐)입니다. teacher forcing이 정답 접두사만
보여 주는 문제의 본학습 단계 대책인데, 디코더 조건부를 바꾸는 개입이라
측정 없이 켜지 않았습니다.

시도한다면 0.1부터 보고 검증 loss로 A/B하십시오. labels는 건드리지 않으므로
목적함수는 그대로입니다.
