# foundation 사전학습 (단일어 복원)

번역 학습 **이전** 단계입니다. 언어별 단일어 텍스트로 span-corruption 복원만
학습해 encoder-decoder 를 먼저 만들고, 그 위에서 번역을 학습합니다.

단계 순서:

```
foundation (단일어 복원)  →  SFT (번역)  →  MRT (사후학습)
runs/*/foundation/           runs/*/pretrain/   runs/*/posttrain/
```

## 코퍼스 배치

`data/corpus/` 아래에 **언어 코드 폴더**를 만들고 그 안에 파일을 둡니다.
하위 폴더도 훑습니다.

```
data/corpus/
  ko/
    kowiki_corpus.txt          # 한 줄에 문장/문단 하나
    2026/news.jsonl            # {"text": "..."} 한 줄에 하나
  ja/
    wiki.txt
```

허용 형식은 **`.txt` 와 `.jsonl` 둘뿐**이고, `.jsonl` 은 **`text` 키**를 씁니다.
다른 확장자, 언어 코드가 아닌 폴더 이름, 최상위에 떠 있는 파일은 전부
**건너뛰고 그 사실을 출력**합니다. 조용히 무시하지 않는 이유는 이 단계가
가장 오래 걸리는데 입력 오류는 티가 나지 않기 때문입니다 — 키 이름 하나가
틀리면 그 파일만 0문장이 되고, 학습이 끝난 뒤에야 드러납니다.

`python easy_run.py` 를 실행하면 학습 전에 보고가 나옵니다. 예를 들어 corpus를
잘못 배치해 일본어가 누락된 경우에는 다음처럼 경고합니다.

```
[easy_run] foundation(단일어 사전학습) 코퍼스를 확인합니다.
[easy_run]   단일어 코퍼스 루트: data/corpus
[easy_run]     ko: 파일 4개, 5.31 GB
[easy_run]     데이터 없는 언어: ja
[easy_run]     건너뛴 항목 3개:
[easy_run]       - data/corpus/a.py: 언어 폴더가 아닌 최상위 파일
[easy_run]       - data/corpus/korean_tech_corpus_130m: 언어 코드 형식이 아닌 폴더 이름
[easy_run] [경고] 단일어 데이터가 전혀 없는 언어: ja …
```

## 어떤 언어가 대상인가

설정된 언어에서 **source-only 언어를 뺀 것**입니다. 현재 설정에서는
`kj`/`kd`/`jd` 가 빠지고 `ko`, `ja` 만 남습니다.

이유는 데이터 계약입니다. 복원 과제는 그 언어를 **디코더 출력**으로 만드는
학습인데, source-only 는 "번역 결과로 나오면 안 되는 언어" 라는 뜻입니다.
걸러 두지 않으면 foundation 이 이후 번역 단계가 금지하는 것을 먼저 가르칩니다.

## 언어 불균형

언어별 분량 차이는 온도 샘플링(`foundation.language_sampling_alpha`, 기본 0.7)
으로 눕히지만, **없는 데이터를 만들어 내지는 못합니다.** 분량이 0 인 언어는
가중치도 0 이고, 그건 가중치가 아니라 경고로 다뤄야 할 문제입니다.

한쪽 언어만 학습하면 그 언어를 목표로 하는 번역 방향만 강해집니다. 이
저장소는 이미 ko→ja 59.81 대 ja→ko 49.87 로 그 증상을 갖고 있으므로,
한국어만 있는 상태로 foundation 을 돌리면 격차가 벌어질 수 있습니다.
그래도 기본은 **경고 후 진행** 입니다 — 언어를 나중에 채우는 것이 정상적인
작업 흐름이기 때문입니다. 막으려면:

```yaml
foundation:
  require_all_languages: true
```

## 토크나이저

단일어 코퍼스도 토크나이저 학습에 들어갑니다. 넣지 않으면 foundation 단계가
자기 코퍼스에 없는 어휘로 학습하게 됩니다 — 실측으로, 단일어에만 있는 낱말이
19 조각(byte fallback 18 개)이 됐다가 넣으면 1 조각이 됩니다.

전량 넣지도 않습니다. 언어별로 **그 언어의 병렬 코퍼스 문장 수 ×
`foundation.tokenizer_sample_ratio`**(기본 0.4)까지만 뽑습니다. 전량 넣으면
분량이 큰 언어가 vocab 을 독식합니다. 2026-08-08 production 실행은 해시 표본으로
ja 3,445,471문장과 ko 3,308,940문장을 골라 두 언어를 함께 반영했습니다.

표본은 파일 앞에서 자르지 않고 해시로 고르게 뽑습니다. 단일어 파일은 출처별로
나뉘어 있어(위키 → 뉴스 → 커뮤니티) 앞을 자르면 한 출처만 뽑히고 그 편향이
그대로 어휘에 박힙니다.

0.4는 어휘 균형을 위한 값입니다. 예전 0.13은 SentencePiece 0.2.2의 native
crash를 우연히 피하려고 낮춘 값이었고 안전 경계가 아니었습니다. 원인을 고친
버전 정책과 실측은 [`sentencepiece-sigsegv.md`](sentencepiece-sigsegv.md)에
남겼습니다. 이 비율은 토크나이저 학습 문장뿐 아니라 `required_chars` 문자 빈도
스캔에도 동일하게 적용됩니다.

## 산출물은 번역 모델이 아닙니다

이 단계의 결과물은 **`sion`** 이라는 별도 이름으로 나갑니다. 번역 모델은
그것에서 파생된 **`sion_translate`** 입니다.

이름만 다른 것이 아닙니다. foundation 체크포인트는 번역 체크포인트와
아키텍처가 완전히 같아서, 막지 않으면 `Translator` 가 그대로 싣고 방향 태그를
받아들여 **유창한 헛소리**를 냅니다 — 번역쌍을 한 번도 본 적 없는 가중치인데도
그렇습니다. 그래서 export metadata 에 능력 계약을 적습니다.

```json
{"release_name": "sion", "translation_capable": false, "languages": ["ko", "ja"]}
```

`language_pairs` 와 `translation_directions` 는 **적지 않습니다.** 복원 모델에게
쌍과 방향은 존재하지 않습니다. `Translator` 는 `translation_capable: false` 인
export 를 거부하고 `runs/*/pretrain` 이나 `posttrain` 을 가리킵니다.

## 다시 돌리지 않는다

foundation 이 끝나면 `runs/*/foundation/stage_complete.json` 이 남습니다. 이후
실행은 학습을 건너뛰고 best 가중치만 물려받습니다.

이 표시가 필요한 이유는 번역 학습이 실패해 재실행되기 때문입니다 — 코퍼스
지문 변경, export 의존성 누락, 용량 검사 실패. 표시가 없으면 그때마다 며칠짜리
사전학습을 반복합니다. **중단된** 실행은 표시가 없으므로 정상적으로 재개됩니다
(표시는 학습이 끝난 뒤에만 씁니다).

가중치 인계는 재개가 아닙니다. optimizer moment·scheduler·step 을 물려받지
않습니다 — 두 단계는 목적함수가 다르므로 momentum 이 다른 loss 표면을 가리키고,
step 을 물려받으면 warmup 이 통째로 건너뛰어집니다. 대신 **토크나이저와 model
config 가 같은지 검증**합니다. 토크나이저가 다르면 텐서 모양은 맞아서
`load_state_dict` 가 성공하고 모든 임베딩 행이 조용히 다른 것을 가리킵니다.

## 설정

```yaml
foundation:
  enabled: true                  # 코퍼스가 있으면 자동 실행. false 면 데이터가 있어도 건너뜀
  corpus_dir: data/corpus
  dataset_dir: artifacts/foundation_dataset
  release_name: sion

  language_sampling_alpha: 0.7   # 1.0 = 분량 정비례, 낮출수록 균등
  minimum_language_share: 0.05   # 이 비중 미만이면 경고
  require_all_languages: false
  minimum_characters: 8
  maximum_characters: 4000
  deduplicate: true

  noise_density: 0.15            # 복원 과제의 손상 비율
  mean_span: 3.0
  tokenizer_sample_ratio: 0.4    # 0 이면 토크나이저 학습에서 단일어 제외

  max_steps: 100000
  batch_size_per_gpu: 16
  learning_rate: 0.0003
  warmup_steps: 2000
  final_export_formats: [fp32, bf16, transformers]
```

`enabled: true` 가 기본이지만 코퍼스가 없으면 이유를 출력하고 건너뜁니다.
