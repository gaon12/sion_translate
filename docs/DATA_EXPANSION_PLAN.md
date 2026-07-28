# 1억 병렬쌍 데이터 확장 계획

기준일은 2026-07-27이다. 이 문서에서 “쌍”은 JSONL 물리 줄 수가 아니라 현재
전처리기의 `expand_parallel_record`가 확장하고 품질 필터·중복 제거를 통과한
`valid_pairs` 한 개를 뜻한다. 한 JSONL 레코드가 여러 언어쌍이나 여러 문장쌍을
담으면 물리 1행에서 유효 쌍이 여러 개 나올 수 있다. 반대로
`bidirectional: true`가 만드는 역방향 학습 예제는 물리 병렬쌍 목표에 다시
더하지 않는다.

## 1. 현재 상태와 정확한 잔여량

검증된 훈련 루트의 현재 합계는 다음과 같다.

| 구분 | 병렬쌍 |
|---|---:|
| 실데이터 | 8,414,581 |
| 규칙 기반 숫자 합성 | 240,000 |
| 현재 검증 완료 합계 | **8,654,581** |
| 번역·왕복검사 전 대기열 | 10,000,000 |
| 대기열 통과 후 예상 합계 | **18,654,581** |
| 100,000,000까지 추가 필요 | **81,345,419** |

계산은 `100,000,000 - 8,654,581 - 10,000,000 = 81,345,419`다.
`translation_queue/namuwiki_ko_to_ja.jsonl`의 10,000,000문장은 아직 일본어
정답과 품질 판정이 없으므로 현재 훈련 합계에 넣지 않는다. 실패하거나 중복으로
탈락한 만큼은 81,345,419 외에 다시 보충해야 한다.

현재 상태의 기준값과 SHA256은 `data/dataset_remediation_20260727.json` 및
각 remediation manifest를 단일 진실 원천으로 삼는다. 원천을 재구축하거나
교정하면 행 수뿐 아니라 hash와 제외 이유도 함께 갱신한다.

## 2. 잔여 81,345,419쌍의 분야별 배분

아래 표의 “실데이터 최소”는 사람 번역, 독립적으로 작성된 다국어 공통 ID,
검증 가능한 문서 정렬, 또는 높은 신뢰도의 병렬 마이닝을 뜻한다. “합성 상한”은
모델 번역, 역번역, 규칙 생성, 대조·교정용 변형을 모두 포함한다. 각 행의 두
열 합과 총합은 정확히 맞는다.

| 분야 | 추가 목표 | 실데이터 최소 | 합성 상한 | 핵심 보강 내용 |
|---|---:|---:|---:|---|
| 일반 웹·뉴스·공공 정보 | 20,000,000 | 20,000,000 | 0 | 시사, 생활, 지역, 문화, 설명문, 최신 용어 |
| 대화·고객지원·메신저 | 12,000,000 | 10,200,000 | 1,800,000 | 구어체, 생략, 존댓말, 다중 turn, 문의·답변 |
| 기술·소프트웨어·UI | 8,000,000 | 7,500,000 | 500,000 | 문서, 오류문, CLI, placeholder, 코드 혼합 |
| 법률·행정·정부 | 6,000,000 | 5,700,000 | 300,000 | 법령, 계약, 민원, 기관명, 조문·버전 |
| 의료·제약·보건 | 5,000,000 | 4,700,000 | 300,000 | 증상, 진료 안내, 약품·용량, 안전 문구 |
| 금융·상거래·전자상거래 | 5,000,000 | 4,700,000 | 300,000 | 금액, 결제, 배송, 환불, 상품 속성 |
| 과학·교육·학술 | 5,000,000 | 4,700,000 | 300,000 | 교재, 논문 초록, 수식 주변 문장, 단위 |
| 문학·웹소설·서브컬처 | 10,000,000 | 9,500,000 | 500,000 | 서술체, 대사, 캐릭터 말투, 장르별 어휘 |
| 음성 전사·강연·인터뷰 | 5,000,000 | 5,000,000 | 0 | 발화체, 망설임 정리, 문장 경계, 장문 |
| 구조 문자열·숫자·코드 혼합 | 2,000,000 | 500,000 | 1,500,000 | 날짜, 시간, 단위, ID, URL, placeholder |
| 용어·짧은 검색·명령 | 1,500,000 | 1,250,000 | 250,000 | 짧은 UI, 검색어, 고유명사, 동음이의어 |
| 방언·높임·격식 register | 1,000,000 | 1,000,000 | 0 | 반말/존댓말, 문어/구어, 지역·세대 표현 |
| 적대적 교정·오류 대조 | 845,419 | 250,000 | 595,419 | 숫자 변조, 누락, 복사, 반복, 언어 혼입 |
| **합계** | **81,345,419** | **75,000,000** | **6,345,419** |  |

대기열 10,000,000쌍을 모델 번역으로 만들고 전부 통과한다고 가정하면 1억
최종 구성은 다음과 같다.

| provenance | 최종 쌍 | 물리 비율 |
|---|---:|---:|
| 실데이터 | 83,414,581 | 83.415% |
| 합성 데이터 | 16,585,419 | 16.585% |
| 합계 | **100,000,000** | **100%** |

현재 기본 `synthetic_sampling_weight: 0.5`를 유지하면 위 물리 구성에서 합성
레코드의 단순 가중 유효 비중은 약 9.043%다. 이 값은 source temperature sampling과
실제 source별 크기에 따라 달라질 수 있으므로 indexed dataset manifest와 sampler
통계로 다시 계산한다. 합성 상한을 채우기 위해 품질이 낮은 번역을 억지로
통과시키지 않는다. 합성 탈락분은 우선 실데이터로 보충한다.

## 3. 언어쌍 원칙

기본 제품 목표가 한↔일 번역이라면 추가분의 대부분을 직접 `ko-ja` 쌍으로
확보한다. 영어·러시아어 등 보조 edge를 넣는 것은 parser가 지원한다는 이유만으로
자동 승인하지 않는다. 토크나이저 vocab, 모델 용량, 각 edge의 검증 세트와
최소 데이터량이 함께 준비됐을 때만 `data.language_pairs`에 추가한다.

다국어 목표를 선택하면 다음 규칙을 지킨다.

- 1억 목표는 설정된 각 edge를 확장한 뒤의 직접 병렬쌍 합이다. pivot으로
  추론한 `ko-ru`를 실제 direct pair처럼 세지 않는다.
- 각 edge는 독립된 provenance, 원천 revision, 유효 쌍 수, 합성 비율,
  validation/test 지표를 가진다.
- `language_pairs`에는 무방향 edge를 한 번만 쓴다. `ko-ja`와 `ja-ko`를 함께
  쓰지 않고 `bidirectional: true`로 양방향 예제를 만든다.
- 다국어 한 레코드에서 파생된 모든 edge는 같은 split에 둬 같은 내용이
  train과 validation에 갈라지지 않게 한다. 현재 전처리기가 다언어 레코드의
  raw-record hash를 split key로 사용한다.
- 새 언어를 추가하면 기존 토크나이저·dataset·checkpoint를 재사용하지 않고,
  모든 언어의 `<2xx>`와 `<denoise_xx>` 태그를 가진 토크나이저부터 새로 만든다.

YAML 예시는 다음과 같다.

```yaml
data:
  language_pairs:
    - [ko, ja]
    - [en, ru]
  bidirectional: true
  synthetic_sampling_weight: 0.5
```

## 4. 지원하는 JSONL 레코드

언어 키는 ASCII 영문자로 시작하는 1~16자의 영숫자 문자열이어야 한다. 아래
형식은 한 파일 안에서 섞여 있어도 현재 전처리기가 확장한다.

### 평면 언어 키

```json
{"ko":"안녕하세요.","ja":"こんにちは。"}
```

한 레코드에 설정된 edge가 여러 개 있어도 된다.

```json
{"ko":"안녕하세요.","ja":"こんにちは。","en":"Hello.","ru":"Здравствуйте."}
```

`language_pairs: [[ko, ja], [en, ru]]`이면 위 한 행에서 두 쌍을 만든다. 설정에
없는 언어 조합은 임의로 생성하지 않는다.

### 같은 길이의 문장 배열

```json
{"ko":["첫 문장.","둘째 문장."],"ja":["一文目。","二文目。"]}
```

양쪽 배열 길이가 다르면 `unaligned_lists`로 거부한다.

### 명시적 source/target 언어

```json
{"source_language":"ko","target_language":"ja","source":"원문","target":"訳文"}
```

`src_language`/`tgt_language`도 쓸 수 있다. 원문 필드 alias는
`source`, `src`, `input`이고 번역 필드 alias는 `target`, `tgt`,
`reference`, `translation`, `output`이다.

### 한 레코드 안의 여러 항목

```json
{"records":[
  {"source_language":"ko","target_language":"ja","source":"안녕","target":"こんにちは"},
  {"source_language":"en","target_language":"ru","source":"Hello","target":"Привет"}
]}
```

컨테이너 이름은 `records`, `items`, `pairs`, `translations`를 지원한다.

### 언어쌍 이름 컨테이너

```json
{"pairs":{
  "ko-ja":[{"source":"안녕","target":"こんにちは"}],
  "en-ru":[{"source":"Hello","target":"Привет"}]
}}
```

`ko-ja`, `ja-ko`, `ko/ja`, `ja/ko`, `ko_to_ja`, `ja_to_ko` 형태를 인식하고
역방향 표기는 설정의 canonical edge 순서로 되돌린다.

### 합성 provenance

```json
{"ko":"회의는 14:30에 시작합니다.","ja":"会議は14時30分に始まります。","synthetic":true}
```

또는 다음처럼 쓸 수 있다.

```json
{"ko":"원문","ja":"訳文","metadata":{"synthetic":true}}
```

`synthetic`은 문자열 `"true"`가 아니라 JSON boolean `true`여야 한다. 레코드
표시 또는 `bt_`, `concat_`, `revise_`, `synthetic_` 파일 접두사 중 하나가 있으면
train-only가 되고 기본 0.5 sampling weight가 적용된다. 원천 모델, prompt,
생성 시각, decoding 설정, 원문 hash는 별도 source manifest에도 기록한다.

## 5. 수집·정렬 단계

새 source마다 다음 순서를 독립적으로 통과시킨다.

1. **원본 격리:** 다운로드 URL, 원천 revision/commit, 파일 hash, 취득일,
   추출 도구 버전을 기록하고 원본은 직접 편집하지 않는다.
2. **결정적 정렬:** 공통 문서·문단·문장 ID, 동일 리소스 키, 검증된 alignment
   score를 사용한다. MASSIVE 사례처럼 “같은 intent”만으로 번역 등가라고
   가정하지 않는다.
3. **source 전용 정규화:** HTML/markup은 양쪽 구조를 보존하고, UI placeholder,
   법령 조문, 성경 절, 게임 resource key처럼 domain별 join key를 manifest에
   남긴다.
4. **공통 레코드 확장:** 위 JSONL schema 중 하나로 만들고 설정된 language
   edge만 추출한다.
5. **자동 품질 gate:** UTF-8/JSON, 빈 값, type, 길이, script, 반복, 구조 문자열,
   중복을 검사한다.
6. **표본 의미 검수:** 자동 점수만으로 번역 등가성을 승인하지 않는다.
7. **indexed dataset 생성:** 새 output 디렉터리에 fingerprint와 통계를 기록한다.
8. **고정 평가:** source 내부 holdout뿐 아니라 완전히 별도인 domain/time
   benchmark를 측정한다.

소스 하나를 통째로 없애야 할 정도의 체계적 오정렬이면 일부 행만 조용히
남기지 않는다. `data/excluded/`로 이동해 원본, 제외 사유, 행 수를 보존하고
`data/data.txt`와 remediation manifest에 함께 반영한다.

## 6. 자동 품질 gate

현재 `sion-prepare-data` 기본값은 다음을 제공한다.

- UTF-8·JSON·dict 레코드와 문자열 type 확인
- 최소 양쪽 2자, 최대 문자 길이 비율 5.0
- 한·일 script 비율 검사
- 동일 문자열, control character, 과도한 반복 거부
- 숫자·URL·email·placeholder 등 structured span 불일치 경고
- 토큰화 후 한쪽 최대 510 token
- SQLite 기반 bounded-memory exact pair dedup
- target 문장이 서로 다른 split에 들어가는 것을 막는 leakage guard
- raw 레코드 단위 결정적 split
- 합성 레코드와 합성 접두사 파일 train-only 처리

다음은 source별 전처리에서 추가로 **거부 조건**으로 둔다.

| 유형 | 필수 검사 |
|---|---|
| 숫자·금융·의료 | 숫자 값, 부호, 소수점, 통화, 단위, 범위, 날짜·시간 보존 |
| UI·기술 | `{name}`, `%s`, `%1$d`, `${var}`, XML/HTML tag, URL, 명령·코드 span 보존 |
| 법률·행정 | 조·항·호, 기관명, 시행일, 문서 version, 부정 표현 보존 |
| 대화 | speaker/turn 순서, 경어 register, 응답과 장면 metadata 분리 |
| 게임·현지화 | 같은 build/revision의 동일 resource key, 캐릭터·고유명사 일관성 |
| 자막·음성 | 시간 겹침만으로 정렬하지 않고 문장 병합·분할과 화자 전환 확인 |
| 다언어 레코드 | 각 edge가 실제 번역 등가인지 별도 검사; 같은 ID만으로 승인 금지 |

`structured_span_mismatch`는 현재 공통 품질기에서 경고이므로 숫자·UI·의료처럼
정확 보존이 필수인 source는 ingest 단계에서 hard rejection으로 승격한다.
라틴 문자 언어끼리는 script만으로 언어를 구분하지 못하므로 language-ID와
표본 검수를 별도로 적용한다.

## 7. 의미 품질 표본검수

새 source는 먼저 최대 10만 쌍의 pilot만 만들고 다음 표본을 사람이 확인한다.

- source별 최소 1,000쌍
- 규모가 큰 source는 유효 쌍의 최소 0.02%
- 각 domain, 길이 decile, 품질 점수 구간, 언어 edge에서 층화 추출
- 숫자·placeholder·고유명사·부정·존댓말 사례를 별도 oversampling

아래 중 하나면 전체 자동 투입을 중지하고 정렬기나 원천을 수정한다.

- 심각한 오정렬·반대 의미·다른 장면이 표본의 0.5% 이상
- 의미 누락·숫자/단위/부정 오류가 1.0% 이상
- 언어 혼입·깨진 markup·speaker mismatch 같은 material error가 2.0% 이상
- 특정 문서·게임 build·언어 edge에 오류가 군집

수정 뒤에는 같은 표본을 재사용하지 않고 새 seed의 표본으로 다시 승인한다.
오류가 체계적이면 파일 전체를 제외하고 이유를 manifest에 남긴다. 단순 자동
번역 fluency 점수 하나로 이 절차를 대체하지 않는다.

## 8. 중복과 split 누수 방지

중복 제거 단위는 다음과 같이 계층화한다.

1. raw asset hash로 같은 archive/revision 재수집 방지
2. NFC 저장 정규화와 compatibility 기반 dedup key로 exact pair 제거
3. source·target가 뒤집힌 동일 edge도 canonical edge 기준으로 제거
4. source 한쪽만 같은 one-to-many 후보는 의미 검수 후 유지하거나 source
   family 단위로 묶음
5. 문자 5-gram MinHash/SimHash 등으로 template·near duplicate cluster 생성
6. cluster 또는 문서 단위로 한 split에만 배치
7. evaluation-only 원문·번역 양쪽 hash를 금지 목록으로 만들어 train ingest 전에
   차단

현재 indexed 전처리는 exact pair dedup와 target-side split guard를 제공한다.
near duplicate와 외부 benchmark 금지 목록은 raw ingest 단계에서 먼저 처리해야
한다. 특히 템플릿에서 숫자·이름만 바꾼 수백만 행은 “서로 다른 문장”으로
과대계상하지 않고 cluster sampling cap을 둔다.

한 레코드의 여러 번역, 같은 문서의 인접 문장, 같은 대화의 turn은 모두 같은
split에 둔다. train source를 잘라 만든 validation은 in-domain 지표일 뿐이므로
대외 점수와 조기 종료 판단에는 별도 고정 benchmark도 함께 사용한다.

## 9. Holdout 설계

기본 indexed dataset 생성은 실데이터에 validation 0.5%, test 0.5%를 결정적으로
할당한다. 1억 규모에서 무조건 50만/50만을 생성할 필요는 없으므로 최종 규모와
평가 비용에 맞춰 fraction을 고정하되, 한 번 정한 hash split은 재학습 사이에
바꾸지 않는다. 합성 데이터는 validation/test에 절대 넣지 않는다.

최소 평가 묶음은 다음과 같다.

- domain별 실데이터 validation/test
- source·문서가 train과 겹치지 않는 out-of-source test
- 최신 문서를 날짜 기준으로 분리한 temporal test
- 숫자·단위·날짜·ID·placeholder 보존 challenge set
- 장문·다문장·대화 turn challenge set
- 존댓말·반말·문어체·방언 register set
- 각 language edge별 독립 test

현재 `data/data.txt`의 MKQA, PAWS-X dev/test, WMT24++, NTREX, BOUQuET,
FLORES-200은 훈련에서 제외하고 평가 전용 상태를 유지한다. 평가 세트를 학습에
넣은 뒤 그 점수를 일반화 성능으로 보고하지 않는다.

## 10. 재현 가능한 전처리와 집계

다국어 예시는 다음과 같다.

```bash
sion-train-tokenizer --input "data/*.jsonl" \
  --output-dir artifacts/tokenizer-v2 \
  --language-pairs ko ja \
  --language-pairs en ru

sion-prepare-data --input "data/*.jsonl" \
  --tokenizer artifacts/tokenizer-v2/sion.model \
  --output-dir artifacts/dataset-v2 \
  --language-pairs ko ja \
  --language-pairs en ru \
  --validation-fraction 0.005 \
  --test-fraction 0.005 \
  --dedup-backend sqlite
```

기존 데이터셋 디렉터리에 새 shard를 섞지 않는다. 현재 코드는 non-empty output
디렉터리를 거부하므로 매 remediation마다 새 디렉터리를 사용하고, 승인 후 config
경로를 바꾼다.

최종 집계는 `artifacts/dataset-v2/manifest.json`의 전체 `stats.valid_pairs`,
`stats.synthetic_pairs`, source별 stats와 raw remediation manifest를
대조한다. 다음 등식이 모두 성립해야 1억 달성을 선언한다.

```text
sum(source valid_pairs) = global valid_pairs
real_pairs + synthetic_pairs = 100,000,000
train + validation + test = valid_pairs
excluded/invalid/duplicate/too_long 행은 valid_pairs에 미포함
```

매 추가 batch마다 다음을 기록한다.

- 원천명, URL, revision/commit, archive/file SHA256, 취득일
- 물리 JSONL 행 수와 확장 전·후 pair 수
- 언어 edge별 실데이터·합성 pair 수
- invalid, quality_filtered, duplicate, too_long, split_conflicts 수
- 자동 gate별 거부 수와 의미 검수 표본·오류율
- train/validation/test 수와 benchmark 금지 목록 검사 결과
- 최종 output SHA256과 재구축 명령

이 기록 없이 “파일이 크다”거나 `wc -l`이 1억이라는 이유로 목표 달성을
선언하지 않는다.
