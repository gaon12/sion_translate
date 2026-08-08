# SentencePiece 0.2.2 SIGSEGV 조사 기록

## 결론

원인은 코퍼스 크기나 특정 문자가 아니라 SentencePiece **0.2.2의 다중 스레드
trainer normalization 회귀**입니다. upstream commit
[`de32a1e`](https://github.com/google/sentencepiece/commit/de32a1eb2e8ae1d63380e586685db4863b84a9a2)가 학습용
`Normalizer::Normalize(string_view)`에서 byte-offset 벡터 생성을 생략하도록
바꾼 뒤 발생합니다.

프로젝트는 `sentencepiece==0.2.1`로 고정합니다. `train_tokenizer()`는 0.2.2가
설치된 채 `num_threads > 1`이면 코퍼스를 읽기 전에 중단합니다. 같은 입력을
0.2.2에서 꼭 처리해야 한다면 실측으로 통과한 `num_threads=1`이 느린 우회입니다.

## 재현과 반증

실패 입력은 `corpus_balanced_short.txt`입니다. 물리 20,355,467줄 중 예약 문자가
있는 12줄을 SentencePiece가 건너뛰므로 실제 trainer 입력은 20,355,455문장,
UTF-8 2,774,987,960바이트, Unicode scalar 1,057,602,757개입니다.

| 조건 | 결과 | 실행 시간 |
|---|---:|---:|
| 0.2.2 wheel, 4 threads, file input, char | SIGSEGV (`-11`) | 74.12초 |
| 0.2.2 wheel, 1 thread, 같은 입력 | 통과 | 259.66초 |
| 0.2.1 wheel, 4 threads, 같은 입력 | 통과 | 138.40초 |
| 0.2.2 C++ CLI, 4 threads | SIGSEGV | 재현 |
| 0.2.2 + 새 thread pool만 직접 `std::thread`로 교체 | SIGSEGV (`-11`) | 78.16초 |
| 0.2.2 + normalization offset만 복원 | 통과 | 117.27초 |

마지막 두 행은 나머지 0.2.2 소스를 고정한 one-change A/B입니다. 따라서 같은
시기에 들어온 thread-pool 변경
[`c5ed56a`](https://github.com/google/sentencepiece/commit/c5ed56a5501676f38804662893ec345b7fa1570b)는
원인이 아니며, offset 생략 변경 `de32a1e`가 필요조건입니다.

RelWithDebInfo로 빌드한 0.2.2 `spm_train`의 native stack은 다음 순서였습니다.

```text
glibc sysmalloc
std::string::_M_append
sentencepiece::normalizer::PrefixMatcher::GlobalReplace (normalizer.cc:384)
TrainerInterface::LoadSentences normalization worker
sentencepiece::ThreadPool::Impl::WorkLoop
```

`Done! preprocessed`와 `Making suffix array`보다 전이므로 seed-piece/suffix-array
경로에는 도달하지 않았습니다. Python binding을 쓰지 않는 C++ CLI에서도 같아
`sentence_iterator`도 원인이 아닙니다.

다음 가설도 실측으로 배제했습니다.

- 더 작은 실패 입력(1.08G 원소)이 더 큰 성공 입력(1.97G)보다 작으므로 크기,
  문장 수, 문자 수, 2³¹ 경계가 아닙니다.
- 256GiB를 주어도 같은 위치에서 죽고, 성공 실행 peak는 49.8GiB 이하여서 단순
  OOM이 아닙니다.
- 실패 입력은 strict UTF-8입니다. C0/C1 문자가 있는 8개 행을 각각/함께 학습한
  36개 조합은 모두 통과했습니다.
- 4KiB 초과 행을 없앤 입력이 원본 실패 입력과 정확히 같은 trainer stream을
  만드므로 긴 행/긴 낱말 문제가 아닙니다.
- 실패 코퍼스 전체를 기존 모델의 normalizer로 순차 및 16 threads 처리해도
  통과했습니다. 문제는 0.2.2 trainer의 offset 없는 병렬 경로입니다.

### 최소화가 멈춘 지점

ordered prefix를 `sentence_iterator`로 이분 탐색했습니다. 물리 20,350,000행
(예약 문자 12행 제외 후 20,349,988문장)은 통과했고 20,350,172행
(20,350,160문장)은 두 번 모두 `-11`로 죽었습니다. 그러나 바로 앞의 **동일한
20,350,171행 prefix**(2,790,882,460바이트, 20,350,159문장)는 한 번 통과한 뒤
재시험에서 `-11`로 죽었습니다. 따라서 행 단위의 결정적 최소 경계는 없습니다.
스케줄링/allocator 배치에 따라 경계 부근 결과가 달라지는 native memory bug입니다.

20,350,172번째 행은 Unicode category `Lo/Po/Zs`만 있는 평범한 302자 한국어
문장(SHA-256
`2eab3a8f81b7e209681db5bcc63b1effa6f4052f78d817f2dadf90b61087af35`)입니다.
그 행 하나만 0.2.2/4 threads로 학습하면 0.21초에 통과했습니다. 즉 마지막 행이
독립적인 bad row인 것도 아닙니다. 작고 안정적인 corpus reproducer까지 줄이는
일은 여기서 막혔고, 보존한 **안정 재현물**은 8회 연속 실패한 전체
`corpus_balanced_short.txt`입니다. 대신 원인 판별에는 나머지 코드를 고정한
offset 복원 one-change A/B가 결정적입니다.

## 단일어 비중 복구 실증

기존 실패 코퍼스는 병렬 11,997,047문장과 단일어 ja 4,251,477문장, ko
4,251,476문장으로 구성됩니다. 단일어/병렬 비율은 0.709이며, 새 설정 0.40이
요구하는 언어별 약 3.62M문장보다 더 강한 스트레스 조건입니다. 이 코퍼스로
SentencePiece 0.2.1, 16 threads, unigram 48,000 모델을 실제 학습했고 1,308.99초에
완료했습니다.

`easy_run._verify_tokenizer(model, Path("data"))` 결과는 1,312,459개 토큰 중
byte fallback 670개, **0.0510%**로 0.2% 상한을 통과했습니다. `넼`은 dummy
prefix를 제외하면 `넼` 한 조각이고, `38,720`은 `3 8 , 7 2 0`으로 나뉩니다.
모델 SHA-256은
`6149c905411a38ca900ee7d45fd29594ea895a01d140c950ddc1fb1854735454`입니다.

foundation 입력도 따로 확인했습니다. 언어별 결정적 표본에서 기존 0.13 모델과
후보를 비교한 결과입니다.

| 언어 | 문장 | 모델 | byte fallback | pieces/character |
|---|---:|---|---:|---:|
| ko | 18,500 | 기존 0.13 | 0.0255% | 0.475843 |
| ko | 18,500 | 고단일어 후보 | 0.0267% | **0.453025** |
| ja | 19,166 | 기존 0.13 | 0.0038% | 0.539918 |
| ja | 19,166 | 고단일어 후보 | 0.0039% | **0.526961** |

fallback은 사실상 같고, 조각 수는 ko에서 4.79%, ja에서 2.40% 줄었습니다. 이
절의 스트레스 후보는 당시 재현 작업 폴더에만 보존했고 배포 artifact는 교체하지
않았습니다. 다음 절의 production artifact를 만들 때 tokenizer와 token-id dataset을
함께 교체했습니다.

### 2026-08-08 production 0.40 검증

실제 production 경로인 `scripts/modal_train_tokenizer.py`로 SentencePiece 0.2.1,
preprocess workers 16개, SentencePiece threads 16개, unigram 48,000 모델을
학습했습니다. run ID는
`ratio-040-fe9a4799de05-6b2fd43b3111-20260808t081122z-d0d1ac`입니다.

- 병렬 18,177,344문장
- 단일어 ja 3,445,471문장 + ko 3,308,940문장
- 총 24,931,755문장; 문자 계획 pass와 최종 trainer pass 모두 정확히
  24,931,755문장
- required characters 9,751개, SHA-256
  `bc1e656d90cd109cb8631583c588aeb9b51125dafa31744c9016eba392ed1e86`
- 모델 SHA-256
  `082695f2d42314061fe3c5431816ef501cb2257d6af6c334f726816aea1bdc98`

`easy_run._verify_tokenizer()`는 1,304,691개 표본 토큰 중 byte fallback 736개,
**0.0564%**로 0.2% 상한을 통과했습니다. `넼`은 dummy prefix를 제외하면 한
조각이고, `38,720`은 `3 8 , 7 2 0`으로 나뉩니다. 같은 모델로 병렬 dataset도
다시 만들었고, dataset manifest와 freshness fingerprint가 위 모델 SHA를
기록하는지 확인했습니다. 정확한 source/artifact hash와 두 pass count는 tokenizer
옆의 `training_manifest.json`에 보존합니다.

## 다시 실행하는 법

진단은 프로젝트의 고정 버전을 바꾸지 않도록 별도 가상환경에서 합니다.
`tokenizer_plan.json`과 코퍼스는 크기 때문에 Git에는 넣지 않고, 프로젝트의
ignored 로컬 산출물 경로인 `artifacts/sentencepiece_repro/`에 둡니다.

```powershell
py -3.11 -m venv C:\tmp\spm-022
C:\tmp\spm-022\Scripts\python.exe -m pip install sentencepiece==0.2.2
C:\tmp\spm-022\Scripts\python.exe scripts\diagnose_sentencepiece_crash.py `
  --corpus artifacts\sentencepiece_repro\corpus_balanced_short.txt `
  --plan artifacts\sentencepiece_repro\tokenizer_plan.json `
  --model-type char --threads 4
```

보존한 입력의 SHA-256은 corpus
`855cdc67378272490691279deb2ce5d4160dc56d4d4e4373e889d683f98ad12a`, plan
`ef286833960dd6ac5cac22852b8a9fdcc0a0762fd14a97edd71fb1785a7858f6`입니다.

부모 프로세스가 trainer를 자식으로 실행하므로 native crash 뒤에도 return code와
시간을 JSON으로 남깁니다. `--maximum-sentences N`은 같은 파일의 ordered prefix를
`sentence_iterator`로 시험해 축소할 때 씁니다. file/iterator 양쪽이 같은 native
정규화 경로로 들어간다는 점을 감안해, 최종 확인은 옵션 없이 file input으로
합니다.

## 운영상 의미

기존 `foundation.tokenizer_sample_ratio: 0.13`은 회귀를 우연히 피한 코퍼스 조합일
뿐 안전 상한이 아니었습니다. 버전을 0.2.1로 고정한 뒤 0.40으로 복원합니다.
비율은 학습에 들어가는 단일어 문장과 `required_chars`를 계산하는 단일어 문장
모두에 적용됩니다. 즉 표본 밖 문자의 빈도까지 전량 세는 설정은 아니며, 예전
주석의 “내용 문자는 전량 보존” 주장은 사실이 아니었습니다.

모델 sidecar에는 SentencePiece 버전, 실제 언어별 표본 수, required-character
개수와 SHA-256을 기록합니다. metadata version은 2를 유지해 기존 v1/v2 숫자 분리
정책 로딩을 깨지 않습니다.
