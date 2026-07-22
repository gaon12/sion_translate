# 번역 시스템 비교 가이드

이 문서는 특정 시스템이 항상 더 낫다고 단정하지 않고, 같은 입력과 같은 기준으로
재현 가능한 비교를 만드는 방법을 설명합니다. 서비스 출력은 호출 시점, 제품 버전,
요금제와 옵션에 따라 바뀔 수 있으므로 결과 JSONL에는 생성 날짜와 설정을 별도 메모로
남기는 편이 좋습니다.

## 비교 대상별 관점

| 시스템 | 확인할 강점 | 주의할 점 |
|---|---|---|
| KJ-X | 한↔일 전용, 완전 로컬 실행, 코드·생성 옵션 통제, slot 기반 용어 강제 | 작은 전용 모델의 일반화 한계, custom PyTorch loader 필요, 공개 점수만으로 프로덕션 품질을 보장할 수 없음 |
| [LibreTranslate](https://docs.libretranslate.com/) | 오픈 소스 API, self-host 가능, 외부 전송 없이 운영 가능 | 설치된 언어 모델에 따라 지원 범위가 달라지고 직접 언어쌍이 없으면 보통 영어를 경유하므로 `/languages` 결과를 기록해야 함 |
| [Papago](https://api.ncloud-docs.com/docs/en/ai-naver-papagowebsitetranslation-translation) | 공식 API가 한국어↔일본어 방향을 명시적으로 지원 | 폐쇄형 클라우드 서비스이므로 모델 버전 고정이 어렵고 인증·요금·이용약관 확인 필요 |
| [Google Cloud Translation](https://cloud.google.com/translate/docs) | 폭넓은 언어 지원, glossary와 adaptive translation 같은 옵션 | 클라우드 인증·비용이 필요하고 Basic/Advanced 및 적응형 설정을 섞으면 공정 비교가 아님 |
| [DeepL](https://developers.deepl.com/docs) | 문맥(`context`), glossary, 언어별 style rule 등 번역 제어 기능 | 기능 지원 범위가 언어·API 버전에 따라 다르고 인증·비용이 필요함 |
| [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | 100개 언어 사이 직접 번역, 로컬 재현, 버전 고정 가능 | 418M 가중치의 메모리·지연 비용, 오래된 범용 checkpoint, upstream 조건을 별도로 확인해야 함 |
| [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 매우 넓은 언어 범위, 로컬 재현, 저자원 언어 연구 baseline | CC-BY-NC 4.0, 모델 카드가 연구용·비프로덕션을 명시, 512토큰을 넘는 입력에서 품질 저하 가능 |

표의 강점은 품질 우승을 뜻하지 않습니다. 각 시스템이 제공하는 배포·제어 특성을
뜻하며, 실제 품질은 동일한 JSONL 결과로 확인해야 합니다.

## 공정한 실행 규칙

1. `examples/comparison_cases.jsonl`을 수정했다면 모든 시스템을 다시 실행합니다.
2. 원문 언어와 목표 언어를 명시하고 자동 언어 감지는 끕니다.
3. 문장별 번역을 사용하고, 어떤 시스템만 추가 문맥이나 glossary를 받지 않게 합니다.
4. beam 수, 모델 revision, API 제품명, 호출 날짜를 함께 기록합니다.
5. 실패한 문장도 삭제하지 말고 오류로 기록한 뒤 같은 조건으로 재시도합니다.
6. API 키, 응답 헤더, 계정 정보는 JSONL에 넣지 않습니다.

## 사람이 볼 항목

- 의미 누락·추가와 부정 표현 반전
- 존댓말과 화자 관계
- 동음이의어의 문맥 해소
- 숫자, 통화, 날짜, 파일명, HTTP 상태 코드 보존
- 고유명사 음역과 문서 전체 일관성
- 일본어 조사·한국어 조사, 어순과 자연스러움
- 장문에서 주어·조건절·인과 관계 유지

chrF/BLEU가 높아도 정답 표현과 다른 올바른 번역이 낮게 채점될 수 있습니다. 반대로
표면이 비슷해도 숫자나 부정이 틀릴 수 있으므로 문장별 표를 반드시 같이 검토합니다.

## 라이선스와 데이터 경계

비교 코드는 MIT이지만 각 서비스, 모델, 입력 문장과 생성 출력의 조건은 별개입니다.
제3자 benchmark나 API 출력을 공개 커밋하기 전에는 재배포 가능 여부를 확인하세요.
이 저장소는 `benchmarks/`, `comparison_outputs/`, `reports/`를 기본적으로 무시합니다.
