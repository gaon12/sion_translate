# Corpus Coverage Gaps and Real-Data Sources

Survey date: 2026-07-31. This document records source URLs and licensing notes only. Corpus contents and row-level statistics remain in `data/data.txt`, the untracked local data ledger.

## Why this document exists

The release diagnostic showed that category-level chrF rank closely followed category-level row-count rank. Dialect had 3,409 rows and chrF 9.89; idioms had 2,424 rows and chrF 36.69; customer support had 4,895 rows and chrF 35.96. By comparison, academic text had 1,330,632 rows and chrF 78.36, while structured strings had 346,352 rows and chrF 93.58. These failures therefore point to coverage gaps rather than only architecture or hyperparameter problems.

From 2026-07-30 through 2026-07-31, rule-based generation added 88,000 dialect rows, 30,994 internet-colloquial rows, and 4,013 structured-domain rows. However, **synthetic data cannot replace real data**. The dialect and colloquial generators modify endings in standard-language sentences, so their vocabulary and pragmatics remain standard. The structured-domain generator contains only templates, leaving unstructured legal and medical text, such as judicial narratives and clinical notes, uncovered.

The `[verified]` label below means the page was opened and reviewed directly. `[search-only]` means only search results were reviewed. Do not erase this distinction. Presenting an unverified source as verified forces the next reviewer to repeat the research.

## Priority 1: available by application and directly addresses a gap

### AIHub 71263 Korean-Chinese and Korean-Japanese broadcast-content parallel corpus `[verified]`

<https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71263>

- **750,000 Korean-Japanese pairs**, divided into ko→ja and ja→ko directions
- Domains: educational programming, observational entertainment, and reality/variety entertainment
- Format: text plus JSON labels in split archives; Linux commands are required to combine the parts
- Access: login and approval required; **applications are limited to Korean nationals**

This is the largest single candidate for improving the measured colloquial chrF score of 30.45 with real data. Observational and reality/variety programming is substantially closer to spontaneous speech than the existing broadcast-dialogue shards.

**Ingestion caution:** because this is broadcast/subtitle data, measure overlap with existing broadcast and transcription shards before admitting it.

### AIHub 71411 Korean-Chinese and Korean-Japanese daily-life and colloquial parallel corpus `[verified]`

<https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71411>

- 90,003 total pairs divided evenly across six language pairs at 16.67% each, yielding approximately 30,000 Korean-Japanese pairs
- Three styles: **chat**, colloquial, and written
- Access: login and approval required; limited to Korean nationals

The **chat style is the only identified real-data source for the internet-colloquial gap**. At the survey date, this register had zero real rows and only rule-generated synthetic data. Although the dataset is small, it can serve as a ground-truth reference for behavior the synthetic generator attempts to imitate.

## Priority 2: publicly downloadable, but licensing still requires confirmation

### Japan Tourism Agency multilingual regional-tourism commentary database `[verified]`

<https://www.mlit.go.jp/tagengo-db/>

- Commentary in five languages: Japanese, English, Simplified Chinese, Traditional Chinese, and Korean
- Provides a **13.8 MB bulk Excel download containing all commentary**

This source addresses tourism and transportation coverage. Resolve both issues below before ingestion.

1. The footer says `Japan Tourism Agency. All Rights Reserved`, and the terms-of-use page was not reviewed. **Read the terms before using the data.**
2. The site states that the Japanese edition is a provisional translation of the English original. The Korean-Japanese relation may therefore be an **English-pivoted translation** rather than a direct translation. Inspect a sample manually.

### Japan Tourism Agency Korean commentary glossary `[search-only]`

- <https://www.mlit.go.jp/kankocho/jirei_shien/tagengo_kor.html>
- <https://www.mlit.go.jp/kankocho/content/001732567.pdf> (glossary, March 2026 edition)

This is a glossary, not a corpus, and should be used as such. The multilingual guidance contains English, Chinese, and Korean equivalents for more than 400 terms.

### Korea Tourism Organization multilingual smart-tourism-city POI data `[search-only]`

- <https://www.data.go.kr/data/15124972/fileData.do> (Daegu)
- <https://www.data.go.kr/data/15124975/fileData.do> (Ulsan)

The files provide attraction names, short descriptions, detailed descriptions, and operating hours in Korean, English, Japanese, and Chinese. They are marked as Korea Open Government License Type 1, which permits commercial use and modification, but this still requires confirmation.

The city-level datasets are small. Their direction complements the Japan Tourism Agency database because they describe Korean destinations. Using both would strengthen tourism vocabulary in both directions.

## Priority 3: real sources with low yield or high processing cost

### Ministry of Foreign Affairs Treaty Information System `[verified]`

<https://treatyweb.mofa.go.kr>

The system contains 2,835 treaties. Korean-Japanese bilateral treaties have **equally authentic Korean and Japanese texts**; for example, the 1965 Treaty on Basic Relations states that it was written in Korean and Japanese, both equally authentic. This is genuine legal-domain parallel data.

However, the side-by-side viewer supports **Korean and English**, not Korean and Japanese, and there is no Korean-Japanese parallel export. The Excel export contains only list-level metadata. Filtering to bilateral Korean-Japanese treaties would likely yield only hundreds of documents, and article-level alignment would have to be built manually. The effort-to-yield ratio is poor, but this appears to be the only practical source of genuinely authentic legal translations in this domain.

### COJADS Corpus of Japanese Dialects, National Institute for Japanese Language and Linguistics `[verified]`

<https://www2.ninjal.ac.jp/cojads/>

The corpus covers the 1977–1985 Emergency Survey of Regional Dialects conducted by the Agency for Cultural Affairs: approximately 200 locations across all 47 prefectures and about 2,500 hours of discourse. Dialect speech is paired with standard Japanese equivalents.

This is **monolingual Japanese** dialect↔standard data, not Korean-Japanese parallel data. It is useful only as real data for the Japanese-dialect-to-standard-Japanese direction. At the survey date, that direction was entirely synthetic, so COJADS would be valuable as a ground-truth reference. Chūnagon user registration and a separate access application are required.

### OpenWHO document-level parallel corpus `[search-only]`

<https://aclanthology.org/2025.wmt-1.8/>

The corpus contains 2,978 WHO e-learning documents and 26,824 sentences in more than 20 languages. **Korean and Japanese inclusion was not confirmed.** If both are present, this would be an unusual resource that addresses both medical-domain coverage and document-level context; at the survey date, only 0.26% of examples exceeded 201 characters.

## Sources investigated and excluded

Record exclusions so future work does not repeat the same search.

### Japanese Law Translation database `[verified]`

<https://www.japaneselawtranslation.go.jp/>

The database publishes more than 800 laws, but translations are **English only**. It has no Korean translations, and its downloadable bilingual dictionary is Japanese-English. It cannot provide legal Korean-Japanese pairs.

### Korean statutory translations and the Korean Law Information Center `[verified]`

<https://elaw.klri.re.kr/>

These services likewise provide **Korean-English only** and do not publish Japanese translations.

In other words, **both governments translate legislation only into English**. There is no known path to direct Korean-Japanese parallel statutes. Pivoting through English does not solve the problem because the two sides translate different laws, so sentence alignment is not valid. Outside treaties, no real legal-domain source was identified.

### Tatoeba `[search-only]`

The licensing is clear (CC BY or CC0), but the local ledger already records an exclusion decision because of translation-mining noise and duplicates. Any reconsideration should begin with a separate quality sample of the Korean-Japanese subset.

## Gaps with no identified source

- **E-commerce** (product descriptions, reviews, and shipping notices): no public Korean-Japanese parallel corpus was found. Product data is platform-owned, and scraping would violate terms of service.
- **Administrative civil-service requests:** multilingual local-government guidance is scattered across sites and is not distributed in aligned parallel form.
- **IT technical documentation:** the only practical route is joining Korean and Japanese localization files from open-source projects by identical resource key. This method has already been validated with the Firefox and VS Code localization shards. Candidates include KDE, Django, and Kubernetes documentation. GNOME was previously excluded for quality reasons, so inspect a sample before reconsidering it.
- **Additional languages outside Korean-Japanese:** the stated goal is general translation, but the survey snapshot had zero rows outside Korean-Japanese. AIHub Korean-English and Korean-Chinese corpora (dataSetSn 124, 125, 128, and 129; 1.3–1.5 million pairs each) were available immediately. At that time, however, the tokenizer was Korean-Japanese-specific, so adding language pairs required a redesign and full retraining rather than a simple data append. The current release 1.5 software contract supports configuration-driven language graphs, but every new graph still requires a newly trained compatible tokenizer, dataset, and model.

## Recommended acquisition order

1. Apply for AIHub 71263 (750,000 pairs) and 71411 (approximately 30,000 pairs). Both require approval, so apply first and work on the remaining sources during review.
2. Review the Tourism Agency database terms, then parse the Excel file and inspect whether samples are English-pivoted.
3. Apply for COJADS access.
4. Confirm whether OpenWHO contains both Korean and Japanese.
5. Estimate the manual effort for authentic Korean-Japanese treaty alignment before committing to it.

## Ingestion gates

Every new dataset must pass the existing gates without exception.

```text
scripts/data/audit_generated_shards.py      # Template collapse and holdout leakage
sion-check-preservation                     # Numbers, signs, units, and writing systems
scripts/data/screen_protected_content.py    # Minor-content screening
```

For sources with questionable alignment, first run the preparation stage in `scripts/data/recover_shard.py` to remove placeholder gaps and segmentation whitespace.
