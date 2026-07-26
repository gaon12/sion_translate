"""추론 공용 도우미.

학습이 끝난 모델(exports/)을 찾아 불러오고, 문장 목록을 배치로 번역합니다.
`sion-translate`(대화형 번역)와 `sion-augment`(역번역 데이터 증강)가 공유합니다.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import torch

from sion_translate.glossary import Glossary, apply_source_placeholders, restore_targets
from sion_translate.rerank import select as rerank_select
from sion_translate.revision import DRAFT_SEPARATOR, serialize_revision_input
from sion_translate.tokenizer import SLOT_SYMBOLS, SionTokenizer
from sion_translate.training.export import load_exported_model


def find_exported_model(
    output_dir: str | Path,
    *,
    int8: bool = False,
) -> Path:
    """가장 좋은 내보내기 모델을 찾습니다.

    우선순위: 사후학습 best/latest → 사전학습 best/latest → 기존 단일-stage 경로.
    (EMA 가중치가 보통 번역 품질이 더 좋습니다. --int8 이면 양자화본을 찾습니다.)
    """
    output_dir = Path(output_dir)
    filenames = ["model_int8.pt"] if int8 else ["model_ema.pt", "model.pt"]
    export_roots = [
        output_dir / "posttrain" / "exports",
        output_dir / "pretrain" / "exports",
        output_dir / "exports",  # 이전 버전 산출물과의 호환
    ]
    for exports in export_roots:
        for stage in ("best", "latest"):
            for filename in filenames:
                candidate = exports / stage / filename
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(
        f"{output_dir} 아래에 내보낸 모델이 없습니다. 먼저 sion-train 으로 학습하세요."
    )


class Translator:
    """내보낸 모델 + 토크나이저로 문장을 번역하는 얇은 래퍼."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        *,
        device: str | torch.device | None = None,
    ):
        self.tokenizer = SionTokenizer(tokenizer_path)
        if not self.tokenizer.splits_digits:
            # split_digits 없이 학습된 토크나이저는 숫자를 덩어리로 암기하므로
            # 금액·용량·날짜가 조용히 다른 값으로 바뀔 수 있습니다. 출력만 보고는
            # 알아채기 어려우므로 로드 시점에 한 번 알립니다.
            warnings.warn(
                f"{tokenizer_path} 는 숫자를 자릿수로 분리하지 않습니다. "
                "금액·용량·날짜가 다른 값으로 바뀔 수 있으니 숫자가 중요한 문장은 "
                "사람이 검토하세요. 재학습 시에는 split_digits 를 켜십시오.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.model, self.model_config, self.pad_id = load_exported_model(model_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # 양자화 모델은 CPU 전용 커널을 쓰므로 CPU 에 남깁니다.
        self.quantized = any(
            "quantized" in type(module).__module__ for module in self.model.modules()
        )
        if not self.quantized:
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    @property
    def languages(self) -> tuple[str, ...]:
        """이 모델이 지원하는 언어 (토크나이저의 <2xx> 태그에서 자동 인식)."""
        return self.tokenizer.languages

    def _other_language(self, target_language: str) -> str:
        """양방향 모델에서 목표 언어가 아닌 쪽을 원문 언어로 간주합니다."""
        others = [lang for lang in self.languages if lang != target_language]
        return others[0] if len(others) == 1 else ""

    @torch.no_grad()
    def translate(
        self,
        texts: Sequence[str],
        *,
        target_language: str,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        max_new_tokens: int = 256,
        batch_size: int = 16,
        glossary: Glossary | None = None,
        append_missing_glossary: bool = True,
        num_candidates: int = 0,
        rerank: str = "mbr+qe",
        temperature: float = 0.7,
        top_k: int = 0,
        return_rerank_details: bool = False,
    ) -> list[str]:
        """문장 목록을 ``target_language`` 로 번역합니다.

        입력 언어는 지정할 필요가 없습니다 — 모델 입력의 <2xx> 태그가
        '어느 언어로 번역할지'만 지시하며 (학습 때와 같은 방식),
        나머지 한쪽 언어가 입력이라고 가정합니다.

        ``glossary`` 를 주면 지정한 용어를 정해진 대응어로 강제합니다.
        (원문에서 slot 토큰으로 치환 → 번역 → 대응어로 복원.)
        모델이 slot 을 보존하지 못해 누락된 용어는 ``append_missing_glossary``
        가 참이면 문장 끝에 괄호로 덧붙여 최소한의 강제를 보장합니다.

        ``num_candidates`` 를 1 이상으로 두면 beam 결과에 더해 그 수만큼 확률적
        후보를 뽑고 ``rerank`` 방식으로 하나를 고릅니다 (``sion_translate.rerank``
        참고). 재학습 없이 추론 계산량만 늘리는 경로이며, 후보 목록의 첫 번째는
        항상 beam 결과이므로 동점이면 기존 동작이 유지됩니다.

        ``return_rerank_details`` 가 참이면 문자열 대신 ``RerankResult`` 목록을
        돌려줍니다 — 어느 후보가 왜 뽑혔는지 확인할 때 씁니다.
        """
        if num_candidates < 0:
            raise ValueError("num_candidates 는 0 이상이어야 합니다")
        if return_rerank_details and num_candidates < 1:
            raise ValueError("return_rerank_details 는 num_candidates 가 1 이상일 때만 씁니다")
        tag_id = self.tokenizer.language_tags.get(target_language)
        if tag_id is None:
            raise ValueError(
                f"지원하지 않는 언어: {target_language} (지원: {sorted(self.languages)})"
            )
        source_language = self._other_language(target_language)
        eos = self.tokenizer.eos_id
        results: list[str] = []
        special_ids = {
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            eos,
            self.tokenizer.mask_id,
            *self.tokenizer.language_tags.values(),
            *self.tokenizer.denoise_tags.values(),
        }
        def restore(row: Sequence[int], slot_map: dict[str, str] | None) -> str:
            """생성 토큰을 문자열로 되돌리고 글로서리 slot 을 복원합니다."""
            tokens = [token for token in row if token not in special_ids]
            text = self.tokenizer.decode(tokens)
            if slot_map:
                text, missing = restore_targets(text, slot_map)
                if missing and append_missing_glossary:
                    # 모델이 slot 을 흘린 경우: 최소한의 용어 보존을 위해
                    # 강제 용어를 괄호로 덧붙입니다.
                    text = f"{text} ({', '.join(missing)})"
            return text

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            # QE 는 원문과 대조하므로 slot 치환 전의 문장을 따로 보관합니다.
            sources = list(chunk)
            # 글로서리 적용: 원문의 용어를 slot 으로 치환하고 문장별 매핑을 보관.
            slot_maps: list[dict[str, str]] = []
            if glossary is not None and source_language:
                prepared: list[str] = []
                for text in chunk:
                    masked, slot_map = apply_source_placeholders(
                        text,
                        glossary,
                        source_language=source_language,
                        target_language=target_language,
                        slot_symbols=SLOT_SYMBOLS,
                    )
                    prepared.append(masked)
                    slot_maps.append(slot_map)
                chunk = prepared
            encoded = [
                [tag_id, *self.tokenizer.encode(text), eos] for text in chunk
            ]
            longest = max(len(ids) for ids in encoded)
            input_ids = torch.full(
                (len(encoded), longest), self.pad_id, dtype=torch.long
            )
            attention_mask = torch.zeros(
                (len(encoded), longest), dtype=torch.bool
            )
            for row, ids in enumerate(encoded):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = True
            device_inputs = input_ids.to(self.device)
            device_mask = attention_mask.to(self.device)
            generated = self.model.generate(
                device_inputs,
                device_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=eos,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )
            beam_texts = [
                restore(row, slot_maps[index] if slot_maps else None)
                for index, row in enumerate(generated.tolist())
            ]

            if num_candidates < 1:
                results.extend(beam_texts)
                continue

            # beam 결과를 첫 후보로 두고 확률적 후보를 덧붙입니다. 동점이면
            # 첫 후보가 유지되므로 재순위가 기존 동작보다 나빠질 일이 없습니다.
            sampled = self.model.sample(
                device_inputs,
                device_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=eos,
                num_samples=num_candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            for row_index, source_text in enumerate(sources):
                slot_map = slot_maps[row_index] if slot_maps else None
                candidates = [beam_texts[row_index]]
                for sample_row in sampled[row_index].tolist():
                    candidate = restore(sample_row, slot_map)
                    # 같은 문장을 여러 번 채점할 이유가 없습니다.
                    if candidate not in candidates:
                        candidates.append(candidate)
                outcome = rerank_select(
                    source_text,
                    candidates,
                    strategy=rerank,
                    target_language=target_language,
                )
                results.append(outcome if return_rerank_details else outcome.text)
        return results

    @torch.no_grad()
    def revise(
        self,
        texts: Sequence[str],
        drafts: Sequence[str],
        *,
        target_language: str,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        max_new_tokens: int = 256,
        batch_size: int = 16,
    ) -> list[str]:
        """``원문 + 초안`` 을 받아 고친 번역을 돌려줍니다.

        ``sion-revise-data`` 로 만든 ``원문 <draft> 초안 → 번역`` 예제로 학습한
        모델에서만 의미가 있습니다. 그렇게 학습하지 않은 모델에 쓰면 ``<draft>``
        뒤를 그냥 원문의 일부로 읽으므로 결과가 나빠집니다.

        토크나이저에 ``<draft>`` 가 없으면 (2026-07 이전 토크나이저) 오류를 냅니다 —
        구분자가 여러 토큰으로 쪼개져 학습 때와 다른 입력이 되기 때문입니다.
        """
        if self.tokenizer.draft_id is None:
            raise ValueError(
                f"이 토크나이저에는 {DRAFT_SEPARATOR} 제어 토큰이 없어 초안 수정을 "
                "쓸 수 없습니다. sion-train-tokenizer 로 다시 학습하십시오."
            )
        if len(texts) != len(drafts):
            raise ValueError(f"원문 {len(texts)}개와 초안 {len(drafts)}개의 수가 다릅니다")
        return self.translate(
            [
                serialize_revision_input(source, draft)
                for source, draft in zip(texts, drafts, strict=True)
            ],
            target_language=target_language,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
