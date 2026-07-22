"""추론 공용 도우미.

학습이 끝난 모델(exports/)을 찾아 불러오고, 문장 목록을 배치로 번역합니다.
`kjx-translate`(대화형 번역)와 `kjx-augment`(역번역 데이터 증강)가 공유합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from kjx.glossary import Glossary, apply_source_placeholders, restore_targets
from kjx.tokenizer import SLOT_SYMBOLS, KJTokenizer
from kjx.training.export import load_exported_model


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
        f"{output_dir} 아래에 내보낸 모델이 없습니다. 먼저 kjx-train 으로 학습하세요."
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
        self.tokenizer = KJTokenizer(tokenizer_path)
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
    ) -> list[str]:
        """문장 목록을 ``target_language`` 로 번역합니다.

        입력 언어는 지정할 필요가 없습니다 — 모델 입력의 <2xx> 태그가
        '어느 언어로 번역할지'만 지시하며 (학습 때와 같은 방식),
        나머지 한쪽 언어가 입력이라고 가정합니다.

        ``glossary`` 를 주면 지정한 용어를 정해진 대응어로 강제합니다.
        (원문에서 slot 토큰으로 치환 → 번역 → 대응어로 복원.)
        모델이 slot 을 보존하지 못해 누락된 용어는 ``append_missing_glossary``
        가 참이면 문장 끝에 괄호로 덧붙여 최소한의 강제를 보장합니다.
        """
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
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
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
            generated = self.model.generate(
                input_ids.to(self.device),
                attention_mask.to(self.device),
                bos_id=self.tokenizer.bos_id,
                eos_id=eos,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )
            for row_index, row in enumerate(generated.tolist()):
                # BOS/EOS/패딩 등 특수 토큰을 걷어내고 문자열로 복원
                tokens = [token for token in row if token not in special_ids]
                text = self.tokenizer.decode(tokens)
                if slot_maps and slot_maps[row_index]:
                    text, missing = restore_targets(text, slot_maps[row_index])
                    if missing and append_missing_glossary:
                        # 모델이 slot 을 흘린 경우: 최소한의 용어 보존을 위해
                        # 강제 용어를 괄호로 덧붙입니다.
                        text = f"{text} ({', '.join(missing)})"
                results.append(text)
        return results
