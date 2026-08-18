"""추론 공용 도우미.

학습이 끝난 모델(exports/)을 찾아 불러오고, 문장 목록을 배치로 번역합니다.
`sion-translate`(대화형 번역)와 `sion-augment`(역번역 데이터 증강)가 공유합니다.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch

from sion_translate.glossary import Glossary, apply_source_placeholders, restore_targets
from sion_translate.model import SionForConditionalGeneration
from sion_translate.rerank import select as rerank_select
from sion_translate.revision import DRAFT_SEPARATOR, serialize_revision_input
from sion_translate.structured import mask_structured_spans
from sion_translate.tokenizer import (
    SLOT_SYMBOLS,
    SionTokenizer,
    load_tokenizer_metadata,
    tokenizer_split_digits_policy,
)
from sion_translate.fp8_runtime import describe_runtime, prepare_fp8_model_for_device
from sion_translate.training.export import load_exported_model, resolve_manifest_artifact


def _language_pairs_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    raw_pairs: object = metadata.get("language_pairs")
    if raw_pairs is None and metadata.get("language_pair") is not None:
        raw_pairs = [metadata["language_pair"]]
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        return ()
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for raw_pair in cast(Sequence[object], raw_pairs):
        if not isinstance(raw_pair, Sequence) or isinstance(raw_pair, (str, bytes)):
            continue
        pair_items = cast(Sequence[object], raw_pair)
        if len(pair_items) != 2:
            continue
        pair = (str(pair_items[0]), str(pair_items[1]))
        edge = frozenset(pair)
        if len(edge) == 2 and edge not in seen:
            seen.add(edge)
            pairs.append(pair)
    return tuple(pairs)


def _translation_directions_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    pairs = _language_pairs_from_metadata(metadata)
    raw_directions: object = metadata.get("translation_directions")
    if raw_directions is None:
        return tuple(direction for pair in pairs for direction in (pair, (pair[1], pair[0])))
    if not isinstance(raw_directions, Sequence) or isinstance(
        raw_directions,
        (str, bytes),
    ):
        raise ValueError("translation_directions metadata must be a sequence")
    if pairs and not raw_directions:
        raise ValueError(
            "translation_directions metadata cannot be empty when language pairs are configured"
        )
    allowed_edges = {frozenset(pair) for pair in pairs}
    directions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_direction in cast(Sequence[object], raw_directions):
        if not isinstance(raw_direction, Sequence) or isinstance(raw_direction, (str, bytes)):
            raise ValueError(f"invalid translation direction metadata: {raw_direction!r}")
        direction_items = cast(Sequence[object], raw_direction)
        if len(direction_items) != 2:
            raise ValueError(f"invalid translation direction metadata: {raw_direction!r}")
        direction = (str(direction_items[0]), str(direction_items[1]))
        if direction[0] == direction[1] or frozenset(direction) not in allowed_edges:
            raise ValueError(f"invalid translation direction metadata: {raw_direction!r}")
        if direction not in seen:
            seen.add(direction)
            directions.append(direction)
    return tuple(directions)


def _manifest_artifact(directory: Path, *, int8: bool) -> Path | None:
    format_names = ("int8",) if int8 else ("fp32", "bf16", "fp16")
    return resolve_manifest_artifact(directory, format_names)


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
            directory = exports / stage
            manifested = _manifest_artifact(directory, int8=int8)
            if manifested is not None:
                return manifested
            # A v2 manifest is authoritative. If it has no successful requested
            # format, do not silently select a stale file that it did not verify.
            if (directory / "export_manifest.json").exists():
                continue
            for filename in filenames:
                candidate = directory / filename
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
        token_features_path: str | Path | None = None,
    ):
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        self.tokenizer = SionTokenizer(tokenizer_path)
        declared_split_digits = tokenizer_split_digits_policy(tokenizer_path)
        if declared_split_digits is False or (
            declared_split_digits is None and not self.tokenizer.splits_digits
        ):
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
        loaded = load_exported_model(model_path, return_metadata=True)
        if len(loaded) == 3:
            self.model, self.model_config, self.pad_id = cast(
                tuple[SionForConditionalGeneration, Any, int], loaded
            )
            self.export_metadata: dict[str, Any] = {}
        else:
            self.model, self.model_config, self.pad_id, raw_metadata = cast(
                tuple[SionForConditionalGeneration, Any, int, object], loaded
            )
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("model export metadata must be an object")
            self.export_metadata = cast(
                dict[str, Any], dict(cast(Mapping[object, object], raw_metadata))
            )
        # 번역할 수 없는 산출물을 번역기로 싣지 않습니다. foundation 모델은
        # 번역쌍을 한 번도 보지 않았지만 구조가 같아서, 막지 않으면 방향
        # 태그를 받아들이고 그럴듯한 쓰레기를 냅니다.
        if self.export_metadata.get("translation_capable") is False:
            release = self.export_metadata.get("release_name", "unknown")
            raise ValueError(
                f"이 export 는 번역 모델이 아닙니다 (release_name={release!r}). "
                "단일어 복원만 학습한 foundation 산출물이므로 번역에 쓸 수 없습니다. "
                "번역 단계(runs/*/pretrain 또는 posttrain)의 export 를 지정하세요."
            )
        self.tokenizer_metadata = cast(
            dict[str, Any] | None, load_tokenizer_metadata(tokenizer_path)
        )
        self.language_pairs = _language_pairs_from_metadata(self.export_metadata)
        if not self.language_pairs:
            self.language_pairs = _language_pairs_from_metadata(self.tokenizer_metadata)
        if not self.language_pairs and len(self.tokenizer.languages) == 2:
            self.language_pairs = ((self.tokenizer.languages[0], self.tokenizer.languages[1]),)
        self.translation_directions = _translation_directions_from_metadata(self.export_metadata)
        if not self.translation_directions:
            self.translation_directions = _translation_directions_from_metadata(
                self.tokenizer_metadata
            )
        if not self.translation_directions:
            self.translation_directions = tuple(
                direction
                for pair in self.language_pairs
                for direction in (pair, (pair[1], pair[0]))
            )
        self._translation_direction_edges: set[tuple[str, str]] = set(self.translation_directions)
        self._validate_compatibility(tokenizer_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # 양자화 모델은 CPU 전용 커널을 쓰므로 CPU 에 남깁니다.
        quantization = self.export_metadata.get("quantization")
        quantization_mapping = (
            cast(Mapping[object, object], quantization)
            if isinstance(quantization, Mapping)
            else None
        )
        runtime_device = (
            quantization_mapping.get("runtime_device") if quantization_mapping is not None else None
        )
        self.quantized = runtime_device == "cpu" or any(
            "quantized" in type(module).__module__ for module in self.model.modules()
        )
        # FP8 export 는 CPU 전용이 아닙니다. 현재는 가중치를 BF16(미지원 CUDA는
        # FP16)으로 즉시 역양자화한 뒤 dense GEMM 을 하므로 실제 선택을 로그에
        # 남깁니다. 상주 FP8 형식과 계산 dtype은 서로 독립입니다.
        fp8_export = (
            quantization_mapping is not None and quantization_mapping.get("format") == "fp8"
        )
        self.fp8_runtime: str | None = describe_runtime(self.device) if fp8_export else None
        if not self.quantized:
            if fp8_export:
                prepare_fp8_model_for_device(self.model, self.device)
            else:
                self.model.to(self.device)
        else:
            if self.device.type != "cpu":
                warnings.warn(
                    "이 양자화 export는 CPU 전용이므로 요청한 CUDA 장치 대신 CPU에서 실행합니다.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.device = torch.device("cpu")
        self.model.eval()

        feature_path = self._resolve_token_features_path(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            explicit_path=token_features_path,
        )
        self.token_features = self._load_token_features(
            feature_path,
            required=self.model_config.experimental.morphoscript_enabled,
            explicit=token_features_path is not None,
            expected_identity=self.export_metadata.get("token_features"),
        )

    def _resolve_token_features_path(
        self,
        *,
        model_path: Path,
        tokenizer_path: Path,
        explicit_path: str | Path | None,
    ) -> Path:
        if explicit_path is not None:
            return Path(explicit_path)

        filenames: list[str] = []
        export_identity = self.export_metadata.get("token_features")
        if isinstance(export_identity, Mapping):
            filename = cast(Mapping[object, object], export_identity).get("filename")
            if isinstance(filename, str) and filename and Path(filename).name == filename:
                filenames.append(filename)
        if self.tokenizer_metadata is not None:
            filename = self.tokenizer_metadata.get("token_features_file")
            if (
                isinstance(filename, str)
                and filename
                and Path(filename).name == filename
                and filename not in filenames
            ):
                filenames.append(filename)
        if "token_features.npz" not in filenames:
            filenames.append("token_features.npz")

        candidates = [
            parent / filename
            for filename in filenames
            for parent in (model_path.parent, tokenizer_path.parent)
        ]
        return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    def _validate_compatibility(self, tokenizer_path: Path) -> None:
        tokenizer_vocab = len(self.tokenizer)
        if tokenizer_vocab != self.model_config.vocab_size:
            raise ValueError(
                "tokenizer vocab does not match model config: "
                f"{tokenizer_vocab} != {self.model_config.vocab_size}"
            )
        if self.tokenizer.pad_id != self.pad_id:
            raise ValueError(
                "tokenizer pad ID does not match model export: "
                f"{self.tokenizer.pad_id} != {self.pad_id}"
            )

        tokenizer_identity = self.export_metadata.get("tokenizer")
        if isinstance(tokenizer_identity, Mapping):
            identity = cast(Mapping[object, object], tokenizer_identity)
            expected_size = identity.get("size")
            if isinstance(expected_size, int) and tokenizer_path.stat().st_size != expected_size:
                raise ValueError("tokenizer metadata size does not match the selected tokenizer")
            expected_sha256 = identity.get("sha256")
            if isinstance(expected_sha256, str):
                digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
                if digest != expected_sha256:
                    raise ValueError(
                        "tokenizer metadata SHA256 does not match the selected tokenizer"
                    )

        raw_pairs: object = self.export_metadata.get("language_pairs")
        if raw_pairs is None and self.export_metadata.get("language_pair") is not None:
            raw_pairs = [self.export_metadata["language_pair"]]
        if isinstance(raw_pairs, Sequence) and not isinstance(raw_pairs, (str, bytes)):
            expected_languages = {
                str(language)
                for pair in cast(Sequence[object], raw_pairs)
                if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes))
                for language in cast(Sequence[object], pair)
            }
            if expected_languages and expected_languages != set(self.tokenizer.languages):
                raise ValueError(
                    "tokenizer languages do not match model metadata: "
                    f"{sorted(self.tokenizer.languages)} != {sorted(expected_languages)}"
                )

        feature_flags = self.export_metadata.get("feature_flags")
        if isinstance(feature_flags, Mapping):
            typed_feature_flags = cast(Mapping[object, object], feature_flags)
            experimental = self.model_config.experimental
            expected_flags = {
                "bats": bool(experimental.bats_enabled),
                "core": bool(experimental.core_enabled),
                "tetm": bool(experimental.tetm_enabled),
                "morphoscript": bool(experimental.morphoscript_enabled),
                "evidence_repair": bool(experimental.evidence_repair_enabled),
                "semantic_parity": bool(experimental.semantic_parity_enabled),
                "situglu": bool(experimental.situglu_enabled),
                "recurrent_block": bool(experimental.recurrent_block_layers),
            }
            mismatches = {
                name: (bool(typed_feature_flags[name]), enabled)
                for name, enabled in expected_flags.items()
                if name in typed_feature_flags and bool(typed_feature_flags[name]) != enabled
            }
            if mismatches:
                raise ValueError(
                    f"model feature metadata does not match model config: {mismatches}"
                )

        capabilities = self.export_metadata.get("capabilities")
        if capabilities is not None:
            if not isinstance(capabilities, Mapping):
                raise ValueError("model capabilities metadata must be an object")
            if "revision_trained" in capabilities and not isinstance(
                capabilities["revision_trained"], bool
            ):
                raise ValueError(
                    "model capabilities.revision_trained must be a boolean when present"
                )

        tokenizer_metadata = self.tokenizer_metadata
        if tokenizer_metadata is None:
            return
        metadata_vocab = tokenizer_metadata.get("vocab_size")
        if isinstance(metadata_vocab, int) and metadata_vocab != tokenizer_vocab:
            raise ValueError(
                "tokenizer metadata vocab does not match tokenizer model: "
                f"{metadata_vocab} != {tokenizer_vocab}"
            )
        metadata_sha256 = tokenizer_metadata.get("model_sha256")
        if isinstance(metadata_sha256, str):
            digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            if digest != metadata_sha256:
                raise ValueError("tokenizer sidecar model identity does not match tokenizer.model")
        metadata_pairs: object = tokenizer_metadata.get("language_pairs")
        if isinstance(metadata_pairs, Sequence) and not isinstance(metadata_pairs, (str, bytes)):
            metadata_languages = {
                str(language)
                for pair in cast(Sequence[object], metadata_pairs)
                if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes))
                for language in cast(Sequence[object], pair)
            }
            if metadata_languages and metadata_languages != set(self.tokenizer.languages):
                raise ValueError(
                    "tokenizer sidecar languages do not match tokenizer.model: "
                    f"{sorted(metadata_languages)} != {sorted(self.tokenizer.languages)}"
                )
        sidecar_pairs = _language_pairs_from_metadata(tokenizer_metadata)
        export_pairs = _language_pairs_from_metadata(self.export_metadata)
        if (
            sidecar_pairs
            and export_pairs
            and {frozenset(pair) for pair in sidecar_pairs}
            != {frozenset(pair) for pair in export_pairs}
        ):
            raise ValueError(
                "tokenizer sidecar language pairs do not match model metadata: "
                f"{sidecar_pairs} != {export_pairs}"
            )

    def _load_token_features(
        self,
        path: Path,
        *,
        required: bool,
        explicit: bool,
        expected_identity: object,
    ) -> dict[str, torch.Tensor] | None:
        if not required and not explicit:
            return None
        if not path.is_file():
            if required or explicit:
                raise FileNotFoundError(
                    f"MorphoScript token feature file does not exist: {path}" if required else path
                )
            return None
        identity = (
            cast(Mapping[object, object], expected_identity)
            if isinstance(expected_identity, Mapping)
            else None
        )
        if identity is None and self.tokenizer_metadata is not None:
            expected_sha = self.tokenizer_metadata.get("token_features_sha256")
            if isinstance(expected_sha, str):
                identity = {
                    "filename": self.tokenizer_metadata.get("token_features_file"),
                    "size": self.tokenizer_metadata.get("token_features_size"),
                    "sha256": expected_sha,
                }
        if identity is not None:
            expected_filename = identity.get("filename")
            if isinstance(expected_filename, str) and path.name != expected_filename:
                raise ValueError(
                    "token feature filename does not match model/tokenizer metadata: "
                    f"{path.name} != {expected_filename}"
                )
            expected_size = identity.get("size")
            if isinstance(expected_size, int) and path.stat().st_size != expected_size:
                raise ValueError("token feature size does not match model/tokenizer metadata")
            expected_sha256 = identity.get("sha256")
            if isinstance(expected_sha256, str):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != expected_sha256:
                    raise ValueError("token feature SHA256 does not match model/tokenizer metadata")
        expected_length = len(self.tokenizer)
        features: dict[str, torch.Tensor] = {}
        maximum_ids = {
            "script": self.model_config.experimental.script_classes,
            "onset": 20,
            "vowel": 22,
            "coda": 29,
        }
        with np.load(path, allow_pickle=False) as loaded:
            required_names = {"script", "onset", "vowel", "coda"}
            if set(loaded.files) != required_names:
                raise ValueError(
                    "token feature file must contain exactly "
                    f"{', '.join(sorted(required_names))}; got {sorted(loaded.files)}"
                )
            for name in ("script", "onset", "vowel", "coda"):
                values = np.asarray(loaded[name])
                if values.ndim != 1 or len(values) != expected_length:
                    raise ValueError(
                        f"token feature {name} has shape {values.shape}; "
                        f"expected ({expected_length},)"
                    )
                if not np.issubdtype(values.dtype, np.integer):
                    raise ValueError(f"token feature {name} must use an integer dtype")
                values = values.astype(np.int64, copy=True)
                if values.size and (
                    int(values.min()) < 0 or int(values.max()) >= maximum_ids[name]
                ):
                    raise ValueError(
                        f"token feature {name} contains IDs outside [0, {maximum_ids[name]})"
                    )
                features[name] = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                    values
                )
        return features

    def _generation_features(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        if self.token_features is not None:
            for source_name, target_name in (
                ("script", "src_script_ids"),
                ("onset", "src_onset_ids"),
                ("vowel", "src_vowel_ids"),
                ("coda", "src_coda_ids"),
            ):
                features[target_name] = self.token_features[source_name][input_ids].to(self.device)

        if self.model_config.experimental.tetm_enabled:
            slot_ids = set(self.tokenizer.slot_ids)
            rows = [
                [int(token_id) for token_id in row if int(token_id) in slot_ids][:64]
                for row in input_ids
            ]
            memory_length = max(1, max(map(len, rows), default=0))
            memory_token_ids = torch.full(
                (len(rows), memory_length, 1),
                self.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            memory_mask = torch.zeros(
                (len(rows), memory_length),
                dtype=torch.bool,
                device=self.device,
            )
            memory_type_ids = torch.zeros_like(memory_mask, dtype=torch.long)
            memory_mode_ids = torch.zeros_like(memory_mask, dtype=torch.long)
            for row_index, row in enumerate(rows):
                if not row:
                    continue
                length = len(row)
                memory_token_ids[row_index, :length, 0] = torch.tensor(
                    row, dtype=torch.long, device=self.device
                )
                memory_mask[row_index, :length] = True
                memory_type_ids[row_index, :length] = min(
                    8, self.model_config.experimental.tetm_types - 1
                )
                memory_mode_ids[row_index, :length] = min(
                    4, self.model_config.experimental.tetm_modes - 1
                )
            features.update(
                memory_token_ids=memory_token_ids,
                memory_mask=memory_mask,
                memory_type_ids=memory_type_ids,
                memory_mode_ids=memory_mode_ids,
            )
        return features

    @property
    def languages(self) -> tuple[str, ...]:
        """이 모델이 지원하는 언어 (토크나이저의 <2xx> 태그에서 자동 인식)."""
        return self.tokenizer.languages

    def _other_language(self, target_language: str) -> str:
        """양방향 모델에서 목표 언어가 아닌 쪽을 원문 언어로 간주합니다."""
        others = [lang for lang in self.languages if lang != target_language]
        return others[0] if len(others) == 1 else ""

    def _resolve_source_language(
        self,
        source_language: str | None,
        target_language: str,
    ) -> str:
        if source_language is None:
            source_language = self._other_language(target_language)
            if not source_language:
                raise ValueError(
                    "다국어 모델은 source_language를 명시해야 합니다 "
                    f"(지원: {sorted(self.languages)})"
                )
        if source_language not in self.languages:
            raise ValueError(
                f"지원하지 않는 원문 언어: {source_language} (지원: {sorted(self.languages)})"
            )
        if source_language == target_language:
            raise ValueError("source_language와 target_language는 달라야 합니다")
        empty_directions: set[tuple[str, str]] = set()
        direction_edges = cast(
            set[tuple[str, str]],
            getattr(self, "_translation_direction_edges", empty_directions),
        )
        if direction_edges and (source_language, target_language) not in direction_edges:
            translation_directions = getattr(self, "translation_directions", ())
            supported = ", ".join(f"{source}→{target}" for source, target in translation_directions)
            raise ValueError(
                f"학습되지 않은 번역 방향: {source_language}→{target_language} "
                f"(지원 방향: {supported})"
            )
        return source_language

    @torch.no_grad()
    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str | None = None,
        target_language: str,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        max_new_tokens: int = 256,
        batch_size: int = 16,
        glossary: Glossary | None = None,
        append_missing_glossary: bool = True,
        append_missing_structured: bool = True,
        num_candidates: int = 0,
        rerank: str = "mbr+qe",
        temperature: float = 0.3,
        top_k: int = 0,
        seed: int | None = None,
        sampling_seed: int | None = None,
        generator: torch.Generator | None = None,
        return_rerank_details: bool = False,
        min_new_tokens: int = 1,
        no_repeat_ngram_size: int = 4,
        max_output_length_ratio: float | None = 3.0,
        max_output_length_margin: int = 16,
        reasoning_level: int | None = None,
    ) -> list[str]:
        """문장 목록을 ``target_language`` 로 번역합니다.

        입력 언어는 지정할 필요가 없습니다 — 모델 입력의 <2xx> 태그가
        '어느 언어로 번역할지'만 지시하며 (학습 때와 같은 방식),
        나머지 한쪽 언어가 입력이라고 가정합니다.

        ``glossary`` 를 주면 지정한 용어를 정해진 대응어로 강제합니다.
        (원문에서 slot 토큰으로 치환 → 번역 → 대응어로 복원.)
        모델이 slot 을 보존하지 못해 누락된 용어는 ``append_missing_glossary``
        가 참이면 문장 끝에 괄호로 덧붙여 최소한의 강제를 보장합니다.

        숫자·단위·URL·현지화 플레이스홀더도 같은 slot 경로로 자동 보호하고
        원문의 정확한 표면형으로 복원합니다. 모델이 slot 을 누락했을 때에도
        ``append_missing_structured`` 가 참이면 값 자체가 사라지지 않게 덧붙입니다.

        ``num_candidates`` 를 1 이상으로 두면 beam 결과에 더해 그 수만큼 확률적
        후보를 뽑고 ``rerank`` 방식으로 하나를 고릅니다 (``sion_translate.rerank``
        참고). 재학습 없이 추론 계산량만 늘리는 경로이며, 후보 목록의 첫 번째는
        항상 beam 결과이므로 동점이면 기존 동작이 유지됩니다.

        ``return_rerank_details`` 가 참이면 문자열 대신 ``RerankResult`` 목록을
        돌려줍니다 — 어느 후보가 왜 뽑혔는지 확인할 때 씁니다.

        ``reasoning_level`` 은 SSRT의 선택적 evidence repair 예산입니다. 0은
        auditor/repair를 완전히 우회하고, 1-9는 허용 예산을 단조롭게 늘립니다.
        ``None``은 이전 체크포인트와 호출자의 기존 동작을 보존합니다.

        생성 중에는 학습 제어 토큰을 금지하고 ``no_repeat_ngram_size`` 크기의
        반복을 차단합니다. ``max_output_length_ratio``는 원문 토큰 수에 비해
        비정상적으로 긴 디코딩만 일찍 잘라 정상적인 EOS 종료에는 관여하지 않습니다.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_new_tokens <= 0 or max_new_tokens > self.model_config.max_seq_len:
            raise ValueError(
                "max_new_tokens must be positive and no larger than model max_seq_len "
                f"({self.model_config.max_seq_len})"
            )
        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if length_penalty <= 0:
            raise ValueError("length_penalty must be positive")
        if num_candidates < 0:
            raise ValueError("num_candidates 는 0 이상이어야 합니다")
        if return_rerank_details and num_candidates < 1:
            raise ValueError("return_rerank_details 는 num_candidates 가 1 이상일 때만 씁니다")
        if num_candidates and temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        if max_output_length_ratio is not None and max_output_length_ratio <= 0:
            raise ValueError("max_output_length_ratio must be positive or None")
        if max_output_length_margin < 0:
            raise ValueError("max_output_length_margin must be non-negative")
        if reasoning_level is not None:
            if type(reasoning_level) is not int:
                raise TypeError("reasoning_level must be an integer from 0 to 9 or None")
            if not 0 <= reasoning_level <= 9:
                raise ValueError("reasoning_level must be between 0 and 9")
        if seed is not None and sampling_seed is not None:
            raise ValueError("seed and sampling_seed are aliases; provide only one")
        resolved_seed = seed if seed is not None else sampling_seed
        if resolved_seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if resolved_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(resolved_seed))
        tag_id = self.tokenizer.language_tags.get(target_language)
        if tag_id is None:
            raise ValueError(
                f"지원하지 않는 언어: {target_language} (지원: {sorted(self.languages)})"
            )
        source_language = self._resolve_source_language(
            source_language,
            target_language,
        )
        eos = self.tokenizer.eos_id
        results: list[Any] = []
        special_ids = {
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            eos,
            self.tokenizer.mask_id,
            *self.tokenizer.language_tags.values(),
            *self.tokenizer.denoise_tags.values(),
        }
        if self.tokenizer.draft_id is not None:
            special_ids.add(self.tokenizer.draft_id)
        forbidden_token_ids = tuple(sorted(special_ids - {eos}))

        def restore(
            row: Sequence[int],
            structured_map: dict[str, str] | None,
            glossary_map: dict[str, str] | None,
        ) -> str:
            """생성 토큰을 문자열로 되돌리고 보호 slot 을 복원합니다."""
            tokens = [token for token in row if token not in special_ids]
            text = self.tokenizer.decode(tokens)
            if structured_map:
                text, missing = restore_targets(text, structured_map)
                if missing and append_missing_structured:
                    text = f"{text} ({', '.join(missing)})"
            if glossary_map:
                text, missing = restore_targets(text, glossary_map)
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
            structured_maps: list[dict[str, str]] = []
            glossary_maps: list[dict[str, str]] = []
            prepared: list[str] = []
            for text in chunk:
                masked, structured_map = mask_structured_spans(
                    text,
                    slot_symbols=SLOT_SYMBOLS,
                )
                structured_maps.append(structured_map)
                glossary_map: dict[str, str] = {}
                if glossary is not None and source_language:
                    remaining_slots = SLOT_SYMBOLS[len(structured_map) :]
                    masked, slot_map = apply_source_placeholders(
                        masked,
                        glossary,
                        source_language=source_language,
                        target_language=target_language,
                        slot_symbols=remaining_slots,
                    )
                    glossary_map = slot_map
                glossary_maps.append(glossary_map)
                prepared.append(masked)
            chunk = prepared
            encoded = [[tag_id, *self.tokenizer.encode(text), eos] for text in chunk]
            longest = max(len(ids) for ids in encoded)
            if longest > self.model_config.max_seq_len:
                raise ValueError(
                    "encoded source length exceeds model max_seq_len: "
                    f"{longest} > {self.model_config.max_seq_len}"
                )
            input_ids = torch.full((len(encoded), longest), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), longest), dtype=torch.bool)
            for row, ids in enumerate(encoded):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = True
            device_inputs = input_ids.to(self.device)
            device_mask = attention_mask.to(self.device)
            generation_features = self._generation_features(input_ids)
            generation_context = (
                self.model.prepare_generation(
                    device_inputs,
                    device_mask,
                    reasoning_level=reasoning_level,
                    **generation_features,
                )
                if num_candidates > 0
                else None
            )
            chunk_max_new_tokens = max_new_tokens
            row_max_new_tokens: torch.Tensor | None = None
            if max_output_length_ratio is not None:
                row_limits = [
                    min(
                        max_new_tokens,
                        max(
                            min_new_tokens + 1,
                            math.ceil((len(ids) - 2) * max_output_length_ratio)
                            + max_output_length_margin,
                        ),
                    )
                    for ids in encoded
                ]
                chunk_max_new_tokens = max(row_limits)
                row_max_new_tokens = torch.tensor(
                    row_limits,
                    dtype=torch.long,
                    device=self.device,
                )
            chunk_min_new_tokens = min(
                min_new_tokens,
                max(0, chunk_max_new_tokens - 1),
            )
            generated = self.model.generate(
                device_inputs,
                device_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=eos,
                max_new_tokens=chunk_max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                generation_context=generation_context,
                forbidden_token_ids=forbidden_token_ids,
                min_new_tokens=chunk_min_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens_per_row=row_max_new_tokens,
                reasoning_level=reasoning_level,
                **({} if generation_context is not None else generation_features),
            )
            generated_rows = cast(
                list[list[int]],
                generated.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            beam_texts = [
                restore(row, structured_maps[index], glossary_maps[index])
                for index, row in enumerate(generated_rows)
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
                max_new_tokens=chunk_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                forbidden_token_ids=forbidden_token_ids,
                min_new_tokens=chunk_min_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                generator=generator,
                generation_context=generation_context,
                max_new_tokens_per_row=row_max_new_tokens,
                reasoning_level=reasoning_level,
            )
            sampled_rows = cast(
                list[list[list[int]]],
                sampled.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            for row_index, source_text in enumerate(sources):
                candidates = [beam_texts[row_index]]
                for sample_row in sampled_rows[row_index]:
                    candidate = restore(
                        sample_row,
                        structured_maps[row_index],
                        glossary_maps[row_index],
                    )
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
        source_language: str | None = None,
        target_language: str,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        max_new_tokens: int = 256,
        batch_size: int = 16,
        reasoning_level: int | None = None,
    ) -> list[str]:
        """``원문 + 초안`` 을 받아 고친 번역을 돌려줍니다.

        ``sion-revise-data`` 로 만든 ``원문 <draft> 초안 → 번역`` 예제로 학습한
        모델에서만 의미가 있습니다. 그렇게 학습하지 않은 모델에 쓰면 ``<draft>``
        뒤를 그냥 원문의 일부로 읽으므로 결과가 나빠집니다.

        토크나이저에 ``<draft>`` 가 없으면 (2026-07 이전 토크나이저) 오류를 냅니다 —
        구분자가 여러 토큰으로 쪼개져 학습 때와 다른 입력이 되기 때문입니다.
        """
        capabilities = self.export_metadata.get("capabilities")
        capabilities_mapping = (
            cast(Mapping[object, object], capabilities)
            if isinstance(capabilities, Mapping)
            else None
        )
        revision_trained = (
            capabilities_mapping.get("revision_trained")
            if capabilities_mapping is not None
            else None
        )
        if revision_trained is False:
            raise ValueError(
                "The exported model does not declare the revision capability; "
                "export a checkpoint trained on revision examples."
            )
        if revision_trained is None:
            warnings.warn(
                "이전 export에는 revision 학습 여부가 기록되어 있지 않습니다. "
                "호환성을 위해 실행하지만, revision 예제로 학습하지 않은 모델이면 "
                "결과 품질을 보장할 수 없습니다.",
                RuntimeWarning,
                stacklevel=2,
            )
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
            source_language=source_language,
            target_language=target_language,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            reasoning_level=reasoning_level,
        )
