from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DistributedBucketBatchSampler": ("sion_translate.data.indexed", "DistributedBucketBatchSampler"),
    "IndexedParallelDataset": ("sion_translate.data.indexed", "IndexedParallelDataset"),
    "SionBatchCollator": ("sion_translate.data.collate", "SionBatchCollator"),
    "QualityPolicy": ("sion_translate.data.quality", "QualityPolicy"),
    "assess_pair": ("sion_translate.data.quality", "assess_pair"),
    "audit_dataset": ("sion_translate.data.audit", "audit_dataset"),
    "prepare_dataset": ("sion_translate.data.prepare", "prepare_dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
