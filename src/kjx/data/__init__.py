from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DistributedBucketBatchSampler": ("kjx.data.indexed", "DistributedBucketBatchSampler"),
    "IndexedParallelDataset": ("kjx.data.indexed", "IndexedParallelDataset"),
    "KJBatchCollator": ("kjx.data.collate", "KJBatchCollator"),
    "QualityPolicy": ("kjx.data.quality", "QualityPolicy"),
    "assess_pair": ("kjx.data.quality", "assess_pair"),
    "audit_dataset": ("kjx.data.audit", "audit_dataset"),
    "prepare_dataset": ("kjx.data.prepare", "prepare_dataset"),
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
