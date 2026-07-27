from __future__ import annotations

import os
from collections import deque
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, TypeVar


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def available_cpu_count() -> int:
    """Return CPUs this process can actually use, including container affinity."""

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass

    # Windows exposes the process affinity mask through stdlib only indirectly;
    # NUMBER_OF_PROCESSORS is container-aware on common Windows containers.
    if os.name == "nt":
        try:
            reported = int(os.environ.get("NUMBER_OF_PROCESSORS", "0"))
            if reported > 0:
                return reported
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class CpuPlan:
    available: int
    preprocess_workers: int
    dataset_workers: int
    sentencepiece_threads: int
    dataloader_workers_per_rank: int


def build_cpu_plan(*, world_size: int = 1, input_files: int | None = None) -> CpuPlan:
    """Divide allocated CPUs between concurrent work without oversubscription."""

    available = available_cpu_count()
    ranks = max(1, world_size)
    # Tokenizer ingestion and SentencePiece training run concurrently. Give both
    # sides CPU capacity; a one-core machine remains fully functional.
    if available == 1:
        preprocess_workers = 1
        sentencepiece_threads = 1
    else:
        preprocess_workers = max(1, available // 2)
        sentencepiece_threads = max(1, available - preprocess_workers)
    # Work is batched within files, so one large file can still occupy every
    # worker. ``input_files`` is accepted for future I/O-aware tuning.

    per_rank = max(1, available // ranks)
    # Each rank owns its own DataLoader pool. Leave one execution slot per rank
    # for the training process and cap workers at the rank's CPU allocation.
    # Hundreds of workers per rank increase process scheduling and leave several
    # persistent pools alive during validation/stage transitions. A bounded pool
    # with deeper prefetching feeds H100-class GPUs more reliably.
    dataloader_workers = min(16, max(0, per_rank - 1))
    return CpuPlan(
        available=available,
        preprocess_workers=preprocess_workers,
        dataset_workers=max(1, available - 1),
        sentencepiece_threads=sentencepiece_threads,
        dataloader_workers_per_rank=dataloader_workers,
    )


def bounded_ordered_map(
    executor: Executor,
    function: Callable[[_Input], _Output],
    inputs: Iterable[_Input],
    *,
    max_pending: int,
) -> Iterator[_Output]:
    """Ordered executor map that does not enqueue a multi-GiB input eagerly."""

    iterator = iter(inputs)
    pending = deque()
    for _ in range(max(1, max_pending)):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass
