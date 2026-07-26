from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from sion_translate.performance import bounded_ordered_map, build_cpu_plan


def _double(value: int) -> int:
    return value * 2


def test_cpu_plan_uses_allocated_capacity_without_oversubscription(monkeypatch) -> None:
    monkeypatch.setattr("sion_translate.performance.available_cpu_count", lambda: 32)
    plan = build_cpu_plan(world_size=4, input_files=1)
    assert plan.available == 32
    assert plan.preprocess_workers + plan.sentencepiece_threads == 32
    assert plan.dataset_workers == 31
    assert plan.dataloader_workers_per_rank == 7


def test_bounded_ordered_map_preserves_order_and_laziness() -> None:
    consumed = 0

    def inputs():
        nonlocal consumed
        for value in range(20):
            consumed += 1
            yield value

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = bounded_ordered_map(executor, _double, inputs(), max_pending=4)
        assert next(results) == 0
        assert consumed <= 5
        assert [0, *results] == [value * 2 for value in range(20)]
