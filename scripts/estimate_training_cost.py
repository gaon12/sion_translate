"""Project single-GPU training time from completed, stress-tested Modal probes.

This is a scenario calculator, not a convergence or full-pipeline guarantee.
Failed/OOM candidates are never used, even when they contain attractive timings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


# Modal GPU Tasks list prices checked on 2026-09-03: https://modal.com/pricing
GPU_USD_PER_SECOND = {"a100-80gb": 0.000694, "h100": 0.001097}
CPU_USD_PER_CORE_SECOND = 0.0000131
RAM_USD_PER_GIB_SECOND = 0.00000222
STAGES = ("foundation", "sft", "mrt")


def _positive(value: Any, name: str, *, integer: bool = False) -> float | int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}")
    return value


def projected_examples(case: dict[str, Any]) -> int:
    """Count complete epochs and a possibly partial final accumulation window."""
    rows = int(_positive(case["examples_per_epoch"], "examples_per_epoch", integer=True))
    if case.get("drop_last", False):
        batch = int(_positive(case["batch_size"], "batch_size", integer=True))
        rows = rows // batch * batch
        if rows == 0:
            raise ValueError("drop_last leaves no training examples")
    if case.get("max_steps") is None:
        return rows * int(_positive(case["epochs"], "epochs", integer=True))
    steps = int(_positive(case["max_steps"], "max_steps", integer=True))
    effective = int(
        _positive(case["summary"]["projected_effective_batch"], "effective_batch", integer=True)
    )
    updates_per_epoch = (rows + effective - 1) // effective
    epochs, remaining = divmod(steps, updates_per_epoch)
    return epochs * rows + min(rows, remaining * effective)


def estimate_run(
    report: dict[str, Any],
    *,
    overhead_low: float = 0.2,
    overhead_high: float = 0.5,
    cpu_cores: float = 8,
    memory_gib: float = 48,
    max_reserved_fraction: float = 0.85,
    mrt_max_steps: int | None = None,
) -> dict[str, Any]:
    if report.get("status") != "completed":
        raise ValueError("estimate requires a completed probe")
    if report.get("target") not in GPU_USD_PER_SECOND:
        raise ValueError("estimate supports one A100 80GB or H100")
    devices = report["runtime"]["devices"]
    if len(devices) != 1:
        raise ValueError("single-GPU timings cannot establish multi-GPU scaling")
    total_vram = _positive(devices[0]["total_memory_gib"], "total_memory_gib")
    if (
        not math.isfinite(overhead_low)
        or not math.isfinite(overhead_high)
        or not 0 <= overhead_low <= overhead_high
        or not math.isfinite(max_reserved_fraction)
        or not 0 < max_reserved_fraction < 1
    ):
        raise ValueError("invalid overhead or VRAM headroom range")
    if mrt_max_steps is not None:
        _positive(mrt_max_steps, "mrt_max_steps", integer=True)
    cpu_cores = _positive(cpu_cores, "cpu_cores")
    memory_gib = _positive(memory_gib, "memory_gib")
    hourly = 3600 * (
        GPU_USD_PER_SECOND[report["target"]]
        + cpu_cores * CPU_USD_PER_CORE_SECOND
        + memory_gib * RAM_USD_PER_GIB_SECOND
    )
    chosen = []
    excluded = []
    for stage in STAGES:
        eligible = []
        for case in report["cases"]:
            if case.get("stage") != stage:
                continue
            if case.get("status") != "passed" or case.get("returncode") != 0:
                excluded.append({"name": case.get("name"), "reason": case.get("status")})
                continue
            peak = max(
                _positive(case["stress_peak_reserved_gib"], "stress_peak_reserved_gib"),
                _positive(
                    case["representative_peak_reserved_gib"], "representative_peak_reserved_gib"
                ),
                case.get("checkpoint", {}).get("peak_reserved_gib", 0),
            )
            if peak > total_vram * max_reserved_fraction:
                excluded.append({"name": case.get("name"), "reason": "insufficient VRAM headroom"})
                continue
            if case.get("precision") != "bf16" or case.get("float32_matmul_precision") != "high":
                raise ValueError("compare only production BF16/high matmul precision measurements")
            speed = _positive(case["summary"]["examples_per_second"], "examples_per_second")
            eligible.append((speed, peak, case))
        if not eligible:
            raise ValueError(f"no passing candidate with VRAM headroom for {stage}")
        speed, peak, case = max(eligible, key=lambda entry: entry[0])
        schedule_case = dict(case)
        if stage == "mrt" and mrt_max_steps is not None:
            schedule_case["max_steps"] = mrt_max_steps
        examples = projected_examples(schedule_case)
        hours = examples / speed / 3600
        chosen.append(
            {
                "stage": stage,
                "case": case["name"],
                "batch_size": case["batch_size"],
                "candidate_micro_batch": case.get("candidate_micro_batch"),
                "accumulation": case["summary"]["projected_accumulation"],
                "effective_batch_changed": case["summary"]["effective_batch_changed"],
                "examples": examples,
                "max_steps_override": mrt_max_steps if stage == "mrt" else None,
                "examples_per_second": speed,
                "peak_reserved_gib": peak,
                "compute_hours": hours,
                "compute_usd": hours * hourly,
                "scenario_hours": [hours * (1 + overhead_low), hours * (1 + overhead_high)],
                "scenario_usd": [
                    hours * hourly * (1 + overhead_low),
                    hours * hourly * (1 + overhead_high),
                ],
            }
        )
    hours = sum(stage["compute_hours"] for stage in chosen)
    return {
        "schema": "sion-training-estimate-v1",
        "target": report["target"],
        "run_id": report.get("run_id"),
        "plan_sha256": report.get("plan_sha256"),
        "source_contract_sha256": report.get("source_contract_sha256"),
        "probe_sha256": report.get("probe_sha256"),
        "pricing_date": "2026-09-03",
        "pricing_source": "https://modal.com/pricing",
        "cpu_cores": cpu_cores,
        "memory_gib": memory_gib,
        "hourly_compute_usd": hourly,
        "max_reserved_fraction": max_reserved_fraction,
        "assumed_overhead_fraction": [overhead_low, overhead_high],
        "schedule_overrides": {"mrt_max_steps": mrt_max_steps},
        "stages": chosen,
        "excluded_candidates": excluded,
        "total_compute_hours": hours,
        "total_compute_usd": hours * hourly,
        "total_scenario_days": [hours * (1 + overhead_low) / 24, hours * (1 + overhead_high) / 24],
        "total_scenario_usd": [
            hours * hourly * (1 + overhead_low),
            hours * hourly * (1 + overhead_high),
        ],
        "limitations": [
            "Resident cohort timings exclude full-corpus I/O, validation, checkpoints and exports.",
            "Overhead percentages are sensitivity assumptions, not measured confidence bounds.",
            "Random-weight MRT generation is not representative of a trained SFT model's termination.",
            "Changing effective batch changes optimization; selected cases are scenarios, not automatic config updates.",
            "The recorded prepared snapshot may differ from currently edited raw data; re-prepare and re-measure after changes.",
            "Prices exclude storage, image builds, networking, discounts, credits and region/preemption surcharges; CPU/RAM usage above these assumptions costs extra.",
            "Passing short probes does not establish full-training completion or convergence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overhead-low", type=float, default=0.2)
    parser.add_argument("--overhead-high", type=float, default=0.5)
    parser.add_argument("--mrt-max-steps", type=int)
    args = parser.parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    result = estimate_run(
        report,
        overhead_low=args.overhead_low,
        overhead_high=args.overhead_high,
        mrt_max_steps=args.mrt_max_steps,
    )
    rendered = json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
