from copy import deepcopy

import pytest

from scripts.estimate_training_cost import estimate_run, projected_examples


def report_fixture():
    case = {
        "status": "passed",
        "returncode": 0,
        "precision": "bf16",
        "float32_matmul_precision": "high",
        "batch_size": 8,
        "stress_peak_reserved_gib": 50,
        "representative_peak_reserved_gib": 20,
        "epochs": 3,
        "max_steps": None,
        "examples_per_epoch": 1000,
        "summary": {
            "examples_per_second": 10,
            "projected_effective_batch": 16,
            "projected_accumulation": 2,
            "effective_batch_changed": False,
        },
    }
    return {
        "status": "completed",
        "target": "a100-80gb",
        "runtime": {"devices": [{"total_memory_gib": 80}]},
        "cases": [
            dict(deepcopy(case), stage=stage, name=stage + "-b8")
            for stage in ("foundation", "sft", "mrt")
        ],
    }


def test_cost_counts_all_stages_and_cpu_ram_without_mutating_measurements():
    report = report_fixture()
    before = deepcopy(report)
    result = estimate_run(report)
    assert result["total_compute_hours"] == pytest.approx(900 / 3600)
    assert result["hourly_compute_usd"] == pytest.approx(3.259296)
    assert result["total_scenario_usd"] == pytest.approx([0.9777888, 1.222236])
    assert report == before


@pytest.mark.parametrize("status", ["capacity_oom", "failed", "running"])
def test_attractive_incomplete_timings_never_win(status):
    report = report_fixture()
    bad = deepcopy(report["cases"][0])
    bad.update(status=status, name="unsafe-fast", returncode=20)
    bad["summary"]["examples_per_second"] = 10000
    report["cases"].append(bad)
    result = estimate_run(report)
    assert result["stages"][0]["case"] == "foundation-b8"
    assert result["excluded_candidates"][0]["name"] == "unsafe-fast"


def test_near_capacity_candidate_is_rejected_even_after_passing_stress():
    report = report_fixture()
    report["cases"][0]["stress_peak_reserved_gib"] = 79
    with pytest.raises(ValueError, match="no passing candidate"):
        estimate_run(report)


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf")])
def test_invalid_speed_cannot_produce_a_budget(speed):
    report = report_fixture()
    report["cases"][0]["summary"]["examples_per_second"] = speed
    with pytest.raises(ValueError, match="examples_per_second"):
        estimate_run(report)


def test_precision_and_success_are_required():
    report = report_fixture()
    report["status"] = "failed"
    with pytest.raises(ValueError, match="completed"):
        estimate_run(report)
    report["status"] = "completed"
    report["cases"][0].pop("float32_matmul_precision")
    with pytest.raises(ValueError, match="precision"):
        estimate_run(report)


@pytest.mark.parametrize("steps,expected", [(1, 16), (2, 20), (3, 36), (4, 40)])
def test_step_override_counts_partial_epoch_update(steps, expected):
    case = report_fixture()["cases"][0]
    case.update(examples_per_epoch=20, max_steps=steps)
    assert projected_examples(case) == expected


def test_drop_last_counts_only_complete_microbatches():
    case = report_fixture()["cases"][0]
    case.update(examples_per_epoch=27, drop_last=True)
    assert projected_examples(case) == 24 * 3
    case["max_steps"] = 3
    assert projected_examples(case) == 24 + 16


def test_explicit_mrt_step_scenario_does_not_change_other_stages_or_report():
    report = report_fixture()
    before = deepcopy(report)
    result = estimate_run(report, mrt_max_steps=25)
    examples = {stage["stage"]: stage["examples"] for stage in result["stages"]}
    assert examples == {"foundation": 3000, "sft": 3000, "mrt": 400}
    assert result["schedule_overrides"] == {"mrt_max_steps": 25}
    assert report == before


@pytest.mark.parametrize("steps", [0, -1, True])
def test_invalid_mrt_step_scenario_is_rejected(steps):
    with pytest.raises(ValueError, match="mrt_max_steps"):
        estimate_run(report_fixture(), mrt_max_steps=steps)
