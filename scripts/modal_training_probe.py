"""Measure real-data training capacity without starting an unbounded training run.

Each GPU call has a finite deadline, no application retries, and a durable receipt.
Every candidate runs in its own process. CUDA OOM is an explicit capacity result;
all other failures stop the run and retain their traceback. This is a performance
probe, not evidence of convergence or a guarantee that full training will finish.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/opt/sion")
VOLUME = "sion-training-probe-results"
TIMEOUT = 1200
STARTUP = 180
CHILD_TIMEOUT = 210
MRT_CHILD_TIMEOUT = 360
CPU = 8
MEMORY_GIB = 48
PRICES = {"a100-80gb": 0.000694, "h100": 0.001097}
GPU_NAMES = {"a100-80gb": "A100-80GB", "h100": "H100!"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def smoke_module():
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("modal_gpu_smoke")


def compute_contingency(target: str) -> float:
    """Cover two startup/function windows; this is not a provider hard cap."""
    rate = PRICES[target] + CPU * 0.0000131 + MEMORY_GIB * 0.00000222
    return 2 * (TIMEOUT + STARTUP + 10) * rate


def verify_plan(directory: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256(directory / "plan.json") != expected_sha256:
        raise ValueError("prepared probe plan SHA-256 mismatch")
    plan = read_json(directory / "plan.json")
    if plan["schema"] != "sion-training-probe-data-v1":
        raise ValueError("unsupported probe data schema")
    if not plan["provenance"]["complete_indexed_inventories_verified"]:
        raise ValueError("complete prepared inventories were not authenticated")
    for name, record in plan["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or path.is_symlink():
            raise ValueError("probe asset escapes the prepared directory")
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"probe asset integrity mismatch: {name}")
    return plan


def batch_at_fraction(items: list[dict[str, Any]], batch_size: int, fraction: float):
    """Select one contiguous length bucket, never pad by duplicating rows."""
    if batch_size < 1 or batch_size > len(items) or not 0 <= fraction <= 1:
        raise ValueError("invalid cohort batch request")
    ordered = sorted(items, key=lambda item: max(len(item["src"]), len(item["tgt"])))
    start = int((len(ordered) - batch_size) * fraction)
    return ordered[start : start + batch_size]


def summarize_measurements(
    samples: list[dict[str, Any]], optimizer_seconds: float, batch_size: int, effective_batch: int
) -> dict[str, float | int]:
    if (
        not samples
        or not math.isfinite(optimizer_seconds)
        or optimizer_seconds < 0
        or batch_size < 1
        or effective_batch < 1
        or any(not math.isfinite(item["seconds"]) or item["seconds"] <= 0 for item in samples)
    ):
        raise ValueError("invalid timing inputs")
    accumulation = max(1, round(effective_batch / batch_size))
    seconds = sum(float(item["seconds"]) for item in samples)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("timings must be finite and positive")
    projected_seconds = seconds + len(samples) * optimizer_seconds / accumulation
    return {
        "micro_batch": batch_size,
        "projected_accumulation": accumulation,
        "projected_effective_batch": batch_size * accumulation,
        "examples_per_second": len(samples) * batch_size / projected_seconds,
        "tokens_per_second": sum(item["tokens"] for item in samples) / projected_seconds,
        "mean_micro_seconds": statistics.mean(item["seconds"] for item in samples),
        "optimizer_ema_seconds": optimizer_seconds,
        "measurement_microbatches": len(samples),
    }


def resolved_config(plan: dict[str, Any], directory: Path):
    from sion_translate.auto import apply_auto_settings, probe_environment
    from sion_translate.config import config_from_raw

    raw = deepcopy(plan["config"])
    raw["data"].pop("language_pair", None)
    raw["training"].pop("fsdp2", None)
    config = config_from_raw(raw)
    stage = plan["stages"]["sft"]
    decisions = apply_auto_settings(
        config,
        plan["raw_config"],
        probe_environment(),
        train_examples=stage["examples_per_epoch"],
        validation_examples=1,
        physical_train_pairs=stage["physical_pairs"],
        physical_train_tokens=stage["physical_tokens"],
    )
    config.data.tokenizer_model = str(directory / "tokenizer/sion.model")
    config.data.tokenizer_features = str(directory / "tokenizer/token_features.npz")
    # Validation cadence is reported from the prepared plan separately. The
    # microbenchmark never runs validation against its training cohort.
    if config.training.precision != "bf16" or config.training.compile:
        raise ValueError("this probe currently requires eager BF16 production settings")
    config.validate()
    return config, decisions


def run_case(directory: Path, plan_hash: str, stage_name: str, batch_size: int, output: Path):
    import torch

    from sion_translate.cli.train import _build_posttraining_config, build_collator_args
    from sion_translate.data.collate import SionBatchCollator
    from sion_translate.foundation import build_foundation_config
    from sion_translate.model import SionForConditionalGeneration
    from sion_translate.tokenizer import SionTokenizer
    from sion_translate.training.checkpoint import load_checkpoint, save_checkpoint
    from sion_translate.training.distributed import DistributedContext
    from sion_translate.training.ema import EMAWeights
    from sion_translate.training.objectives import MinimumRiskObjective
    from sion_translate.training.trainer import (
        _normalize_and_clip_finite_gradients,
        build_optimizer_param_groups,
        cosine_scheduler,
    )

    started = time.perf_counter()
    torch.set_num_threads(CPU)
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    plan = verify_plan(directory, plan_hash)
    config, decisions = resolved_config(plan, directory)
    if stage_name == "foundation":
        config = build_foundation_config(config)
    elif stage_name == "mrt":
        config = _build_posttraining_config(config, output.parent)
    elif stage_name != "sft":
        raise ValueError("unknown training stage")
    training = config.training
    effective_batch = training.batch_size_per_gpu * training.gradient_accumulation_steps
    tokenizer = SionTokenizer(directory / "tokenizer/sion.model")
    collator = SionBatchCollator(
        **build_collator_args(config, tokenizer),
        denoise_probability=0.0 if stage_name == "mrt" else config.data.denoise_probability,
        source_token_dropout=0.0 if stage_name == "mrt" else config.data.source_token_dropout,
        decoder_input_noise=0.0 if stage_name == "mrt" else config.data.decoder_input_noise,
    )
    cohort = plan["stages"]["sft" if stage_name == "mrt" else stage_name]
    objective = (
        MinimumRiskObjective(tokenizer, config.posttraining) if stage_name == "mrt" else None
    )
    device = torch.device("cuda", 0)
    context = DistributedContext(0, 0, 1, device, False)
    with torch.device("meta"):
        model = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, training.weight_decay),
        lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_eps,
        weight_decay=0.0,
        fused=True,
    )
    scheduler = cosine_scheduler(
        optimizer, warmup_steps=1, max_steps=100, min_ratio=training.min_learning_rate_ratio
    )
    ema = EMAWeights(model, training.ema_decay) if training.ema_decay > 0 else None
    total_updates = 0

    def micro(items):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        batch = {key: value.to(device) for key, value in collator(items).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if objective is None:
                result = model(**batch)
                loss = result.lm_loss_sum + result.auxiliary_loss * result.token_count.detach()
                normalizer = result.token_count.detach()
                tokens = normalizer
            else:
                result = objective(model, batch)
                loss = result.loss_sum
                normalizer = result.normalizer.detach()
                tokens = result.processed_tokens.detach()
        if not bool(torch.isfinite(loss).item()) or float(normalizer.item()) <= 0:
            raise FloatingPointError("non-finite loss or empty normalization mass")
        loss.backward()
        torch.cuda.synchronize()
        measurement = {
            "seconds": time.perf_counter() - begin,
            "normalizer": float(normalizer.item()),
            "tokens": int(tokens.item()),
            "loss": float((loss.detach() / normalizer).item()),
            "source_shape": list(batch["input_ids"].shape),
            "target_shape": list(batch["labels"].shape),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        print(json.dumps({"stage": stage_name, "batch": batch_size, **measurement}), flush=True)
        return measurement

    def update(normalizer):
        nonlocal total_updates
        torch.cuda.synchronize()
        begin = time.perf_counter()
        gradients = [p for p in model.parameters() if p.grad is not None]
        if not gradients:
            raise RuntimeError("model has no gradients")
        _normalize_and_clip_finite_gradients(
            gradients,
            global_normalizer=normalizer,
            max_norm=training.grad_clip,
            context=context,
            stage_name=stage_name,
            next_step=total_updates + 1,
        )
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        total_updates += 1
        torch.cuda.synchronize()
        return time.perf_counter() - begin

    warmup = micro(batch_at_fraction(cohort["representative"], batch_size, 0.5))
    update(warmup["normalizer"])
    torch.cuda.reset_peak_memory_stats()
    # Equal-probability length strata approximate the sampled distribution.
    # Inputs are collated/transferred inside the timing. Whole-corpus disk and
    # DataLoader stalls are not measured by this resident-cohort experiment.
    fractions = [0.125, 0.375, 0.625, 0.875]
    samples = [micro(batch_at_fraction(cohort["representative"], batch_size, f)) for f in fractions]
    optimizer_seconds = update(sum(item["normalizer"] for item in samples))
    summary = summarize_measurements(samples, optimizer_seconds, batch_size, effective_batch)
    summary["effective_batch_changed"] = summary["projected_effective_batch"] != effective_batch
    report = {
        "status": "running",
        "stage": stage_name,
        "batch_size": batch_size,
        "parameter_count": model.parameter_count(),
        "gradient_checkpointing": config.model.gradient_checkpointing,
        "precision": training.precision,
        "epochs": training.num_train_epochs,
        "max_steps": training.max_steps,
        "examples_per_epoch": cohort["examples_per_epoch"],
        "decisions": decisions,
        "summary": summary,
        "samples": samples,
        "representative_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "representative_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "timing_scope": "resident cohort, eager BF16, random initialization, forward/backward and projected optimizer/EMA; excludes whole-corpus I/O, validation, export and startup",
    }
    write_json(output, report)
    torch.cuda.reset_peak_memory_stats()
    stress = micro(cohort["stress"][:batch_size])
    update(stress["normalizer"])
    report["stress"] = stress
    report["stress_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["stress_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    write_json(output, report)

    # Authenticate an actual full optimizer/EMA checkpoint once per stage.
    # Save and resume are outside steady-state throughput, but retain timings.
    if batch_size == (1 if stage_name == "mrt" else 8):
        checkpoint = output.parent / f"checkpoint-{stage_name}"
        begin = time.perf_counter()
        save_checkpoint(checkpoint, model, optimizer, scheduler, total_updates, context, ema=ema)
        save_seconds = time.perf_counter() - begin
        parameter = next(model.parameters())
        before = parameter.detach().clone()
        with torch.no_grad():
            parameter.zero_()
        begin = time.perf_counter()
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, context, ema=ema)
        load_seconds = time.perf_counter() - begin
        if restored != total_updates or not torch.equal(parameter, before):
            raise RuntimeError("real-data checkpoint failed to restore step and weights")
        resumed = micro(batch_at_fraction(cohort["representative"], batch_size, 0.5))
        update(resumed["normalizer"])
        report["checkpoint"] = {
            "save_seconds": save_seconds,
            "load_seconds": load_seconds,
            "resumed_update": True,
            "bytes": sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        # This checkpoint is disposable random-weight probe state, not a model.
        # Keep only its byte count and validated resume result on the durable volume.
        import shutil

        shutil.rmtree(checkpoint)
    report["status"] = "passed"
    report["total_seconds"] = time.perf_counter() - started
    write_json(output, report)
    return report


def child(args) -> int:
    try:
        run_case(args.data, args.plan_sha256, args.stage, args.batch, args.output)
    except BaseException as error:
        import torch

        status = "capacity_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        previous = read_json(args.output) if args.output.is_file() else {}
        previous.update(status=status, error=repr(error), traceback=traceback.format_exc())
        write_json(args.output, previous)
        traceback.print_exc()
        return 20 if status == "capacity_oom" else 1
    return 0


def execute(target: str, run_id: str, plan_hash: str, source_hash: str, probe_hash: str):
    """Modal entrypoint: run bounded children and publish progress after each one."""
    import modal
    import torch

    smoke = smoke_module()
    volume = modal.Volume.from_name(VOLUME)
    root = Path("/probe-results") / run_id
    root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "run_id": run_id,
        "target": target,
        "call_id": modal.current_function_call_id(),
        "plan_sha256": plan_hash,
        "source_contract_sha256": source_hash,
        "probe_sha256": probe_hash,
        "status": "running",
        "cases": [],
    }
    started = time.monotonic()

    def publish():
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "result.json", report)
        volume.commit()

    publish()
    try:
        smoke._verify_executed_entrypoint(
            Path(__file__), REMOTE / "scripts/modal_training_probe.py"
        )
        if sha256(REMOTE / "scripts/modal_training_probe.py") != probe_hash:
            raise RuntimeError("benchmark executable hash mismatch")
        if smoke.gpu_smoke_contract_sha256(REMOTE) != source_hash:
            raise RuntimeError("training source contract mismatch")
        verify_plan(REMOTE / "probe", plan_hash)
        runtime = smoke._runtime_report(torch)
        smoke._validate_target_hardware(target, runtime)
        report["runtime"] = runtime
        publish()
        for stage in ("foundation", "sft", "mrt"):
            child_timeout = MRT_CHILD_TIMEOUT if stage == "mrt" else CHILD_TIMEOUT
            for batch_size in (1, 2, 4, 8) if stage == "mrt" else (8, 16, 32, 64):
                remaining = TIMEOUT - (time.monotonic() - started) - 45
                if remaining < child_timeout:
                    report["status"] = "incomplete_deadline"
                    publish()
                    return report
                name = f"{stage}-b{batch_size}"
                case_path = root / f"{name}.json"
                report["active_case"] = name
                publish()
                command = [
                    sys.executable,
                    "-W",
                    "error",
                    str(REMOTE / "scripts/modal_training_probe.py"),
                    "child",
                    "--data",
                    str(REMOTE / "probe"),
                    "--plan-sha256",
                    plan_hash,
                    "--stage",
                    stage,
                    "--batch",
                    str(batch_size),
                    "--output",
                    str(case_path),
                ]
                with (
                    (root / f"{name}.stdout.log").open("w") as stdout,
                    (root / f"{name}.stderr.log").open("w") as stderr,
                ):
                    process = subprocess.Popen(
                        command, stdout=stdout, stderr=stderr, start_new_session=True
                    )
                    try:
                        returncode = process.wait(timeout=child_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=15)
                        raise TimeoutError(f"{name} exceeded {child_timeout} seconds") from None
                case = read_json(case_path) if case_path.is_file() else {}
                case["name"] = name
                case["returncode"] = returncode
                report["cases"].append(case)
                publish()
                if returncode == 20 and case.get("status") == "capacity_oom":
                    break
                if returncode != 0 or case.get("status") != "passed":
                    raise RuntimeError(f"{name} failed; inspect its preserved stderr and JSON")
                # Leave at least 15% for runtime variation and evaluation. This
                # threshold is a search stop, not a safety or speed guarantee.
                total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if case["stress_peak_reserved_gib"] > total_gib * 0.85:
                    break
        report.pop("active_case", None)
        passed_stages = {
            case.get("stage") for case in report["cases"] if case.get("status") == "passed"
        }
        report["status"] = (
            "completed" if passed_stages == {"foundation", "sft", "mrt"} else "failed"
        )
        publish()
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        publish()
        raise


def build_app(data: Path, target: str):
    import modal

    smoke = smoke_module()
    smoke._validate_modal_client_version()
    image = (
        smoke.image.add_local_file(
            str(Path(__file__)), (REMOTE / "scripts/modal_training_probe.py").as_posix(), copy=True
        )
        .add_local_dir(str(data), (REMOTE / "probe").as_posix(), copy=True)
        .env({"PYTHONPATH": "/opt/sion/src:/opt/sion/scripts"})
    )
    app = modal.App("sion-real-data-training-probe", include_source=False)
    function = app.function(
        image=image,
        gpu=GPU_NAMES[target],
        cpu=CPU,
        memory=MEMORY_GIB * 1024,
        timeout=TIMEOUT,
        startup_timeout=STARTUP,
        retries=0,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        single_use_containers=True,
        include_source=True,
        volumes={"/probe-results": modal.Volume.from_name(VOLUME, create_if_missing=True)},
    )(execute)
    return app, function


def submit(args):
    import modal

    if not math.isfinite(args.max_dollars) or args.max_dollars < compute_contingency(args.target):
        raise ValueError("insufficient two-attempt compute authorization")
    if args.workspace_hard_budget != 30:
        raise ValueError(
            "this run is authorized only under the confirmed $30 Workspace hard budget"
        )
    root = args.receipts.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "submission.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        for old in root.glob("*/receipt.json"):
            if read_json(old).get("state") not in {"completed", "failed", "incomplete_deadline"}:
                raise RuntimeError(
                    f"reconcile the previous receipt before another submission: {old}"
                )
        active = json.loads(
            subprocess.check_output(
                [sys.executable, "-m", "modal", "app", "list", "--json"], text=True
            )
        )
        if any(
            str(item.get("State", item.get("state", ""))).lower() != "stopped" for item in active
        ):
            raise RuntimeError("another Modal app is active; inspect it before spending")
        billing = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "billing",
                    "report",
                    "--for",
                    "this month",
                    "--show-resources",
                    "--json",
                ],
                text=True,
            )
        )
        observed_cost = sum(float(item["cost"]) for item in billing)
        if not math.isfinite(observed_cost) or observed_cost < 0:
            raise ValueError("provider billing report is not finite and non-negative")
        if observed_cost + args.max_dollars > args.workspace_hard_budget:
            raise ValueError("fresh reported usage plus authorization exceeds the Workspace budget")
        # Keep the fresh provider response verbatim. Exact credit balance is a
        # user-confirmed value; this CLI reports charges, not remaining credits.
        data = args.data.resolve(strict=True)
        plan_hash = sha256(data / "plan.json")
        verify_plan(data, plan_hash)
        smoke = smoke_module()
        contract = smoke.gpu_smoke_contract_sha256(ROOT)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
        ).strip()
        tracked = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src",
                "scripts/modal_training_probe.py",
                "scripts/modal_gpu_smoke.py",
                "requirements",
            ],
            cwd=ROOT,
            text=True,
        )
        if tracked.strip():
            raise RuntimeError("commit the reviewed GPU runtime files before submission")
        run_id = (
            "probe-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
            + "-"
            + uuid.uuid4().hex[:12]
        )
        receipt_path = root / run_id / "receipt.json"
        receipt = {
            "run_id": run_id,
            "target": args.target,
            "state": "submission_intent",
            "commit": commit,
            "tree": tree,
            "source_contract_sha256": contract,
            "probe_sha256": sha256(Path(__file__)),
            "plan_sha256": plan_hash,
            "data": str(data),
            "billing_before": billing,
            "reported_monthly_usage_usd": observed_cost,
            "max_dollars": args.max_dollars,
            "workspace_hard_budget": args.workspace_hard_budget,
            "compute_contingency_usd": compute_contingency(args.target),
            "created_utc": datetime.now(UTC).isoformat(),
            "volume": VOLUME,
        }
        write_json(receipt_path, receipt)
        # Canonical registration makes Modal import this module remotely instead
        # of serializing a __main__ closure with a client-side filesystem path.
        canonical = importlib.import_module("modal_training_probe")
        try:
            app, function = canonical.build_app(data, args.target)
        except Exception as error:
            receipt.update(state="failed", phase="local_app_definition", error=repr(error))
            write_json(receipt_path, receipt)
            raise
        with modal.enable_output(), app.run(detach=True):
            receipt["app_id"] = app.app_id
            write_json(receipt_path, receipt)
            call = function.spawn(args.target, run_id, plan_hash, contract, receipt["probe_sha256"])
            receipt.update(state="submitted", call_id=call.object_id)
            write_json(receipt_path, receipt)
        print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2))
    finally:
        os.close(descriptor)
        lock.unlink()


def status(args):
    import modal

    receipt = read_json(args.receipt)
    call = modal.FunctionCall.from_id(receipt["call_id"])
    result = None
    call_error = None
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        pass
    except Exception as error:
        if type(error) is not modal.exception.TimeoutError:
            call_error = repr(error)
            receipt["call_error"] = call_error
    volume = modal.Volume.from_name(receipt["volume"])
    try:
        journal = json.loads(b"".join(volume.read_file(f"{receipt['run_id']}/result.json")))
    except FileNotFoundError:
        journal = None
    if journal is not None:
        for key in (
            "run_id",
            "target",
            "call_id",
            "plan_sha256",
            "probe_sha256",
            "source_contract_sha256",
        ):
            if journal[key] != receipt[key]:
                raise RuntimeError(f"remote journal identity mismatch: {key}")
        if result is not None and result != journal:
            raise RuntimeError("FunctionCall result differs from the durable journal")
        write_json(args.receipt.parent / "result.json", journal)
        if result is not None and call_error is None:
            receipt["state"] = journal["status"]
        elif call_error is not None:
            receipt["state"] = "failed" if journal["status"] == "failed" else "needs_reconciliation"
    if result is not None and journal is None:
        raise RuntimeError("function returned without a durable journal")
    logs = [
        {
            "timestamp": entry.timestamp.isoformat(),
            "message": entry.message,
            "source": str(entry.source),
        }
        for entry in call.logs.tail(entries=500)
    ]
    write_json(args.receipt.parent / "function-logs.json", logs)
    write_json(args.receipt, receipt)
    print(
        json.dumps(
            {"receipt": str(args.receipt), "state": receipt["state"], "journal": journal}, indent=2
        )
    )


def main():
    # Modal's status output includes Unicode symbols. Korean Windows consoles
    # may default to CP949, which cannot encode them even in redirected logs.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("submit")
    launch.add_argument("--data", type=Path, required=True)
    launch.add_argument("--target", choices=GPU_NAMES, required=True)
    launch.add_argument("--max-dollars", type=float, required=True)
    launch.add_argument("--workspace-hard-budget", type=float, required=True)
    launch.add_argument("--receipts", type=Path, default=ROOT / "artifacts/modal-probes")
    inspect = commands.add_parser("status")
    inspect.add_argument("--receipt", type=Path, required=True)
    worker = commands.add_parser("child")
    worker.add_argument("--data", type=Path, required=True)
    worker.add_argument("--plan-sha256", required=True)
    worker.add_argument("--stage", choices=("foundation", "sft", "mrt"), required=True)
    worker.add_argument("--batch", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "child":
        raise SystemExit(child(args))
    if args.command == "submit":
        submit(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
