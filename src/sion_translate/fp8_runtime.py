"""Run inference while keeping exported weights resident in FP8.

Exported FP8 weights and their block scales remain in their packed dtypes in
model buffers, which reduces resident weight memory. ``Fp8Linear`` currently
has one compute path: each forward pass dequantizes the complete weight to the
device's supported ``compute_dtype`` (BF16 by default, or FP16 on CUDA devices
without BF16 support) and calls ``torch.nn.functional.linear``.

This implementation does not call ``torch._scaled_mm`` or use native FP8
Tensor Core GEMM. It creates a temporary dense dequantized weight, so it also
does not promise lower runtime bandwidth or fewer operations. Benchmark the
target device before claiming a performance gain.

Activations remain high precision. Weight-only FP8 was more accurate in the
measurements (2.57% versus 3.63% output error) and is insensitive to activation
outliers. See :mod:`sion_translate.fp8` for the supporting measurements.
"""

# Torch FP8 primitives and optional packed-module hooks are incompletely typed.
# pyright: reportArgumentType=false, reportCallIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import torch
from torch import nn

from sion_translate.fp8 import DEFAULT_BLOCK, FORWARD_DTYPE, fp8_gemm_supported


def _cuda_bf16_supported(device: torch.device) -> bool:
    """Return whether ``device`` has native BF16 CUDA arithmetic.

    CUDA BF16 tensor-core arithmetic starts with Ampere (SM 8.x).  Looking up
    the requested device directly also avoids accidentally inspecting CUDA's
    current device in a heterogeneous process.
    """

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8


def resolve_fp8_compute_dtype(device: torch.device) -> torch.dtype:
    """Choose the dense dtype used after restoring resident FP8 weights.

    FP8 is only the resident weight format in this runtime.  CUDA devices use
    BF16 when the architecture supports it (including A100) and otherwise use
    FP16, so loading an FP8 export never requires native FP8 tensor cores.  The
    existing CPU path remains BF16; other accelerators conservatively use FP16.
    """

    if device.type == "cpu":
        return torch.bfloat16
    if device.type == "cuda" and _cuda_bf16_supported(device):
        return torch.bfloat16
    return torch.float16


class Fp8Linear(nn.Module):
    """Replace ``nn.Linear`` while storing weights as E4M3 plus block scales.

    Bias is intentionally unsupported because every projection in this model
    is bias-free and an unused branch would add unnecessary state and checks.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        *,
        block: int = DEFAULT_BLOCK,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if weight.dtype is not FORWARD_DTYPE:
            raise ValueError(f"Fp8Linear expects {FORWARD_DTYPE} weights, got {weight.dtype}")
        if weight.ndim != 2:
            raise ValueError("Fp8Linear expects a 2-D weight")
        out_features, in_features = weight.shape
        if in_features % block:
            raise ValueError(
                f"in_features {in_features} is not a multiple of the FP8 block size {block}"
            )
        if scales.shape != (out_features, in_features // block):
            raise ValueError(
                f"scale shape {tuple(scales.shape)} does not match "
                f"{(out_features, in_features // block)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block = block
        self.compute_dtype = compute_dtype
        # Buffers are not trainable, but remain available in ``state_dict``.
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)

    def dequantized_weight(self) -> torch.Tensor:
        compute_dtype = self.compute_dtype or resolve_fp8_compute_dtype(self.weight.device)
        grouped = self.weight.to(compute_dtype).reshape(self.out_features, -1, self.block)
        restored = grouped * self.scales.to(compute_dtype).unsqueeze(-1)
        return restored.reshape(self.out_features, self.in_features)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight()
        output = torch.nn.functional.linear(hidden.to(weight.dtype), weight)
        # Fp8Linear is embedded between residual branches.  Returning FP16 into
        # a BF16 residual promotes the sum to FP32 and makes the next BF16
        # Linear fail with a mixed-dtype matmul.  Keep the module boundary in
        # the caller's dtype while using the selected fallback dtype internally.
        return output.to(hidden.dtype)

    def extra_repr(self) -> str:
        compute_dtype = (
            str(self.compute_dtype).removeprefix("torch.")
            if self.compute_dtype is not None
            else "auto(bfloat16->float16)"
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block={self.block}, weight_dtype={FORWARD_DTYPE}, "
            f"compute_dtype={compute_dtype}"
        )


def _replace_child(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attribute, replacement)


def apply_fp8_weights(
    model: nn.Module,
    packed_state: dict[str, dict[str, object]],
    *,
    compute_dtype: torch.dtype | None = None,
) -> int:
    """Replace modules backed by packed FP8 entries with ``Fp8Linear``.

    ``packed_state`` is produced by ``training.export._pack_fp8_state``. Only
    ``block_fp8`` entries are replaced. The export policy therefore remains
    authoritative about which tensors stay in high precision.

    When ``compute_dtype`` is omitted, the forward path checks BF16 support on
    the active device and falls back to FP16 on unsupported CUDA devices. The
    return value is the number of replaced modules.
    """

    replaced = 0
    for name, entry in packed_state.items():
        if entry.get("kind") != "block_fp8":
            continue
        if not name.endswith(".weight"):
            raise ValueError(f"FP8 packed entry is not a linear weight: {name}")
        module_name = name.removesuffix(".weight")
        target = model.get_submodule(module_name)
        if not isinstance(target, nn.Linear):
            raise ValueError(
                f"{module_name} is a {type(target).__name__}, not nn.Linear; "
                "the FP8 export and this model do not agree"
            )
        if target.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(f"{module_name} has a bias, which Fp8Linear does not carry")
        _replace_child(
            model,
            module_name,
            Fp8Linear(
                entry["values"],
                entry["scales"],
                block=int(entry.get("block", DEFAULT_BLOCK)),
                compute_dtype=compute_dtype,
            ),
        )
        replaced += 1
    return replaced


def prepare_fp8_model_for_device(model: nn.Module, device: torch.device) -> torch.dtype:
    """Move an FP8-resident model and align its dense state with ``device``.

    ``nn.Module.to(dtype=...)`` would also cast ``Fp8Linear.weight`` and
    ``Fp8Linear.scales``, destroying the resident FP8 representation and its
    full-precision block scales.  Cast only ordinary parameters and non-FP8
    floating buffers on their current device first, then move every tensor
    without changing dtype.  In the common CPU-load path this also avoids ever
    placing BF16 dense tensors on a CUDA device that cannot execute them.

    The returned dtype is the dense compute dtype selected for telemetry and
    tests.  This helper is intended for inference models, before any gradients
    have been accumulated.
    """

    compute_dtype = resolve_fp8_compute_dtype(device)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype is not compute_dtype:
                parameter.data = parameter.data.to(dtype=compute_dtype)
            if parameter.grad is not None and parameter.grad.is_floating_point():
                parameter.grad.data = parameter.grad.data.to(dtype=compute_dtype)
        for module in model.modules():
            if isinstance(module, Fp8Linear):
                module.compute_dtype = compute_dtype
                continue
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point() and buffer.dtype is not compute_dtype:
                    setattr(module, name, buffer.to(dtype=compute_dtype))
    model.to(device=device)
    return compute_dtype


def describe_runtime(device: torch.device) -> str:
    """Describe the actual compute path in one deployment-log message.

    The message separates resident format, dense compute dtype, and hardware
    FP8 support. The current kernel does not use native FP8 even when the
    hardware supports it.
    """

    compute_dtype = resolve_fp8_compute_dtype(device)
    compute_name = "BF16" if compute_dtype is torch.bfloat16 else "FP16"
    if device.type == "cuda":
        if fp8_gemm_supported(device):
            hardware = "native FP8 Tensor Cores are available but unused by this path"
        else:
            hardware = "device lacks native FP8 Tensor Cores; using the dense fallback"
    else:
        hardware = f"{device.type.upper()} dense fallback"
    return (
        f"FP8-resident weights with on-demand {compute_name} dequantization and dense GEMM "
        f"(reduced resident weight memory; {hardware})"
    )


__all__ = [
    "Fp8Linear",
    "apply_fp8_weights",
    "describe_runtime",
    "prepare_fp8_model_for_device",
    "resolve_fp8_compute_dtype",
]
