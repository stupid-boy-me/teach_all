"""Multi-precision quantized linear layers for ViT comparison.

Supported weight modes:
  - fp32: standard nn.Linear
  - int8 / int4 / int2: symmetric uniform quantization
  - bit158: BitNet-style ternary weights {-1, 0, 1} (~1.58 bit)

Activations use per-token absmax int8 (BitNet-style) when quantizing.
Training uses STE (straight-through estimator).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_PRECISIONS = ("fp32", "int8", "int4", "int2", "bit158")


def _activation_quant_int8(x: torch.Tensor) -> torch.Tensor:
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def _weight_quant_uniform(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric absmax quantization to `bits` (including sign)."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax().clamp_(min=1e-5) / qmax
    q = (w / scale).round().clamp_(-qmax, qmax)
    return q * scale


def _weight_quant_bit158(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 ternary quantization: {-scale, 0, +scale}."""
    scale = w.abs().mean().clamp_(min=1e-5)
    q = (w / scale).round().clamp_(-1, 1)
    return q * scale


class QuantLinear(nn.Linear):
    """Drop-in Linear with selectable quantization precision."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision: str = "fp32",
        quant_activation: bool = True,
        device=None,
        dtype=None,
    ):
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"precision must be one of {SUPPORTED_PRECISIONS}, got {precision}")
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.precision = precision
        self.quant_activation = quant_activation and precision != "fp32"
        self.norm = nn.LayerNorm(in_features) if precision != "fp32" else nn.Identity()

    def _quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp32":
            return w
        if self.precision == "int8":
            return _weight_quant_uniform(w, 8)
        if self.precision == "int4":
            return _weight_quant_uniform(w, 4)
        if self.precision == "int2":
            return _weight_quant_uniform(w, 2)
        if self.precision == "bit158":
            return _weight_quant_bit158(w)
        raise RuntimeError(f"unknown precision: {self.precision}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp32":
            return F.linear(x, self.weight, self.bias)

        x_norm = self.norm(x)
        if self.quant_activation:
            x_q = _activation_quant_int8(x_norm)
            x_use = x_norm + (x_q - x_norm).detach()
        else:
            x_use = x_norm

        w_q = self._quantize_weight(self.weight)
        w_use = self.weight + (w_q - self.weight).detach()
        return F.linear(x_use, w_use, self.bias)


def count_effective_bits(precision: str) -> float:
    mapping = {
        "fp32": 32.0,
        "int8": 8.0,
        "int4": 4.0,
        "int2": 2.0,
        "bit158": math.log2(3),  # ~1.585
    }
    return mapping[precision]
