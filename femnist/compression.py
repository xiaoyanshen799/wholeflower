"""Lightweight uniform quantizer with optional error feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

NDArray = np.ndarray
_FLT_EPS = 1e-8
_INT4_OFFSET = 8


@dataclass(frozen=True)
class QuantizationReport:
    compression_ratio: float
    quant_error_rmse: float
    avg_scale: float
    max_scale: float
    num_tensors: int
    num_bits: int

    def as_metrics(self) -> Dict[str, float]:
        return {
            "compression_ratio": self.compression_ratio,
            "quant_error_rmse": self.quant_error_rmse,
            "quant_avg_scale": self.avg_scale,
            "quant_max_scale": self.max_scale,
        }


class ErrorFeedbackQuantizer:
    """Uniform per-tensor quantization with optional residual feedback."""

    SUPPORTED_BITS = (16, 8, 4)

    def __init__(self, num_bits: int = 8, *, error_feedback: bool = True) -> None:
        if num_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported bit-width {num_bits}. Supported values: {self.SUPPORTED_BITS}"
            )
        self.num_bits = num_bits
        self.error_feedback = error_feedback
        self.level = (1 << (num_bits - 1)) - 1
        self._residuals: Optional[List[NDArray]] = None
        self._last_report: Optional[QuantizationReport] = None

    def encode(self, weights: Iterable[NDArray]) -> Tuple[List[NDArray], QuantizationReport]:
        tensors = [self._ensure_float32(w) for w in weights]
        if self._residuals is None or len(self._residuals) != len(tensors):
            self._residuals = [np.zeros_like(w, dtype=np.float32) for w in tensors]
        else:
            self._ensure_residual_shapes(tensors)

        if not self.error_feedback and self._residuals is not None:
            self._residuals = [np.zeros_like(w, dtype=np.float32) for w in tensors]

        packed: List[NDArray] = []
        scales: List[float] = []
        squared_error = 0.0
        elem_count = 0
        original_bits = 0
        compressed_bits = 0

        for idx, (baseline, feedback) in enumerate(zip(tensors, self._residuals)):
            compensated = baseline + feedback
            absmax = float(np.max(np.abs(compensated)))
            if not np.isfinite(absmax):
                absmax = 0.0
            scale = absmax / self.level if absmax > 0 else 1.0
            scale = max(scale, _FLT_EPS)

            q_dtype = np.int8 if self.num_bits <= 8 else np.int16
            quantized = np.clip(
                np.round(compensated / scale), -self.level, self.level
            ).astype(q_dtype)
            dequantized = quantized.astype(np.float32) * scale

            if self.error_feedback:
                self._residuals[idx] = compensated - dequantized
            else:
                self._residuals[idx].fill(0.0)

            if self.num_bits == 16:
                packed.append(quantized.astype(np.int16, copy=False))
                packed.append(np.array([scale], dtype=np.float32))
                compressed_bits += quantized.size * self.num_bits + 32
            elif self.num_bits == 8:
                packed.append(quantized.astype(np.int8, copy=False))
                packed.append(np.array([scale], dtype=np.float32))
                compressed_bits += quantized.size * self.num_bits + 32
            else:
                packed_bytes = _pack_int4(quantized)
                packed.append(packed_bytes)
                packed.append(np.array([scale], dtype=np.float32))
                packed.append(np.array(baseline.shape, dtype=np.int32))
                compressed_bits += quantized.size * self.num_bits + 32 + baseline.ndim * 32

            scales.append(scale)
            squared_error += float(np.sum((dequantized - baseline) ** 2))
            elem_count += baseline.size
            original_bits += baseline.size * baseline.itemsize * 8

        compression_ratio = (
            (original_bits / compressed_bits) if compressed_bits > 0 else 1.0
        )
        quant_error_rmse = (
            float(np.sqrt(squared_error / elem_count)) if elem_count > 0 else 0.0
        )
        avg_scale = float(np.mean(scales)) if scales else 0.0
        max_scale = float(np.max(scales)) if scales else 0.0

        report = QuantizationReport(
            compression_ratio=compression_ratio,
            quant_error_rmse=quant_error_rmse,
            avg_scale=avg_scale,
            max_scale=max_scale,
            num_tensors=len(tensors),
            num_bits=self.num_bits,
        )
        self._last_report = report
        return packed, report

    def set_state(self, residuals: Optional[List[NDArray]]) -> None:
        if residuals is None:
            self._residuals = None
            return
        self._residuals = [np.array(r, dtype=np.float32, copy=True) for r in residuals]

    def get_state(self) -> Optional[List[NDArray]]:
        if self._residuals is None:
            return None
        return [r.astype(np.float32, copy=True) for r in self._residuals]

    def last_report(self) -> Optional[QuantizationReport]:
        return self._last_report

    @staticmethod
    def maybe_unpack(packed: List[NDArray]) -> Tuple[List[NDArray], bool]:
        unpacked: List[NDArray] = []
        idx = 0
        changed = False
        while idx < len(packed):
            tensor = packed[idx]
            if (
                np.issubdtype(tensor.dtype, np.signedinteger)
                and tensor.dtype.itemsize in (1, 2)
                and idx + 1 < len(packed)
                and _is_scale_tensor(packed[idx + 1])
            ):
                scale = float(packed[idx + 1][0])
                unpacked.append(tensor.astype(np.float32) * scale)
                idx += 2
                changed = True
                continue

            if (
                tensor.dtype == np.uint8
                and idx + 2 < len(packed)
                and _is_scale_tensor(packed[idx + 1])
                and _is_shape_tensor(packed[idx + 2])
            ):
                scale = float(packed[idx + 1][0])
                shape = tuple(int(dim) for dim in packed[idx + 2])
                restored = _unpack_int4(tensor, shape)
                unpacked.append(restored.astype(np.float32) * scale)
                idx += 3
                changed = True
                continue

            unpacked.append(tensor)
            idx += 1

        return unpacked, changed

    @staticmethod
    def is_packed(packed: List[NDArray]) -> bool:
        _, changed = ErrorFeedbackQuantizer.maybe_unpack(list(packed))
        return changed

    @staticmethod
    def _ensure_float32(weight: NDArray) -> NDArray:
        return weight.astype(np.float32, copy=False)

    def _ensure_residual_shapes(self, tensors: List[NDArray]) -> None:
        assert self._residuals is not None
        for idx, tensor in enumerate(tensors):
            if self._residuals[idx].shape != tensor.shape:
                self._residuals[idx] = np.zeros_like(tensor, dtype=np.float32)


def maybe_unpack_quantized(weights: List[NDArray]) -> Tuple[List[NDArray], bool]:
    return ErrorFeedbackQuantizer.maybe_unpack(weights)


def _pack_int4(values: NDArray) -> NDArray:
    flat = values.astype(np.int8, copy=False).ravel()
    shifted = (flat + _INT4_OFFSET).astype(np.uint8)
    if shifted.size % 2 != 0:
        shifted = np.concatenate([shifted, np.array([_INT4_OFFSET], dtype=np.uint8)])
    high = shifted[0::2]
    low = shifted[1::2]
    packed = (high << 4) | (low & 0x0F)
    return packed


def _unpack_int4(packed: NDArray, shape: Sequence[int]) -> NDArray:
    bytes_flat = packed.ravel()
    expanded = np.empty(bytes_flat.size * 2, dtype=np.uint8)
    expanded[0::2] = (bytes_flat >> 4) & 0x0F
    expanded[1::2] = bytes_flat & 0x0F
    numel = int(np.prod(shape))
    expanded = expanded[:numel]
    int_vals = expanded.astype(np.int16) - _INT4_OFFSET
    return int_vals.astype(np.int8).reshape(shape)


def _is_scale_tensor(tensor: NDArray) -> bool:
    return tensor.shape == (1,) and np.issubdtype(tensor.dtype, np.floating)


def _is_shape_tensor(tensor: NDArray) -> bool:
    return tensor.ndim == 1 and np.issubdtype(tensor.dtype, np.integer)
