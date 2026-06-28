"""DCT helpers used by the educational libjxl-tiny Python port."""

from __future__ import annotations

import numpy as np


BLOCK_DIM = 8
K_SQRT2 = np.float32(1.41421356237)
WC_MULTIPLIERS_8 = np.asarray(
    (0.5097955791041592, 0.6013448869350453, 0.8999762231364156,
     2.5629154477415055),
    dtype=np.float32,
)
WC_MULTIPLIERS = {
    4: np.asarray((0.541196100146197, 1.3065629648763764), dtype=np.float32),
    8: WC_MULTIPLIERS_8,
}


def _f32(value: float | np.float32) -> np.float32:
  return np.float32(value)


def _dct_1d(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  n = values.shape[0]
  if n == 1:
    return values.copy()
  if n == 2:
    return np.asarray((values[0] + values[1], values[0] - values[1]),
                      dtype=np.float32)

  half = n // 2
  tmp = np.empty(n, dtype=np.float32)
  for i in range(half):
    tmp[i] = _f32(values[i] + values[n - i - 1])
    tmp[half + i] = _f32(values[i] - values[n - i - 1])
  tmp[:half] = _dct_1d(tmp[:half])
  tmp[half:] = _dct_1d(tmp[half:] * WC_MULTIPLIERS[n])

  tmp[half] = _f32(tmp[half] * K_SQRT2 + tmp[half + 1])
  for i in range(half + 1, n - 1):
    tmp[i] = _f32(tmp[i] + tmp[i + 1])

  out = np.empty(n, dtype=np.float32)
  for i in range(half):
    out[2 * i] = tmp[i]
    out[2 * i + 1] = tmp[half + i]
  return out


def scaled_dct_8x8(block: np.ndarray) -> np.ndarray:
  """Compute the scaled 8x8 DCT used by `TransformFromPixels(DCT)`."""
  block = np.asarray(block, dtype=np.float32)
  if block.shape != (BLOCK_DIM, BLOCK_DIM):
    raise ValueError("expected an 8x8 block")

  tmp = np.empty_like(block)
  for x in range(BLOCK_DIM):
    tmp[:, x] = _dct_1d(block[:, x]) * _f32(1.0 / BLOCK_DIM)

  transposed = tmp.T.copy()
  out = np.empty_like(block)
  for x in range(BLOCK_DIM):
    out[:, x] = _dct_1d(transposed[:, x]) * _f32(1.0 / BLOCK_DIM)
  return out
