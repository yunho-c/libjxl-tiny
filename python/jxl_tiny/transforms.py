"""Scaled DCT helpers for the VarDCT subset used by libjxl-tiny.

The encoder only needs 8x8, 16x8, and 8x16 transforms. These routines preserve
the scaled coefficient layout expected by quantization and AC-strategy scoring
instead of presenting a general-purpose orthonormal DCT API.
"""

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
    16: np.asarray(
        (0.5024192861881557, 0.5224986149396889, 0.5669440348163577,
         0.6468217833599901, 0.7881546234512502, 1.060677685990347,
         1.7224470982383342, 5.101148618689155),
        dtype=np.float32,
    ),
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


def _dct_columns(block: np.ndarray) -> np.ndarray:
  rows, cols = block.shape
  out = np.empty_like(block)
  scale = _f32(1.0 / rows)
  for x in range(cols):
    out[:, x] = _dct_1d(block[:, x]) * scale
  return out


def scaled_dct(block: np.ndarray) -> np.ndarray:
  """Compute the scaled DCT used by `TransformFromPixels`.

  For 16x8 input this returns the encoder's 8x16 coefficient layout.
  """
  block = np.asarray(block, dtype=np.float32)
  if block.shape not in ((8, 8), (16, 8), (8, 16)):
    raise ValueError("expected an 8x8, 16x8, or 8x16 block")

  rows, cols = block.shape
  if rows < cols:
    tmp = _dct_columns(block)
    coeff = _dct_columns(tmp.T.copy()).T.copy()
  else:
    # The missing final transpose is intentional: libjxl-tiny stores 16x8
    # transform coefficients in this transposed canonical layout.
    tmp = _dct_columns(block)
    coeff = _dct_columns(tmp.T.copy())
  return coeff


def scaled_dct_8x8(block: np.ndarray) -> np.ndarray:
  """Compute the scaled 8x8 DCT used by `TransformFromPixels(DCT)`."""
  if np.asarray(block).shape != (BLOCK_DIM, BLOCK_DIM):
    raise ValueError("expected an 8x8 block")
  return scaled_dct(block)
