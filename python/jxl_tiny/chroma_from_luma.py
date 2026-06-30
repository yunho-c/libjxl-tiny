"""Estimate tile-local chroma-from-luma multipliers.

The encoder predicts some X and B channel AC energy from the Y channel before
quantizing chroma. This module mirrors `encoder/enc_chroma_from_luma.cc` and
returns the byte-sized Y-to-X and Y-to-B multipliers serialized in AC metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transforms import BLOCK_DIM, scaled_dct_8x8


K_INV_COLOR_FACTOR = np.float32(1.0 / 84.0)
K_DISTANCE_MULTIPLIER_AC = np.float32(1e-3)

_DCT_X_WEIGHTS = np.asarray((
    0.00031746033, 0.00031746057, 0.00031854658, 0.00037755401,
    0.00044749113, 0.00053038419, 0.00062863121, 0.00074507861,
    0.00031746057, 0.00031746062, 0.00033158599, 0.00038811122,
    0.00045695182, 0.00053938502, 0.00063753547, 0.00075413194,
    0.00031854658, 0.00033158599, 0.00036670428, 0.0004184779,
    0.00048487642, 0.00056626293, 0.00066427846, 0.00078140449,
    0.00037755401, 0.00038811122, 0.0004184779, 0.00046632939,
    0.00053038419, 0.00061082945, 0.00070903177, 0.00082727504,
    0.00044749113, 0.00045695182, 0.00048487642, 0.00053038419,
    0.00059302151, 0.00067320757, 0.00077229418, 0.00094286882,
    0.00053038419, 0.00053938502, 0.00056626293, 0.00061082945,
    0.00067320757, 0.00075413194, 0.00085507357, 0.0012723245,
    0.00062863121, 0.00063753547, 0.00066427846, 0.00070903177,
    0.00077229418, 0.00085507357, 0.0011923184, 0.001791994,
    0.00074507861, 0.00075413194, 0.00078140449, 0.00082727504,
    0.00094286882, 0.0012723245, 0.001791994, 0.0026133191,
), dtype=np.float32)

_DCT_B_WEIGHTS = np.asarray((
    0.0019531252, 0.0034018266, 0.0059007513, 0.0083743408,
    0.011718751, 0.011718759, 0.011968765, 0.016986061,
    0.0034018266, 0.0042808522, 0.0064091417, 0.0088638803,
    0.011718752, 0.011718759, 0.012320629, 0.017413978,
    0.0059007513, 0.0064091417, 0.0078861341, 0.010351914,
    0.011718754, 0.011718762, 0.013408982, 0.018736197,
    0.0083743408, 0.0088638803, 0.010351914, 0.011718752,
    0.011718759, 0.011718766, 0.015336527, 0.021072537,
    0.011718751, 0.011718752, 0.011718754, 0.011718759,
    0.011718764, 0.013782934, 0.018288977, 0.025368163,
    0.011718759, 0.011718759, 0.011718762, 0.011718766,
    0.013782934, 0.017413978, 0.022557227, 0.034232263,
    0.011968765, 0.012320629, 0.013408982, 0.015336527,
    0.018288977, 0.022557227, 0.032079678, 0.048214123,
    0.016986061, 0.017413978, 0.018736197, 0.021072537,
    0.025368163, 0.034232263, 0.048214123, 0.07031212,
), dtype=np.float32)


@dataclass(frozen=True)
class ChromaFromLumaResult:
  ytox: np.int8
  ytob: np.int8


def _inv_matrix(weights: np.ndarray) -> np.ndarray:
  matrix = (np.float32(1.0) / weights).astype(np.float32)
  matrix[0] = np.float32(0.0)
  return matrix


INV_MATRIX_X = _inv_matrix(_DCT_X_WEIGHTS)
INV_MATRIX_B = _inv_matrix(_DCT_B_WEIGHTS)


def _roundf(value: np.float32) -> int:
  value_f = float(value)
  if value_f >= 0.0:
    return math.floor(value_f + 0.5)
  return math.ceil(value_f - 0.5)


def find_best_multiplier(values_m: np.ndarray, values_s: np.ndarray,
                         base: float) -> np.int8:
  if values_m.size == 0:
    return np.int8(0)
  values_m = np.asarray(values_m, dtype=np.float32)
  values_s = np.asarray(values_s, dtype=np.float32)
  # Regularized least-squares fit for predicting one chroma coefficient stream
  # from luma. The result is quantized to the byte-sized CFL multiplier range.
  a = values_m * K_INV_COLOR_FACTOR
  b = np.float32(base) * values_m - values_s
  ca = np.sum(a * a, dtype=np.float32)
  cb = np.sum(a * b, dtype=np.float32)
  denom = (ca + np.float32(values_m.size) * K_DISTANCE_MULTIPLIER_AC *
           np.float32(0.5))
  multiplier = np.float32(-cb / denom)
  return np.int8(max(-128, min(127, _roundf(multiplier))))


def compute_chroma_from_luma(xyb: np.ndarray) -> ChromaFromLumaResult:
  """Compute Y-to-X and Y-to-B multipliers for one 64x64 tile."""
  if xyb.ndim != 3 or xyb.shape[0] != 3:
    raise ValueError("expected channel-first XYB image with shape (3, y, x)")
  if xyb.shape[1] % BLOCK_DIM != 0 or xyb.shape[2] % BLOCK_DIM != 0:
    raise ValueError("xyb dimensions must be multiples of 8")

  source = np.asarray(xyb, dtype=np.float32)
  coeffs_yx: list[np.ndarray] = []
  coeffs_x: list[np.ndarray] = []
  coeffs_yb: list[np.ndarray] = []
  coeffs_b: list[np.ndarray] = []
  for by in range(source.shape[1] // BLOCK_DIM):
    for bx in range(source.shape[2] // BLOCK_DIM):
      y0 = by * BLOCK_DIM
      x0 = bx * BLOCK_DIM
      block_y = scaled_dct_8x8(source[1, y0:y0 + BLOCK_DIM, x0:x0 + BLOCK_DIM])
      block_x = scaled_dct_8x8(source[0, y0:y0 + BLOCK_DIM, x0:x0 + BLOCK_DIM])
      block_b = scaled_dct_8x8(source[2, y0:y0 + BLOCK_DIM, x0:x0 + BLOCK_DIM])
      block_y = block_y.reshape(-1).copy()
      block_x = block_x.reshape(-1).copy()
      block_b = block_b.reshape(-1).copy()
      # CFL models AC correlation only; average color stays in DC.
      block_y[0] = np.float32(0.0)
      block_x[0] = np.float32(0.0)
      block_b[0] = np.float32(0.0)
      coeffs_yx.append(block_y * INV_MATRIX_X)
      coeffs_x.append(block_x * INV_MATRIX_X)
      coeffs_yb.append(block_y * INV_MATRIX_B)
      coeffs_b.append(block_b * INV_MATRIX_B)

  values_yx = np.concatenate(coeffs_yx).astype(np.float32)
  values_x = np.concatenate(coeffs_x).astype(np.float32)
  values_yb = np.concatenate(coeffs_yb).astype(np.float32)
  values_b = np.concatenate(coeffs_b).astype(np.float32)
  return ChromaFromLumaResult(
      ytox=find_best_multiplier(values_yx, values_x, base=0.0),
      ytob=find_best_multiplier(values_yb, values_b, base=1.0),
  )
