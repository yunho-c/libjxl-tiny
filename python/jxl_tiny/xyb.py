"""Convert linear RGB pixels into the XYB color space used by VarDCT.

XYB is the perceptual color transform used by the lossy encoder path. The
constants mirror `encoder/enc_xyb.cc`; the NumPy implementation intentionally
uses array operations while trace tests allow a small tolerance for the cube
root approximation difference.
"""

from __future__ import annotations

import numpy as np


K_M02 = np.float32(0.078)
K_M00 = np.float32(0.30)
K_M01 = np.float32(1.0) - K_M02 - K_M00
K_M12 = np.float32(0.078)
K_M10 = np.float32(0.23)
K_M11 = np.float32(1.0) - K_M12 - K_M10
K_M20 = np.float32(0.24342268924547819)
K_M21 = np.float32(0.20476744424496821)
K_M22 = np.float32(1.0) - K_M20 - K_M21
K_OPSIN_ABSORBANCE_BIAS = np.float32(0.0037930732552754493)
K_NEG_BIAS_CBRT = np.float32(-0.15595420054)


def _cube_root_and_add(x: np.ndarray, add: np.float32) -> np.ndarray:
  return np.cbrt(x, dtype=np.float32) + add


def to_xyb(image: np.ndarray) -> np.ndarray:
  """Convert a channel-first linear RGB image to XYB.

  The C++ encoder uses a fast approximate cube root. This educational version
  uses NumPy's correctly rounded cube root and is therefore compared with a
  small tolerance in trace-backed tests.
  """
  if image.ndim != 3 or image.shape[0] != 3:
    raise ValueError("expected channel-first RGB image with shape (3, y, x)")

  source = np.asarray(image, dtype=np.float32)
  r = source[0]
  g = source[1]
  b = source[2]

  mixed0 = K_M00 * r + K_M01 * g + K_M02 * b + K_OPSIN_ABSORBANCE_BIAS
  mixed1 = K_M10 * r + K_M11 * g + K_M12 * b + K_OPSIN_ABSORBANCE_BIAS
  mixed2 = K_M20 * r + K_M21 * g + K_M22 * b + K_OPSIN_ABSORBANCE_BIAS

  tm0 = _cube_root_and_add(np.maximum(mixed0, np.float32(0.0)), K_NEG_BIAS_CBRT)
  tm1 = _cube_root_and_add(np.maximum(mixed1, np.float32(0.0)), K_NEG_BIAS_CBRT)
  tm2 = _cube_root_and_add(np.maximum(mixed2, np.float32(0.0)), K_NEG_BIAS_CBRT)

  out = np.empty_like(source)
  out[0] = np.float32(0.5) * (tm0 - tm1)
  out[1] = np.float32(0.5) * (tm0 + tm1)
  out[2] = tm2
  return out
