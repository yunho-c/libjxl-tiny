"""Image layout helpers that mirror libjxl-tiny encoder behavior."""

from __future__ import annotations

import numpy as np


def copy_and_pad_image(image: np.ndarray, block_dim: int = 8) -> np.ndarray:
  """Copy a channel-first RGB image and pad it to whole blocks.

  This mirrors `CopyAndPadImage` in `encoder/enc_frame.cc`: the last source
  column is repeated across the padded right edge, then the last padded row is
  repeated across the padded bottom edge.
  """
  if image.ndim != 3 or image.shape[0] != 3:
    raise ValueError("expected channel-first RGB image with shape (3, y, x)")
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")

  _, ysize, xsize = image.shape
  xsize_padded = ((xsize + block_dim - 1) // block_dim) * block_dim
  ysize_padded = ((ysize + block_dim - 1) // block_dim) * block_dim

  out = np.empty((3, ysize_padded, xsize_padded), dtype=np.float32)
  source = np.asarray(image, dtype=np.float32)
  out[:, :ysize, :xsize] = source

  if xsize < xsize_padded:
    out[:, :ysize, xsize:] = source[:, :, xsize - 1:xsize]
  if ysize < ysize_padded:
    out[:, ysize:, :] = out[:, ysize - 1:ysize, :]
  return out
