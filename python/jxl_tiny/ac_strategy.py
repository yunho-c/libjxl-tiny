"""AC strategy helpers for libjxl-tiny trace parity tests."""

from __future__ import annotations

import numpy as np


DCT = 0
DCT16X8 = 1
DCT8X16 = 2


def raw_strategy(encoded: np.ndarray | np.uint8 | int) -> np.ndarray:
  return np.asarray(encoded, dtype=np.uint8) >> np.uint8(1)


def is_first_block(encoded: np.ndarray | np.uint8 | int) -> np.ndarray:
  return (np.asarray(encoded, dtype=np.uint8) & np.uint8(1)) != 0


def covered_blocks_x(strategy: int) -> int:
  lut = (1, 1, 2)
  return lut[int(strategy)]


def covered_blocks_y(strategy: int) -> int:
  lut = (1, 2, 1)
  return lut[int(strategy)]


def adjust_quant_field(raw_quant_field: np.ndarray,
                       ac_strategy: np.ndarray) -> np.ndarray:
  """Apply `AdjustQuantField` using a traced AC strategy image.

  `ac_strategy` uses the trace encoding from `TraceAcStrategy`:
  `(raw_strategy << 1) | is_first_block`.
  """
  quant = np.array(raw_quant_field, dtype=np.uint8, copy=True)
  strategy = np.asarray(ac_strategy, dtype=np.uint8)
  if quant.shape != strategy.shape:
    raise ValueError("raw_quant_field and ac_strategy must have the same shape")

  ysize, xsize = quant.shape
  for y in range(ysize):
    for x in range(xsize):
      encoded = strategy[y, x]
      if not bool(is_first_block(encoded)):
        continue
      raw = int(raw_strategy(encoded))
      if raw not in (DCT, DCT16X8, DCT8X16):
        raise ValueError(f"invalid AC strategy at ({x}, {y}): {raw}")
      blocks_x = covered_blocks_x(raw)
      blocks_y = covered_blocks_y(raw)
      if x + blocks_x > xsize or y + blocks_y > ysize:
        raise ValueError(f"AC strategy at ({x}, {y}) exceeds quant field")
      block = quant[y:y + blocks_y, x:x + blocks_x]
      block[:, :] = np.max(block)
  return quant
