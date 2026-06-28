"""Small NumPy PFM reader for libjxl-tiny reproduction tests."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def _read_token(data: bytes, pos: int) -> tuple[bytes, int]:
  while pos < len(data) and data[pos] in b" \t\r\n":
    pos += 1
  if pos >= len(data):
    raise ValueError("unexpected end of PFM header")
  start = pos
  while pos < len(data) and data[pos] not in b" \t\r\n":
    pos += 1
  return data[start:pos], pos


def read_pfm(path: str | Path) -> np.ndarray:
  """Read an RGB PFM file as channel-first float32 data.

  libjxl-tiny consumes linear RGB PFM input. PFM stores rows bottom-up, so the
  returned array is flipped back to top-down order with shape `(3, y, x)`.
  """
  data = Path(path).read_bytes()
  token, pos = _read_token(data, 0)
  if token != b"PF":
    raise ValueError(f"{path}: expected RGB PFM header 'PF'")

  width_token, pos = _read_token(data, pos)
  height_token, pos = _read_token(data, pos)
  scale_token, pos = _read_token(data, pos)
  width = int(width_token)
  height = int(height_token)
  scale = float(scale_token)
  if scale not in (-1.0, 1.0):
    raise ValueError(f"{path}: expected PFM scale 1.0 or -1.0")
  if pos >= len(data) or data[pos] not in b" \t\r\n":
    raise ValueError(f"{path}: expected whitespace after PFM scale")
  pos += 1

  expected_size = width * height * 3 * 4
  payload = data[pos:]
  if len(payload) != expected_size:
    raise ValueError(
        f"{path}: expected {expected_size} pixel bytes, got {len(payload)}")

  endian = "<" if scale < 0 else ">"
  values = struct.unpack(f"{endian}{width * height * 3}f", payload)
  rows_bottom_up = np.asarray(values, dtype=np.float32).reshape(height, width, 3)
  rows_top_down = rows_bottom_up[::-1, :, :]
  return np.ascontiguousarray(np.transpose(rows_top_down, (2, 0, 1)))
