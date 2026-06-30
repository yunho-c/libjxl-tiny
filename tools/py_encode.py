#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Encode PFM/PNG/JPEG inputs with the educational Python JPEG XL encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(PYTHON_ROOT) not in sys.path:
  sys.path.insert(0, str(PYTHON_ROOT))
if str(TOOLS_ROOT) not in sys.path:
  sys.path.insert(0, str(TOOLS_ROOT))

import pfm_tools
from jxl_tiny import encode_from_image, read_pfm


def default_output_path(input_path: Path) -> Path:
  if input_path.suffix:
    return input_path.with_suffix(".jxl")
  return input_path.with_name(input_path.name + ".jxl")


def _linearize_srgb_uint8(rgb: np.ndarray) -> np.ndarray:
  lut = np.asarray([pfm_tools.srgb_to_linear(i) for i in range(256)],
                   dtype=np.float32)
  return lut[np.asarray(rgb, dtype=np.uint8)]


def load_raster_image(path: Path, background: str) -> np.ndarray:
  try:
    from PIL import Image, ImageOps
  except ImportError as exc:
    raise RuntimeError(
        "Pillow is required for PNG/JPEG input: python3 -m pip install Pillow"
    ) from exc

  image = ImageOps.exif_transpose(Image.open(path))
  if image.mode in ("RGBA", "LA") or (
      image.mode == "P" and "transparency" in image.info):
    background_color = pfm_tools.parse_hex_color(background)
    base = Image.new("RGBA", image.size, background_color)
    image = Image.alpha_composite(base, image.convert("RGBA"))

  image = image.convert("RGB")
  rgb = np.asarray(image, dtype=np.uint8)
  linear = _linearize_srgb_uint8(rgb)
  return np.ascontiguousarray(np.transpose(linear, (2, 0, 1)))


def load_input_image(path: Path, background: str) -> np.ndarray:
  if path.suffix.lower() == ".pfm":
    return read_pfm(path)
  return load_raster_image(path, background)


def encode_path(input_path: Path, output_path: Path, distance: float,
                background: str) -> int:
  image = load_input_image(input_path, background)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_bytes(encode_from_image(image, distance))
  print(f"Wrote {output_path}")
  return 0


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("output", nargs="?", type=Path)
  parser.add_argument("-d", "--distance", type=float, default=1.0)
  parser.add_argument(
      "--background",
      default="#ffffff",
      help="background color for transparent PNG inputs, as #rrggbb",
  )
  args = parser.parse_args(argv)

  output = args.output if args.output is not None else default_output_path(args.input)
  try:
    return encode_path(args.input, output, args.distance, args.background)
  except (OSError, RuntimeError, ValueError) as exc:
    print(f"py_encode failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
