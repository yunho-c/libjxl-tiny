#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Smoke-test the educational Python encoder CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

import pfm_tools


def require_nonempty(path: Path) -> None:
  if not path.exists():
    raise RuntimeError(f"missing output: {path}")
  if path.stat().st_size == 0:
    raise RuntimeError(f"empty output: {path}")


def require_equal(reference: Path, candidate: Path) -> None:
  expected = reference.read_bytes()
  actual = candidate.read_bytes()
  if expected == actual:
    return
  limit = min(len(expected), len(actual))
  first_diff = next((i for i in range(limit) if expected[i] != actual[i]), limit)
  raise RuntimeError(
      f"{candidate.name} differs from {reference.name} at byte {first_diff}: "
      f"expected_len={len(expected)} actual_len={len(actual)}")


def run_smoke(work_dir: Path) -> None:
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True)

  pfm = work_dir / "gradient.pfm"
  png_equivalent_pfm = work_dir / "gradient_from_png_pixels.pfm"
  png = work_dir / "gradient.png"
  pfm_out = work_dir / "gradient_from_pfm.jxl"
  png_equivalent_pfm_out = work_dir / "gradient_from_png_pixels.jxl"
  png_out = work_dir / "gradient_from_png.jxl"

  pfm_tools.write_pfm(pfm, 17, 9, pfm_tools.generate_pixels("gradient", 17, 9))
  pixels = [
      (x * 15, y * 28, (x * 9 + y * 13) % 256)
      for y in range(9)
      for x in range(17)
  ]
  image = Image.new("RGB", (17, 9))
  image.putdata(pixels)
  image.save(png)
  pfm_tools.write_pfm(
      png_equivalent_pfm,
      17,
      9,
      (
          tuple(pfm_tools.srgb_to_linear(channel) for channel in pixel)
          for pixel in pixels
      ),
  )

  script = Path(__file__).resolve().parent / "py_encode.py"
  subprocess.run(
      [sys.executable, str(script), str(pfm), str(pfm_out), "-d", "1.0"],
      check=True,
  )
  subprocess.run(
      [
          sys.executable,
          str(script),
          str(png_equivalent_pfm),
          str(png_equivalent_pfm_out),
          "-d",
          "1.0",
      ],
      check=True,
  )
  subprocess.run(
      [sys.executable, str(script), str(png), str(png_out), "-d", "1.0"],
      check=True,
  )
  require_nonempty(pfm_out)
  require_nonempty(png_equivalent_pfm_out)
  require_nonempty(png_out)
  require_equal(png_equivalent_pfm_out, png_out)
  print(f"py_encode smoke test passed: {work_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--work-dir", type=Path, required=True)
  args = parser.parse_args(argv)

  try:
    run_smoke(args.work_dir)
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"py_encode smoke test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
