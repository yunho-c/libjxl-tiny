#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Compare educational Python encoder output with cjxl_tiny."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(PYTHON_ROOT) not in sys.path:
  sys.path.insert(0, str(PYTHON_ROOT))
if str(TOOLS_ROOT) not in sys.path:
  sys.path.insert(0, str(TOOLS_ROOT))

import pfm_tools
from jxl_tiny import encode_from_image, read_pfm


def default_cjxl_path() -> str:
  return pfm_tools.default_cjxl_path()


def copy_or_convert_input(input_path: Path, source_pfm: Path,
                          background: str) -> None:
  if input_path.suffix.lower() == ".pfm":
    shutil.copyfile(input_path, source_pfm)
    return

  try:
    from PIL import Image, ImageOps
  except ImportError as exc:
    raise RuntimeError(
        "Pillow is required for PNG/JPEG input: python3 -m pip install Pillow"
    ) from exc

  image = ImageOps.exif_transpose(Image.open(input_path))
  if image.mode in ("RGBA", "LA") or (
      image.mode == "P" and "transparency" in image.info):
    background_color = pfm_tools.parse_hex_color(background)
    base = Image.new("RGBA", image.size, background_color)
    image = Image.alpha_composite(base, image.convert("RGBA"))

  image = image.convert("RGB")
  width, height = image.size

  def converted_pixels():
    for r, g, b in image.getdata():
      yield pfm_tools.srgb_to_linear(r), pfm_tools.srgb_to_linear(
          g), pfm_tools.srgb_to_linear(b)

  pfm_tools.write_pfm(source_pfm, width, height, converted_pixels())


def encode_cjxl(cjxl: str, source_pfm: Path, output: Path,
                distance: float) -> None:
  subprocess.run([cjxl, str(source_pfm), str(output), "-d", str(distance)],
                 check=True)


def encode_python(source_pfm: Path, output: Path, distance: float) -> None:
  output.write_bytes(encode_from_image(read_pfm(source_pfm), distance))


def print_size(label: str, path: Path) -> None:
  print(f"{label}={path} bytes={path.stat().st_size}")


def decode_with_djxl(djxl: str, encoded: Path, decoded: Path) -> None:
  subprocess.run([djxl, str(encoded), str(decoded)], check=True)


def compare_decoded_outputs(source_pfm: Path, cjxl_decoded: Path,
                            py_decoded: Path) -> None:
  print()
  print("Decoded cjxl_tiny vs source:")
  pfm_tools.print_metrics(pfm_tools.compare_pfm(source_pfm, cjxl_decoded))
  print()
  print("Decoded Python vs source:")
  pfm_tools.print_metrics(pfm_tools.compare_pfm(source_pfm, py_decoded))
  print()
  print("Decoded Python vs decoded cjxl_tiny:")
  pfm_tools.print_metrics(pfm_tools.compare_pfm(cjxl_decoded, py_decoded))


def run_compare(args: argparse.Namespace) -> None:
  work_dir = args.work_dir
  work_dir.mkdir(parents=True, exist_ok=True)

  source_pfm = work_dir / "source.pfm"
  cjxl_output = work_dir / "cjxl_tiny.jxl"
  py_output = work_dir / "python.jxl"
  cjxl_decoded = work_dir / "cjxl_tiny_decoded.pfm"
  py_decoded = work_dir / "python_decoded.pfm"

  copy_or_convert_input(args.input, source_pfm, args.background)
  cjxl = pfm_tools.resolve_executable(args.cjxl)
  encode_cjxl(cjxl, source_pfm, cjxl_output, args.distance)
  encode_python(source_pfm, py_output, args.distance)

  print(f"source={source_pfm}")
  print_size("cjxl_tiny", cjxl_output)
  print_size("python", py_output)

  try:
    djxl = pfm_tools.resolve_executable(args.djxl)
  except ValueError as exc:
    if args.require_djxl:
      raise RuntimeError(str(exc)) from exc
    print()
    print(f"decode comparison skipped: {exc}")
    return

  decode_with_djxl(djxl, cjxl_output, cjxl_decoded)
  decode_with_djxl(djxl, py_output, py_decoded)
  compare_decoded_outputs(source_pfm, cjxl_decoded, py_decoded)


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("--work-dir", type=Path,
                      default=Path("build-codex/compare-py-cjxl"))
  parser.add_argument("-d", "--distance", type=float, default=1.0)
  parser.add_argument("--cjxl", default=default_cjxl_path())
  parser.add_argument("--djxl", default="djxl")
  parser.add_argument("--require-djxl", action="store_true")
  parser.add_argument(
      "--background",
      default="#ffffff",
      help="background color for transparent PNG inputs, as #rrggbb",
  )
  args = parser.parse_args(argv)

  try:
    run_compare(args)
  except (OSError, RuntimeError, ValueError,
          subprocess.CalledProcessError) as exc:
    print(f"compare_py_cjxl failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
