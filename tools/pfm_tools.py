#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Create and compare RGB PFM files for cjxl_tiny.

The synthetic generator has no third-party dependencies. Conversion from common
image formats requires Pillow and assumes the input RGB values are sRGB.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Iterable


Pixel = tuple[float, float, float]


def srgb_to_linear(v: int) -> float:
  v = v / 255.0
  if v <= 0.04045:
    return v / 12.92
  return ((v + 0.055) / 1.055) ** 2.4


def parse_hex_color(value: str) -> tuple[int, int, int]:
  if value.startswith("#"):
    value = value[1:]
  if len(value) != 6:
    raise ValueError("expected a color like #ffffff")
  try:
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
  except ValueError as exc:
    raise ValueError("expected a color like #ffffff") from exc


def pattern_pixel(pattern: str, x: int, y: int, width: int, height: int) -> Pixel:
  fx = x / max(width - 1, 1)
  fy = y / max(height - 1, 1)

  if pattern == "gradient":
    blue = 0.5 + 0.5 * math.sin(x * 0.15) * math.cos(y * 0.12)
    return fx, fy, blue

  if pattern == "checker":
    block = 16
    v = 0.15 if ((x // block) + (y // block)) % 2 else 0.85
    return v, 0.25 + 0.5 * fx, 0.25 + 0.5 * fy

  if pattern == "rings":
    dx = fx - 0.5
    dy = fy - 0.5
    radius = math.sqrt(dx * dx + dy * dy)
    wave = 0.5 + 0.5 * math.cos(radius * 60.0)
    return fx, wave, fy

  raise ValueError(f"unknown pattern: {pattern}")


def generate_pixels(pattern: str, width: int, height: int) -> Iterable[Pixel]:
  for y in range(height):
    for x in range(width):
      yield pattern_pixel(pattern, x, y, width, height)


def write_pfm(path: Path, width: int, height: int, pixels: Iterable[Pixel]) -> None:
  rows: list[list[Pixel]] = []
  iterator = iter(pixels)
  for _ in range(height):
    row = []
    for _ in range(width):
      row.append(next(iterator))
    rows.append(row)

  with path.open("wb") as f:
    f.write(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))
    for row in reversed(rows):
      for r, g, b in row:
        f.write(struct.pack("<fff", r, g, b))


def read_token(data: bytes, pos: int) -> tuple[bytes, int]:
  while pos < len(data) and data[pos] in b" \t\r\n":
    pos += 1
  if pos >= len(data):
    raise ValueError("unexpected end of PFM header")
  start = pos
  while pos < len(data) and data[pos] not in b" \t\r\n":
    pos += 1
  return data[start:pos], pos


def read_pfm(path: Path) -> tuple[int, int, list[Pixel]]:
  data = path.read_bytes()
  token, pos = read_token(data, 0)
  if token != b"PF":
    raise ValueError(f"{path}: expected RGB PFM header 'PF'")

  width_token, pos = read_token(data, pos)
  height_token, pos = read_token(data, pos)
  scale_token, pos = read_token(data, pos)
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

  endian = ">" if scale > 0 else "<"
  values = struct.unpack(f"{endian}{width * height * 3}f", payload)
  pixels_bottom_up = [
      (values[i], values[i + 1], values[i + 2])
      for i in range(0, len(values), 3)
  ]

  pixels: list[Pixel] = []
  for y in range(height):
    row_start = (height - 1 - y) * width
    pixels.extend(pixels_bottom_up[row_start:row_start + width])
  return width, height, pixels


def metrics(reference: list[Pixel], candidate: list[Pixel]) -> dict[str, float]:
  dot = 0.0
  norm_ref = 0.0
  norm_candidate = 0.0
  sum_sq = 0.0
  sum_abs = 0.0
  max_abs = 0.0
  count = 0

  for ref_pixel, candidate_pixel in zip(reference, candidate, strict=True):
    for ref, got in zip(ref_pixel, candidate_pixel, strict=True):
      dot += ref * got
      norm_ref += ref * ref
      norm_candidate += got * got
      diff = got - ref
      abs_diff = abs(diff)
      sum_sq += diff * diff
      sum_abs += abs_diff
      max_abs = max(max_abs, abs_diff)
      count += 1

  if norm_ref == 0.0 or norm_candidate == 0.0:
    cosine = 1.0 if norm_ref == norm_candidate else 0.0
  else:
    cosine = dot / math.sqrt(norm_ref * norm_candidate)
  return {
      "cosine": cosine,
      "rmse": math.sqrt(sum_sq / count),
      "mae": sum_abs / count,
      "max_abs": max_abs,
  }


def compare_pfm(reference_path: Path, candidate_path: Path) -> dict[str, float]:
  ref_width, ref_height, reference = read_pfm(reference_path)
  got_width, got_height, candidate = read_pfm(candidate_path)
  if (ref_width, ref_height) != (got_width, got_height):
    raise ValueError(
        "image dimensions differ: "
        f"{ref_width}x{ref_height} vs {got_width}x{got_height}")
  return metrics(reference, candidate)


def require_thresholds(
    values: dict[str, float],
    min_cosine: float,
    max_rmse: float,
    max_mae: float,
    max_abs: float,
) -> None:
  failures = []
  if values["cosine"] < min_cosine:
    failures.append(f"cosine {values['cosine']:.9f} < {min_cosine:.9f}")
  if values["rmse"] > max_rmse:
    failures.append(f"rmse {values['rmse']:.9f} > {max_rmse:.9f}")
  if values["mae"] > max_mae:
    failures.append(f"mae {values['mae']:.9f} > {max_mae:.9f}")
  if values["max_abs"] > max_abs:
    failures.append(f"max_abs {values['max_abs']:.9f} > {max_abs:.9f}")
  if failures:
    raise ValueError("; ".join(failures))


def print_metrics(values: dict[str, float]) -> None:
  print(f"cosine={values['cosine']:.9f}")
  print(f"rmse={values['rmse']:.9f}")
  print(f"mae={values['mae']:.9f}")
  print(f"max_abs={values['max_abs']:.9f}")


def cmd_generate(args: argparse.Namespace) -> int:
  write_pfm(
      args.output,
      args.width,
      args.height,
      generate_pixels(args.pattern, args.width, args.height),
  )
  return 0


def cmd_convert(args: argparse.Namespace) -> int:
  try:
    from PIL import Image
  except ImportError:
    print("Pillow is required for image conversion: python3 -m pip install Pillow",
          file=sys.stderr)
    return 1

  image = Image.open(args.input)
  if image.mode in ("RGBA", "LA") or (
      image.mode == "P" and "transparency" in image.info):
    try:
      background_color = parse_hex_color(args.background)
    except ValueError as exc:
      print(f"invalid --background: {exc}", file=sys.stderr)
      return 1
    background = Image.new("RGBA", image.size, background_color)
    image = Image.alpha_composite(background, image.convert("RGBA"))

  image = image.convert("RGB")
  width, height = image.size

  def converted_pixels() -> Iterable[Pixel]:
    for r, g, b in image.getdata():
      yield srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)

  write_pfm(args.output, width, height, converted_pixels())
  return 0


def cmd_compare(args: argparse.Namespace) -> int:
  try:
    values = compare_pfm(args.reference, args.candidate)
    print_metrics(values)
    require_thresholds(
        values,
        args.min_cosine,
        args.max_rmse,
        args.max_mae,
        args.max_abs,
    )
  except ValueError as exc:
    print(f"compare failed: {exc}", file=sys.stderr)
    return 1
  return 0


def resolve_executable(path: str) -> str:
  if os.sep in path:
    return path
  resolved = shutil.which(path)
  if resolved is None:
    raise ValueError(f"could not find executable on PATH: {path}")
  return resolved


def default_cjxl_path() -> str:
  for path in ("build/encoder/cjxl_tiny", "build/encoder/cjxl_tiny"):
    if Path(path).exists():
      return path
  return "cjxl_tiny"


def cmd_roundtrip(args: argparse.Namespace) -> int:
  try:
    cjxl = resolve_executable(args.cjxl)
    djxl = resolve_executable(args.djxl)
  except ValueError as exc:
    print(f"roundtrip failed: {exc}", file=sys.stderr)
    return 1

  work_dir = args.work_dir
  work_dir.mkdir(parents=True, exist_ok=True)

  source = work_dir / "source.pfm"
  encoded = work_dir / "encoded.jxl"
  decoded = work_dir / "decoded.pfm"

  try:
    write_pfm(
        source,
        args.width,
        args.height,
        generate_pixels(args.pattern, args.width, args.height),
    )
    subprocess.run(
        [cjxl, str(source), str(encoded), "-d", str(args.distance)],
        check=True,
    )
    subprocess.run([djxl, str(encoded), str(decoded)], check=True)

    values = compare_pfm(source, decoded)
    print_metrics(values)
    print(f"source={source}")
    print(f"encoded={encoded}")
    print(f"decoded={decoded}")
    require_thresholds(
        values,
        args.min_cosine,
        args.max_rmse,
        args.max_mae,
        args.max_abs,
    )
  except (OSError, subprocess.CalledProcessError, ValueError) as exc:
    print(f"roundtrip failed: {exc}", file=sys.stderr)
    return 1
  return 0


def add_metric_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--min-cosine", type=float, default=0.995)
  parser.add_argument("--max-rmse", type=float, default=0.08)
  parser.add_argument("--max-mae", type=float, default=0.04)
  parser.add_argument("--max-abs", type=float, default=0.60)


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  generate = subparsers.add_parser(
      "generate", help="write a deterministic synthetic linear-sRGB PFM")
  generate.add_argument("output", type=Path)
  generate.add_argument("--width", type=int, default=256)
  generate.add_argument("--height", type=int, default=192)
  generate.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  generate.set_defaults(func=cmd_generate)

  convert = subparsers.add_parser(
      "convert",
      help="convert an sRGB PNG/JPEG/etc. to a linear-sRGB PFM; requires Pillow",
  )
  convert.add_argument("input", type=Path)
  convert.add_argument("output", type=Path)
  convert.add_argument(
      "--background",
      default="#ffffff",
      help="background for alpha compositing, as #rrggbb",
  )
  convert.set_defaults(func=cmd_convert)

  compare = subparsers.add_parser("compare", help="compare two RGB PFM files")
  compare.add_argument("reference", type=Path)
  compare.add_argument("candidate", type=Path)
  add_metric_args(compare)
  compare.set_defaults(func=cmd_compare)

  roundtrip = subparsers.add_parser(
      "roundtrip",
      help="generate a PFM, encode with cjxl_tiny, decode with djxl, compare",
  )
  roundtrip.add_argument("--cjxl", default=default_cjxl_path())
  roundtrip.add_argument("--djxl", default="djxl")
  roundtrip.add_argument("--distance", type=float, default=1.0)
  roundtrip.add_argument("--width", type=int, default=256)
  roundtrip.add_argument("--height", type=int, default=192)
  roundtrip.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  roundtrip.add_argument(
      "--work-dir",
      type=Path,
      default=Path("/tmp/libjxl-tiny-roundtrip"),
      help="directory for source.pfm, encoded.jxl, and decoded.pfm",
  )
  add_metric_args(roundtrip)
  roundtrip.set_defaults(func=cmd_roundtrip)

  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
