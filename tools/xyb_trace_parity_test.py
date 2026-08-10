#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python padding and XYB helpers against jxl_tiny_trace artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from trace_test_utils import (
    assert_close,
    load_artifact,
    load_manifest,
    resolve_executable,
)

import pfm_tools
from jxl_tiny import copy_and_pad_image, read_pfm, to_xyb


def run_parity(args: argparse.Namespace, work_dir: Path) -> None:
  trace = resolve_executable(args.trace)
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True)

  source = work_dir / "source.pfm"
  trace_dir = work_dir / "trace"
  pfm_tools.write_pfm(
      source,
      args.width,
      args.height,
      pfm_tools.generate_pixels(args.pattern, args.width, args.height),
  )

  subprocess.run(
      [trace, str(source), str(trace_dir), "-d", str(args.distance)],
      check=True,
  )

  manifest = load_manifest(trace_dir)
  image = manifest.get("image")
  if image != {"xsize": args.width, "ysize": args.height}:
    raise RuntimeError(f"unexpected image metadata: {image}")

  rgb = read_pfm(source)
  padded = copy_and_pad_image(rgb)
  traced_padded = load_artifact(trace_dir, manifest, "input_padded")
  traced_xyb = load_artifact(trace_dir, manifest, "xyb")

  assert_close("input_padded", traced_padded, padded, atol=0.0, rtol=0.0)
  assert_close(
      "xyb", traced_xyb, to_xyb(padded), atol=args.xyb_atol,
      rtol=args.xyb_rtol)

  print(f"xyb trace parity test passed: {trace_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", default="build/encoder/jxl_tiny_trace")
  parser.add_argument("--work-dir", type=Path)
  parser.add_argument("--width", type=int, default=17)
  parser.add_argument("--height", type=int, default=9)
  parser.add_argument("--distance", type=float, default=1.0)
  parser.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  parser.add_argument("--xyb-atol", type=float, default=2e-5)
  parser.add_argument("--xyb-rtol", type=float, default=2e-5)
  args = parser.parse_args(argv)

  try:
    if args.work_dir is not None:
      run_parity(args, args.work_dir)
    else:
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-xyb-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"xyb trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
