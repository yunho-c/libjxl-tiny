#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python adaptive quantization helpers against jxl_tiny_trace."""

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
from jxl_tiny import (
    adjust_quant_field,
    compute_adaptive_quantization,
    copy_and_pad_image,
    read_pfm,
    to_xyb,
)


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

  xyb = to_xyb(copy_and_pad_image(read_pfm(source)))
  result = compute_adaptive_quantization(xyb, args.distance)

  assert_close(
      "aq_map",
      load_artifact(trace_dir, manifest, "aq_map"),
      result.aq_map,
      atol=args.aq_atol,
      rtol=args.aq_rtol,
  )
  assert_close(
      "mask",
      load_artifact(trace_dir, manifest, "mask"),
      result.mask,
      atol=args.mask_atol,
      rtol=args.mask_rtol,
  )
  assert_close(
      "raw_quant_field_pre_adjust",
      load_artifact(trace_dir, manifest, "raw_quant_field_pre_adjust"),
      result.raw_quant_field,
      atol=0,
      rtol=0,
  )
  adjusted_raw_quant_field = adjust_quant_field(
      result.raw_quant_field,
      load_artifact(trace_dir, manifest, "ac_strategy"),
  )
  assert_close(
      "raw_quant_field_post_adjust",
      load_artifact(trace_dir, manifest, "raw_quant_field_post_adjust"),
      adjusted_raw_quant_field,
      atol=0,
      rtol=0,
  )
  assert_close(
      "raw_quant_field",
      load_artifact(trace_dir, manifest, "raw_quant_field"),
      adjusted_raw_quant_field,
      atol=0,
      rtol=0,
  )
  ytox = load_artifact(trace_dir, manifest, "ytox")
  ytob = load_artifact(trace_dir, manifest, "ytob")
  assert_close(
      "ytox_map",
      load_artifact(trace_dir, manifest, "ytox_map"),
      ytox.reshape((1, 1)),
      atol=0,
      rtol=0,
  )
  assert_close(
      "ytob_map",
      load_artifact(trace_dir, manifest, "ytob_map"),
      ytob.reshape((1, 1)),
      atol=0,
      rtol=0,
  )

  print(f"aq trace parity test passed: {trace_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", default="build-codex/encoder/jxl_tiny_trace")
  parser.add_argument("--work-dir", type=Path)
  parser.add_argument("--width", type=int, default=17)
  parser.add_argument("--height", type=int, default=9)
  parser.add_argument("--distance", type=float, default=1.0)
  parser.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  parser.add_argument("--aq-atol", type=float, default=5e-4)
  parser.add_argument("--aq-rtol", type=float, default=5e-4)
  parser.add_argument("--mask-atol", type=float, default=5e-4)
  parser.add_argument("--mask-rtol", type=float, default=5e-4)
  args = parser.parse_args(argv)

  try:
    if args.work_dir is not None:
      run_parity(args, args.work_dir)
    else:
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-aq-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"aq trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
