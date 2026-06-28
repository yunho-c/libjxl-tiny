#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python entropy-code optimization against jxl_tiny_trace."""

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
from jxl_tiny import ac_entropy_code, dc_entropy_code


def compare_entropy_code(name: str, trace_dir: Path, manifest: dict[str, object],
                         actual) -> None:
  assert_close(
      f"{name}_context_map",
      load_artifact(trace_dir, manifest, f"{name}_context_map"),
      actual.context_map,
      atol=0,
      rtol=0,
  )
  assert_close(
      f"{name}_prefix_depths",
      load_artifact(trace_dir, manifest, f"{name}_prefix_depths"),
      actual.prefix_depths,
      atol=0,
      rtol=0,
  )
  assert_close(
      f"{name}_prefix_bits",
      load_artifact(trace_dir, manifest, f"{name}_prefix_bits"),
      actual.prefix_bits,
      atol=0,
      rtol=0,
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
  dc = dc_entropy_code(
      load_artifact(trace_dir, manifest, "dc_tokens"),
      load_artifact(trace_dir, manifest, "ac_metadata_tokens"),
  )
  ac = ac_entropy_code(load_artifact(trace_dir, manifest, "ac_tokens"))
  compare_entropy_code("dc_entropy", trace_dir, manifest, dc)
  compare_entropy_code("ac_entropy", trace_dir, manifest, ac)

  print(f"entropy code trace parity test passed: {trace_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", default="build-codex/encoder/jxl_tiny_trace")
  parser.add_argument("--work-dir", type=Path)
  parser.add_argument("--width", type=int, default=17)
  parser.add_argument("--height", type=int, default=9)
  parser.add_argument("--distance", type=float, default=1.0)
  parser.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  args = parser.parse_args(argv)

  try:
    if args.work_dir is not None:
      run_parity(args, args.work_dir)
    else:
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-entropy-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"entropy code trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
