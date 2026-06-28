#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python tokenization helpers against jxl_tiny_trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from trace_test_utils import (
    assert_close,
    load_artifact,
    load_manifest,
    resolve_executable,
)

import pfm_tools
from jxl_tiny import ac_metadata_tokens, ac_tokens, dc_tokens


def load_exact_artifact(trace_dir: Path, manifest: dict[str, object],
                        name: str) -> np.ndarray:
  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, list):
    raise RuntimeError("manifest has no artifact list")
  matches = [
      artifact for artifact in artifacts
      if isinstance(artifact, dict) and artifact.get("name") == name
  ]
  if len(matches) != 1:
    raise RuntimeError(
        f"expected one exact {name} artifact, found {len(matches)}")
  artifact = matches[0]
  path = trace_dir / str(artifact["path"])
  array = np.load(path)
  expected_shape = tuple(int(dim) for dim in artifact["shape"])
  if array.shape != expected_shape:
    raise RuntimeError(
        f"{name}: expected shape {expected_shape}, got {array.shape}")
  return array


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
  xyb = load_artifact(trace_dir, manifest, "xyb")
  raw_quant_field = load_artifact(trace_dir, manifest, "raw_quant_field")
  ac_strategy = load_artifact(trace_dir, manifest, "ac_strategy")
  ytox_map = load_artifact(trace_dir, manifest, "ytox_map")
  ytob_map = load_artifact(trace_dir, manifest, "ytob_map")
  quant_dc = load_exact_artifact(trace_dir, manifest, "dcg_0_quant_dc")

  assert_close(
      "dc_tokens",
      load_artifact(trace_dir, manifest, "dc_tokens"),
      dc_tokens(quant_dc),
      atol=0,
      rtol=0,
  )
  assert_close(
      "ac_metadata_tokens",
      load_artifact(trace_dir, manifest, "ac_metadata_tokens"),
      ac_metadata_tokens(ytox_map, ytob_map, ac_strategy, raw_quant_field),
      atol=0,
      rtol=0,
  )
  assert_close(
      "ac_tokens",
      load_artifact(trace_dir, manifest, "ac_tokens"),
      ac_tokens(xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map,
                args.distance),
      atol=0,
      rtol=0,
  )

  print(f"token trace parity test passed: {trace_dir}")


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
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-token-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"token trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
