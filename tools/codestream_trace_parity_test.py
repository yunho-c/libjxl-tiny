#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python final codestream assembly against jxl_tiny_trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from trace_test_utils import (
    artifact_with_suffix,
    load_artifact,
    load_manifest,
    resolve_executable,
)

from trace_fixture_matrix import fixtures_from_args, prepare_work_dir, run_trace
from jxl_tiny import codestream_bytes


def load_bytes_artifact(trace_dir: Path, manifest: dict[str, object],
                        suffix: str) -> bytes:
  artifact = artifact_with_suffix(manifest, suffix)
  path = trace_dir / str(artifact["path"])
  data = path.read_bytes()
  expected_shape = tuple(int(dim) for dim in artifact["shape"])
  if expected_shape != (len(data),):
    raise RuntimeError(
        f"{suffix}: expected shape {expected_shape}, got ({len(data)},)")
  return data


def assert_bytes_equal(name: str, expected: bytes, actual: bytes) -> None:
  if expected == actual:
    return
  limit = min(len(expected), len(actual))
  first_diff = next(
      (i for i in range(limit) if expected[i] != actual[i]), limit)
  expected_byte = expected[first_diff] if first_diff < len(expected) else None
  actual_byte = actual[first_diff] if first_diff < len(actual) else None
  raise RuntimeError(
      f"{name}: byte mismatch at {first_diff}, "
      f"expected={expected_byte}, actual={actual_byte}, "
      f"expected_len={len(expected)}, actual_len={len(actual)}")


def compare_fixture(trace: str, work_dir: Path, fixture) -> None:
  _, trace_dir = run_trace(trace, fixture, work_dir)
  manifest = load_manifest(trace_dir)
  image = manifest.get("image")
  if not isinstance(image, dict):
    raise RuntimeError("manifest has no image metadata")

  actual = codestream_bytes(
      int(image["xsize"]),
      int(image["ysize"]),
      load_artifact(trace_dir, manifest, "dc_tokens"),
      load_artifact(trace_dir, manifest, "ac_metadata_tokens"),
      load_artifact(trace_dir, manifest, "ac_tokens"),
      load_artifact(trace_dir, manifest, "ac_strategy"),
      float(manifest["distance"]),
  )
  assert_bytes_equal("codestream",
                     load_bytes_artifact(trace_dir, manifest, "codestream"),
                     actual)

  print(f"codestream trace parity fixture passed: {fixture.name}")


def run_parity(args: argparse.Namespace, work_dir: Path) -> None:
  trace = resolve_executable(args.trace)
  prepare_work_dir(work_dir)
  fixtures = fixtures_from_args(args)
  for fixture in fixtures:
    compare_fixture(trace, work_dir / fixture.name, fixture)

  print(f"codestream trace parity test passed: {work_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", default="build-codex/encoder/jxl_tiny_trace")
  parser.add_argument("--work-dir", type=Path)
  parser.add_argument("--width", type=int, default=17)
  parser.add_argument("--height", type=int, default=9)
  parser.add_argument("--distance", type=float, default=1.0)
  parser.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  parser.add_argument("--fixture-matrix", action="store_true")
  args = parser.parse_args(argv)

  try:
    if args.work_dir is not None:
      run_parity(args, args.work_dir)
    else:
      with tempfile.TemporaryDirectory(
          prefix="libjxl-tiny-codestream-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"codestream trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
