#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python section-byte serialization against jxl_tiny_trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from trace_test_utils import (
    artifact_with_suffix,
    load_artifact,
    load_manifest,
    resolve_executable,
)

import pfm_tools
from jxl_tiny import (
    ac_entropy_code,
    ac_global_section,
    ac_group_section,
    compute_distance_params,
    dc_entropy_code,
    dc_global_section,
    dc_group_section,
)


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
  dc_tokens = load_artifact(trace_dir, manifest, "dc_tokens")
  ac_metadata_tokens = load_artifact(trace_dir, manifest, "ac_metadata_tokens")
  ac_tokens = load_artifact(trace_dir, manifest, "ac_tokens")
  ac_strategy = load_artifact(trace_dir, manifest, "ac_strategy")

  dc_code = dc_entropy_code(dc_tokens, ac_metadata_tokens)
  ac_code = ac_entropy_code(ac_tokens)
  dist = compute_distance_params(args.distance)

  expected_sections = {
      "dc_global_section": dc_global_section(dist, 1, dc_code),
      "dc_group_section_0": dc_group_section(dc_tokens, ac_metadata_tokens,
                                             ac_strategy, dc_code),
      "ac_global_section": ac_global_section(1, ac_code),
      "ac_group_section_0": ac_group_section(ac_tokens, ac_code),
  }

  for name, actual in expected_sections.items():
    assert_bytes_equal(name, load_bytes_artifact(trace_dir, manifest, name),
                       actual)

  print(f"section trace parity test passed: {trace_dir}")


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
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-section-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"section trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
