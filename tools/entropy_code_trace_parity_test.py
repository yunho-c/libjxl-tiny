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
import subprocess
import sys
import tempfile

import numpy as np

from trace_test_utils import (
    assert_close,
    artifacts_matching_numbered,
    load_artifact,
    load_ac_token_groups,
    load_named_artifact,
    load_manifest,
    resolve_executable,
)

from trace_fixture_matrix import fixtures_from_args, prepare_work_dir, run_trace
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


def compare_fixture(trace: str, work_dir: Path, fixture) -> None:
  _, trace_dir = run_trace(trace, fixture, work_dir)
  manifest = load_manifest(trace_dir)
  dc_token_groups = [
      load_named_artifact(trace_dir, artifact)
      for artifact in artifacts_matching_numbered(
          manifest, r"dcg_(?P<index>\d+)_tokens_dc_tokens")
  ]
  ac_metadata_token_groups = [
      load_named_artifact(trace_dir, artifact)
      for artifact in artifacts_matching_numbered(
          manifest, r"dcg_(?P<index>\d+)_tokens_ac_metadata_tokens")
  ]
  dc = dc_entropy_code(
      np.concatenate(dc_token_groups, axis=0),
      np.concatenate(ac_metadata_token_groups, axis=0),
  )
  ac_token_groups = load_ac_token_groups(trace_dir, manifest)
  ac = ac_entropy_code(np.concatenate(ac_token_groups, axis=0))
  compare_entropy_code("dc_entropy", trace_dir, manifest, dc)
  compare_entropy_code("ac_entropy", trace_dir, manifest, ac)

  print(f"entropy code trace parity fixture passed: {fixture.name}")


def run_parity(args: argparse.Namespace, work_dir: Path) -> None:
  trace = resolve_executable(args.trace)
  prepare_work_dir(work_dir)
  fixtures = fixtures_from_args(args)
  for fixture in fixtures:
    compare_fixture(trace, work_dir / fixture.name, fixture)

  print(f"entropy code trace parity test passed: {work_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", default="build/encoder/jxl_tiny_trace")
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
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-entropy-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"entropy code trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
