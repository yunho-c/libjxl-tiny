#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python padding and XYB helpers against jxl_tiny_trace artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
  sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
  sys.path.insert(0, str(REPO_ROOT / "tools"))

import pfm_tools
from jxl_tiny import copy_and_pad_image, read_pfm, to_xyb


def resolve_executable(path: str) -> str:
  try:
    return pfm_tools.resolve_executable(path)
  except ValueError as exc:
    raise RuntimeError(str(exc)) from exc


def artifact_with_suffix(manifest: dict[str, object],
                         suffix: str) -> dict[str, object]:
  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, list):
    raise RuntimeError("manifest has no artifact list")
  matches = [
      artifact for artifact in artifacts
      if str(artifact.get("name", "")).endswith("_" + suffix)
      or str(artifact.get("name", "")) == suffix
  ]
  if len(matches) != 1:
    names = [str(artifact.get("name", "")) for artifact in matches]
    raise RuntimeError(
        f"expected one {suffix} artifact, found {len(matches)}: {names}")
  artifact = matches[0]
  if not isinstance(artifact, dict):
    raise RuntimeError(f"unexpected artifact entry for {suffix}: {artifact}")
  return artifact


def load_artifact(trace_dir: Path, manifest: dict[str, object],
                  suffix: str) -> np.ndarray:
  artifact = artifact_with_suffix(manifest, suffix)
  path = trace_dir / str(artifact["path"])
  array = np.load(path)
  expected_shape = tuple(int(dim) for dim in artifact["shape"])
  if array.shape != expected_shape:
    raise RuntimeError(
        f"{suffix}: expected shape {expected_shape}, got {array.shape}")
  return array


def assert_close(name: str, expected: np.ndarray, actual: np.ndarray,
                 atol: float, rtol: float) -> None:
  try:
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
  except AssertionError as exc:
    diff = np.abs(actual - expected)
    raise RuntimeError(
        f"{name}: max_abs={float(diff.max())} mean_abs={float(diff.mean())}"
    ) from exc


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

  manifest = json.loads((trace_dir / "manifest.json").read_text())
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
  parser.add_argument("--trace", default="build-codex/encoder/jxl_tiny_trace")
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
