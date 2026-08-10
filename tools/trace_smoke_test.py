#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Smoke-test jxl_tiny_trace output against cjxl_tiny output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import pfm_tools


REQUIRED_ARTIFACT_SUFFIXES = (
    "codestream",
    "input_padded",
    "xyb",
    "aq_map",
    "mask",
    "raw_quant_field",
    "ytox_map",
    "ytob_map",
    "ac_strategy",
    "quant_dc",
)


def resolve_executable(path: str) -> str:
  try:
    return pfm_tools.resolve_executable(path)
  except ValueError as exc:
    raise RuntimeError(str(exc)) from exc


def artifact_stage(name: str) -> str:
  for suffix in REQUIRED_ARTIFACT_SUFFIXES:
    if name == suffix or name.endswith("_" + suffix):
      return suffix
  return name


def validate_manifest(manifest_path: Path, width: int, height: int,
                      distance: float) -> list[dict[str, object]]:
  if not manifest_path.exists():
    raise RuntimeError(f"missing manifest: {manifest_path}")
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("schema_version") != 1:
    raise RuntimeError(f"unexpected schema_version: {manifest.get('schema_version')}")
  image = manifest.get("image")
  if image != {"xsize": width, "ysize": height}:
    raise RuntimeError(f"unexpected image metadata: {image}")
  if abs(float(manifest.get("distance")) - distance) > 1e-6:
    raise RuntimeError(f"unexpected distance: {manifest.get('distance')}")

  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, list) or not artifacts:
    raise RuntimeError("manifest has no artifacts")

  seen = {artifact_stage(str(artifact.get("name"))) for artifact in artifacts}
  missing = [name for name in REQUIRED_ARTIFACT_SUFFIXES if name not in seen]
  if missing:
    raise RuntimeError("missing trace artifacts: " + ", ".join(missing))

  return artifacts


def validate_artifacts(trace_dir: Path, artifacts: list[dict[str, object]]) -> None:
  for artifact in artifacts:
    name = str(artifact["name"])
    path = trace_dir / str(artifact["path"])
    if not path.exists():
      raise RuntimeError(f"{name}: missing artifact file {path}")

    expected_bytes = int(artifact["bytes"])
    if path.stat().st_size != expected_bytes and path.suffix == ".bin":
      raise RuntimeError(
          f"{name}: expected {expected_bytes} bytes, got {path.stat().st_size}")

    if path.suffix != ".npy":
      continue

    array = np.load(path)
    expected_shape = tuple(int(dim) for dim in artifact["shape"])
    if array.shape != expected_shape:
      raise RuntimeError(
          f"{name}: expected shape {expected_shape}, got {array.shape}")
    expected_dtype = str(artifact["dtype"])
    if array.dtype.name != expected_dtype:
      raise RuntimeError(
          f"{name}: expected dtype {expected_dtype}, got {array.dtype.name}")


def run_smoke(args: argparse.Namespace, work_dir: Path) -> None:
  cjxl = resolve_executable(args.cjxl)
  trace = resolve_executable(args.trace)
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True)

  source = work_dir / "source.pfm"
  encoded = work_dir / "encoded.jxl"
  trace_dir = work_dir / "trace"

  pfm_tools.write_pfm(
      source,
      args.width,
      args.height,
      pfm_tools.generate_pixels(args.pattern, args.width, args.height),
  )

  subprocess.run(
      [cjxl, str(source), str(encoded), "-d", str(args.distance)],
      check=True,
  )
  subprocess.run(
      [trace, str(source), str(trace_dir), "-d", str(args.distance)],
      check=True,
  )

  traced_codestream = trace_dir / "codestream.bin"
  if encoded.read_bytes() != traced_codestream.read_bytes():
    raise RuntimeError("codestream.bin does not match cjxl_tiny output")

  artifacts = validate_manifest(
      trace_dir / "manifest.json", args.width, args.height, args.distance)
  validate_artifacts(trace_dir, artifacts)

  print(f"trace smoke test passed: {trace_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cjxl", default=pfm_tools.default_cjxl_path())
  parser.add_argument("--trace", default="build/encoder/jxl_tiny_trace")
  parser.add_argument("--work-dir", type=Path)
  parser.add_argument("--width", type=int, default=17)
  parser.add_argument("--height", type=int, default=9)
  parser.add_argument("--distance", type=float, default=1.0)
  parser.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  args = parser.parse_args(argv)

  try:
    if args.work_dir is not None:
      run_smoke(args, args.work_dir)
    else:
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-trace-") as work_dir:
        run_smoke(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"trace smoke test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
