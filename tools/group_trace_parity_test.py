#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check Python AC group quantization helpers against jxl_tiny_trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
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
from jxl_tiny import quantize_ac_group


BLOCK_ARTIFACT_RE = re.compile(
    r"_block_y_(?P<y>[0-9]+)_x_(?P<x>[0-9]+)_(?P<kind>.+)$")


def load_named_artifact(trace_dir: Path, artifact: dict[str, object],
                        expected_name: str) -> np.ndarray:
  path = trace_dir / str(artifact["path"])
  array = np.load(path)
  expected_shape = tuple(int(dim) for dim in artifact["shape"])
  if array.shape != expected_shape:
    raise RuntimeError(
        f"{expected_name}: expected shape {expected_shape}, got {array.shape}")
  return array


def collect_block_artifacts(
    manifest: dict[str, object],
) -> dict[tuple[int, int], dict[str, dict[str, object]]]:
  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, list):
    raise RuntimeError("manifest has no artifact list")
  blocks: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
  for artifact in artifacts:
    if not isinstance(artifact, dict):
      continue
    name = str(artifact.get("name", ""))
    match = BLOCK_ARTIFACT_RE.search(name)
    if match is None:
      continue
    key = (int(match.group("y")), int(match.group("x")))
    kind = match.group("kind")
    blocks.setdefault(key, {})[kind] = artifact
  if not blocks:
    raise RuntimeError("no AC group block trace artifacts found")
  return blocks


def compare_block(trace_dir: Path, block_key: tuple[int, int],
                  artifacts: dict[str, dict[str, object]], actual,
                  args: argparse.Namespace) -> None:
  required = {
      "raw_coefficients",
      "quant_input_coefficients",
      "quantized_ac",
      "block_quant_dc",
      "num_nonzeros",
      "num_nonzeros_map",
  }
  missing = sorted(required.difference(artifacts))
  if missing:
    raise RuntimeError(f"block {block_key}: missing artifacts {missing}")

  prefix = f"block_y_{block_key[0]}_x_{block_key[1]}"
  assert_close(
      f"{prefix}_raw_coefficients",
      load_named_artifact(trace_dir, artifacts["raw_coefficients"],
                          "raw_coefficients"),
      actual.raw_coefficients,
      atol=args.coeff_atol,
      rtol=args.coeff_rtol,
  )
  assert_close(
      f"{prefix}_quant_input_coefficients",
      load_named_artifact(trace_dir, artifacts["quant_input_coefficients"],
                          "quant_input_coefficients"),
      actual.quant_input_coefficients,
      atol=args.coeff_atol,
      rtol=args.coeff_rtol,
  )
  assert_close(
      f"{prefix}_quantized_ac",
      load_named_artifact(trace_dir, artifacts["quantized_ac"],
                          "quantized_ac"),
      actual.quantized_ac,
      atol=0,
      rtol=0,
  )
  assert_close(
      f"{prefix}_block_quant_dc",
      load_named_artifact(trace_dir, artifacts["block_quant_dc"],
                          "block_quant_dc"),
      actual.block_quant_dc,
      atol=0,
      rtol=0,
  )
  assert_close(
      f"{prefix}_num_nonzeros",
      load_named_artifact(trace_dir, artifacts["num_nonzeros"],
                          "num_nonzeros"),
      actual.num_nonzeros,
      atol=0,
      rtol=0,
  )
  assert_close(
      f"{prefix}_num_nonzeros_map",
      load_named_artifact(trace_dir, artifacts["num_nonzeros_map"],
                          "num_nonzeros_map"),
      actual.num_nonzeros_map,
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
  distance = float(manifest["distance"])
  blocks = collect_block_artifacts(manifest)
  xyb = load_artifact(trace_dir, manifest, "xyb")
  raw_quant_field = load_artifact(trace_dir, manifest, "raw_quant_field")
  ac_strategy = load_artifact(trace_dir, manifest, "ac_strategy")
  ytox_map = load_artifact(trace_dir, manifest, "ytox_map")
  ytob_map = load_artifact(trace_dir, manifest, "ytob_map")
  actual_blocks = quantize_ac_group(xyb, raw_quant_field, ac_strategy, ytox_map,
                                    ytob_map, distance)

  if sorted(blocks) != sorted(actual_blocks):
    raise RuntimeError(
        f"block key mismatch: trace={sorted(blocks)} python={sorted(actual_blocks)}"
    )
  for block_key in sorted(blocks):
    compare_block(trace_dir, block_key, blocks[block_key],
                  actual_blocks[block_key], args)

  print(f"group trace parity test passed: {trace_dir}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", default="build-codex/encoder/jxl_tiny_trace")
  parser.add_argument("--work-dir", type=Path)
  parser.add_argument("--width", type=int, default=17)
  parser.add_argument("--height", type=int, default=9)
  parser.add_argument("--distance", type=float, default=1.0)
  parser.add_argument(
      "--pattern", choices=("gradient", "checker", "rings"), default="gradient")
  parser.add_argument("--coeff-atol", type=float, default=1e-5)
  parser.add_argument("--coeff-rtol", type=float, default=1e-5)
  args = parser.parse_args(argv)

  try:
    if args.work_dir is not None:
      run_parity(args, args.work_dir)
    else:
      with tempfile.TemporaryDirectory(prefix="libjxl-tiny-group-") as work_dir:
        run_parity(args, Path(work_dir))
  except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"group trace parity test failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
