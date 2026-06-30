#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Summarize jxl_tiny_trace artifacts for educational inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


STAGE_ORDER = (
    ("Input and XYB", "input_xyb"),
    ("AQ, CFL, and AC Strategy", "aq_cfl_strategy"),
    ("Quantization Blocks", "quantization"),
    ("Tokens", "tokens"),
    ("Entropy Tables", "entropy"),
    ("Sections", "sections"),
    ("Codestream", "codestream"),
    ("Other", "other"),
)


def stage_for_name(name: str) -> str:
  if name == "codestream":
    return "codestream"
  if name.endswith("_section") or "_section_" in name:
    return "sections"
  if name.startswith(("dc_entropy_", "ac_entropy_")):
    return "entropy"
  if "_tokens" in name or name.endswith("_ac_tokens"):
    return "tokens"
  if any(part in name for part in (
      "raw_coefficients",
      "quant_input_coefficients",
      "quant_dc",
      "quantized_ac",
      "block_quant_dc",
      "num_nonzeros",
  )):
    return "quantization"
  if any(part in name for part in (
      "aq_map",
      "mask",
      "raw_quant_field",
      "ytox",
      "ytob",
      "ac_strategy",
  )):
    return "aq_cfl_strategy"
  if name.endswith("input_padded") or name.endswith("xyb"):
    return "input_xyb"
  return "other"


def load_manifest(trace_dir: Path) -> dict[str, object]:
  manifest_path = trace_dir / "manifest.json"
  if not manifest_path.exists():
    raise RuntimeError(f"missing manifest: {manifest_path}")
  return json.loads(manifest_path.read_text())


def expected_shape(artifact: dict[str, object]) -> tuple[int, ...] | None:
  shape = artifact.get("shape")
  if not isinstance(shape, list):
    return None
  return tuple(int(dim) for dim in shape)


def validate_array(name: str, artifact: dict[str, object],
                   array: np.ndarray) -> None:
  shape = expected_shape(artifact)
  if shape is not None and array.shape != shape:
    raise RuntimeError(f"{name}: expected shape {shape}, got {array.shape}")
  dtype = artifact.get("dtype")
  if isinstance(dtype, str) and str(array.dtype) != dtype:
    raise RuntimeError(f"{name}: expected dtype {dtype}, got {array.dtype}")


def preview_array(array: np.ndarray, preview: int) -> str:
  if preview <= 0:
    return ""
  flat = array.reshape(-1)[:preview]
  return np.array2string(flat, precision=6, separator=", ")


def summarize_artifact(trace_dir: Path, artifact: dict[str, object],
                       preview: int) -> str:
  name = str(artifact.get("name", ""))
  rel_path = str(artifact.get("path", ""))
  if not name or not rel_path:
    raise RuntimeError(f"artifact missing name/path: {artifact}")
  path = trace_dir / rel_path
  if not path.exists():
    raise RuntimeError(f"{name}: missing artifact file {path}")

  tolerance = artifact.get("tolerance", "")
  shape = artifact.get("shape", "")
  dtype = artifact.get("dtype", "")
  manifest_bytes = artifact.get("bytes", "")
  suffix = path.suffix.lower()
  if suffix == ".npy":
    array = np.load(path)
    validate_array(name, artifact, array)
    parts = [
        f"{name}",
        f"path={rel_path}",
        f"shape={tuple(array.shape)}",
        f"dtype={array.dtype}",
    ]
    if tolerance:
      parts.append(f"tolerance={tolerance}")
    if manifest_bytes != "":
      parts.append(f"bytes={manifest_bytes}")
    if preview > 0:
      parts.append(f"preview={preview_array(array, preview)}")
    return "  - " + " ".join(parts)

  size = path.stat().st_size
  parts = [f"{name}", f"path={rel_path}", f"bytes={size}"]
  if shape:
    parts.append(f"shape={shape}")
  if dtype:
    parts.append(f"dtype={dtype}")
  if tolerance:
    parts.append(f"tolerance={tolerance}")
  if manifest_bytes != "" and int(manifest_bytes) != size:
    raise RuntimeError(f"{name}: expected {manifest_bytes} bytes, got {size}")
  return "  - " + " ".join(parts)


def print_summary(trace_dir: Path, preview: int) -> None:
  manifest = load_manifest(trace_dir)
  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, list):
    raise RuntimeError("manifest has no artifact list")

  print(f"trace={trace_dir}")
  print(f"schema_version={manifest.get('schema_version', 'unknown')}")
  print(f"distance={manifest.get('distance', 'unknown')}")
  image = manifest.get("image", {})
  if isinstance(image, dict):
    print(f"image={image.get('xsize', '?')}x{image.get('ysize', '?')}")
  build = manifest.get("build", {})
  if isinstance(build, dict) and build:
    build_text = ", ".join(f"{key}={value}" for key, value in sorted(build.items()))
    print(f"build={build_text}")
  print(f"artifacts={len(artifacts)}")

  grouped: dict[str, list[dict[str, object]]] = {
      key: [] for _, key in STAGE_ORDER
  }
  for artifact in artifacts:
    if not isinstance(artifact, dict):
      raise RuntimeError(f"unexpected artifact entry: {artifact}")
    grouped[stage_for_name(str(artifact.get("name", "")))].append(artifact)

  for title, key in STAGE_ORDER:
    entries = grouped[key]
    if not entries:
      continue
    print()
    print(title)
    for artifact in entries:
      print(summarize_artifact(trace_dir, artifact, preview))


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trace_dir", type=Path)
  parser.add_argument("--preview", type=int, default=16)
  args = parser.parse_args(argv)

  try:
    print_summary(args.trace_dir, args.preview)
  except (OSError, RuntimeError, ValueError) as exc:
    print(f"trace_summary failed: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
