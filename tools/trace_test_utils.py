#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Shared helpers for Python tests that consume jxl_tiny_trace artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(PYTHON_ROOT) not in sys.path:
  sys.path.insert(0, str(PYTHON_ROOT))
if str(TOOLS_ROOT) not in sys.path:
  sys.path.insert(0, str(TOOLS_ROOT))

import pfm_tools


def resolve_executable(path: str) -> str:
  try:
    return pfm_tools.resolve_executable(path)
  except ValueError as exc:
    raise RuntimeError(str(exc)) from exc


def load_manifest(trace_dir: Path) -> dict[str, object]:
  manifest_path = trace_dir / "manifest.json"
  if not manifest_path.exists():
    raise RuntimeError(f"missing manifest: {manifest_path}")
  return json.loads(manifest_path.read_text())


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


def artifact_with_exact_name(manifest: dict[str, object],
                             name: str) -> dict[str, object]:
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
  return matches[0]


def artifacts_matching_numbered(manifest: dict[str, object],
                                pattern: str) -> list[dict[str, object]]:
  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, list):
    raise RuntimeError("manifest has no artifact list")
  regex = re.compile(pattern)
  matches: list[tuple[int, dict[str, object]]] = []
  for artifact in artifacts:
    if not isinstance(artifact, dict):
      continue
    match = regex.fullmatch(str(artifact.get("name", "")))
    if match is None:
      continue
    if "index" in match.groupdict():
      index = int(match.group("index"))
    else:
      index = int(match.group(1))
    matches.append((index, artifact))
  if not matches:
    raise RuntimeError(f"no artifacts matched {pattern}")
  return [artifact for _, artifact in sorted(matches)]


def load_named_artifact(trace_dir: Path,
                        artifact: dict[str, object]) -> np.ndarray:
  path = trace_dir / str(artifact["path"])
  array = np.load(path)
  expected_shape = tuple(int(dim) for dim in artifact["shape"])
  if array.shape != expected_shape:
    name = str(artifact.get("name", artifact["path"]))
    raise RuntimeError(
        f"{name}: expected shape {expected_shape}, got {array.shape}")
  return array


def load_exact_artifact(trace_dir: Path, manifest: dict[str, object],
                        name: str) -> np.ndarray:
  return load_named_artifact(trace_dir, artifact_with_exact_name(manifest, name))


def load_artifact(trace_dir: Path, manifest: dict[str, object],
                  suffix: str) -> np.ndarray:
  artifact = artifact_with_suffix(manifest, suffix)
  return load_named_artifact(trace_dir, artifact)


def load_exact_bytes_artifact(trace_dir: Path, manifest: dict[str, object],
                              name: str) -> bytes:
  artifact = artifact_with_exact_name(manifest, name)
  path = trace_dir / str(artifact["path"])
  data = path.read_bytes()
  expected_shape = tuple(int(dim) for dim in artifact["shape"])
  if expected_shape != (len(data),):
    raise RuntimeError(
        f"{name}: expected shape {expected_shape}, got ({len(data)},)")
  return data


def assert_close(name: str, expected: np.ndarray, actual: np.ndarray,
                 atol: float, rtol: float) -> None:
  try:
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
  except AssertionError as exc:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    raise RuntimeError(
        f"{name}: max_abs={float(diff.max())} mean_abs={float(diff.mean())}"
    ) from exc
