#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Small deterministic fixture sets for trace parity tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import pfm_tools


@dataclass(frozen=True)
class TraceFixture:
  name: str
  width: int
  height: int
  distance: float
  pattern: str


DEFAULT_FIXTURES = (
    TraceFixture("gradient_17x9_d1", 17, 9, 1.0, "gradient"),
    TraceFixture("checker_17x9_d1", 17, 9, 1.0, "checker"),
    TraceFixture("rings_17x9_d1", 17, 9, 1.0, "rings"),
    TraceFixture("gradient_65x33_d0_5", 65, 33, 0.5, "gradient"),
    TraceFixture("checker_16x16_d2", 16, 16, 2.0, "checker"),
    TraceFixture("rings_32x16_d1", 32, 16, 1.0, "rings"),
    TraceFixture("gradient_129x129_d1", 129, 129, 1.0, "gradient"),
    TraceFixture("gradient_257x17_d1", 257, 17, 1.0, "gradient"),
    TraceFixture("gradient_2049x17_d1", 2049, 17, 1.0, "gradient"),
)


def fixture_from_args(args) -> TraceFixture:
  return TraceFixture("single", args.width, args.height, args.distance,
                      args.pattern)


def fixtures_from_args(args) -> tuple[TraceFixture, ...]:
  if getattr(args, "fixture_matrix", False):
    return DEFAULT_FIXTURES
  return (fixture_from_args(args),)


def prepare_work_dir(work_dir: Path) -> None:
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True)


def run_trace(trace: str, fixture: TraceFixture,
              work_dir: Path) -> tuple[Path, Path]:
  work_dir.mkdir(parents=True, exist_ok=True)
  source = work_dir / "source.pfm"
  trace_dir = work_dir / "trace"
  pfm_tools.write_pfm(
      source,
      fixture.width,
      fixture.height,
      pfm_tools.generate_pixels(fixture.pattern, fixture.width,
                                fixture.height),
  )
  subprocess.run(
      [trace, str(source), str(trace_dir), "-d", str(fixture.distance)],
      check=True,
  )
  return source, trace_dir
