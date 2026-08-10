#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Check the Python one-DC-group encoder against jxl_tiny_trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from trace_test_utils import (
    load_exact_bytes_artifact,
    load_manifest,
    resolve_executable,
)

from trace_fixture_matrix import TraceFixture, fixture_from_args, prepare_work_dir
from trace_fixture_matrix import run_trace
from jxl_tiny import encode_from_image, read_pfm


ONE_DC_GROUP_FIXTURES = (
    TraceFixture("gradient_17x9_d1", 17, 9, 1.0, "gradient"),
    TraceFixture("checker_17x9_d1", 17, 9, 1.0, "checker"),
    TraceFixture("rings_17x9_d1", 17, 9, 1.0, "rings"),
    TraceFixture("gradient_65x33_d0_5", 65, 33, 0.5, "gradient"),
    TraceFixture("checker_16x16_d2", 16, 16, 2.0, "checker"),
    TraceFixture("rings_32x16_d1", 32, 16, 1.0, "rings"),
    TraceFixture("gradient_129x129_d1", 129, 129, 1.0, "gradient"),
    TraceFixture("gradient_257x17_d1", 257, 17, 1.0, "gradient"),
    TraceFixture("gradient_2049x17_d1", 2049, 17, 1.0, "gradient"),
    TraceFixture("gradient_16x16_d0_01", 16, 16, 0.01, "gradient"),
)


def assert_bytes_equal(name: str, expected: bytes, actual: bytes) -> None:
    if expected == actual:
        return
    limit = min(len(expected), len(actual))
    first_diff = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    expected_byte = expected[first_diff] if first_diff < len(expected) else None
    actual_byte = actual[first_diff] if first_diff < len(actual) else None
    raise RuntimeError(
        f"{name}: byte mismatch at {first_diff}, "
        f"expected={expected_byte}, actual={actual_byte}, "
        f"expected_len={len(expected)}, actual_len={len(actual)}"
    )


def fixtures_from_args(args: argparse.Namespace) -> tuple[TraceFixture, ...]:
    if args.fixture_matrix:
        return ONE_DC_GROUP_FIXTURES
    return (fixture_from_args(args),)


def compare_fixture(trace: str, work_dir: Path, fixture: TraceFixture) -> None:
    source, trace_dir = run_trace(trace, fixture, work_dir)
    manifest = load_manifest(trace_dir)
    image = manifest.get("image")
    if not isinstance(image, dict):
        raise RuntimeError("manifest has no image metadata")
    if int(image["xsize"]) != fixture.width or int(image["ysize"]) != fixture.height:
        raise RuntimeError(f"unexpected image metadata: {image}")

    actual = encode_from_image(read_pfm(source), fixture.distance)
    expected = load_exact_bytes_artifact(trace_dir, manifest, "codestream")
    assert_bytes_equal("codestream", expected, actual)

    print(f"encode trace parity fixture passed: {fixture.name}")


def run_parity(args: argparse.Namespace, work_dir: Path) -> None:
    trace = resolve_executable(args.trace)
    prepare_work_dir(work_dir)
    for fixture in fixtures_from_args(args):
        compare_fixture(trace, work_dir / fixture.name, fixture)

    print(f"encode trace parity test passed: {work_dir}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="build/encoder/jxl_tiny_trace")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--width", type=int, default=17)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument(
        "--pattern", choices=("gradient", "checker", "rings"), default="gradient"
    )
    parser.add_argument("--fixture-matrix", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.work_dir is not None:
            run_parity(args, args.work_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="libjxl-tiny-encode-") as work_dir:
                run_parity(args, Path(work_dir))
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"encode trace parity test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
