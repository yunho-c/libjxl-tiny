set positional-arguments

default:
    @just --list

# Convert an image to linear-sRGB PFM and encode it with cjxl_tiny.
encode input output="" distance="1.0":
    #!/usr/bin/env bash
    set -euo pipefail

    input="$1"
    output="$2"
    distance="$3"

    if [[ -z "$output" ]]; then
      if [[ "$input" == *.* ]]; then
        output="${input%.*}.jxl"
      else
        output="${input}.jxl"
      fi
    fi

    cjxl="${CJXL_TINY:-}"
    if [[ -z "$cjxl" ]]; then
      if [[ -x build-codex/encoder/cjxl_tiny ]]; then
        cjxl="build-codex/encoder/cjxl_tiny"
      elif [[ -x build/encoder/cjxl_tiny ]]; then
        cjxl="build/encoder/cjxl_tiny"
      elif command -v cjxl_tiny >/dev/null 2>&1; then
        cjxl="cjxl_tiny"
      else
        echo "Could not find cjxl_tiny. Build it first or set CJXL_TINY=/path/to/cjxl_tiny." >&2
        exit 1
      fi
    fi

    pfm="$(mktemp "${TMPDIR:-/tmp}/libjxl-tiny.XXXXXX.pfm")"
    trap 'rm -f "$pfm"' EXIT

    tools/pfm_tools.py convert "$input" "$pfm"
    "$cjxl" "$pfm" "$output" -d "$distance"
    printf 'Wrote %s\n' "$output"

# Encode PFM/PNG/JPEG inputs with the educational Python port.
py-encode input output="" distance="1.0":
    #!/usr/bin/env bash
    set -euo pipefail

    input="$1"
    output="$2"
    distance="$3"

    if [[ -z "$output" ]]; then
      if [[ "$input" == *.* ]]; then
        output="${input%.*}.jxl"
      else
        output="${input}.jxl"
      fi
    fi

    tools/py_encode.py "$input" "$output" -d "$distance"

# Summarize a C++ trace directory produced by the parity helpers.
trace-summary trace_dir preview="16":
    @tools/trace_summary.py "$1" --preview "$2"

# Compare cjxl_tiny and educational Python encoder outputs from one input.
py-compare input work_dir="build-codex/compare-py-cjxl" distance="1.0":
    @tools/compare_py_cjxl.py "$1" --work-dir "$2" -d "$3"

# Run the Python-port syntax, parity, smoke, and optional decode checks.
py-check work_dir="build-codex/python-port-check":
    #!/usr/bin/env bash
    set -euo pipefail

    work_dir="$1"
    export PYTHONDONTWRITEBYTECODE=1

    resolve_tool() {
      local name="$1"
      if [[ -x "build-codex/encoder/${name}" ]]; then
        printf 'build-codex/encoder/%s\n' "$name"
      elif [[ -x "build/encoder/${name}" ]]; then
        printf 'build/encoder/%s\n' "$name"
      elif command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
      else
        return 1
      fi
    }

    trace="$(resolve_tool jxl_tiny_trace || true)"
    if [[ -z "$trace" ]]; then
      echo "Could not find jxl_tiny_trace. Build it first with tracing enabled." >&2
      echo "Expected build-codex/encoder/jxl_tiny_trace, build/encoder/jxl_tiny_trace, or a PATH entry." >&2
      exit 1
    fi

    echo "== py_compile =="
    python3 -m py_compile tools/*.py python/jxl_tiny/*.py

    echo "== encode trace parity =="
    python3 tools/encode_trace_parity_test.py \
      --trace "$trace" \
      --work-dir "${work_dir}/encode_trace_parity_test" \
      --fixture-matrix

    echo "== py_encode smoke =="
    if python3 -c 'import PIL' >/dev/null 2>&1; then
      python3 tools/py_encode_smoke_test.py \
        --work-dir "${work_dir}/py_encode_smoke_test"
    else
      echo "Skipping py_encode_smoke_test because Pillow is not installed."
    fi

    echo "== Python vs cjxl_tiny compare =="
    cjxl="$(resolve_tool cjxl_tiny || true)"
    if [[ -n "$cjxl" ]]; then
      source="${work_dir}/compare-source.pfm"
      mkdir -p "$(dirname "$source")"
      python3 tools/pfm_tools.py generate "$source" \
        --width 17 \
        --height 9 \
        --pattern gradient
      python3 tools/compare_py_cjxl.py "$source" \
        --cjxl "$cjxl" \
        --work-dir "${work_dir}/compare" \
        -d 1.0
    else
      echo "Skipping compare_py_cjxl because cjxl_tiny is not available."
    fi

# Profile the educational Python encoder CLI with pyinstrument.
py-profile input output="" distance="1.0" profile="py_encode_profile.html":
    #!/usr/bin/env bash
    set -euo pipefail

    input="$1"
    output="$2"
    distance="$3"
    profile="$4"

    if [[ -z "$output" ]]; then
      if [[ "$input" == *.* ]]; then
        output="${input%.*}.jxl"
      else
        output="${input}.jxl"
      fi
    fi

    if ! python3 -m pyinstrument --version >/dev/null 2>&1; then
      echo "pyinstrument is required: python3 -m pip install pyinstrument" >&2
      exit 1
    fi

    mkdir -p "$(dirname "$profile")"
    python3 -m pyinstrument -r html -o "$profile" tools/py_encode.py "$input" "$output" -d "$distance"
    printf 'Wrote profile %s\n' "$profile"
