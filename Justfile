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
