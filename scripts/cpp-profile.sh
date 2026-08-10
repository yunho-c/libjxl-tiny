#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <input image> [distance]" >&2
  exit 1
fi

input="$1"
distance="${2:-1.0}"
case "$input" in
  /*) ;;
  *) input="$PWD/$input" ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d-%H%M%S)"
output_prefix="$repo_root/build/profiles/cjxl-tiny-$timestamp"
output_jxl="$output_prefix.jxl"
output_trace="$output_prefix.trace"

mkdir -p "$(dirname "$output_prefix")"

case "$input" in
  *.pfm|*.PFM)
    input_pfm="$input"
    ;;
  *)
    input_pfm="$output_prefix.pfm"
    "$repo_root/tools/pfm_tools.py" convert "$input" "$input_pfm"
    ;;
esac

xcrun xctrace record \
  --template 'Time Profiler' \
  --output "$output_trace" \
  --target-stdout - \
  --launch -- \
  "$repo_root/build/encoder/cjxl_tiny" \
  "$input_pfm" \
  "$output_jxl" \
  -d "$distance"

printf 'Wrote encoded image %s\n' "$output_jxl"
printf 'Wrote profile %s\n' "$output_trace"
printf 'Open it with: open %q\n' "$output_trace"
