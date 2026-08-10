# PFM tools

`cjxl_tiny` expects RGB PFM input in linear sRGB. The helper script
`tools/pfm_tools.py` can generate deterministic test inputs, convert common
image formats to compatible PFM files, compare PFM files, and run a simple
encode/decode similarity check.

## Generate a test PFM

Create a deterministic input without third-party dependencies:

```bash
tools/pfm_tools.py generate /tmp/libjxl-tiny-test.pfm
build/encoder/cjxl_tiny /tmp/libjxl-tiny-test.pfm /tmp/libjxl-tiny-test.jxl
```

## Convert an image to PFM

To convert an sRGB PNG/JPEG/etc. to a linear-sRGB PFM, install Pillow and run:

```bash
python3 -m pip install Pillow
tools/pfm_tools.py convert input.png /tmp/input.pfm
```

## Convert and encode in one step

The `Justfile` wraps conversion and encoding:

```bash
just encode input.png
just encode input.png output.jxl 0.8
```

## Encode with the Python port

The educational Python encoder can write `.jxl` files directly from PFM,
PNG, JPEG, and other Pillow-readable images:

```bash
tools/py_encode.py input.png output.jxl -d 1.0
just py-encode input.jpg output.jxl 0.8
```

For PFM input, pixels are read as linear RGB. For PNG/JPEG input, Pillow reads
the image as sRGB and the helper converts it to linear RGB before calling
`jxl_tiny.encode_from_image`. Transparent PNGs are composited on white by
default; pass `--background '#000000'` to choose another background.

## Compare the Python and C++ encoders

Use `tools/compare_py_cjxl.py` to encode the same PFM/PNG/JPEG input with
`cjxl_tiny` and the educational Python port:

```bash
tools/compare_py_cjxl.py input.png --work-dir build/compare-py-cjxl -d 1.0
just py-compare input.jpg build/compare-py-cjxl 0.8
```

The helper normalizes the input to one shared PFM, writes both `.jxl` files,
and prints their byte sizes. If `djxl` is available, it also decodes both
outputs and reports cosine similarity, RMSE, MAE, and maximum absolute channel
error against the source and against each other. Add `--require-djxl` when a
missing decoder should make the command fail instead of skipping those metrics.

## Round-trip check

To run a simple encode/decode similarity check, install a `djxl` decoder
(for example from a full JPEG XL package) and run:

```bash
tools/pfm_tools.py roundtrip --cjxl build/encoder/cjxl_tiny --djxl djxl
```

The round-trip check reports cosine similarity, RMSE, MAE, and maximum absolute
channel error.
