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

## Round-trip check

To run a simple encode/decode similarity check, install a `djxl` decoder
(for example from a full JPEG XL package) and run:

```bash
tools/pfm_tools.py roundtrip --cjxl build/encoder/cjxl_tiny --djxl djxl
```

The round-trip check reports cosine similarity, RMSE, MAE, and maximum absolute
channel error.
