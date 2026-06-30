# Educational Python Encoder Port

This document explains the Python/NumPy reproduction under `python/jxl_tiny`.
The port is intended for learning and trace-backed validation against
`libjxl-tiny`; it is not a separate attempt to design a better encoder.

## Data Model

The Python encoder operates on channel-first linear RGB arrays:

```text
(3, ysize, xsize), dtype=float32
```

PFM inputs already use this linear RGB contract. PNG and JPEG inputs accepted by
`tools/py_encode.py` are converted from 8-bit sRGB into the same linear RGB
domain before encoding.

Intermediate image-like arrays are also channel-first. Block and tile maps use
row-major `(y, x)` order.

## Pipeline Overview

`encode_from_image()` in `python/jxl_tiny/encoder.py` orchestrates the same
lossy VarDCT path as `cjxl_tiny`:

1. Validate the linear RGB input and clamp the effective distance.
2. Split the image into DC groups and AC groups.
3. Process each AC group as 256x64 stripes, mirroring the C++ encoder's
   low-memory working set.
4. For each stripe, pad RGB to whole blocks and convert to XYB.
5. For each 64x64 tile, compute adaptive quantization, chroma-from-luma, and
   AC strategy decisions.
6. Quantize selected DCT blocks, filling quantized DC planes and AC token
   streams.
7. Tokenize DC/control data and AC coefficients.
8. Optimize entropy tables, serialize sections, and write the JPEG XL
   codestream bytes.

The stripe layout is an encoder implementation choice in `libjxl-tiny`, not a
fundamental compression requirement. It reduces the transient XYB buffer from a
full 256x256 AC group to one 256x64 row of tiles. The choice is still observable
because stripe-local padding, tile halos, and row-context token prediction affect
the produced tokens.

## Module Map

The modules mirror the C++ encoder stages:

| Python module | C++ reference | Purpose |
| --- | --- | --- |
| `image.py` | `enc_frame.cc` | Block-aligned image copying and edge padding. |
| `xyb.py` | `enc_xyb.cc` | Linear RGB to XYB conversion. |
| `adaptive_quantization.py` | `enc_adaptive_quantization.cc` | Per-tile AQ maps, masking, and raw quant fields. |
| `chroma_from_luma.py` | `enc_chroma_from_luma.cc` | Tile-local Y-to-X and Y-to-B multipliers. |
| `ac_strategy.py` | `enc_ac_strategy.cc` | Transform-size scoring and quant-field adjustment. |
| `transforms.py` | `enc_transforms-inl.h` | Scaled DCT helpers for 8x8, 16x8, and 8x16 blocks. |
| `quantization.py` | `enc_group.cc` | AC/DC coefficient quantization and nonzero maps. |
| `tokenization.py` | `enc_frame.cc`, `enc_group.cc` | Logical token streams before entropy coding. |
| `entropy.py` | `enc_entropy_code.cc` | Context clustering and prefix-code construction. |
| `bitstream.py` | `enc_frame.cc`, `enc_bit_writer.cc` | Section and codestream serialization. |

## Trace Parity

The C++ tracing executable emits `.npy` arrays and byte streams for intermediate
stages. Python tests compare each stage before relying on final codestream
equality.

Use exact equality for:

* integer maps,
* quantized coefficients,
* token streams,
* entropy-coded sections,
* final codestream bytes.

Use explicit tolerances for floating-point stages such as XYB, AQ maps, masks,
and AC strategy candidate costs. Some real images can expose tiny AC-strategy
tie differences because C++ and NumPy do not always reduce floating-point
expressions in the same order. Those are distinct from large functional failures
such as incorrect input linearization.

## Reading Path

For a first pass, read modules in pipeline order:

1. `encoder.py`
2. `image.py` and `xyb.py`
3. `adaptive_quantization.py`, `chroma_from_luma.py`, and `ac_strategy.py`
4. `transforms.py` and `quantization.py`
5. `tokenization.py`
6. `entropy.py` and `bitstream.py`

For debugging, start with the trace parity test that owns the first mismatching
stage instead of starting from the final `.jxl` bytes.
