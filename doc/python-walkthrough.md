# Python Encoder Walkthrough

This walkthrough follows one tiny image through the educational Python encoder.
It is meant to make the JPEG XL VarDCT path concrete before reading the code in
`python/jxl_tiny`.

The example uses a 17x9 gradient at distance 1.0. That size is intentionally
awkward: it is small enough to inspect by hand, but it still exercises padding
because neither dimension is a multiple of 8.

## Generate A Trace

Run the parity helper to produce a C++ reference trace and compare it with the
Python encoder:

```bash
python3 tools/encode_trace_parity_test.py \
  --width 17 \
  --height 9 \
  --pattern gradient \
  --distance 1.0 \
  --work-dir build/learn/python-walkthrough
```

The trace files are written under:

```text
build/learn/python-walkthrough/single/trace
```

`build/` is disposable build output. Use it for local exploration, but do
not commit generated trace artifacts.

## Inspect The Artifacts

Use the trace summary helper for a first pass through the C++ reference
artifacts:

```bash
python3 tools/trace_summary.py \
  build/learn/python-walkthrough/single/trace \
  --preview 16
```

Or use the `just` recipe:

```bash
just trace-summary build/learn/python-walkthrough/single/trace 16
```

The goal is not to memorize every value. The useful exercise is to watch the
representation change from pixels, to coefficients, to tokens, to bytes. The
summary groups artifacts by stage, prints array shapes and dtypes, previews the
first few values in `.npy` files, and reports byte sizes for entropy sections
and the final codestream.

To run the broader Python-port validation suite after making changes, use:

```bash
just py-check
```

## Step By Step

1. `source.pfm` is a 17x9 linear RGB image. PFM inputs already satisfy the
   encoder's contract: channel-first float RGB after loading, in linear light.
2. `dcg_0_acg_0_stripe_0_input_padded.npy` is the stripe input after edge
   padding. The 17x9 image becomes 24x16 pixels so every dimension contains a
   whole number of 8x8 blocks.
3. `dcg_0_acg_0_stripe_0_xyb.npy` is the padded image after the XYB perceptual
   color transform. Later stages mostly work in this color space, not in RGB.
4. `dcg_0_acg_0_stripe_0_tile_0_aq_map.npy` estimates how much AC detail each
   8x8 block can lose. `dcg_0_acg_0_stripe_0_tile_0_raw_quant_field.npy` is the
   byte-valued version that drives AC quantization.
5. `dcg_0_acg_0_stripe_0_tile_0_ytob.npy` and
   `dcg_0_acg_0_stripe_0_tile_0_ytox.npy` are chroma-from-luma multipliers.
   They let the encoder predict some X and B channel AC energy from Y.
6. `dcg_0_ac_strategy.npy` records which transform owns each 8x8 cell. In this
   port, each entry uses `(raw_strategy << 1) | is_first_block`, so multi-block
   transforms mark only their top-left cell as the owner.
7. `dcg_0_quant_dc.npy` is the low-frequency image that will be tokenized with
   a gradient predictor. For larger transforms, DC covers the low-frequency
   coefficients distributed back over the covered 8x8 cells.
8. Block-local files such as
   `dcg_0_acg_0_stripe_0_ac_block_y_0_x_0_quantized_ac.npy` hold quantized AC
   coefficients for one transform owner. Matching `num_nonzeros` and
   `num_nonzeros_map` files explain the contexts used when those coefficients
   become AC tokens.
9. `dcg_0_tokens_dc_tokens.npy`,
   `dcg_0_tokens_ac_metadata_tokens.npy`, and
   `dcg_0_acg_0_stripe_0_ac_ac_tokens.npy` are logical `(context, value)` token
   streams. They are still integers, not compressed bits.
10. `dc_entropy_*` and `ac_entropy_*` files describe the optimized context maps
    and prefix-code tables. The section `.bin` files are the entropy-coded frame
    sections, and `codestream.bin` is the final bare JPEG XL codestream.

## Layout Mental Model

The encoder uses several overlapping grids:

| Unit | Size in this port | What it is for |
| --- | --- | --- |
| Pixel | 1 sample position | RGB/XYB image data. |
| Block | 8x8 pixels | The base DCT and quantization unit. |
| Tile | 64x64 pixels | AQ, CFL, and AC-strategy decisions. |
| AC group | Up to 256x256 pixels | Independent AC coefficient/token group. |
| Stripe | 256x64 pixels | `libjxl-tiny` streaming unit inside an AC group. |
| DC group | Up to 2048x2048 pixels | Low-frequency/control data group. |

Stripes are an implementation detail, not a compression ideal. They keep the
temporary XYB working set small, but they still affect the exact token stream
because padding, AQ halos, and row-context prediction happen in stripe order.

## JPEG To Tiny JPEG XL

If you know baseline JPEG, this port is easiest to read as a familiar pipeline
with different choices:

| Baseline JPEG idea | Tiny JPEG XL VarDCT counterpart |
| --- | --- |
| 8-bit gamma-coded input | Linear RGB input, then XYB. |
| Usually fixed 8x8 DCT | 8x8, 16x8, or 8x16 DCT decisions. |
| Quantization tables | Per-block adaptive quant field plus distance settings. |
| YCbCr chroma handling | XYB plus chroma-from-luma prediction. |
| Zigzag coefficient order | JPEG XL coefficient orders and context modeling. |
| Huffman-coded scan | Logical tokens, clustered contexts, prefix codes, sections. |

The important conceptual shift is that JPEG XL spends more effort modeling
perceptual importance and token contexts before entropy coding. The Python port
keeps those stages visible so each representation can be inspected separately.

## More Things To Inspect

These exercises are small enough to run quickly, but each one highlights a
different encoder behavior. Put all generated output under `build/` so it
stays disposable.

### Padding: 17x9

The walkthrough fixture is intentionally not block-aligned:

```bash
python3 tools/encode_trace_parity_test.py \
  --width 17 \
  --height 9 \
  --pattern gradient \
  --distance 1.0 \
  --work-dir build/learn/padding-17x9

just trace-summary build/learn/padding-17x9/single/trace 16
```

Look at `dcg_0_acg_0_stripe_0_input_padded.npy`. The image grows from 17x9
pixels to 24x16 pixels because transform coding works on complete 8x8 blocks.
The extra samples come from edge padding, not from introducing new image
content.

### AC Strategy: 129x129

Use a larger, uneven image to make transform decisions more interesting:

```bash
python3 tools/encode_trace_parity_test.py \
  --width 129 \
  --height 129 \
  --pattern rings \
  --distance 1.0 \
  --work-dir build/learn/ac-strategy-129

just trace-summary build/learn/ac-strategy-129/single/trace 8
```

Focus on `dcg_0_ac_strategy.npy` and the
`*_ac_strategy_entropy_*`, `*_ac_strategy_costs`, and
`*_ac_strategy_decision` arrays. The strategy map records which 8x8 cells are
encoded as standalone blocks and which cells are covered by wider transforms.

### Distance Sweep: 0.5, 1.0, 2.0

Run the same input at several quality settings:

```bash
for distance in 0.5 1.0 2.0; do
  python3 tools/encode_trace_parity_test.py \
    --width 129 \
    --height 129 \
    --pattern gradient \
    --distance "$distance" \
    --work-dir "build/learn/distance-${distance}"
  just trace-summary "build/learn/distance-${distance}/single/trace" 0
done
```

Compare the `raw_quant_field`, `quantized_ac`, token counts, section byte
sizes, and final `codestream.bin` sizes. Lower distances preserve more detail;
higher distances usually produce coarser quantization and smaller byte streams.

### Compare Real Inputs

For a PNG or JPEG, compare the Python encoder with `cjxl_tiny`:

```bash
just py-compare input.png build/learn/compare-real-input 1.0
```

The helper converts both encoders to the same linear-RGB PFM input first. If
`djxl` is installed, it also decodes both outputs and reports image-space
similarity metrics.

### Profile One Encode

Use the `py-profile` recipe when you want to see where the educational encoder
spends time:

```bash
tools/pfm_tools.py generate build/learn/profile-source.pfm \
  --width 257 \
  --height 193 \
  --pattern rings

just py-profile \
  build/learn/profile-source.pfm \
  build/learn/profile-source.jxl \
  1.0 \
  build/learn/py-encode-profile.html
```

Open the generated HTML profile and look for whole-stage costs first. The goal
is to understand the pipeline shape, not to micro-optimize the Python port.
