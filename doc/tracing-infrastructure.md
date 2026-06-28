# Tracing Infrastructure for a Python Reproduction

This document describes a plan for adding reference tracing infrastructure to
`libjxl-tiny` so that a minimal Python/NumPy reproduction can be validated
incrementally against the C++ encoder.

The intended workflow is not to debug the Python port only by comparing final
JPEG XL bytes. Final codestream equality is useful, but it is a late-stage
acceptance test. The practical path is to expose deterministic intermediate
artifacts from the C++ encoder, compare the Python implementation stage by
stage, and only then require byte-identical `.jxl` output.

## Goal

The tracing infrastructure should make the existing encoder observable without
changing normal encoder behavior.

The reference C++ encoder should be able to emit small, stable fixtures for the
important stages of the VarDCT path:

* RGB padding and RGB-to-XYB conversion
* adaptive quantization
* chroma-from-luma map selection
* AC strategy selection
* transform coefficients and quantization
* DC prediction inputs and quantized DC values
* AC/DC token streams
* entropy-coded sections and final codestream bytes

The Python/NumPy reproduction can then implement each stage and compare against
the corresponding C++ trace artifact before moving on to the next stage.

## Why Trace Instead of Only Comparing JXL Bytes

Final `.jxl` byte comparison is too coarse for early development. A byte
mismatch can be caused by RGB-to-XYB math, adaptive quantization, transform
choice, coefficient rounding, chroma-from-luma maps, token contexts, entropy
code optimization, section assembly, or header writing. Once entropy coding is
involved, a small upstream difference can change many downstream bytes.

Tracing gives each stage a clear contract:

* floating-point stages compare with explicit tolerances;
* rounded or quantized integer stages compare exactly;
* token streams compare exactly before entropy coding;
* section bytes compare exactly before final codestream assembly;
* final codestream bytes compare exactly after all prior stages match.

This keeps failures local and makes the Python port useful for education,
because each fixture explains which part of the encoder behavior is being
reproduced.

## Reference Encoder Trace Design

Add an optional trace sink to the encoder pipeline. The sink should be disabled
by default and must not change the behavior of `cjxl_tiny`.

Recommended shape:

* define an `EncoderTraceSink` interface in the encoder library;
* pass `EncoderTraceSink* trace` through the frame, DC-group, tile, and AC-group
  processing path;
* use `nullptr` as the default, with no trace work performed when disabled;
* keep the public `EncodeFile` and `cjxl_tiny` behavior unchanged;
* add a separate trace executable, tentatively named `jxl_tiny_trace`.

The trace executable should accept:

```text
jxl_tiny_trace <input.pfm> <trace-dir> [-d distance]
```

It should read the same PFM input and distance value as `cjxl_tiny`, run the
same encoder pipeline, write the final codestream bytes, and emit trace
artifacts under `<trace-dir>`.

The implementation should avoid mixing trace serialization into the numerical
code. The numerical path should call small trace hooks with already-computed
values, while the concrete trace sink owns naming, serialization, and manifest
bookkeeping.

## Trace Artifact Format

Use a directory containing one `manifest.json` file plus array and byte-stream
files.

Preferred conventions:

* use `.npy` for numeric arrays so Python can load fixtures directly;
* use `.bin` for raw byte streams such as sections and codestream output;
* use channel-first order for image-like arrays: `(3, height, width)`;
* use row-major block order for block-like arrays;
* store all integer arrays with explicit-width dtypes such as `uint8`, `int8`,
  `int16`, `int32`, and `uint32`;
* store floating arrays as `float32` unless a stage needs `float64` for
  diagnostics.

The manifest should contain:

* trace schema version;
* input filename or input label;
* image width and height;
* encoder distance;
* build flags that affect behavior, especially `OPTIMIZE_CODE`,
  `OPTIMIZE_CHROMA_FROM_LUMA`, and `OPTIMIZE_BLOCK_SIZES`;
* target architecture or Highway target if available;
* list of artifacts with path, dtype, shape, stage name, and tolerance class;
* final codestream path and byte length.

Example manifest shape:

```json
{
  "schema_version": 1,
  "distance": 1.0,
  "image": {
    "xsize": 17,
    "ysize": 9
  },
  "build": {
    "OPTIMIZE_CODE": 1,
    "OPTIMIZE_CHROMA_FROM_LUMA": 1,
    "OPTIMIZE_BLOCK_SIZES": 1
  },
  "artifacts": [
    {
      "name": "xyb",
      "path": "xyb.npy",
      "dtype": "float32",
      "shape": [3, 16, 24],
      "tolerance": "float_stage"
    },
    {
      "name": "raw_quant_field",
      "path": "raw_quant_field.npy",
      "dtype": "uint8",
      "shape": [2, 3],
      "tolerance": "exact"
    }
  ]
}
```

Keep schema versioning from the start. Fixture readers should reject unknown
major schema versions instead of silently misinterpreting files.

## Trace Points

Start with coarse traces that explain the main pipeline, then add fine-grained
per-block traces when the Python port reaches those stages.

Recommended coarse trace points:

* `input_padded`: the padded linear RGB stripe or group input used by the
  encoder;
* `xyb`: RGB-to-XYB output;
* `aq_map`: floating adaptive quantization map for a tile;
* `mask`: masking values used by AC strategy selection;
* `raw_quant_field_pre_adjust`: quantized per-block AC quantization values
  before AC strategy adjustment;
* `raw_quant_field_post_adjust`: quantized per-block AC quantization values
  after AC strategy adjustment;
* `raw_quant_field`: compatibility alias for the post-adjusted quantization
  field;
* tile-local `ytox` and `ytob`: chroma-from-luma multipliers before storage
  in the DC-group maps;
* `ytox_map` and `ytob_map`: chroma-from-luma tile multipliers;
* `ac_strategy`: raw per-block AC strategy representation;
* `quant_dc`: quantized DC image;
* `sections`: DC global, DC groups, AC global, and AC groups;
* `codestream`: final encoded bytes.

Recommended fine-grained trace points:

* `ac_strategy_entropy_8x8`, `ac_strategy_entropy_16x8`,
  `ac_strategy_entropy_8x16`, and `ac_strategy_costs`: candidate scores used
  when selecting a transform layout for a 16x16 region;
* `ac_strategy_decision`: the selected 2x2 raw AC strategy encoding for a
  traced 16x16 region;
* `raw_coefficients`: selected per-block transform coefficients before
  roundtrip quantization and chroma-from-luma removal;
* `quant_input_coefficients`: per-block transform coefficients after
  Y roundtrip quantization and chroma-from-luma removal, immediately before
  X/B quantization;
* `quantized_ac`: quantized AC coefficients after chroma-from-luma removal;
* `block_quant_dc`: block-local quantized DC coefficients derived while
  quantizing AC groups;
* `num_nonzeros` and `num_nonzeros_map`: nonzero counts used for AC contexts;
* `dc_tokens`: DC token stream before entropy coding;
* `ac_tokens`: AC token stream before entropy coding.

For fine-grained block traces, prefer small fixture images and scoped dumps.
Dumping every coefficient for large images will make fixtures noisy and slow.

## Python Test Harness

Add a Python-side fixture runner after the C++ trace executable exists.

The runner should:

1. generate or locate a deterministic PFM input;
2. invoke `jxl_tiny_trace` with a fixed distance and output directory;
3. load `manifest.json` and all referenced artifacts;
4. run the Python/NumPy implementation through the matching stage;
5. compare the Python result to the C++ artifact;
6. stop at the first mismatching stage with a clear diagnostic.

Comparison policy:

* use exact equality for integer arrays, token streams, byte streams, and final
  codestream bytes;
* use absolute and relative tolerances for floating-point arrays;
* define tolerances by stage in the Python test helper, keyed by manifest
  tolerance class;
* keep tolerance values explicit and conservative;
* require exact equality once a floating result is rounded or quantized.

The test harness should make it easy to run one fixture and one stage during
development. Full end-to-end byte equality should run only after the relevant
upstream stages already pass.

## Implementation Phases

### Phase 1: Trace Sink and Trace Executable

Add the minimal tracing plumbing:

* define the trace sink interface and a no-op default;
* add a filesystem trace sink that writes `manifest.json`, `.npy`, and `.bin`
  files;
* thread the optional sink through the encode path without changing existing
  caller behavior;
* add the `jxl_tiny_trace` executable to CMake;
* verify that `cjxl_tiny` output is unchanged when tracing is disabled.

### Phase 2: Coarse Stage Traces

Emit the high-value stage artifacts first:

* padded RGB input;
* XYB output;
* adaptive quantization map and raw quant field;
* chroma-from-luma maps;
* AC strategy map;
* quantized DC image;
* section bytes and final codestream bytes.

These traces are enough to start the Python implementation and catch most major
pipeline errors.

### Phase 3: Per-Block Numerical Traces

Add detailed traces for the hardest numerical stages:

* AC strategy candidate entropies, candidate costs, and selected 2x2 decision;
* transform coefficients before quantization, quantization-input coefficients,
  quantized AC coefficients, block-local quantized DC values, and nonzero
  counts;
* AC/DC tokens before entropy coding.

Use these only for small fixtures by default. They are mainly for debugging
rounding, coefficient order, and context modeling.

### Phase 4: Python Fixture Runner

Add Python tests around the trace schema:

* fixture generation for small deterministic images;
* trace invocation helper;
* `.npy`, `.bin`, and manifest loading;
* per-stage comparison helpers;
* short diagnostics for first mismatch location, dtype, shape, and max error.

### Phase 5: Python/NumPy Port

Port the encoder in the same order as the traces:

1. bit writer, size headers, constants, and distance parameters;
2. image padding and RGB-to-XYB;
3. DCT/IDCT primitives and quant matrices;
4. adaptive quantization;
5. chroma-from-luma maps;
6. AC strategy selection;
7. AC/DC quantization and tokenization;
8. entropy writing and section assembly;
9. final codestream byte equality.

Each stage should land with fixture comparisons before the next stage depends
on it.

## Suggested Fixtures

Use tiny synthetic inputs first:

* constant gray and constant color images;
* single-pixel impulse;
* horizontal and vertical gradients;
* checkerboard;
* seeded random image;
* odd dimensions such as `17x9` and `65x33`;
* dimensions near block, tile, and group boundaries after small fixtures pass.

Keep fixture generation deterministic and source-controlled. Generated trace
outputs can be regenerated locally rather than checked in unless a specific
test needs stable committed golden data.

## Acceptance Criteria

Tracing infrastructure is ready when:

* `cjxl_tiny` output is byte-identical before and after tracing changes when
  tracing is disabled;
* `jxl_tiny_trace` builds through CMake;
* the trace executable writes a manifest and artifacts for at least one small
  fixture;
* trace output is deterministic for the same input, distance, build flags, and
  architecture;
* Python tests can validate at least one fixture through coarse stage traces;
* integer arrays, tokens, section bytes, and codestream bytes compare exactly;
* floating stages compare within documented tolerances.

The Python reproduction is ready for final byte checks only after the relevant
intermediate stages already match.

## Caveats

Some floating-point traces can be architecture-sensitive. The encoder uses
Highway dispatch and fast math helpers, so exact floating equality against
NumPy should not be assumed before values are rounded or quantized.

Record behavior-affecting build flags in the manifest. In this checkout,
`OPTIMIZE_CODE`, `OPTIMIZE_CHROMA_FROM_LUMA`, and `OPTIMIZE_BLOCK_SIZES` affect
the path being reproduced and therefore must be part of the fixture identity.

Avoid making trace hooks part of the stable encoder API. They are development
and education infrastructure, not a supported encoding interface.
