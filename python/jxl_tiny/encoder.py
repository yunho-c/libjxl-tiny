"""Whole-image orchestration for the educational Python encoder.

This module ties the stage helpers together into the same lossy VarDCT path as
`cjxl_tiny`. It is intentionally compact: each helper owns the stage-specific
math, while this file owns image grouping, stripe order, and the data flow from
linear RGB arrays to final codestream bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ac_strategy import DCT, adjust_quant_field, find_best_16x16_transform
from .adaptive_quantization import compute_adaptive_quantization
from .bitstream import codestream_bytes_from_groups
from .chroma_from_luma import compute_chroma_from_luma
from .image import copy_and_pad_image
from .quantization import QuantizedBlock, quantize_ac_group
from .tokenization import ac_metadata_tokens, ac_tokens_from_quantized_blocks, dc_tokens
from .xyb import to_xyb


BLOCK_DIM = 8
GROUP_DIM = 256
GROUP_DIM_IN_BLOCKS = GROUP_DIM // BLOCK_DIM
DC_GROUP_DIM = GROUP_DIM * BLOCK_DIM
TILE_DIM = 64
TILE_DIM_IN_BLOCKS = TILE_DIM // BLOCK_DIM
GROUP_DIM_IN_TILES = GROUP_DIM // TILE_DIM


@dataclass(frozen=True)
class _DcGroupEncoding:
    dc_tokens: np.ndarray
    ac_metadata_tokens: np.ndarray
    ac_strategy: np.ndarray
    ac_token_groups: list[np.ndarray]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_image(image: np.ndarray, distance: float) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected channel-first RGB image with shape (3, y, x)")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("image dimensions must be nonzero")
    if distance <= 0.0:
        raise ValueError("lossless compression is not supported")


def _effective_distance(distance: float) -> float:
    return 0.03 if distance <= 0.03 else float(distance)


def _compute_ac_group_fields(
    xyb: np.ndarray, xsize: int, ysize: int, distance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute quant fields, AC strategy, and CFL maps for one AC stripe."""
    x_blocks = _ceil_div(xsize, BLOCK_DIM)
    y_blocks = _ceil_div(ysize, BLOCK_DIM)
    x_tiles = _ceil_div(xsize, TILE_DIM)
    y_tiles = _ceil_div(ysize, TILE_DIM)

    raw_quant_field = np.empty((y_blocks, x_blocks), dtype=np.uint8)
    ac_strategy = np.full((y_blocks, x_blocks), np.uint8((DCT << 1) | 1))
    ytox_map = np.zeros((y_tiles, x_tiles), dtype=np.int8)
    ytob_map = np.zeros((y_tiles, x_tiles), dtype=np.int8)

    for tile_y in range(y_tiles):
        by0 = tile_y * TILE_DIM_IN_BLOCKS
        tile_blocks_y = min(TILE_DIM_IN_BLOCKS, y_blocks - by0)
        py0 = by0 * BLOCK_DIM
        py1 = py0 + tile_blocks_y * BLOCK_DIM
        for tile_x in range(x_tiles):
            bx0 = tile_x * TILE_DIM_IN_BLOCKS
            tile_blocks_x = min(TILE_DIM_IN_BLOCKS, x_blocks - bx0)
            px0 = bx0 * BLOCK_DIM
            px1 = px0 + tile_blocks_x * BLOCK_DIM

            tile = xyb[:, py0:py1, px0:px1]
            # AQ sees the full padded stripe so its four-pixel halo can cross
            # tile boundaries; CFL is fit only from the current 64x64 tile.
            aq = compute_adaptive_quantization(
                xyb,
                distance,
                block_x0=bx0,
                block_y0=by0,
                block_width=tile_blocks_x,
                block_height=tile_blocks_y,
            )
            raw_quant_field[by0 : by0 + tile_blocks_y, bx0 : bx0 + tile_blocks_x] = (
                aq.raw_quant_field
            )

            cfl = compute_chroma_from_luma(tile)
            ytox_map[tile_y, tile_x] = cfl.ytox
            ytob_map[tile_y, tile_x] = cfl.ytob

            for cy in range(0, tile_blocks_y - 1, 2):
                for cx in range(0, tile_blocks_x - 1, 2):
                    decision = find_best_16x16_transform(
                        xyb,
                        aq.aq_map,
                        aq.mask,
                        distance,
                        int(cfl.ytox),
                        int(cfl.ytob),
                        bx0=bx0,
                        by0=by0,
                        cx=cx,
                        cy=cy,
                    )
                    ac_strategy[
                        by0 + cy : by0 + cy + 2, bx0 + cx : bx0 + cx + 2
                    ] = decision.decision

    raw_quant_field = adjust_quant_field(raw_quant_field, ac_strategy)
    return raw_quant_field, ac_strategy, ytox_map, ytob_map


def _quantized_blocks_to_quant_dc(
    raw_quant_field: np.ndarray,
    quantized_blocks: dict[tuple[int, int], QuantizedBlock],
) -> np.ndarray:
    quant_dc = np.zeros(
        (3, raw_quant_field.shape[0], raw_quant_field.shape[1]), dtype=np.int16
    )
    for (by, bx), block in quantized_blocks.items():
        covered_y = block.block_quant_dc.shape[1]
        covered_x = block.block_quant_dc.shape[2]
        quant_dc[:, by : by + covered_y, bx : bx + covered_x] = block.block_quant_dc
    return quant_dc


def _ac_group_tokens_and_quant_dc(
    xyb: np.ndarray,
    raw_quant_field: np.ndarray,
    ac_strategy: np.ndarray,
    ytox_map: np.ndarray,
    ytob_map: np.ndarray,
    distance: float,
    nzeros_map: np.ndarray | None = None,
    by_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize one stripe and produce AC tokens plus local quantized DC.

    `nzeros_map` may be a full AC-group map when stripes are being streamed.
    Passing the group map lets row-context prediction see the previous stripe,
    while `by_offset` tells tokenization where this stripe starts in that map.
    """
    quantized_blocks = quantize_ac_group(
        xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map, distance
    )
    local_nzeros_map = np.zeros(
        (3, raw_quant_field.shape[0], raw_quant_field.shape[1]), dtype=np.uint8
    )
    for (by, bx), block in quantized_blocks.items():
        covered_y = block.num_nonzeros_map.shape[1]
        covered_x = block.num_nonzeros_map.shape[2]
        local_nzeros_map[:, by : by + covered_y, bx : bx + covered_x] = (
            block.num_nonzeros_map
        )
    token_nzeros_map = local_nzeros_map
    if nzeros_map is not None:
        _copy_into_global(nzeros_map, local_nzeros_map, by_offset, 0)
        token_nzeros_map = nzeros_map
    return (
        ac_tokens_from_quantized_blocks(
            ac_strategy, quantized_blocks, token_nzeros_map, by_offset=by_offset
        ),
        _quantized_blocks_to_quant_dc(raw_quant_field, quantized_blocks),
        local_nzeros_map,
    )


def _copy_into_global(
    global_array: np.ndarray, local_array: np.ndarray, y0: int, x0: int
) -> None:
    ysize, xsize = local_array.shape[-2:]
    global_array[..., y0 : y0 + ysize, x0 : x0 + xsize] = local_array


def _encode_dc_group(rgb: np.ndarray, distance: float) -> _DcGroupEncoding:
    _, ysize, xsize = rgb.shape
    x_blocks = _ceil_div(xsize, BLOCK_DIM)
    y_blocks = _ceil_div(ysize, BLOCK_DIM)
    x_tiles = _ceil_div(xsize, TILE_DIM)
    y_tiles = _ceil_div(ysize, TILE_DIM)
    x_groups = _ceil_div(xsize, GROUP_DIM)
    y_groups = _ceil_div(ysize, GROUP_DIM)

    raw_quant_field = np.empty((y_blocks, x_blocks), dtype=np.uint8)
    ac_strategy = np.full((y_blocks, x_blocks), np.uint8((DCT << 1) | 1))
    ytox_map = np.zeros((y_tiles, x_tiles), dtype=np.int8)
    ytob_map = np.zeros((y_tiles, x_tiles), dtype=np.int8)
    quant_dc = np.zeros((3, y_blocks, x_blocks), dtype=np.int16)
    ac_token_groups: list[np.ndarray] = []

    for group_y in range(y_groups):
        group_py0 = group_y * GROUP_DIM
        group_height = min(GROUP_DIM, ysize - group_py0)
        group_by0 = group_y * GROUP_DIM_IN_BLOCKS
        group_ty0 = group_y * GROUP_DIM_IN_TILES
        for group_x in range(x_groups):
            group_px0 = group_x * GROUP_DIM
            group_width = min(GROUP_DIM, xsize - group_px0)
            group_bx0 = group_x * GROUP_DIM_IN_BLOCKS
            group_tx0 = group_x * GROUP_DIM_IN_TILES
            group_rgb = rgb[
                :,
                group_py0 : group_py0 + group_height,
                group_px0 : group_px0 + group_width,
            ]
            group_y_blocks = _ceil_div(group_height, BLOCK_DIM)
            group_x_blocks = _ceil_div(group_width, BLOCK_DIM)
            group_nzeros_map = np.zeros(
                (3, group_y_blocks, group_x_blocks), dtype=np.uint8
            )
            group_token_stripes: list[np.ndarray] = []

            # libjxl-tiny streams each 256x256 AC group as 256x64 stripes to
            # keep the transient XYB buffer small. This is an encoder memory
            # optimization rather than an inherent compression requirement, but
            # it affects padding, AQ halos, and row-context token prediction.
            for stripe_y in range(_ceil_div(group_height, TILE_DIM)):
                stripe_py0 = stripe_y * TILE_DIM
                stripe_height = min(TILE_DIM, group_height - stripe_py0)
                stripe_by0 = stripe_y * TILE_DIM_IN_BLOCKS
                stripe_rgb = group_rgb[
                    :, stripe_py0 : stripe_py0 + stripe_height, :
                ]
                stripe_xyb = to_xyb(copy_and_pad_image(stripe_rgb))
                stripe_qf, stripe_acs, stripe_ytox, stripe_ytob = (
                    _compute_ac_group_fields(
                        stripe_xyb, group_width, stripe_height, distance
                    )
                )
                stripe_ac_tokens, stripe_quant_dc, _ = _ac_group_tokens_and_quant_dc(
                    stripe_xyb,
                    stripe_qf,
                    stripe_acs,
                    stripe_ytox,
                    stripe_ytob,
                    distance,
                    group_nzeros_map,
                    stripe_by0,
                )
                group_token_stripes.append(stripe_ac_tokens)

                _copy_into_global(
                    raw_quant_field, stripe_qf, group_by0 + stripe_by0, group_bx0
                )
                _copy_into_global(
                    ac_strategy, stripe_acs, group_by0 + stripe_by0, group_bx0
                )
                _copy_into_global(ytox_map, stripe_ytox, group_ty0 + stripe_y, group_tx0)
                _copy_into_global(ytob_map, stripe_ytob, group_ty0 + stripe_y, group_tx0)
                _copy_into_global(
                    quant_dc, stripe_quant_dc, group_by0 + stripe_by0, group_bx0
                )

            ac_token_groups.append(np.concatenate(group_token_stripes, axis=0))

    return _DcGroupEncoding(
        dc_tokens=dc_tokens(quant_dc),
        ac_metadata_tokens=ac_metadata_tokens(
            ytox_map, ytob_map, ac_strategy, raw_quant_field
        ),
        ac_strategy=ac_strategy,
        ac_token_groups=ac_token_groups,
    )


def encode_from_image(image: np.ndarray, distance: float = 1.0) -> bytes:
    """Encode one RGB image to a JPEG XL codestream with libjxl-tiny behavior.

    The input must be channel-first linear RGB float data with shape `(3, y, x)`.
    """
    rgb = np.asarray(image, dtype=np.float32)
    _validate_image(rgb, distance)
    distance = _effective_distance(float(distance))

    _, ysize, xsize = rgb.shape
    x_dc_groups = _ceil_div(xsize, DC_GROUP_DIM)
    y_dc_groups = _ceil_div(ysize, DC_GROUP_DIM)
    x_ac_groups = _ceil_div(xsize, GROUP_DIM)
    y_ac_groups = _ceil_div(ysize, GROUP_DIM)
    dc_group_encodings: list[_DcGroupEncoding] = []
    ac_token_groups: list[np.ndarray | None] = [None] * (x_ac_groups * y_ac_groups)

    for dc_group_y in range(y_dc_groups):
        dc_py0 = dc_group_y * DC_GROUP_DIM
        dc_height = min(DC_GROUP_DIM, ysize - dc_py0)
        for dc_group_x in range(x_dc_groups):
            dc_px0 = dc_group_x * DC_GROUP_DIM
            dc_width = min(DC_GROUP_DIM, xsize - dc_px0)
            dc_rgb = rgb[
                :,
                dc_py0 : dc_py0 + dc_height,
                dc_px0 : dc_px0 + dc_width,
            ]
            dc_encoding = _encode_dc_group(dc_rgb, distance)
            dc_group_encodings.append(dc_encoding)

            local_x_ac_groups = _ceil_div(dc_width, GROUP_DIM)
            local_y_ac_groups = _ceil_div(dc_height, GROUP_DIM)
            base_ac_group_x = dc_group_x * (DC_GROUP_DIM // GROUP_DIM)
            base_ac_group_y = dc_group_y * (DC_GROUP_DIM // GROUP_DIM)
            for local_group_y in range(local_y_ac_groups):
                global_group_y = base_ac_group_y + local_group_y
                for local_group_x in range(local_x_ac_groups):
                    global_group_x = base_ac_group_x + local_group_x
                    local_group_id = local_group_y * local_x_ac_groups + local_group_x
                    global_group_id = global_group_y * x_ac_groups + global_group_x
                    # DC groups own several AC groups. The bitstream wants AC
                    # groups in global raster order, so remap the local list.
                    ac_token_groups[global_group_id] = (
                        dc_encoding.ac_token_groups[local_group_id]
                    )

    if any(group is None for group in ac_token_groups):
        raise RuntimeError("internal error: missing AC token group")

    return codestream_bytes_from_groups(
        xsize,
        ysize,
        [encoding.dc_tokens for encoding in dc_group_encodings],
        [encoding.ac_metadata_tokens for encoding in dc_group_encodings],
        [group for group in ac_token_groups if group is not None],
        [encoding.ac_strategy for encoding in dc_group_encodings],
        distance,
    )
