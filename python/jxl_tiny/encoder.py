"""Minimal whole-image encoder orchestration for the Python reproduction."""

from __future__ import annotations

import numpy as np

from .ac_strategy import DCT, adjust_quant_field, find_best_16x16_transform
from .adaptive_quantization import compute_adaptive_quantization
from .bitstream import codestream_bytes_from_ac_groups
from .chroma_from_luma import compute_chroma_from_luma
from .image import copy_and_pad_image
from .quantization import QuantizedBlock, quantize_ac_group
from .tokenization import ac_metadata_tokens, ac_tokens, dc_tokens
from .xyb import to_xyb


BLOCK_DIM = 8
GROUP_DIM = 256
GROUP_DIM_IN_BLOCKS = GROUP_DIM // BLOCK_DIM
DC_GROUP_DIM = GROUP_DIM * BLOCK_DIM
TILE_DIM = 64
TILE_DIM_IN_BLOCKS = TILE_DIM // BLOCK_DIM
GROUP_DIM_IN_TILES = GROUP_DIM // TILE_DIM


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_single_dc_group_image(image: np.ndarray, distance: float) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected channel-first RGB image with shape (3, y, x)")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("image dimensions must be nonzero")
    if image.shape[1] > DC_GROUP_DIM or image.shape[2] > DC_GROUP_DIM:
        raise ValueError("encode_from_image currently supports one DC group only")
    if distance <= 0.0:
        raise ValueError("lossless compression is not supported")


def _effective_distance(distance: float) -> float:
    return 0.03 if distance <= 0.03 else float(distance)


def _compute_ac_group_fields(
    xyb: np.ndarray, xsize: int, ysize: int, distance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            aq = compute_adaptive_quantization(tile, distance)
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
    distance: float
) -> tuple[np.ndarray, np.ndarray]:
    quantized_blocks = quantize_ac_group(
        xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map, distance
    )
    return (
        ac_tokens(xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map, distance),
        _quantized_blocks_to_quant_dc(raw_quant_field, quantized_blocks),
    )


def _copy_into_global(
    global_array: np.ndarray, local_array: np.ndarray, y0: int, x0: int
) -> None:
    ysize, xsize = local_array.shape[-2:]
    global_array[..., y0 : y0 + ysize, x0 : x0 + xsize] = local_array


def encode_from_image(image: np.ndarray, distance: float = 1.0) -> bytes:
    """Encode one RGB image to a JPEG XL codestream with libjxl-tiny behavior.

    This educational entrypoint intentionally starts with one DC group. Images
    may span multiple AC groups, but must fit within the 2048x2048 DC-group
    extent used by libjxl-tiny. The input must be channel-first linear RGB float
    data with shape `(3, y, x)`.
    """
    rgb = np.asarray(image, dtype=np.float32)
    _validate_single_dc_group_image(rgb, distance)
    distance = _effective_distance(float(distance))

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
            group_xyb = to_xyb(copy_and_pad_image(group_rgb))
            group_qf, group_acs, group_ytox, group_ytob = _compute_ac_group_fields(
                group_xyb, group_width, group_height, distance
            )
            group_ac_tokens, group_quant_dc = _ac_group_tokens_and_quant_dc(
                group_xyb, group_qf, group_acs, group_ytox, group_ytob, distance
            )
            ac_token_groups.append(group_ac_tokens)

            _copy_into_global(raw_quant_field, group_qf, group_by0, group_bx0)
            _copy_into_global(ac_strategy, group_acs, group_by0, group_bx0)
            _copy_into_global(ytox_map, group_ytox, group_ty0, group_tx0)
            _copy_into_global(ytob_map, group_ytob, group_ty0, group_tx0)
            _copy_into_global(quant_dc, group_quant_dc, group_by0, group_bx0)

    return codestream_bytes_from_ac_groups(
        xsize,
        ysize,
        dc_tokens(quant_dc),
        ac_metadata_tokens(ytox_map, ytob_map, ac_strategy, raw_quant_field),
        ac_token_groups,
        ac_strategy,
        distance,
    )
