"""Minimal whole-image encoder orchestration for the Python reproduction."""

from __future__ import annotations

import numpy as np

from .ac_strategy import DCT, adjust_quant_field, find_best_16x16_transform
from .adaptive_quantization import compute_adaptive_quantization
from .bitstream import codestream_bytes
from .chroma_from_luma import compute_chroma_from_luma
from .image import copy_and_pad_image
from .quantization import quantize_ac_group
from .tokenization import ac_metadata_tokens, ac_tokens, dc_tokens
from .xyb import to_xyb


BLOCK_DIM = 8
GROUP_DIM = 256
TILE_DIM = 64
TILE_DIM_IN_BLOCKS = TILE_DIM // BLOCK_DIM


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_single_group_image(image: np.ndarray, distance: float) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected channel-first RGB image with shape (3, y, x)")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("image dimensions must be nonzero")
    if image.shape[1] > GROUP_DIM or image.shape[2] > GROUP_DIM:
        raise ValueError("encode_from_image currently supports one AC group only")
    if distance <= 0.0:
        raise ValueError("lossless compression is not supported")


def _effective_distance(distance: float) -> float:
    return 0.03 if distance <= 0.03 else float(distance)


def _compute_single_group_fields(
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


def _quant_dc_from_blocks(
    xyb: np.ndarray,
    raw_quant_field: np.ndarray,
    ac_strategy: np.ndarray,
    ytox_map: np.ndarray,
    ytob_map: np.ndarray,
    distance: float,
) -> np.ndarray:
    quantized_blocks = quantize_ac_group(
        xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map, distance
    )
    quant_dc = np.zeros((3, raw_quant_field.shape[0], raw_quant_field.shape[1]),
                        dtype=np.int16)
    for (by, bx), block in quantized_blocks.items():
        covered_y = block.block_quant_dc.shape[1]
        covered_x = block.block_quant_dc.shape[2]
        quant_dc[:, by : by + covered_y, bx : bx + covered_x] = block.block_quant_dc
    return quant_dc


def encode_from_image(image: np.ndarray, distance: float = 1.0) -> bytes:
    """Encode one RGB image to a JPEG XL codestream with libjxl-tiny behavior.

    This educational entrypoint intentionally starts with the smallest complete
    integration target: one DC group and one AC group. The input must be
    channel-first linear RGB float data with shape `(3, y, x)`.
    """
    rgb = np.asarray(image, dtype=np.float32)
    _validate_single_group_image(rgb, distance)
    distance = _effective_distance(float(distance))

    _, ysize, xsize = rgb.shape
    xyb = to_xyb(copy_and_pad_image(rgb))
    raw_quant_field, ac_strategy, ytox_map, ytob_map = _compute_single_group_fields(
        xyb, xsize, ysize, distance
    )
    quant_dc = _quant_dc_from_blocks(
        xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map, distance
    )

    return codestream_bytes(
        xsize,
        ysize,
        dc_tokens(quant_dc),
        ac_metadata_tokens(ytox_map, ytob_map, ac_strategy, raw_quant_field),
        ac_tokens(xyb, raw_quant_field, ac_strategy, ytox_map, ytob_map, distance),
        ac_strategy,
        distance,
    )
