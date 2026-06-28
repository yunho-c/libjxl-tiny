"""AC/DC quantization helpers mirroring `encoder/enc_group.cc`."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .ac_strategy import (
    DCT,
    DCT16X8,
    DCT8X16,
    DCT2_B_WEIGHTS,
    DCT2_X_WEIGHTS,
    DCT2_Y_WEIGHTS,
    DCT_Y_WEIGHTS,
    INV_MATRICES,
    covered_blocks_x,
    covered_blocks_y,
    is_first_block,
    raw_strategy,
)
from .chroma_from_luma import K_INV_COLOR_FACTOR, _DCT_B_WEIGHTS, _DCT_X_WEIGHTS
from .transforms import BLOCK_DIM, scaled_dct


K_DCT_BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM
K_INV_DC_QUANT = np.asarray((4096.0, 512.0, 256.0), dtype=np.float32)
K_DC_QUANT = (np.float32(1.0) / K_INV_DC_QUANT).astype(np.float32)
K_DCT_16_TO_2_SCALE = np.float32(0.901764195028874394)

MATRICES = {
    DCT: (
        _DCT_X_WEIGHTS,
        DCT_Y_WEIGHTS,
        _DCT_B_WEIGHTS,
    ),
    DCT16X8: (
        DCT2_X_WEIGHTS,
        DCT2_Y_WEIGHTS,
        DCT2_B_WEIGHTS,
    ),
    DCT8X16: (
        DCT2_X_WEIGHTS,
        DCT2_Y_WEIGHTS,
        DCT2_B_WEIGHTS,
    ),
}


@dataclass(frozen=True)
class DistanceParams:
  global_scale: int
  quant_dc: int
  scale: np.float32
  scale_dc: np.float32
  x_qm_scale: int
  epf_iters: int


@dataclass(frozen=True)
class QuantizedBlock:
  raw_coefficients: np.ndarray
  quant_input_coefficients: np.ndarray
  quantized_ac: np.ndarray
  block_quant_dc: np.ndarray
  num_nonzeros: np.ndarray
  num_nonzeros_map: np.ndarray


def _clamp(value: float, low: float, high: float) -> float:
  return low if value < low else high if value > high else value


def _round_away_from_zero(value: np.ndarray | np.float32) -> np.ndarray:
  value = np.asarray(value, dtype=np.float32)
  return np.where(value >= 0.0, np.floor(value + 0.5),
                  np.ceil(value - 0.5)).astype(np.int32)


def compute_distance_params(distance: float) -> DistanceParams:
  distance = float(distance)
  dc_quant_pow = 0.57
  dc_quant = 1.12
  dc_mul = 2.9
  effective_dist = dc_mul * math.pow(distance / dc_mul, dc_quant_pow)
  effective_dist = _clamp(effective_dist, 0.5 * distance, distance)
  quant_dc = min(dc_quant / effective_dist, 50.0)

  global_scale_denom = 1 << 16
  global_scale_numerator = 4096
  ac_quant = 0.8
  quant_field_target = 5.0
  scale = global_scale_denom * ac_quant / (distance * quant_field_target)
  scale = _clamp(scale, 1.0, float(1 << 15))
  scaled_quant_dc = int(quant_dc * global_scale_numerator * 1.6)
  global_scale = int(_clamp(float(int(scale)), 1.0, float(scaled_quant_dc)))
  scale_f = np.float32(global_scale) * np.float32(1.0 / global_scale_denom)
  quant_dc_i = int(quant_dc / float(scale_f) + 0.5)
  quant_dc_i = int(_clamp(float(quant_dc_i), 1.0, float(1 << 16)))
  scale_dc = np.float32(quant_dc_i) * scale_f

  x_qm_scale = 2
  for step in (1.25, 9.0):
    if distance > step:
      x_qm_scale += 1
  if distance < 0.299:
    x_qm_scale += 1
  epf_iters = 0
  for threshold in (0.7, 1.5, 4.0):
    if distance >= threshold:
      epf_iters += 1

  return DistanceParams(global_scale=global_scale,
                        quant_dc=quant_dc_i,
                        scale=scale_f,
                        scale_dc=np.float32(scale_dc),
                        x_qm_scale=x_qm_scale,
                        epf_iters=epf_iters)


def _canonical_blocks(strategy: int) -> tuple[int, int]:
  xsize = covered_blocks_x(strategy)
  ysize = covered_blocks_y(strategy)
  if ysize > xsize:
    xsize, ysize = ysize, xsize
  return xsize, ysize


def _strategy_shape(strategy: int) -> tuple[int, int]:
  if strategy == DCT:
    return 8, 8
  if strategy == DCT16X8:
    return 16, 8
  if strategy == DCT8X16:
    return 8, 16
  raise ValueError(f"invalid AC strategy: {strategy}")


def _dc_from_lowest_frequencies(strategy: int, block: np.ndarray) -> np.ndarray:
  block = np.asarray(block, dtype=np.float32).reshape(-1)
  if strategy == DCT:
    return np.asarray([[block[0]]], dtype=np.float32)
  low0 = block[0]
  low1 = np.float32(block[1] * K_DCT_16_TO_2_SCALE)
  if strategy == DCT16X8:
    return np.asarray([[low0 + low1], [low0 - low1]], dtype=np.float32)
  if strategy == DCT8X16:
    return np.asarray([[low0 + low1, low0 - low1]], dtype=np.float32)
  raise ValueError(f"invalid AC strategy: {strategy}")


def _thresholds(channel: int, xsize: int, ysize: int) -> np.ndarray:
  thres = np.asarray((0.58, 0.635, 0.66, 0.7), dtype=np.float32)
  if channel == 0:
    thres[1:] += np.float32(0.08)
  if channel == 2:
    thres[1:] = np.float32(0.75)
  if xsize > 1 or ysize > 1:
    reduction = _clamp(0.003 * xsize * ysize, 0.0,
                       0.08 if channel > 0 else 0.12)
    thres -= np.float32(reduction)
  return thres


def quantize_block_ac(block_in: np.ndarray, channel: int, strategy: int,
                      quant: int, scale: float,
                      qm_multiplier: float = 1.0) -> np.ndarray:
  xsize, ysize = _canonical_blocks(strategy)
  size = xsize * ysize * K_DCT_BLOCK_SIZE
  block = np.asarray(block_in, dtype=np.float32).reshape(size)
  qm = INV_MATRICES[strategy][channel][:size]
  q = qm * np.float32(scale * quant * qm_multiplier)
  val = (q * block).astype(np.float32)
  out = np.zeros(size, dtype=np.int32)
  thres = _thresholds(channel, xsize, ysize)
  width = xsize * BLOCK_DIM
  height = ysize * BLOCK_DIM
  for y in range(height):
    yfix = 2 if y >= height // 2 else 0
    for x in range(width):
      if xsize == 1:
        threshold_index = yfix + (1 if x >= 4 else 0)
      else:
        threshold_index = yfix + (1 if x >= width // 2 else 0)
      offset = y * width + x
      if abs(float(val[offset])) >= float(thres[threshold_index]):
        out[offset] = int(np.rint(val[offset]))
  return out


def _adjust_quant_bias(quantized: np.ndarray, channel: int) -> np.ndarray:
  biases = np.asarray((
      1.0 - 0.05465007330715401,
      1.0 - 0.07005449891748593,
      1.0 - 0.049935103337343655,
      0.145,
  ), dtype=np.float32)
  quant = np.asarray(quantized, dtype=np.float32)
  abs_quant = np.abs(quant).astype(np.float32)
  out = np.zeros_like(quant, dtype=np.float32)
  one_mask = (abs_quant < np.float32(1.125)) & (abs_quant > np.float32(0.0))
  out[one_mask] = np.copysign(biases[channel], quant[one_mask])
  other_mask = ~one_mask & (quant != np.float32(0.0))
  out[other_mask] = (quant[other_mask] -
                     biases[3] / quant[other_mask]).astype(np.float32)
  return out


def quantize_roundtrip_y_block_ac(block_in: np.ndarray, strategy: int,
                                  quant: int,
                                  scale: float) -> tuple[np.ndarray, np.ndarray]:
  xsize, ysize = _canonical_blocks(strategy)
  size = xsize * ysize * K_DCT_BLOCK_SIZE
  quantized = quantize_block_ac(block_in, 1, strategy, quant, scale)
  adjusted = _adjust_quant_bias(quantized, 1)
  matrix = MATRICES[strategy][1][:size]
  inv_qac = np.float32(1.0 / (np.float32(scale) * np.float32(quant)))
  roundtripped = (adjusted * matrix * inv_qac).astype(np.float32)
  return roundtripped, quantized


def _num_nonzero_8x8_except_dc(block: np.ndarray) -> int:
  mask = np.ones(K_DCT_BLOCK_SIZE, dtype=bool)
  mask[0] = False
  return int(np.count_nonzero(np.asarray(block, dtype=np.int32)[mask]))


def _num_nonzero_except_llf(block: np.ndarray, xsize: int, ysize: int,
                            covered_blocks: int) -> int:
  coeffs = np.asarray(block, dtype=np.int32).reshape(ysize * BLOCK_DIM,
                                                     xsize * BLOCK_DIM)
  mask = np.ones_like(coeffs, dtype=bool)
  mask[:ysize, :xsize] = False
  return int(np.count_nonzero(coeffs[mask]))


def quantize_ac_group(xyb: np.ndarray, raw_quant_field: np.ndarray,
                      ac_strategy: np.ndarray, ytox_map: np.ndarray,
                      ytob_map: np.ndarray,
                      distance: float) -> dict[tuple[int, int], QuantizedBlock]:
  if xyb.ndim != 3 or xyb.shape[0] != 3:
    raise ValueError("expected channel-first XYB image with shape (3, y, x)")
  qf = np.asarray(raw_quant_field, dtype=np.uint8)
  strategy_grid = np.asarray(ac_strategy, dtype=np.uint8)
  ytox = np.asarray(ytox_map, dtype=np.int8)
  ytob = np.asarray(ytob_map, dtype=np.int8)
  params = compute_distance_params(distance)
  inv_factor = (K_INV_DC_QUANT * params.scale_dc).astype(np.float32)
  cfl_factor = np.asarray((0.0, 0.0, K_INV_DC_QUANT[2] * K_DC_QUANT[1]),
                          dtype=np.float32)
  x_qm_mul = np.float32(math.pow(1.25, params.x_qm_scale - 2.0))
  quant_dc = np.zeros((3, qf.shape[0], qf.shape[1]), dtype=np.int16)
  nzeros_map = np.zeros((3, qf.shape[0], qf.shape[1]), dtype=np.uint8)
  result: dict[tuple[int, int], QuantizedBlock] = {}

  for by in range(qf.shape[0]):
    for bx in range(qf.shape[1]):
      encoded = strategy_grid[by, bx]
      if not bool(is_first_block(encoded)):
        continue
      strategy = int(raw_strategy(encoded))
      xsize, ysize = _canonical_blocks(strategy)
      covered_x = covered_blocks_x(strategy)
      covered_y = covered_blocks_y(strategy)
      size = xsize * ysize * K_DCT_BLOCK_SIZE
      quant_ac = int(qf[by, bx])
      height, width = _strategy_shape(strategy)
      y0 = by * BLOCK_DIM
      x0 = bx * BLOCK_DIM

      raw_coefficients = np.empty((3, size), dtype=np.float32)
      for channel in range(3):
        raw_coefficients[channel] = scaled_dct(
            xyb[channel, y0:y0 + height, x0:x0 + width]).reshape(size)

      coeffs = raw_coefficients.copy()
      dc_y = _dc_from_lowest_frequencies(strategy, coeffs[1])
      quant_dc[1, by:by + covered_y, bx:bx + covered_x] = (
          _round_away_from_zero(inv_factor[1] * dc_y).astype(np.int16))
      coeffs[1], quantized_y = quantize_roundtrip_y_block_ac(
          coeffs[1], strategy, quant_ac, params.scale)

      tile_y = by // 8
      tile_x = bx // 8
      x_factor = np.float32(ytox[tile_y, tile_x]) * K_INV_COLOR_FACTOR
      b_factor = (np.float32(1.0) +
                  np.float32(ytob[tile_y, tile_x]) * K_INV_COLOR_FACTOR)
      coeffs[0] = (coeffs[0] - x_factor * coeffs[1]).astype(np.float32)
      coeffs[2] = (coeffs[2] - b_factor * coeffs[1]).astype(np.float32)

      quantized = np.empty((3, size), dtype=np.int32)
      quantized[1] = quantized_y
      for channel in (0, 2):
        quantized[channel] = quantize_block_ac(
            coeffs[channel],
            channel,
            strategy,
            quant_ac,
            params.scale,
            x_qm_mul if channel == 0 else 1.0,
        )
        dc = _dc_from_lowest_frequencies(strategy, coeffs[channel])
        quant_dc[channel, by:by + covered_y, bx:bx + covered_x] = (
            _round_away_from_zero(dc * inv_factor[channel] -
                                  quant_dc[1, by:by + covered_y,
                                           bx:bx + covered_x] *
                                  cfl_factor[channel]).astype(np.int16))

      num_nonzeros = np.empty(3, dtype=np.int32)
      for channel in (1, 0, 2):
        if size == K_DCT_BLOCK_SIZE:
          count = _num_nonzero_8x8_except_dc(quantized[channel])
          shifted = count
        else:
          covered_blocks = covered_x * covered_y
          count = _num_nonzero_except_llf(quantized[channel], xsize, ysize,
                                          covered_blocks)
          log2_covered_blocks = int(math.log2(covered_blocks))
          shifted = (count + covered_blocks - 1) >> log2_covered_blocks
        num_nonzeros[channel] = count
        nzeros_map[channel, by:by + covered_y, bx:bx + covered_x] = shifted

      result[(by, bx)] = QuantizedBlock(
          raw_coefficients=raw_coefficients,
          quant_input_coefficients=coeffs.copy(),
          quantized_ac=quantized,
          block_quant_dc=quant_dc[:, by:by + covered_y,
                                  bx:bx + covered_x].copy(),
          num_nonzeros=num_nonzeros,
          num_nonzeros_map=nzeros_map[:, by:by + covered_y,
                                      bx:bx + covered_x].copy(),
      )
  return result
