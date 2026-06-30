"""Perceptual adaptive quantization for the educational encoder.

Adaptive quantization estimates how much AC detail each 8x8 block can lose
before the loss becomes visible. The Python version keeps the same intermediate
maps as `encoder/enc_adaptive_quantization.cc`: a floating AQ map, a masking
map used by AC-strategy scoring, and the byte-valued raw quant field that is
serialized later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


BLOCK_DIM = 8
K_AC_QUANT_FIELD = np.float32(0.8294)


@dataclass(frozen=True)
class AdaptiveQuantizationResult:
  aq_map: np.ndarray
  mask: np.ndarray
  raw_quant_field: np.ndarray


def _f32(value: float | np.float32) -> np.float32:
  return np.float32(value)


def _float_to_int32(value: np.float32) -> np.int32:
  return np.asarray(value, dtype=np.float32).view(np.int32)[()]


def _int32_to_float(value: np.int32) -> np.float32:
  return np.asarray(value, dtype=np.int32).view(np.float32)[()]


def fast_log2f(x: float | np.float32) -> np.float32:
  """Approximate log2 from `encoder/fast_math-inl.h`."""
  x = _f32(x)
  p = (
      _f32(-1.8503833400518310e-06),
      _f32(1.4287160470083755e00),
      _f32(7.4245873327820566e-01),
  )
  q = (
      _f32(9.9032814277590719e-01),
      _f32(1.0096718572241148e00),
      _f32(1.7409343003366853e-01),
  )
  x_bits = _float_to_int32(x)
  exp_bits = np.int32(x_bits - np.int32(0x3f2aaaab))
  exp_shifted = np.int32(exp_bits >> np.int32(23))
  mantissa = _int32_to_float(np.int32(x_bits - np.int32(exp_shifted << 23)))
  reduced = _f32(mantissa - _f32(1.0))
  yp = _f32(_f32(p[2] * reduced) + p[1])
  yq = _f32(_f32(q[2] * reduced) + q[1])
  yp = _f32(_f32(yp * reduced) + p[0])
  yq = _f32(_f32(yq * reduced) + q[0])
  return _f32(_f32(yp / yq) + _f32(exp_shifted))


def fast_pow2f(x: float | np.float32) -> np.float32:
  """Approximate exp2 from `encoder/fast_math-inl.h`."""
  x = _f32(x)
  floorx = math.floor(float(x))
  exp_bits = np.int32(np.int32(floorx + 127) << np.int32(23))
  exp = _int32_to_float(exp_bits)
  frac = _f32(x - _f32(floorx))
  num = _f32(frac + _f32(1.01749063e01))
  num = _f32(_f32(num * frac) + _f32(4.88687798e01))
  num = _f32(_f32(num * frac) + _f32(9.85506591e01))
  num = _f32(num * exp)
  den = _f32(_f32(frac * _f32(2.10242958e-01)) + _f32(-2.22328856e-02))
  den = _f32(_f32(den * frac) + _f32(-1.94414990e01))
  den = _f32(_f32(den * frac) + _f32(9.85506633e01))
  return _f32(num / den)


def _ratio_cubic_root_to_simple_gamma(value: np.float32,
                                      invert: bool = False) -> np.float32:
  k_epsilon = _f32(1e-2)
  k_sg_mul = _f32(226.0480446705883)
  k_sg_mul2 = _f32(1.0 / 73.377132366608819)
  k_log2 = _f32(0.693147181)
  k_sg_ret_mul = _f32(k_sg_mul2 * _f32(18.6580932135) * k_log2)
  k_sg_v_offset = _f32(7.14672470003)

  v = max(_f32(value), _f32(0.0))
  num_mul = _f32(k_sg_ret_mul * _f32(3.0) * k_sg_mul)
  v_offset = _f32(_f32(k_sg_v_offset * k_log2) + k_epsilon)
  den_mul = _f32(k_log2 * k_sg_mul)
  v2 = _f32(v * v)
  num = _f32(_f32(num_mul * v2) + k_epsilon)
  den = _f32(_f32(_f32(den_mul * v) * v2) + v_offset)
  return _f32(num / den) if invert else _f32(den / num)


def _masking_sqrt(value: np.float32) -> np.float32:
  k_log_offset = _f32(26.481471032459346)
  k_mul = _f32(211.50759899638012)
  mul_v = _f32(k_mul * _f32(1e8))
  return _f32(_f32(0.25) * np.sqrt(_f32(_f32(value * np.sqrt(mul_v)) +
                                        k_log_offset), dtype=np.float32))


def _compute_mask(out_val: np.float32) -> np.float32:
  k_base = _f32(-0.74174993)
  k_mul4 = _f32(3.2353257320940401)
  k_mul2 = _f32(12.906028311180409)
  k_offset2 = _f32(305.04035728311436)
  k_mul3 = _f32(5.0220313103171232)
  k_offset3 = _f32(2.1925739705298404)
  k_offset4 = _f32(_f32(0.25) * k_offset3)
  k_mul0 = _f32(0.74760422233706747)

  v1 = max(_f32(out_val * k_mul0), _f32(1e-3))
  v2 = _f32(_f32(1.0) / _f32(v1 + k_offset2))
  v3 = _f32(_f32(1.0) / _f32(_f32(v1 * v1) + k_offset3))
  v4 = _f32(_f32(1.0) / _f32(_f32(v1 * v1) + k_offset4))
  return _f32(k_base + _f32(k_mul4 * v4) + _f32(k_mul2 * v2) +
              _f32(k_mul3 * v3))


def _compute_mask_for_ac_strategy_use(out_val: np.float32) -> np.float32:
  return _f32(_f32(1.0) / _f32(out_val + _f32(0.001)))


def _hf_modulation(xyb_y: np.ndarray, block_x: int, block_y: int,
                   out_val: np.float32) -> np.float32:
  total = _f32(0.0)
  for dy in range(BLOCK_DIM):
    y = block_y + dy
    for dx in range(BLOCK_DIM):
      x = block_x + dx
      if dx != BLOCK_DIM - 1:
        total = _f32(total + abs(_f32(xyb_y[y, x] - xyb_y[y, x + 1])))
      if dy != BLOCK_DIM - 1:
        total = _f32(total + abs(_f32(xyb_y[y, x] - xyb_y[y + 1, x])))
  return _f32(_f32(total * _f32(-2.0052193233688884 / 112.0)) + out_val)


def _color_modulation(xyb: np.ndarray, block_x: int, block_y: int,
                      distance: float, out_val: np.float32) -> np.float32:
  k_strength_mul = _f32(2.177823400325309)
  k_red_ramp_start = _f32(0.0073200141118951231)
  k_red_ramp_length = _f32(0.019421555948474039)
  k_blue_ramp_length = _f32(0.086890611400405895)
  k_blue_ramp_start = _f32(0.26973418507870539)
  strength = _f32(k_strength_mul * _f32(_f32(1.0) - _f32(0.25) * distance))
  if strength < 0:
    return out_val

  out_val = _f32(out_val + _f32(strength * _f32(-0.009174542291185913)))
  red_coverage = _f32(0.0)
  blue_coverage = _f32(0.0)
  for dy in range(BLOCK_DIM):
    y = block_y + dy
    for dx in range(BLOCK_DIM):
      x = block_x + dx
      pixel_x = max(_f32(0.0), _f32(xyb[0, y, x] - k_red_ramp_start))
      pixel_y = _f32(xyb[1, y, x])
      pixel_b = max(_f32(0.0),
                    _f32(xyb[2, y, x] - _f32(pixel_y + k_blue_ramp_start)))
      red_coverage = _f32(red_coverage + min(pixel_x, k_red_ramp_length))
      blue_coverage = _f32(blue_coverage + min(pixel_b, k_blue_ramp_length))

  ratio = _f32(30.610615782142737)
  red_strength = _f32(strength * _f32(5.992297772961519))
  blue_strength = strength
  overall_red = min(red_coverage, _f32(ratio * k_red_ramp_length))
  overall_red = _f32(overall_red * _f32(red_strength / ratio))
  overall_blue = min(blue_coverage, _f32(ratio * k_blue_ramp_length))
  overall_blue = _f32(overall_blue * _f32(blue_strength / ratio))
  return _f32(overall_red + overall_blue + out_val)


def _gamma_modulation(xyb: np.ndarray, block_x: int, block_y: int,
                      out_val: np.float32) -> np.float32:
  overall_ratio = _f32(0.0)
  for dy in range(BLOCK_DIM):
    y = block_y + dy
    for dx in range(BLOCK_DIM):
      x = block_x + dx
      iny = _f32(xyb[1, y, x] + _f32(0.16))
      inx = _f32(xyb[0, y, x])
      r = _f32(iny - inx)
      g = _f32(iny + inx)
      ratio_r = _ratio_cubic_root_to_simple_gamma(r, invert=True)
      ratio_g = _ratio_cubic_root_to_simple_gamma(g, invert=True)
      overall_ratio = _f32(overall_ratio +
                           _f32(_f32(0.5) * _f32(ratio_r + ratio_g)))
  overall_ratio = _f32(overall_ratio * _f32(1.0 / 64.0))
  k_gam = _f32(-0.15526878023684174 * 0.693147180559945)
  return _f32(_f32(k_gam * fast_log2f(overall_ratio)) + out_val)


def _per_block_modulations(distance: float, xyb: np.ndarray, scale: np.float32,
                           aq_map: np.ndarray, block_x0: int = 0,
                           block_y0: int = 0) -> None:
  """Fold perceptual per-block heuristics into the erosion-derived AQ map."""
  base_level = _f32(_f32(0.5) * scale)
  dampen = _f32(1.0)
  if distance >= 7.0:
    dampen = _f32(1.0 - ((distance - 7.0) / (14.0 - 7.0)))
    if dampen < 0:
      dampen = _f32(0.0)
  mul = _f32(scale * dampen)
  add = _f32(_f32(_f32(1.0) - dampen) * base_level)

  y_blocks, x_blocks = aq_map.shape
  for iy in range(y_blocks):
    block_y = (block_y0 + iy) * BLOCK_DIM
    for ix in range(x_blocks):
      block_x = (block_x0 + ix) * BLOCK_DIM
      out_val = _compute_mask(_f32(aq_map[iy, ix]))
      out_val = _hf_modulation(xyb[1], block_x, block_y, out_val)
      out_val = _color_modulation(xyb, block_x, block_y, distance, out_val)
      out_val = _gamma_modulation(xyb, block_x, block_y, out_val)
      aq_map[iy, ix] = _f32(_f32(fast_pow2f(_f32(out_val * _f32(1.442695041))) *
                                 mul) + add)


def _store_min4(value: np.float32, mins: list[np.float32]) -> None:
  if value < mins[3]:
    if value < mins[0]:
      mins[3] = mins[2]
      mins[2] = mins[1]
      mins[1] = mins[0]
      mins[0] = value
    elif value < mins[1]:
      mins[3] = mins[2]
      mins[2] = mins[1]
      mins[1] = value
    elif value < mins[2]:
      mins[3] = mins[2]
      mins[2] = value
    else:
      mins[3] = value


def _fuzzy_erosion(pre_erosion: np.ndarray, from_x0: int, from_y0: int,
                   width: int, height: int) -> np.ndarray:
  out = np.zeros((height // 2, width // 2), dtype=np.float32)
  ysize, xsize = pre_erosion.shape
  for fy in range(height):
    y = fy + from_y0
    ym1 = y - 1 if y >= 1 else y
    yp1 = y + 1 if y + 1 < ysize else y
    for fx in range(width):
      x = fx + from_x0
      xm1 = x - 1 if x >= 1 else x
      xp1 = x + 1 if x + 1 < xsize else x
      mins = [
          _f32(pre_erosion[y, x]),
          _f32(pre_erosion[y, xm1]),
          _f32(pre_erosion[y, xp1]),
          _f32(pre_erosion[ym1, xm1]),
      ]
      mins.sort()
      _store_min4(_f32(pre_erosion[ym1, x]), mins)
      _store_min4(_f32(pre_erosion[ym1, xp1]), mins)
      _store_min4(_f32(pre_erosion[yp1, xm1]), mins)
      _store_min4(_f32(pre_erosion[yp1, x]), mins)
      _store_min4(_f32(pre_erosion[yp1, xp1]), mins)
      value = _f32(_f32(0.05) * _f32(pre_erosion[y, x] + mins[0] + mins[1] +
                                      mins[2] + mins[3]))
      oy = fy // 2
      ox = fx // 2
      if (fx % 2) == 0 and (fy % 2) == 0:
        out[oy, ox] = value
      else:
        out[oy, ox] = _f32(out[oy, ox] + value)
  return out


def quant_dc(distance: float) -> np.float32:
  k_dc_quant_pow = 0.57
  k_dc_quant = 1.12
  k_dc_mul = 2.9
  effective_dist = k_dc_mul * ((distance / k_dc_mul) ** k_dc_quant_pow)
  effective_dist = min(max(effective_dist, 0.5 * distance), distance)
  return _f32(min(k_dc_quant / effective_dist, 50.0))


def inverse_global_ac_scale(distance: float) -> np.float32:
  k_global_scale_denom = 1 << 16
  k_ac_quant = 0.8
  k_quant_field_target = 5
  scale = k_global_scale_denom * k_ac_quant / (distance * k_quant_field_target)
  scale = min(max(scale, 1.0), float(1 << 15))
  qdc = quant_dc(distance)
  scaled_quant_dc = int(qdc * 4096 * 1.6)
  global_scale = min(max(int(scale), 1), scaled_quant_dc)
  return _f32(1.0 / _f32(global_scale * (1.0 / k_global_scale_denom)))


def compute_adaptive_quantization(
    xyb: np.ndarray, distance: float,
    inv_scale: float | np.float32 | None = None,
    block_x0: int = 0,
    block_y0: int = 0,
    block_width: int | None = None,
    block_height: int | None = None) -> AdaptiveQuantizationResult:
  """Compute AQ outputs for a block rectangle inside padded XYB data.

  The input is channel-first XYB data whose dimensions are multiples of 8.
  `block_x0`, `block_y0`, `block_width`, and `block_height` describe the tile
  rectangle in 8x8-block coordinates. The pixel pass expands non-edge
  rectangles by four pixels on each side, matching the small C++ halo used by
  `ComputeAdaptiveQuantFieldTile` before fuzzy erosion.
  """
  if xyb.ndim != 3 or xyb.shape[0] != 3:
    raise ValueError("expected channel-first XYB image with shape (3, y, x)")
  if xyb.shape[1] % BLOCK_DIM != 0 or xyb.shape[2] % BLOCK_DIM != 0:
    raise ValueError("xyb dimensions must be multiples of 8")

  source = np.asarray(xyb, dtype=np.float32)
  ysize = source.shape[1]
  xsize = source.shape[2]
  y_blocks = ysize // BLOCK_DIM
  x_blocks = xsize // BLOCK_DIM
  if block_width is None:
    block_width = x_blocks - block_x0
  if block_height is None:
    block_height = y_blocks - block_y0
  if block_x0 < 0 or block_y0 < 0 or block_width <= 0 or block_height <= 0:
    raise ValueError("invalid adaptive quantization block rectangle")
  if block_x0 + block_width > x_blocks or block_y0 + block_height > y_blocks:
    raise ValueError("adaptive quantization block rectangle exceeds image")

  scale = _f32(K_AC_QUANT_FIELD / _f32(distance))
  match_gamma_offset = _f32(0.019)
  k_x_mul = _f32(23.426802998210313)

  x0 = block_x0 * BLOCK_DIM
  x1 = x0 + block_width * BLOCK_DIM
  y_start = block_y0 * BLOCK_DIM
  y_end = y_start + block_height * BLOCK_DIM
  if x0 != 0:
    x0 -= 4
  if x1 != xsize:
    x1 += 4
  if y_start != 0:
    y_start -= 4
  if y_end != ysize:
    y_end += 4
  pre_erosion = np.zeros(((y_end - y_start) // 4, (x1 - x0) // 4),
                         dtype=np.float32)
  diff_buffer = np.zeros(x1 - x0, dtype=np.float32)

  # Build a quarter-resolution contrast map from local Y/X differences. Fuzzy
  # erosion below then spreads low-detail masking limits to neighboring blocks.
  for y in range(y_start, y_end):
    y2 = y + 1 if y + 1 < ysize else y
    y1 = y - 1 if y > 0 else y
    for x in range(x0, x1):
      x2 = x + 1 if x + 1 < xsize else x
      x1_neighbor = x - 1 if x > 0 else x
      base = _f32(_f32(0.25) * _f32(source[1, y2, x] + source[1, y1, x] +
                                    source[1, y, x1_neighbor] +
                                    source[1, y, x2]))
      gammac = _ratio_cubic_root_to_simple_gamma(
          _f32(source[1, y, x] + match_gamma_offset))
      diff = _f32(gammac * _f32(source[1, y, x] - base))
      diff = _f32(diff * diff)
      base_x = _f32(_f32(0.25) * _f32(source[0, y2, x] + source[0, y1, x] +
                                      source[0, y, x1_neighbor] +
                                      source[0, y, x2]))
      diff_x = _f32(gammac * _f32(source[0, y, x] - base_x))
      diff_x = _f32(diff_x * diff_x)
      diff = _f32(diff + _f32(k_x_mul * diff_x))
      diff = _masking_sqrt(diff)
      if (y % 4) != 0:
        diff_buffer[x - x0] = _f32(diff_buffer[x - x0] + diff)
      else:
        diff_buffer[x - x0] = diff

    if (y % 4) == 3:
      out_y = (y - y_start) // 4
      for px in range((x1 - x0) // 4):
        pre_erosion[out_y, px] = _f32(_f32(diff_buffer[px * 4] +
                                           diff_buffer[px * 4 + 1] +
                                           diff_buffer[px * 4 + 2] +
                                           diff_buffer[px * 4 + 3]) *
                                      _f32(0.25))

  from_x0 = 0 if (x0 % BLOCK_DIM) == 0 else 1
  from_y0 = 0 if (y_start % BLOCK_DIM) == 0 else 1
  aq_map = _fuzzy_erosion(pre_erosion, from_x0, from_y0, block_width * 2,
                          block_height * 2)
  mask = np.empty_like(aq_map)
  # AC-strategy scoring uses the pre-modulation mask, while final quantization
  # uses the modulated AQ map converted to byte quant-field values.
  for y in range(block_height):
    for x in range(block_width):
      mask[y, x] = _compute_mask_for_ac_strategy_use(_f32(aq_map[y, x]))

  _per_block_modulations(distance, source, scale, aq_map, block_x0, block_y0)
  if inv_scale is None:
    inv_scale = inverse_global_ac_scale(distance)
  raw = np.clip(np.floor(aq_map * _f32(inv_scale) + _f32(0.5)), 1,
                255).astype(np.uint8)
  return AdaptiveQuantizationResult(aq_map=aq_map, mask=mask,
                                    raw_quant_field=raw)
