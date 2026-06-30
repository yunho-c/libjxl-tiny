"""Tokenization helpers mirroring `enc_frame.cc` and `enc_group.cc`."""

from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np

from .ac_strategy import (
    DCT,
    covered_blocks_x,
    covered_blocks_y,
    is_first_block,
    raw_strategy,
)
from .quantization import K_DCT_BLOCK_SIZE, quantize_ac_group


REPO_ROOT = Path(__file__).resolve().parents[2]
K_GRAD_RANGE_MIN = 0
K_GRAD_RANGE_MID = 512
K_GRAD_RANGE_MAX = 1023
K_NON_ZERO_BUCKETS = 37
K_ZERO_DENSITY_CONTEXT_COUNT = 458
K_NUM_BLOCK_CONTEXTS = 4


def _extract_int_table(path: Path, name: str) -> np.ndarray:
  text = path.read_text()
  match = re.search(rf"{re.escape(name)}[^\n=]*=\s*\{{(?P<body>.*?)\}};",
                    text, re.DOTALL)
  if match is None:
    raise RuntimeError(f"missing C++ table {name} in {path}")
  body = re.sub(r"//.*", "", match.group("body"))
  values = [
      int(token, 0)
      for token in re.findall(r"0x[0-9A-Fa-f]+|-?[0-9]+", body)
  ]
  return np.asarray(values, dtype=np.int64)


K_GRADIENT_CONTEXT_LUT = _extract_int_table(REPO_ROOT / "encoder" /
                                            "enc_frame.cc",
                                            "kGradientContextLut")
K_COEFF_ORDERS = _extract_int_table(REPO_ROOT / "encoder" / "enc_group.cc",
                                    "kCoeffOrders")
K_COEFF_FREQ_CONTEXT = _extract_int_table(REPO_ROOT / "encoder" /
                                          "ac_context.h",
                                          "kCoeffFreqContext")
K_COEFF_NUM_NONZERO_CONTEXT = _extract_int_table(REPO_ROOT / "encoder" /
                                                 "ac_context.h",
                                                 "kCoeffNumNonzeroContext")
K_BLOCK_CONTEXT_MAP = _extract_int_table(REPO_ROOT / "encoder" /
                                         "ac_context.h", "kBlockContextMap")


def pack_signed(value: int) -> int:
  value = int(value)
  if value < 0:
    return -2 * value - 1
  return 2 * value


def clamped_gradient(n: int, w: int, l: int) -> int:
  m = min(n, w)
  max_value = max(n, w)
  grad = n + w - l
  if l < m:
    return max_value
  if l > max_value:
    return m
  return grad


def _strategy_code(strategy: int) -> int:
  return (0, 6, 7)[int(strategy)]


def _block_context(channel: int, strategy_code: int) -> int:
  return int(K_BLOCK_CONTEXT_MAP[channel * 27 + strategy_code])


def _nonzero_context(nonzeros: int, block_ctx: int) -> int:
  if nonzeros < 8:
    bucket = nonzeros
  elif nonzeros >= 64:
    bucket = 36
  else:
    bucket = 4 + nonzeros // 2
  return bucket * K_NUM_BLOCK_CONTEXTS + block_ctx


def _zero_density_context(nonzeros_left: int, k: int, covered_blocks: int,
                          log2_covered_blocks: int, prev: int) -> int:
  nonzeros_left = ((nonzeros_left + covered_blocks - 1) >>
                   log2_covered_blocks)
  k >>= log2_covered_blocks
  return int((K_COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] +
              K_COEFF_FREQ_CONTEXT[k]) * 2 + prev)


def _zero_density_contexts_offset(block_ctx: int) -> int:
  return (K_NUM_BLOCK_CONTEXTS * K_NON_ZERO_BUCKETS +
          K_ZERO_DENSITY_CONTEXT_COUNT * block_ctx)


def _predict_from_top_and_left(row_top: np.ndarray | None, row: np.ndarray,
                               x: int, default_val: int) -> int:
  if x == 0:
    return default_val if row_top is None else int(row_top[x])
  if row_top is None:
    return int(row[x - 1])
  return (int(row_top[x]) + int(row[x - 1]) + 1) // 2


def dc_tokens(quant_dc: np.ndarray) -> np.ndarray:
  quant = np.asarray(quant_dc, dtype=np.int16)
  if quant.ndim != 3 or quant.shape[0] != 3:
    raise ValueError("expected channel-first quant_dc with shape (3, y, x)")
  tokens: list[tuple[int, int]] = []
  for channel in (1, 0, 2):
    plane = quant[channel]
    for y in range(plane.shape[0]):
      for x in range(plane.shape[1]):
        left = int(plane[y, x - 1]) if x else (
            int(plane[y - 1, x]) if y else 0)
        top = int(plane[y - 1, x]) if y else left
        topleft = int(plane[y - 1, x - 1]) if x and y else left
        guess = clamped_gradient(top, left, topleft)
        gradprop = max(K_GRAD_RANGE_MIN,
                       min(K_GRAD_RANGE_MAX,
                           K_GRAD_RANGE_MID + top + left - topleft))
        residual = int(plane[y, x]) - guess
        tokens.append((int(K_GRADIENT_CONTEXT_LUT[gradprop]),
                       pack_signed(residual)))
  return np.asarray(tokens, dtype=np.uint32)


def ac_metadata_tokens(ytox_map: np.ndarray, ytob_map: np.ndarray,
                       ac_strategy: np.ndarray,
                       raw_quant_field: np.ndarray) -> np.ndarray:
  ytox = np.asarray(ytox_map, dtype=np.int8)
  ytob = np.asarray(ytob_map, dtype=np.int8)
  strategy_grid = np.asarray(ac_strategy, dtype=np.uint8)
  qf = np.asarray(raw_quant_field, dtype=np.uint8)
  tokens: list[tuple[int, int]] = []

  for index, cfl_map in enumerate((ytox, ytob)):
    context = 2 - index
    for y in range(cfl_map.shape[0]):
      for x in range(cfl_map.shape[1]):
        left = int(cfl_map[y, x - 1]) if x else (
            int(cfl_map[y - 1, x]) if y else 0)
        top = int(cfl_map[y - 1, x]) if y else left
        topleft = int(cfl_map[y - 1, x - 1]) if x and y else left
        guess = clamped_gradient(top, left, topleft)
        residual = int(cfl_map[y, x]) - guess
        tokens.append((context, pack_signed(residual)))

  left = 0
  for y in range(strategy_grid.shape[0]):
    for x in range(strategy_grid.shape[1]):
      encoded = strategy_grid[y, x]
      if not bool(is_first_block(encoded)):
        continue
      cur = _strategy_code(int(raw_strategy(encoded)))
      ctx = 7 if left > 11 else 8 if left > 5 else 9 if left > 3 else 10
      tokens.append((ctx, pack_signed(cur)))
      left = cur

  left = _strategy_code(int(raw_strategy(strategy_grid[0, 0])))
  for y in range(strategy_grid.shape[0]):
    for x in range(strategy_grid.shape[1]):
      encoded = strategy_grid[y, x]
      if not bool(is_first_block(encoded)):
        continue
      cur = int(qf[y, x]) - 1
      residual = cur - left
      ctx = 3 if left > 11 else 4 if left > 5 else 5 if left > 3 else 6
      tokens.append((ctx, pack_signed(residual)))
      left = cur

  for _ in range(strategy_grid.shape[0] * strategy_grid.shape[1]):
    tokens.append((0, pack_signed(4)))

  return np.asarray(tokens, dtype=np.uint32)


def ac_tokens(xyb: np.ndarray, raw_quant_field: np.ndarray,
              ac_strategy: np.ndarray, ytox_map: np.ndarray,
              ytob_map: np.ndarray, distance: float) -> np.ndarray:
  strategy_grid = np.asarray(ac_strategy, dtype=np.uint8)
  quantized_blocks = quantize_ac_group(xyb, raw_quant_field, ac_strategy,
                                       ytox_map, ytob_map, distance)
  nzeros_map = np.zeros((3, strategy_grid.shape[0], strategy_grid.shape[1]),
                        dtype=np.uint8)
  for (by, bx), block in quantized_blocks.items():
    covered_y = block.num_nonzeros_map.shape[1]
    covered_x = block.num_nonzeros_map.shape[2]
    nzeros_map[:, by:by + covered_y, bx:bx + covered_x] = (
        block.num_nonzeros_map)
  return ac_tokens_from_quantized_blocks(strategy_grid, quantized_blocks,
                                         nzeros_map)


def ac_tokens_from_quantized_blocks(
    ac_strategy: np.ndarray,
    quantized_blocks: dict[tuple[int, int], object],
    nzeros_map: np.ndarray,
    *,
    by_offset: int = 0,
) -> np.ndarray:
  strategy_grid = np.asarray(ac_strategy, dtype=np.uint8)
  group_nzeros = np.asarray(nzeros_map, dtype=np.uint8)
  if group_nzeros.ndim != 3 or group_nzeros.shape[0] != 3:
    raise ValueError("expected channel-first nonzero map with shape (3, y, x)")
  if by_offset < 0:
    raise ValueError("by_offset must be nonnegative")
  if by_offset + strategy_grid.shape[0] > group_nzeros.shape[1]:
    raise ValueError("AC strategy rows exceed nonzero map")
  if strategy_grid.shape[1] > group_nzeros.shape[2]:
    raise ValueError("AC strategy columns exceed nonzero map")

  tokens: list[tuple[int, int]] = []
  for by in range(strategy_grid.shape[0]):
    for bx in range(strategy_grid.shape[1]):
      encoded = strategy_grid[by, bx]
      if not bool(is_first_block(encoded)):
        continue
      strategy = int(raw_strategy(encoded))
      block = quantized_blocks[(by, bx)]
      covered_x = covered_blocks_x(strategy)
      covered_y = covered_blocks_y(strategy)
      covered_blocks = covered_x * covered_y
      log2_covered_blocks = int(math.log2(covered_blocks))
      size = block.quantized_ac.shape[1]
      order_offset = 0 if strategy == DCT else K_DCT_BLOCK_SIZE
      order = K_COEFF_ORDERS[order_offset:order_offset + size]

      for channel in (1, 0, 2):
        coeffs = block.quantized_ac[channel]
        nzeros = int(block.num_nonzeros[channel])
        global_by = by_offset + by
        row_top = None if global_by == 0 else group_nzeros[channel, global_by - 1]
        row = group_nzeros[channel, global_by]
        predicted = _predict_from_top_and_left(row_top, row, bx, 32)
        block_ctx = _block_context(channel, _strategy_code(strategy))
        tokens.append((_nonzero_context(predicted, block_ctx), nzeros))

        histo_offset = _zero_density_contexts_offset(block_ctx)
        prev = 0 if nzeros > size // 16 else 1
        k = covered_blocks
        while k < size and nzeros != 0:
          coeff = int(coeffs[int(order[k])])
          ctx = histo_offset + _zero_density_context(nzeros, k,
                                                     covered_blocks,
                                                     log2_covered_blocks, prev)
          tokens.append((ctx, pack_signed(coeff)))
          prev = 1 if coeff != 0 else 0
          nzeros -= prev
          k += 1
        if nzeros != 0:
          raise RuntimeError(f"AC tokenization ended with {nzeros} nonzeros")

  return np.asarray(tokens, dtype=np.uint32)
