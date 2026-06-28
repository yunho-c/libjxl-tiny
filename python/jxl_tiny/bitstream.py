"""Bitstream serialization helpers for trace-backed section parity tests."""

from __future__ import annotations

import numpy as np

from .ac_strategy import is_first_block
from .entropy import (
    EntropyCodeTables,
    K_AC_CONTEXT_MAP,
    K_ALPHABET_SIZE,
    create_huffman_tree,
    optimize_entropy_code_from_context_values,
    optimize_prefix_code_from_context_values,
    uint_token,
    _convert_bit_depths_to_symbols,
)
from .quantization import DistanceParams
from .tokenization import REPO_ROOT, _extract_int_table, pack_signed


K_MAX_CONTEXTS = 128
K_CODE_LENGTH_CODES = 18
K_COMPACT_BLOCK_CONTEXT_MAP = _extract_int_table(
    REPO_ROOT / "encoder" / "ac_context.h", "kCompactBlockContextMap"
).astype(np.uint8)
K_CONTEXT_TREE_TOKENS = _extract_int_table(
    REPO_ROOT / "encoder" / "enc_frame.cc", "kContextTreeTokens"
).reshape(-1, 2).astype(np.uint32)


class BitWriter:
  """LSB-first byte writer matching `encoder/enc_bit_writer.h`."""

  def __init__(self) -> None:
    self._bytes = bytearray()
    self.bits_written = 0

  def write(self, n_bits: int, bits: int) -> None:
    n_bits = int(n_bits)
    bits = int(bits)
    if n_bits < 0:
      raise ValueError(f"invalid bit count: {n_bits}")
    for i in range(n_bits):
      byte_index = self.bits_written // 8
      bit_index = self.bits_written % 8
      if byte_index == len(self._bytes):
        self._bytes.append(0)
      if (bits >> i) & 1:
        self._bytes[byte_index] |= 1 << bit_index
      self.bits_written += 1

  def bytes_padded(self) -> bytes:
    return bytes(self._bytes)


def ceil_log2_nonzero(value: int) -> int:
  if value <= 1:
    return 0
  return int(value - 1).bit_length()


def uint_encode(value: int) -> tuple[int, int, int]:
  value = int(value)
  token = uint_token(value)
  if value < 16:
    return token, 0, 0
  n = value.bit_length() - 1
  m = value - (1 << n)
  nbits = n - 2
  bits = m & ((1 << nbits) - 1)
  return token, nbits, bits


def write_token(token: tuple[int, int] | np.ndarray, code: EntropyCodeTables,
                writer: BitWriter) -> None:
  context, value = (int(token[0]), int(token[1]))
  mapped_context = int(code.context_map[context])
  tok, nbits, bits = uint_encode(value)
  depth = int(code.prefix_depths[mapped_context, tok])
  prefix = int(code.prefix_bits[mapped_context, tok])
  writer.write(depth + nbits, prefix | (bits << depth))


def _reverse(values: list[int], start: int) -> None:
  values[start:] = reversed(values[start:])


def _write_huffman_tree_repetitions(previous_value: int, value: int,
                                    repetitions: int, tree: list[int],
                                    extra: list[int]) -> None:
  if previous_value != value:
    tree.append(value)
    extra.append(0)
    repetitions -= 1
  if repetitions == 7:
    tree.append(value)
    extra.append(0)
    repetitions -= 1
  if repetitions < 3:
    for _ in range(repetitions):
      tree.append(value)
      extra.append(0)
    return
  repetitions -= 3
  start = len(tree)
  while True:
    tree.append(16)
    extra.append(repetitions & 0x3)
    repetitions >>= 2
    if repetitions == 0:
      break
    repetitions -= 1
  _reverse(tree, start)
  _reverse(extra, start)


def _write_huffman_tree_repetitions_zeros(repetitions: int, tree: list[int],
                                          extra: list[int]) -> None:
  if repetitions == 11:
    tree.append(0)
    extra.append(0)
    repetitions -= 1
  if repetitions < 3:
    for _ in range(repetitions):
      tree.append(0)
      extra.append(0)
    return
  repetitions -= 3
  start = len(tree)
  while True:
    tree.append(17)
    extra.append(repetitions & 0x7)
    repetitions >>= 3
    if repetitions == 0:
      break
    repetitions -= 1
  _reverse(tree, start)
  _reverse(extra, start)


def _decide_over_rle_use(depths: np.ndarray, length: int) -> tuple[bool, bool]:
  total_reps_zero = 0
  total_reps_non_zero = 0
  count_reps_zero = 1
  count_reps_non_zero = 1
  i = 0
  while i < length:
    value = int(depths[i])
    reps = 1
    while i + reps < length and int(depths[i + reps]) == value:
      reps += 1
    if reps >= 3 and value == 0:
      total_reps_zero += reps
      count_reps_zero += 1
    if reps >= 4 and value != 0:
      total_reps_non_zero += reps
      count_reps_non_zero += 1
    i += reps
  return (total_reps_non_zero > count_reps_non_zero * 2,
          total_reps_zero > count_reps_zero * 2)


def _write_huffman_tree(depths: np.ndarray, length: int
                        ) -> tuple[list[int], list[int]]:
  previous_value = 8
  new_length = int(length)
  while new_length > 0 and int(depths[new_length - 1]) == 0:
    new_length -= 1
  use_rle_for_non_zero = False
  use_rle_for_zero = False
  if length > 50:
    use_rle_for_non_zero, use_rle_for_zero = _decide_over_rle_use(
        depths, new_length)

  tree: list[int] = []
  extra: list[int] = []
  i = 0
  while i < new_length:
    value = int(depths[i])
    reps = 1
    if (value != 0 and use_rle_for_non_zero) or (
        value == 0 and use_rle_for_zero):
      while i + reps < new_length and int(depths[i + reps]) == value:
        reps += 1
    if value == 0:
      _write_huffman_tree_repetitions_zeros(reps, tree, extra)
    else:
      _write_huffman_tree_repetitions(previous_value, value, reps, tree, extra)
      previous_value = value
    i += reps
  return tree, extra


def _store_huffman_tree_of_huffman_tree(num_codes: int,
                                        code_length_depths: np.ndarray,
                                        writer: BitWriter) -> None:
  storage_order = (1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14,
                   15)
  code_symbols = (0, 7, 3, 2, 1, 15)
  code_bit_lengths = (2, 4, 3, 2, 2, 4)
  codes_to_store = K_CODE_LENGTH_CODES
  if num_codes > 1:
    while codes_to_store > 0:
      if int(code_length_depths[storage_order[codes_to_store - 1]]) != 0:
        break
      codes_to_store -= 1
  skip_some = 0
  if (int(code_length_depths[storage_order[0]]) == 0 and
      int(code_length_depths[storage_order[1]]) == 0):
    skip_some = 2
    if int(code_length_depths[storage_order[2]]) == 0:
      skip_some = 3
  writer.write(2, skip_some)
  for i in range(skip_some, codes_to_store):
    length = int(code_length_depths[storage_order[i]])
    writer.write(code_bit_lengths[length], code_symbols[length])


def _store_huffman_tree(depths: np.ndarray, num_symbols: int,
                        writer: BitWriter) -> None:
  tree, extra = _write_huffman_tree(depths, num_symbols)
  histogram = np.zeros(K_CODE_LENGTH_CODES, dtype=np.uint32)
  for symbol in tree:
    histogram[symbol] += 1

  num_codes = 0
  code = 0
  for i in range(K_CODE_LENGTH_CODES):
    if int(histogram[i]) != 0:
      if num_codes == 0:
        code = i
        num_codes = 1
      elif num_codes == 1:
        num_codes = 2
        break

  code_length_depths = create_huffman_tree(histogram, K_CODE_LENGTH_CODES, 5)
  code_length_symbols = _convert_bit_depths_to_symbols(
      code_length_depths, K_CODE_LENGTH_CODES)
  _store_huffman_tree_of_huffman_tree(num_codes, code_length_depths, writer)
  if num_codes == 1:
    code_length_depths[code] = 0
  for symbol, extra_bits in zip(tree, extra):
    writer.write(int(code_length_depths[symbol]),
                 int(code_length_symbols[symbol]))
    if symbol == 16:
      writer.write(2, extra_bits)
    elif symbol == 17:
      writer.write(3, extra_bits)


def _store_simple_huffman_tree(depths: np.ndarray, symbols: list[int],
                               max_bits: int, writer: BitWriter) -> None:
  symbols = list(symbols)
  for i in range(len(symbols)):
    for j in range(i + 1, len(symbols)):
      if int(depths[symbols[j]]) < int(depths[symbols[i]]):
        symbols[j], symbols[i] = symbols[i], symbols[j]
  writer.write(2, 1)
  writer.write(2, len(symbols) - 1)
  for symbol in symbols:
    writer.write(max_bits, symbol)
  if len(symbols) == 4:
    writer.write(1, 1 if int(depths[symbols[0]]) == 1 else 0)


def _write_prefix_code(depths: np.ndarray, writer: BitWriter) -> None:
  symbols: list[int] = []
  length = 0
  count = 0
  for i in range(K_ALPHABET_SIZE):
    if int(depths[i]) != 0:
      if count < 4:
        symbols.append(i)
      count += 1
      length = i + 1

  max_bits = int(length - 1).bit_length()
  if count <= 1:
    writer.write(4, 1)
    writer.write(max_bits, symbols[0] if symbols else 0)
  elif count <= 4:
    _store_simple_huffman_tree(depths, symbols, max_bits, writer)
  else:
    _store_huffman_tree(depths, length, writer)


def _store_var_len_uint16(value: int, writer: BitWriter) -> None:
  value = int(value)
  if value == 0:
    writer.write(1, 0)
    return
  writer.write(1, 1)
  nbits = value.bit_length() - 1
  writer.write(4, nbits)
  writer.write(nbits, value - (1 << nbits))


def write_prefix_codes(prefix_depths: np.ndarray, writer: BitWriter) -> None:
  depths = np.asarray(prefix_depths, dtype=np.uint8)
  writer.write(1, 1)
  for _ in range(depths.shape[0]):
    writer.write(4, 4)
    writer.write(3, 2)
    writer.write(2, 0)
  num_symbols: list[int] = []
  for code_depths in depths:
    num_symbol = 1
    for i in range(K_ALPHABET_SIZE):
      if int(code_depths[i]) != 0:
        num_symbol = i + 1
    num_symbols.append(num_symbol)
    _store_var_len_uint16(num_symbol - 1, writer)
  for code_depths, num_symbol in zip(depths, num_symbols):
    if num_symbol > 1:
      _write_prefix_code(code_depths, writer)


def write_context_map(code: EntropyCodeTables, writer: BitWriter) -> None:
  context_map = np.asarray(code.context_map, dtype=np.uint8)
  num_contexts = (int(code.orig_num_contexts) if code.orig_context_map is not None
                  else int(context_map.shape[0]))
  if num_contexts == 0:
    return
  if int(np.max(context_map)) == 0:
    writer.write(3, 1)
    return
  writer.write(3, 0)
  if code.orig_context_map is not None:
    orig = np.asarray(code.orig_context_map, dtype=np.uint8)
    tokens = np.asarray([(0, int(context_map[int(orig[i])]))
                         for i in range(num_contexts)],
                        dtype=np.uint32)
  else:
    tokens = np.asarray([(0, int(context_map[i]))
                         for i in range(context_map.shape[0])],
                        dtype=np.uint32)
  depths, bits = optimize_prefix_code_from_context_values(tokens, 1,
                                                          np.zeros(1,
                                                                   dtype=np.uint8))
  write_prefix_codes(depths, writer)
  ctxmap_code = EntropyCodeTables(context_map=np.zeros(1, dtype=np.uint8),
                                  prefix_depths=depths,
                                  prefix_bits=bits)
  for token in tokens:
    write_token(token, ctxmap_code, writer)


def write_entropy_code(code: EntropyCodeTables, writer: BitWriter) -> None:
  write_context_map(code, writer)
  write_prefix_codes(code.prefix_depths, writer)


def _context_map_code(context_map: np.ndarray) -> EntropyCodeTables:
  return EntropyCodeTables(context_map=np.asarray(context_map, dtype=np.uint8),
                           prefix_depths=np.zeros((0, K_ALPHABET_SIZE),
                                                  dtype=np.uint8),
                           prefix_bits=np.zeros((0, K_ALPHABET_SIZE),
                                                dtype=np.uint16))


def write_quant_scales(global_scale: int, quant_dc: int,
                       writer: BitWriter) -> None:
  if global_scale < 2049:
    writer.write(2, 0)
    writer.write(11, global_scale - 1)
  elif global_scale < 4097:
    writer.write(2, 1)
    writer.write(11, global_scale - 2049)
  elif global_scale < 8193:
    writer.write(2, 2)
    writer.write(12, global_scale - 4097)
  else:
    writer.write(2, 3)
    writer.write(16, global_scale - 8193)

  if quant_dc == 16:
    writer.write(2, 0)
  elif quant_dc < 33:
    writer.write(2, 1)
    writer.write(5, quant_dc - 1)
  elif quant_dc < 257:
    writer.write(2, 2)
    writer.write(8, quant_dc - 1)
  else:
    writer.write(2, 3)
    writer.write(16, quant_dc - 1)


def write_context_tree(num_dc_groups: int, writer: BitWriter) -> None:
  tokens = K_CONTEXT_TREE_TOKENS.copy()
  tokens[1, 1] = pack_signed(1 + int(num_dc_groups))
  code = optimize_entropy_code_from_context_values(tokens, 6)
  writer.write(1, 1)
  writer.write(1, 0)
  write_entropy_code(code, writer)
  for token in tokens:
    write_token(token, code, writer)


def dc_global_section(dist: DistanceParams, num_dc_groups: int,
                      dc_code: EntropyCodeTables) -> bytes:
  writer = BitWriter()
  writer.write(1, 1)
  write_quant_scales(dist.global_scale, dist.quant_dc, writer)
  writer.write(1, 0)
  writer.write(16, 0)
  write_context_map(_context_map_code(K_COMPACT_BLOCK_CONTEXT_MAP), writer)
  writer.write(1, 1)
  write_context_tree(num_dc_groups, writer)
  writer.write(1, 0)
  write_entropy_code(dc_code, writer)
  return writer.bytes_padded()


def ac_global_section(num_groups: int, ac_code: EntropyCodeTables) -> bytes:
  writer = BitWriter()
  writer.write(1, 1)
  num_histo_bits = ceil_log2_nonzero(num_groups)
  if num_histo_bits != 0:
    writer.write(num_histo_bits, 0)
  writer.write(2, 3)
  writer.write(13, 0)
  writer.write(1, 0)
  write_entropy_code(ac_code, writer)
  return writer.bytes_padded()


def _num_ac_blocks(ac_strategy: np.ndarray) -> int:
  strategy_grid = np.asarray(ac_strategy, dtype=np.uint8)
  return int(np.count_nonzero(is_first_block(strategy_grid)))


def dc_group_section(dc_tokens: np.ndarray, ac_metadata_tokens: np.ndarray,
                     ac_strategy: np.ndarray,
                     dc_code: EntropyCodeTables) -> bytes:
  strategy_grid = np.asarray(ac_strategy, dtype=np.uint8)
  num_blocks = int(strategy_grid.size)
  num_ac_blocks = _num_ac_blocks(strategy_grid)
  nb_bits = ceil_log2_nonzero(num_blocks)
  staged: list[tuple[int, int]] = [(K_MAX_CONTEXTS + 6, 12)]
  staged.extend((int(context), int(value)) for context, value in dc_tokens)
  if nb_bits != 0:
    staged.append((K_MAX_CONTEXTS + nb_bits, num_ac_blocks - 1))
  staged.append((K_MAX_CONTEXTS + 4, 3))
  staged.extend((int(context), int(value)) for context, value in ac_metadata_tokens)
  return serialize_optimized_section(staged, dc_code)


def ac_group_section(ac_tokens: np.ndarray, ac_code: EntropyCodeTables) -> bytes:
  logical = np.asarray(ac_tokens, dtype=np.uint32)
  staged = [(int(K_AC_CONTEXT_MAP[int(context)]), int(value))
            for context, value in logical]
  return serialize_optimized_section(staged, ac_code)


def serialize_optimized_section(staged_tokens: list[tuple[int, int]],
                                code: EntropyCodeTables) -> bytes:
  writer = BitWriter()
  for context, value in staged_tokens:
    if context >= K_MAX_CONTEXTS:
      writer.write(context - K_MAX_CONTEXTS, value)
    else:
      write_token((context, value), code, writer)
  return writer.bytes_padded()
