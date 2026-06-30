"""Build context maps and Huffman prefix codes for encoder tokens.

Tokenization produces `(context, value)` pairs, but the bitstream stores those
values through clustered entropy contexts and canonical prefix codes. This
module mirrors the compact optimizer from `encoder/enc_entropy_code.cc` so the
Python port can reproduce section bytes, not just higher-level token streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tokenization import REPO_ROOT, _extract_int_table


K_ALPHABET_SIZE = 64
K_CLUSTERS_LIMIT = 8
K_MIN_DISTANCE_FOR_DISTINCT = 64

K_AC_CONTEXT_MAP = _extract_int_table(REPO_ROOT / "encoder" /
                                      "static_entropy_codes.h",
                                      "kACContextMap").astype(np.uint8)


@dataclass(frozen=True)
class EntropyCodeTables:
  """Prefix-code tables plus the token-context to prefix-code mapping."""

  context_map: np.ndarray
  prefix_depths: np.ndarray
  prefix_bits: np.ndarray
  orig_context_map: np.ndarray | None = None
  orig_num_contexts: int | None = None


def uint_token(value: int) -> int:
  value = int(value)
  if value < 16:
    return value
  n = value.bit_length() - 1
  m = value - (1 << n)
  return (n << 2) + (m >> (n - 2))


def _reverse_bits(num_bits: int, bits: int) -> int:
  out = 0
  for _ in range(num_bits):
    out = (out << 1) | (bits & 1)
    bits >>= 1
  return out


def create_huffman_tree(counts: np.ndarray, length: int,
                        tree_limit: int) -> np.ndarray:
  counts = np.asarray(counts, dtype=np.uint32)
  depth = np.zeros(K_ALPHABET_SIZE, dtype=np.uint8)
  count_limit = 1
  while True:
    tree: list[dict[str, int]] = []
    for i in range(length - 1, -1, -1):
      if counts[i] != 0:
        tree.append({
            "total": max(int(counts[i]), count_limit - 1),
            "left": -1,
            "right": i,
        })

    n = len(tree)
    if n == 1:
      depth[int(tree[0]["right"])] = 1
      break

    tree.sort(key=lambda node: node["total"])
    sentinel = {"total": (1 << 32) - 1, "left": -1, "right": -1}
    tree.append(dict(sentinel))
    tree.append(dict(sentinel))
    i = 0
    j = n + 1
    for k in range(n - 1, 0, -1):
      if tree[i]["total"] <= tree[j]["total"]:
        left = i
        i += 1
      else:
        left = j
        j += 1
      if tree[i]["total"] <= tree[j]["total"]:
        right = i
        i += 1
      else:
        right = j
        j += 1

      j_end = len(tree) - 1
      tree[j_end] = {
          "total": tree[left]["total"] + tree[right]["total"],
          "left": left,
          "right": right,
      }
      tree.append(dict(sentinel))

    depth = np.zeros(K_ALPHABET_SIZE, dtype=np.uint8)

    def set_depth(index: int, level: int) -> None:
      node = tree[index]
      if node["left"] >= 0:
        set_depth(node["left"], level + 1)
        set_depth(node["right"], level + 1)
      else:
        depth[node["right"]] = level

    set_depth(2 * n - 1, 0)
    if int(np.max(depth[:length])) <= tree_limit:
      break
    count_limit *= 2
  return depth


def _convert_bit_depths_to_symbols(depth: np.ndarray,
                                   length: int) -> np.ndarray:
  bits = np.zeros(K_ALPHABET_SIZE, dtype=np.uint16)
  bl_count = [0] * 16
  for i in range(length):
    bl_count[int(depth[i])] += 1
  bl_count[0] = 0
  next_code = [0] * 16
  code = 0
  for i in range(1, 16):
    code = (code + bl_count[i - 1]) << 1
    next_code[i] = code
  for i in range(length):
    d = int(depth[i])
    if d:
      bits[i] = _reverse_bits(d, next_code[d])
      next_code[d] += 1
  return bits


def _histogram_bit_cost(counts: np.ndarray) -> int:
  total = int(np.sum(counts, dtype=np.uint64))
  if total == 0:
    return 0
  depths = create_huffman_tree(counts, K_ALPHABET_SIZE, 15)
  return int(np.sum(counts.astype(np.uint64) * depths.astype(np.uint64)))


def _histogram_distance(a: np.ndarray, b: np.ndarray, bit_cost_a: int,
                        bit_cost_b: int) -> int:
  if int(np.sum(a, dtype=np.uint64)) == 0 or int(np.sum(b, dtype=np.uint64)) == 0:
    return 0
  combined = a + b
  return _histogram_bit_cost(combined) - bit_cost_a - bit_cost_b


def _cluster_histograms(histograms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  if histograms.shape[0] <= 1:
    return histograms.copy(), np.arange(histograms.shape[0], dtype=np.uint8)

  max_histograms = min(K_CLUSTERS_LIMIT, histograms.shape[0])
  in_hist = histograms.copy()
  out: list[np.ndarray] = []
  histogram_symbols = [max_histograms] * in_hist.shape[0]
  dists = [float("inf")] * in_hist.shape[0]
  bit_costs = [0] * in_hist.shape[0]
  totals = [int(np.sum(in_hist[i], dtype=np.uint64)) for i in range(in_hist.shape[0])]
  largest_idx = 0
  for i, total in enumerate(totals):
    if total == 0:
      histogram_symbols[i] = 0
      dists[i] = 0.0
      continue
    bit_costs[i] = _histogram_bit_cost(in_hist[i])
    if total > totals[largest_idx]:
      largest_idx = i

  while len(out) < max_histograms:
    histogram_symbols[largest_idx] = len(out)
    out.append(in_hist[largest_idx].copy())
    dists[largest_idx] = 0.0
    largest_idx = 0
    out_bit_cost = _histogram_bit_cost(out[-1])
    for i in range(in_hist.shape[0]):
      if dists[i] == 0.0:
        continue
      dist = _histogram_distance(in_hist[i], out[-1], bit_costs[i],
                                 out_bit_cost)
      dists[i] = min(float(dist), dists[i])
      if dists[i] > dists[largest_idx]:
        largest_idx = i
    if dists[largest_idx] < K_MIN_DISTANCE_FOR_DISTINCT:
      break

  out_bit_costs = [_histogram_bit_cost(hist) for hist in out]
  for i in range(in_hist.shape[0]):
    if histogram_symbols[i] != max_histograms:
      continue
    best = 0
    best_dist = _histogram_distance(in_hist[i], out[best], bit_costs[i],
                                    out_bit_costs[best])
    for j in range(1, len(out)):
      dist = _histogram_distance(in_hist[i], out[j], bit_costs[i],
                                 out_bit_costs[j])
      if dist < best_dist:
        best = j
        best_dist = dist
    out[best] = out[best] + in_hist[i]
    out_bit_costs[best] = _histogram_bit_cost(out[best])
    histogram_symbols[i] = best

  tmp = [hist.copy() for hist in out]
  new_index: dict[int, int] = {}
  next_index = 0
  for symbol in histogram_symbols:
    if symbol not in new_index:
      new_index[symbol] = next_index
      out[next_index] = tmp[symbol]
      next_index += 1
  out = out[:next_index]
  context_map = np.asarray([new_index[symbol] for symbol in histogram_symbols],
                           dtype=np.uint8)
  return np.stack(out, axis=0).astype(np.uint32), context_map


def _build_huffman_codes(histograms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  num_prefix_codes = histograms.shape[0]
  depths = np.zeros((num_prefix_codes, K_ALPHABET_SIZE), dtype=np.uint8)
  bits = np.zeros((num_prefix_codes, K_ALPHABET_SIZE), dtype=np.uint16)
  for i in range(num_prefix_codes):
    counts = histograms[i]
    length = K_ALPHABET_SIZE
    while length > 0 and counts[length - 1] == 0:
      length -= 1
    depths[i] = create_huffman_tree(counts, length, 15)
    bits[i] = _convert_bit_depths_to_symbols(depths[i], length)
  return depths, bits


def optimize_entropy_code_from_context_values(
    context_values: np.ndarray, num_prefix_codes: int
) -> EntropyCodeTables:
  pairs = np.asarray(context_values, dtype=np.uint32)
  histograms = np.zeros((num_prefix_codes, K_ALPHABET_SIZE), dtype=np.uint32)
  for context, value in pairs:
    histograms[int(context), uint_token(int(value))] += 1
  clustered, context_map = _cluster_histograms(histograms)
  depths, bits = _build_huffman_codes(clustered)
  return EntropyCodeTables(context_map=context_map,
                           prefix_depths=depths,
                           prefix_bits=bits)


def optimize_prefix_code_from_context_values(
    context_values: np.ndarray, num_prefix_codes: int,
    context_map: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
  pairs = np.asarray(context_values, dtype=np.uint32)
  histograms = np.zeros((num_prefix_codes, K_ALPHABET_SIZE), dtype=np.uint32)
  for context, value in pairs:
    mapped_context = int(context)
    if context_map is not None:
      mapped_context = int(context_map[mapped_context])
    histograms[mapped_context, uint_token(int(value))] += 1
  return _build_huffman_codes(histograms)


def dc_entropy_code(dc_tokens: np.ndarray,
                    ac_metadata_tokens: np.ndarray) -> EntropyCodeTables:
  tokens = np.concatenate(
      (np.asarray(dc_tokens, dtype=np.uint32),
       np.asarray(ac_metadata_tokens, dtype=np.uint32)),
      axis=0)
  code = optimize_entropy_code_from_context_values(tokens, 45)
  return EntropyCodeTables(context_map=code.context_map,
                           prefix_depths=code.prefix_depths,
                           prefix_bits=code.prefix_bits,
                           orig_context_map=np.arange(45, dtype=np.uint8),
                           orig_num_contexts=45)


def ac_entropy_code(ac_tokens: np.ndarray) -> EntropyCodeTables:
  logical = np.asarray(ac_tokens, dtype=np.uint32)
  mapped = logical.copy()
  mapped[:, 0] = K_AC_CONTEXT_MAP[mapped[:, 0]]
  code = optimize_entropy_code_from_context_values(mapped, 64)
  return EntropyCodeTables(context_map=code.context_map,
                           prefix_depths=code.prefix_depths,
                           prefix_bits=code.prefix_bits,
                           orig_context_map=K_AC_CONTEXT_MAP,
                           orig_num_contexts=int(K_AC_CONTEXT_MAP.shape[0]))
