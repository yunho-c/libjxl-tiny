"""AC strategy helpers for libjxl-tiny trace parity tests."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .chroma_from_luma import K_INV_COLOR_FACTOR, _DCT_B_WEIGHTS, _DCT_X_WEIGHTS
from .transforms import BLOCK_DIM, scaled_dct


DCT = 0
DCT16X8 = 1
DCT8X16 = 2

DCT_Y_WEIGHTS = np.asarray((
    0.0017857145, 0.0017857157, 0.0017904768, 0.0020441783,
    0.0023338278, 0.0026645192, 0.0030420676, 0.0034731133,
    0.0017857157, 0.001785716, 0.0018473724, 0.0020886122,
    0.0023722122, 0.0026997121, 0.0030756146, 0.0035059757,
    0.0017904768, 0.0018473724, 0.0019982266, 0.0022149722,
    0.0024845072, 0.0028040458, 0.0031757555, 0.0036044519,
    0.0020441783, 0.0020886122, 0.0022149722, 0.0024100873,
    0.0026645192, 0.0029746795, 0.0033413812, 0.0037683977,
    0.0023338278, 0.0023722122, 0.0024845072, 0.0026645192,
    0.0029068382, 0.0032089923, 0.0035716419, 0.003998084,
    0.0026645192, 0.0026997121, 0.0028040458, 0.0029746795,
    0.0032089923, 0.0035059757, 0.0038667743, 0.0042947,
    0.0030420676, 0.0030756146, 0.0031757555, 0.0033413812,
    0.0035716419, 0.0038667743, 0.0042286036, 0.0046607289,
    0.0034731133, 0.0035059757, 0.0036044519, 0.0037683977,
    0.003998084, 0.0042947, 0.0046607289, 0.0051001739,
), dtype=np.float32)

DCT2_X_WEIGHTS = np.asarray((
    0.0001381068, 0.00016047071, 0.00018645605, 0.00021664926,
    0.00025173181, 0.00029249521, 0.00033985957, 0.00039489369,
    0.00041871337, 0.00044087201, 0.00046420316, 0.00048876996,
    0.00051463587, 0.00054187077, 0.00057054684, 0.00060074159,
    0.00019049694, 0.00019694651, 0.00021442315, 0.00024016941,
    0.00027289384, 0.0003124552, 0.00035932945, 0.00040429863,
    0.0004248473, 0.00044662904, 0.00046966935, 0.00049400958,
    0.00051969837, 0.00054679497, 0.00057536521, 0.00060547784,
    0.00026276117, 0.00026734054, 0.00028085473, 0.00030283103,
    0.00033291124, 0.00037106971, 0.0004054005, 0.00042322354,
    0.0004425951, 0.00046344541, 0.00048574654, 0.00050949713,
    0.00053471868, 0.00056144706, 0.00058973121, 0.00061962701,
    0.00036243827, 0.0003666683, 0.00037935356, 0.00039960333,
    0.00040956112, 0.00042183659, 0.00043620329, 0.00045248115,
    0.00047053859, 0.0004902879, 0.00051167433, 0.0005346745,
    0.00055928703, 0.00058552966, 0.00061343284, 0.00064304151,
    0.00043123538, 0.0004325301, 0.00043638589, 0.00044272337,
    0.00045142765, 0.00046236772, 0.00047541404, 0.00049045112,
    0.00050738233, 0.00052613625, 0.00054666388, 0.00056893763,
    0.00059295003, 0.00061870721, 0.00064623309, 0.00067556335,
    0.00048162136, 0.00048277923, 0.00048623976, 0.00049196521,
    0.00049989921, 0.00050997391, 0.00052211789, 0.00053626322,
    0.00055235048, 0.00057033246, 0.00059017621, 0.00061186077,
    0.00063538179, 0.00066074729, 0.00068797835, 0.00075214857,
    0.0005378935, 0.00053897168, 0.00054219965, 0.00054755906,
    0.00055502116, 0.0005645515, 0.00057611306, 0.00058966759,
    0.00060518348, 0.00062263483, 0.00064200454, 0.00066328526,
    0.00068647927, 0.0007393618, 0.000803373, 0.00087534159,
    0.00060074159, 0.00060177385, 0.00060486794, 0.00061001495,
    0.00061720144, 0.00062641077, 0.00063762552, 0.00065082888,
    0.00066600717, 0.00068315107, 0.00071794738, 0.00076673855,
    0.00082213664, 0.00088475435, 0.00095529074, 0.0010345384,
), dtype=np.float32)

DCT2_Y_WEIGHTS = np.asarray((
    0.00069053401, 0.00077444571, 0.00086855399, 0.00097409816,
    0.0010924696, 0.0012252233, 0.0013741088, 0.0015410866,
    0.0017283577, 0.0019383827, 0.0021739292, 0.0023783136,
    0.0025041751, 0.0026366978, 0.002776233, 0.002923158,
    0.00088290084, 0.00090565201, 0.00096644071, 0.0010539154,
    0.0011619731, 0.0012886107, 0.0014338633, 0.0015988132,
    0.0017851711, 0.0019951246, 0.0022312698, 0.0024038092,
    0.0025288085, 0.0026606584, 0.0027996788, 0.0029462043,
    0.0011288587, 0.0011438611, 0.0011877866, 0.0012581701,
    0.00135259, 0.0014695247, 0.0016085195, 0.0017700332,
    0.0019552717, 0.0021660449, 0.0023636019, 0.0024791702,
    0.0026018962, 0.0027319542, 0.0028695827, 0.003015053,
    0.0014433329, 0.0014561869, 0.0014945272, 0.0015578135,
    0.0016454635, 0.0017571596, 0.0018930284, 0.0020537286,
    0.0022404641, 0.0023856999, 0.0024897642, 0.0026016813,
    0.002721444, 0.0028491386, 0.0029849128, 0.0031289863,
    0.0018454153, 0.00185776, 0.0018947916, 0.0019565322,
    0.0020431101, 0.0021548597, 0.0022924175, 0.0023864941,
    0.0024688798, 0.002560135, 0.0026600207, 0.0027684029,
    0.002885245, 0.0030105773, 0.0031445161, 0.0032872346,
    0.0023435291, 0.0023491632, 0.0023660017, 0.0023938618,
    0.0024324679, 0.0024814904, 0.0025405819, 0.002609412,
    0.0026876912, 0.0027751899, 0.0028717481, 0.0029772632,
    0.0030917146, 0.0032151414, 0.0033476453, 0.0034893905,
    0.0026173447, 0.0026225911, 0.0026382981, 0.0026643763,
    0.0027006865, 0.0027470603, 0.002803318, 0.0028692731,
    0.0029447721, 0.003029689, 0.0031239407, 0.0032274905,
    0.0033403505, 0.0034625907, 0.003594313, 0.0037356857,
    0.002923158, 0.0029281813, 0.0029432368, 0.0029682817,
    0.0030032503, 0.0030480621, 0.0031026325, 0.0031668788,
    0.0032407353, 0.0033241559, 0.0034171303, 0.003519665,
    0.0036318223, 0.0037536959, 0.0038854245, 0.0040271855,
), dtype=np.float32)

DCT2_B_WEIGHTS = np.asarray((
    0.0019729543, 0.0025272998, 0.0032374004, 0.0041470206,
    0.0048498721, 0.0051065302, 0.0053767711, 0.005661312,
    0.0063208523, 0.0070889443, 0.0079503711, 0.0089164926,
    0.0099999988, 0.01121517, 0.012578006, 0.015967883,
    0.0033539708, 0.0035433732, 0.004076953, 0.0047721486,
    0.0049862624, 0.005223677, 0.005480676, 0.0058470899,
    0.0065286267, 0.0072964565, 0.0081600742, 0.009130463,
    0.010220082, 0.011443083, 0.012854187, 0.016610704,
    0.0049218577, 0.0049511627, 0.0050357706, 0.0051678251,
    0.0053387447, 0.0055415547, 0.0058825868, 0.0064732647,
    0.0071507092, 0.0079215374, 0.0087942975, 0.0097792931,
    0.010888627, 0.012136214, 0.014550334, 0.018655479,
    0.0054969229, 0.0055188821, 0.0055837538, 0.0056971479,
    0.0060176966, 0.0064261849, 0.0069230762, 0.0075107799,
    0.0081936987, 0.0089781955, 0.0098724691, 0.010886625,
    0.01203262, 0.01403677, 0.017736901, 0.02247834,
    0.0067489492, 0.0067940955, 0.0069295247, 0.0071553192,
    0.007471947, 0.0078806318, 0.0083836997, 0.0089848433,
    0.0096892491, 0.010503774, 0.011436986, 0.012499244,
    0.014953869, 0.018516723, 0.023044668, 0.028803868,
    0.008629065, 0.0086752698, 0.0088141672, 0.0090466458,
    0.009374314, 0.0097996546, 0.0103262, 0.010958694,
    0.011703256, 0.012567491, 0.014605597, 0.01750963,
    0.021164555, 0.025766177, 0.031564441, 0.044291977,
    0.011032925, 0.011082166, 0.011230315, 0.011478675,
    0.01182947, 0.012285955, 0.012938387, 0.014542448,
    0.016570158, 0.019115077, 0.022296766, 0.0262674,
    0.031220267, 0.04152387, 0.056756895, 0.078389272,
    0.015967883, 0.016106272, 0.016526788, 0.017245775,
    0.018291343, 0.019704822, 0.021542856, 0.023880199,
    0.026813647, 0.030466938, 0.037175436, 0.047613274,
    0.06190946, 0.081609353, 0.10892317, 0.14702357,
), dtype=np.float32)


@dataclass(frozen=True)
class AcStrategyDecision:
  entropy_8x8: np.ndarray
  entropy_16x8: np.ndarray
  entropy_8x16: np.ndarray
  costs: np.ndarray
  decision: np.ndarray


def _f32(value: float | np.float32) -> np.float32:
  return np.float32(value)


def _ceil_log2_nonzero(value: int) -> int:
  if value <= 1:
    return 0
  return (value - 1).bit_length()


def _inv_matrix(weights: np.ndarray, zero_count: int) -> np.ndarray:
  matrix = (np.float32(1.0) / weights).astype(np.float32)
  matrix[:zero_count] = np.float32(0.0)
  return matrix


INV_MATRICES = {
    DCT: (
        _inv_matrix(_DCT_X_WEIGHTS, 1),
        _inv_matrix(DCT_Y_WEIGHTS, 1),
        _inv_matrix(_DCT_B_WEIGHTS, 1),
    ),
    DCT16X8: (
        _inv_matrix(DCT2_X_WEIGHTS, 2),
        _inv_matrix(DCT2_Y_WEIGHTS, 2),
        _inv_matrix(DCT2_B_WEIGHTS, 2),
    ),
    DCT8X16: (
        _inv_matrix(DCT2_X_WEIGHTS, 2),
        _inv_matrix(DCT2_Y_WEIGHTS, 2),
        _inv_matrix(DCT2_B_WEIGHTS, 2),
    ),
}


def raw_strategy(encoded: np.ndarray | np.uint8 | int) -> np.ndarray:
  return np.asarray(encoded, dtype=np.uint8) >> np.uint8(1)


def is_first_block(encoded: np.ndarray | np.uint8 | int) -> np.ndarray:
  return (np.asarray(encoded, dtype=np.uint8) & np.uint8(1)) != 0


def covered_blocks_x(strategy: int) -> int:
  lut = (1, 1, 2)
  return lut[int(strategy)]


def covered_blocks_y(strategy: int) -> int:
  lut = (1, 2, 1)
  return lut[int(strategy)]


def _transform_block(xyb: np.ndarray, strategy: int, channel: int, bx: int,
                     by: int) -> np.ndarray:
  if strategy == DCT:
    height, width = 8, 8
  elif strategy == DCT16X8:
    height, width = 16, 8
  elif strategy == DCT8X16:
    height, width = 8, 16
  else:
    raise ValueError(f"invalid AC strategy: {strategy}")
  y0 = by * BLOCK_DIM
  x0 = bx * BLOCK_DIM
  block = xyb[channel, y0:y0 + height, x0:x0 + width]
  return scaled_dct(block).reshape(-1).astype(np.float32)


def estimate_entropy(strategy: int, xyb: np.ndarray, bx0: int, by0: int,
                     cx: int, cy: int, distance: float, qf: np.ndarray,
                     maskf: np.ndarray, ytox: int, ytob: int) -> np.float32:
  num_blocks = covered_blocks_x(strategy) * covered_blocks_y(strategy)
  bx = bx0 + cx
  by = by0 + cy
  blocks = [
      _transform_block(xyb, strategy, channel, bx, by) for channel in range(3)
  ]

  quant = np.float32(0.0)
  masking = np.float32(0.0)
  for iy in range(covered_blocks_y(strategy)):
    for ix in range(covered_blocks_x(strategy)):
      quant = max(quant, np.float32(qf[cy + iy, cx + ix]))
      masking = max(masking, np.float32(maskf[cy + iy, cx + ix]))

  info_loss = np.float32(0.0)
  info_loss2 = np.float32(0.0)
  entropy = np.float32(0.0)
  cmap_factors = (
      np.float32(ytox) * K_INV_COLOR_FACTOR,
      np.float32(0.0),
      np.float32(1.0) + np.float32(ytob) * K_INV_COLOR_FACTOR,
  )
  slope = min(np.float32(1.0), np.float32(distance) * np.float32(1.0 / 3.0))
  cost1 = np.float32(1.0) + slope * np.float32(8.8703248061477744)
  cost2 = np.float32(4.4628149885273363)
  cost_delta = np.float32(5.3359184934516337)
  for channel in range(3):
    val = ((blocks[channel] - blocks[1] * cmap_factors[channel]) *
           (INV_MATRICES[strategy][channel] * quant))
    rounded = np.rint(val).astype(np.float32)
    diff = np.abs(val - rounded).astype(np.float32)
    info_loss = np.float32(info_loss + np.sum(diff, dtype=np.float32))
    info_loss2 = np.float32(info_loss2 +
                            np.sum(diff * diff, dtype=np.float32))

    q = np.abs(rounded).astype(np.float32)
    entropy_v = np.where(q >= np.float32(1.5), cost2,
                         np.float32(0.0)).astype(np.float32)
    entropy_v = (entropy_v + np.sqrt(q, dtype=np.float32) *
                 cost_delta).astype(np.float32)
    num_nonzeros = int(np.sum(q != np.float32(0.0)))
    entropy = np.float32(entropy + np.sum(entropy_v, dtype=np.float32) +
                         np.float32(num_nonzeros) * cost1)
    nbits = _ceil_log2_nonzero(num_nonzeros + 1) + 1
    entropy = np.float32(
        entropy + np.float32(7.565053364251793) *
        np.float32(_ceil_log2_nonzero(nbits + 17) + nbits))

  infoloss2 = np.float32(math.sqrt(num_blocks * float(info_loss2)))
  info_loss_score = (np.float32(138.0) * info_loss +
                     np.float32(50.46839691767866) * infoloss2)
  return np.float32(entropy + masking * info_loss_score)


def _set_strategy(decision: np.ndarray, x: int, y: int, strategy: int) -> None:
  for iy in range(covered_blocks_y(strategy)):
    for ix in range(covered_blocks_x(strategy)):
      decision[y + iy, x + ix] = np.uint8(
          (strategy << 1) | (1 if (iy | ix) == 0 else 0))


def find_best_16x16_transform(xyb: np.ndarray, qf: np.ndarray,
                              maskf: np.ndarray, distance: float, ytox: int,
                              ytob: int, bx0: int = 0, by0: int = 0,
                              cx: int = 0, cy: int = 0) -> AcStrategyDecision:
  mul8x8 = np.float32(
      np.float32(1.0735757687292623 * 0.75) +
      np.float32(-0.55 * 0.75) / (np.float32(distance) + np.float32(1.4)))
  mul16x8 = np.float32(
      np.float32(0.9019587899705066) +
      np.float32(-0.55) / (np.float32(distance) + np.float32(1.6)))

  entropy_8x8 = np.empty((2, 2), dtype=np.float32)
  for dy in range(2):
    for dx in range(2):
      entropy_8x8[dy, dx] = np.float32(
          np.float32(3.0) * mul8x8 + mul8x8 *
          estimate_entropy(DCT, xyb, bx0, by0, cx + dx, cy + dy, distance, qf,
                           maskf, ytox, ytob))

  entropy_16x8 = np.asarray((
      mul16x8 * estimate_entropy(DCT16X8, xyb, bx0, by0, cx, cy, distance, qf,
                                 maskf, ytox, ytob),
      mul16x8 * estimate_entropy(DCT16X8, xyb, bx0, by0, cx + 1, cy, distance,
                                 qf, maskf, ytox, ytob),
  ), dtype=np.float32)
  entropy_8x16 = np.asarray((
      mul16x8 * estimate_entropy(DCT8X16, xyb, bx0, by0, cx, cy, distance, qf,
                                 maskf, ytox, ytob),
      mul16x8 * estimate_entropy(DCT8X16, xyb, bx0, by0, cx, cy + 1, distance,
                                 qf, maskf, ytox, ytob),
  ), dtype=np.float32)

  cost16x8 = np.float32(
      min(entropy_16x8[0], np.float32(entropy_8x8[0, 0] + entropy_8x8[1, 0])) +
      min(entropy_16x8[1], np.float32(entropy_8x8[0, 1] + entropy_8x8[1, 1])))
  cost8x16 = np.float32(
      min(entropy_8x16[0], np.float32(entropy_8x8[0, 0] + entropy_8x8[0, 1])) +
      min(entropy_8x16[1], np.float32(entropy_8x8[1, 0] + entropy_8x8[1, 1])))
  costs = np.asarray((cost16x8, cost8x16), dtype=np.float32)

  decision = np.full((2, 2), np.uint8((DCT << 1) | 1), dtype=np.uint8)
  if cost16x8 < cost8x16:
    if entropy_16x8[0] < entropy_8x8[0, 0] + entropy_8x8[1, 0]:
      _set_strategy(decision, 0, 0, DCT16X8)
    if entropy_16x8[1] < entropy_8x8[0, 1] + entropy_8x8[1, 1]:
      _set_strategy(decision, 1, 0, DCT16X8)
  else:
    if entropy_8x16[0] < entropy_8x8[0, 0] + entropy_8x8[0, 1]:
      _set_strategy(decision, 0, 0, DCT8X16)
    if entropy_8x16[1] < entropy_8x8[1, 0] + entropy_8x8[1, 1]:
      _set_strategy(decision, 0, 1, DCT8X16)

  return AcStrategyDecision(entropy_8x8=entropy_8x8,
                            entropy_16x8=entropy_16x8,
                            entropy_8x16=entropy_8x16,
                            costs=costs,
                            decision=decision)


def adjust_quant_field(raw_quant_field: np.ndarray,
                       ac_strategy: np.ndarray) -> np.ndarray:
  """Apply `AdjustQuantField` using a traced AC strategy image.

  `ac_strategy` uses the trace encoding from `TraceAcStrategy`:
  `(raw_strategy << 1) | is_first_block`.
  """
  quant = np.array(raw_quant_field, dtype=np.uint8, copy=True)
  strategy = np.asarray(ac_strategy, dtype=np.uint8)
  if quant.shape != strategy.shape:
    raise ValueError("raw_quant_field and ac_strategy must have the same shape")

  ysize, xsize = quant.shape
  for y in range(ysize):
    for x in range(xsize):
      encoded = strategy[y, x]
      if not bool(is_first_block(encoded)):
        continue
      raw = int(raw_strategy(encoded))
      if raw not in (DCT, DCT16X8, DCT8X16):
        raise ValueError(f"invalid AC strategy at ({x}, {y}): {raw}")
      blocks_x = covered_blocks_x(raw)
      blocks_y = covered_blocks_y(raw)
      if x + blocks_x > xsize or y + blocks_y > ysize:
        raise ValueError(f"AC strategy at ({x}, {y}) exceeds quant field")
      block = quant[y:y + blocks_y, x:x + blocks_x]
      block[:, :] = np.max(block)
  return quant
