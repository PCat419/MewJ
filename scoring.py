"""Risk-adjusted discard utility and normalized recommendation weights.

The defensive input is a relative danger model, not a calibrated probability.
Consequently ``adjusted_utility`` is a recommendation score expressed on an
EV-like point scale; it must not be presented as a statistically exact net EV.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .noten import candidate_noten_credit, prepare_noten_ctx
from .params import PARAMS as _P

_S = _P["scoring"]
_N = _P["noten"]

# 经验参数统一存于 params.py（此处仅为兼容别名）
DEFAULT_TEMPERATURE = _S["temperature"]
MIN_RECOMMENDATION_WEIGHT = _S["min_recommendation_weight"]
EQUIVALENT_UTILITY_EPSILON = _S["equivalent_utility_epsilon"]
# Actual discard is "尚可" when its Softmax weight is close to the recommended best.
ACCEPTABLE_WEIGHT_RATIO = _S["acceptable_weight_ratio"]

RIICHI_LOSS_CHILD = _S["riichi_loss_child"]
RIICHI_LOSS_DEALER = _S["riichi_loss_dealer"]
FURO_LOSS_CHILD = _S["furo_loss_child"]
FURO_LOSS_DEALER = _S["furo_loss_dealer"]

# Hidden score-situation knob. 1.0 = neutral; lower prefers defense, higher prefers offense.
MIN_OFFENSIVE_DESIRE = _S["min_offensive_desire"]
MAX_OFFENSIVE_DESIRE = _S["max_offensive_desire"]
DEFAULT_OFFENSIVE_DESIRE = _S["default_offensive_desire"]
# Point gap that saturates the margin term (roughly one large hand swing).
OFFENSE_MARGIN_REF = _S["offense_margin_ref"]
MAX_RANK_TERM = _S["max_rank_term"]
MAX_MARGIN_TERM = _S["max_margin_term"]

WINDS = ("east", "south", "west", "north")
SEAT_OFFSETS = {"自家": 0, "下家": 1, "对家": 2, "上家": 3}
WIND_TILES = {"east": "1z", "south": "2z", "west": "3z", "north": "4z"}

# 高向听烂牌（≥4 向听）的役牌单张保有补正：静态表模型纯按进张效率评估，
# 系统性低估役牌单张「摸对/碰出得役」的翻盘价值。δ = (向听−2)×base × 活牌比例，
# 活牌 = 3 − 牌河/副露/指示牌中已见枚数（自家持有 ≥2 张时不补）。
# 仅三元牌/自风/场风等役牌；非役牌字牌不补。
YAKUHAI_KEEP_BASE = _S["yakuhai_keep_base"]
YAKUHAI_KEEP_MIN_SHANTEN = _S["yakuhai_keep_min_shanten"]

# 同向听：EV 与和率/听牌率按交换率合成进攻效用（无绝对 EV 门槛）
NEAR_TIE_WIN_SCALE = _S["near_tie_win_scale"]
NEAR_TIE_TENPAI_SCALE = _S["near_tie_tenpai_scale"]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_tile(tile: Any) -> str:
    value = str(tile or "").strip()
    if value.lower().endswith("r"):
        value = value[:-1]
    if value.startswith("0"):
        value = "5" + value[1:]
    return value


def absolute_wind(self_wind: str, relative_seat: str) -> str:
    """Return an opponent's absolute seat wind from our current seat wind."""
    try:
        self_index = WINDS.index(str(self_wind).strip().lower())
    except ValueError:
        self_index = 0
    offset = SEAT_OFFSETS.get(relative_seat, 0)
    return WINDS[(self_index + offset) % 4]


def estimate_loss(
    self_wind: str,
    relative_seat: str,
    threat_kind: str,
    *,
    honba: int = 0,
) -> float:
    """Estimate points lost when dealing into one threat.

    These transparent constants are deliberately kept separate from the
    relative danger model so they can later be replaced by calibrated values.
    """
    dealer = absolute_wind(self_wind, relative_seat) == "east"
    if threat_kind == "riichi":
        base = RIICHI_LOSS_DEALER if dealer else RIICHI_LOSS_CHILD
    else:
        base = FURO_LOSS_DEALER if dealer else FURO_LOSS_CHILD
    return base + _S["honba_bonus"] * max(0, _as_int(honba))


def fallback_offense_utility(candidate: Dict[str, Any]) -> float:
    """Map efficiency fields to an EV-like scale when nanikiru has no stats."""
    shanten = max(-1, _as_int(candidate.get("shanten"), 6))
    uke = max(0, _as_int(candidate.get("uke")))
    tenpai = min(1.0, max(0.0, _as_float(candidate.get("tenpai_prob"))))
    win = min(1.0, max(0.0, _as_float(candidate.get("win_prob"))))
    # A one-step shanten loss should dominate a modest uke difference.
    return max(
        0.0,
        (4 - shanten) * _S["util_shanten_coef"]
        + math.log1p(uke) * _S["util_uke_coef"]
        + tenpai * _S["util_tenpai_coef"]
        + win * _S["util_win_coef"],
    )


def offense_utility(candidate: Dict[str, Any]) -> float:
    value = candidate.get("exp_score")
    if value is None:
        return fallback_offense_utility(candidate)
    return max(0.0, _as_float(value))


def apply_near_tie_win_break(
    candidates: List[Dict[str, Any]],
    offenses: List[float],
    *,
    group_fallback: bool = False,
) -> List[float]:
    """Blend win/tenpai into min-shanten offense via an EV exchange rate.

    For same-shanten cuts with nanikiru ``exp_score``:
    ``offense = EV + win_prob * win_scale + tenpai_prob * tenpai_scale``.
    Candidate B overturns A iff
    ``ΔEV + Δwin * win_scale + Δtenpai * tenpai_scale > 0`` —
    both gaps are weighed together; there is no absolute EV band.
    Skipped on the fallback formula (it already weights win/tenpai).
    """
    if group_fallback or not candidates or len(candidates) != len(offenses):
        return list(offenses)
    min_s = min(_as_int(c.get("shanten"), 99) for c in candidates)
    win_scale = _as_float(NEAR_TIE_WIN_SCALE, 1000.0)
    tenpai_scale = _as_float(NEAR_TIE_TENPAI_SCALE, 200.0)
    out = list(offenses)
    for i, c in enumerate(candidates):
        if _as_int(c.get("shanten"), 99) != min_s:
            continue
        if c.get("exp_score") is None:
            continue
        win = min(1.0, max(0.0, _as_float(c.get("win_prob"))))
        tenpai = min(1.0, max(0.0, _as_float(c.get("tenpai_prob"))))
        out[i] = out[i] + win * win_scale + tenpai * tenpai_scale
    return out


def softmax_weights(
    utilities: Iterable[float],
    temperature: float = DEFAULT_TEMPERATURE,
    min_weight: float = MIN_RECOMMENDATION_WEIGHT,
) -> List[float]:
    """Return sharp, stable weights with a strictly positive tail floor."""
    values = [_as_float(value) for value in utilities]
    if not values:
        return []
    temp = _as_float(temperature, DEFAULT_TEMPERATURE)
    if temp <= 0:
        raise ValueError("temperature must be greater than zero")

    peak = max(values)
    exps = [math.exp(max(-745.0, min(0.0, (value - peak) / temp))) for value in values]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0:
        return [1.0 / len(values)] * len(values)

    raw_weights = [value / total for value in exps]
    floor = max(0.0, _as_float(min_weight, MIN_RECOMMENDATION_WEIGHT))
    floor = min(floor, (1.0 - 1e-12) / len(values))
    remaining_mass = 1.0 - floor * len(values)
    weights = [floor + remaining_mass * value for value in raw_weights]
    # Remove accumulated floating-point residue without changing ordering.
    residue = 1.0 - sum(weights)
    weights[weights.index(max(weights))] += residue
    return weights


def _honba_from_dp(dp: Any) -> int:
    meta = getattr(dp, "kyoku_meta", None) or []
    return _as_int(meta[1]) if len(meta) > 1 else 0


def _scores_from_dp(dp: Any) -> List[int]:
    raw = getattr(dp, "scores", None) or []
    scores = [_as_int(value) for value in raw]
    if len(scores) < 4:
        scores.extend([25000] * (4 - len(scores)))
    return scores[:4]


def average_rank(scores: List[int], seat: int) -> float:
    """Competition-style average rank (1 = sole first, 4 = sole last)."""
    my = scores[seat]
    better = sum(1 for value in scores if value > my)
    tied = sum(1 for value in scores if value == my)
    return better + (tied + 1) / 2.0


def score_margin(scores: List[int], seat: int) -> int:
    """Positive when leading the field, negative when trailing the leader."""
    my = scores[seat]
    others = [value for index, value in enumerate(scores) if index != seat]
    if not others:
        return 0
    if my >= max(others):
        return my - max(others)
    return my - max(scores)


def clamp_offensive_desire(value: float) -> float:
    return min(MAX_OFFENSIVE_DESIRE, max(MIN_OFFENSIVE_DESIRE, _as_float(value, DEFAULT_OFFENSIVE_DESIRE)))


def compute_offensive_desire(
    scores: Optional[List[int]] = None,
    seat: int = 0,
) -> Dict[str, Any]:
    """Derive a hidden offensive-desire factor from placement and point gap.

    Leading first place lowers desire (more defensive).  Trailing last place
    raises desire (more offensive).  Equal scores stay near 1.0.
    """
    parsed = [_as_int(value, 25000) for value in (scores or [])]
    if len(parsed) < 4:
        parsed.extend([25000] * (4 - len(parsed)))
    parsed = parsed[:4]
    seat_index = min(3, max(0, _as_int(seat)))

    rank = average_rank(parsed, seat_index)
    margin = score_margin(parsed, seat_index)
    # rank 1 → -MAX_RANK_TERM, rank 2.5 → 0, rank 4 → +MAX_RANK_TERM
    rank_term = ((rank - 2.5) / 1.5) * MAX_RANK_TERM
    raw_margin = margin / OFFENSE_MARGIN_REF if OFFENSE_MARGIN_REF else 0.0
    margin_term = -min(MAX_MARGIN_TERM, max(-MAX_MARGIN_TERM, raw_margin))
    desire = clamp_offensive_desire(DEFAULT_OFFENSIVE_DESIRE + rank_term + margin_term)
    risk_scale = 1.0 / desire if desire > 0 else 1.0
    return {
        "offensive_desire": desire,
        "risk_scale": risk_scale,
        "average_rank": rank,
        "margin": margin,
        "rank_term": rank_term,
        "margin_term": margin_term,
        "scores": parsed,
        "seat": seat_index,
        "hidden": True,
    }


def offensive_desire_from_dp(dp: Any) -> Dict[str, Any]:
    return compute_offensive_desire(_scores_from_dp(dp), getattr(dp, "seat", 0))


def score_candidates(
    candidates: List[Dict[str, Any]],
    defense: Optional[Dict[str, Any]],
    dp: Any,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    offensive_desire: Optional[float] = None,
    noten_ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Attach offense, danger cost, adjusted utility and Softmax weight.

    ``offensive_desire`` is a hidden score-situation parameter.  Values below
    1.0 amplify ``risk_cost`` (favor defense); values above 1.0 attenuate it
    (favor offense).  When omitted it is inferred from ``dp.scores``.

    ``noten_ctx``（{"threats": ..., "turn": ...}）存在且 params["noten"]
    启用时，把荒牌流局罚符期望信用注入综合效用（nanikiru 的 exp_score
    不含不听罚符价值，此处补缺口）；为 None 或禁用时 noten_credit 恒为 0，
    候选字段保持稳定。
    """
    defense = defense or {}
    threats = defense.get("threats") or []
    per_seat = defense.get("per_seat") or {}
    alphas = defense.get("alphas") or {}
    combined_all = defense.get("combined") or {}
    honba = _honba_from_dp(dp)
    self_wind = getattr(dp, "seat_wind", "east")

    # 终盘 nanikiru 对全体候选返回 exp_score≈0（而非 None），此时逐候选的
    # offense_utility 不会走 fallback，效用退化为全 0 平推。按组检测：
    # 全体候选 exp_score≈0 且 win_prob≈0 时改用 fallback（向听/听牌率/进张）。
    group_fallback = bool(candidates) and all(
        (c.get("exp_score") is None)
        or (
            _as_float(c.get("exp_score")) <= 1e-9
            and _as_float(c.get("win_prob")) <= 1e-9
        )
        for c in candidates
    )
    desire_info = offensive_desire_from_dp(dp)
    if offensive_desire is not None:
        desire = clamp_offensive_desire(offensive_desire)
        desire_info = {
            **desire_info,
            "offensive_desire": desire,
            "risk_scale": 1.0 / desire if desire > 0 else 1.0,
            "override": True,
        }
    desire = float(desire_info["offensive_desire"])
    risk_scale = float(desire_info["risk_scale"])
    # 对硬威胁（立直或 ≥3 副露）的防守不被进攻欲望稀释：risk_scale 下限 1.0
    # （与 posture.is_hard_threat 同规则；此处就地判断以避免循环依赖）
    def _hard(threat: Dict[str, Any]) -> bool:
        if (threat.get("kind") or "furo") == "riichi":
            return True
        try:
            return int(threat.get("furo_count") or 1) >= 3
        except (TypeError, ValueError):
            return False

    if any(_hard(t) for t in threats):
        risk_scale = max(risk_scale, 1.0)

    # ---- 役牌单张保有补正（仅 ≥4 向听烂牌）----
    yakuhai_keep: Dict[str, float] = {}
    min_shanten = min(
        (_as_int(c.get("shanten"), 99) for c in candidates), default=99
    )
    if min_shanten >= YAKUHAI_KEEP_MIN_SHANTEN:
        yakuhai_tiles = {"5z", "6z", "7z"}
        for wind in (getattr(dp, "seat_wind", None), getattr(dp, "round_wind", None)):
            wt = WIND_TILES.get(str(wind or "").lower())
            if wt:
                yakuhai_tiles.add(wt)
        visible: Dict[str, int] = {}

        def _see(tile: Any) -> None:
            nt = normalize_tile(tile)
            if nt:
                visible[nt] = visible.get(nt, 0) + 1

        for tiles in (getattr(dp, "rivers", None) or {}).values():
            for t in tiles or []:
                _see(t)
        for melds in (getattr(dp, "melds_by_rel", None) or {}).values():
            for m in melds or []:
                for t in (m.get("tiles") or []):
                    _see(t)
        for t in (getattr(dp, "dora_indicators", None) or []):
            _see(t)
        own: Dict[str, int] = {}
        for t in (getattr(dp, "hand", None) or []):
            nt = normalize_tile(t)
            if nt:
                own[nt] = own.get(nt, 0) + 1
        base = (min_shanten - 2) * YAKUHAI_KEEP_BASE
        for t in yakuhai_tiles:
            if own.get(t, 0) == 1:
                live = max(0, 3 - visible.get(t, 0))
                if live > 0:
                    yakuhai_keep[t] = base * live / 3.0

    # ---- 荒牌流局罚符期望（noten.py）：exp_score 缺口的补正 ----
    # 预打包每个决策点只算一次的部分（罚符期望/流局率/推进者数）
    noten_packed: Optional[Dict[str, Any]] = None
    if noten_ctx is not None and bool(_N.get("enabled", False)):
        noten_packed = prepare_noten_ctx(
            noten_ctx.get("threats"), noten_ctx.get("turn"), _N
        )

    utilities: List[float] = []
    base_offenses: List[float] = []
    for candidate in candidates:
        if group_fallback:
            base_offenses.append(fallback_offense_utility(candidate))
        else:
            base_offenses.append(offense_utility(candidate))
    offenses = apply_near_tie_win_break(
        candidates, base_offenses, group_fallback=group_fallback
    )

    for candidate, offense in zip(candidates, offenses):
        tile = normalize_tile(candidate.get("tile"))
        components: Dict[str, Dict[str, float]] = {}
        risk_cost = 0.0
        confidence = "none"

        for threat in threats:
            seat = threat.get("seat")
            if not seat:
                continue
            kind = threat.get("kind") or "furo"
            seat_model = per_seat.get(seat) or {}
            relative_risk = _as_float(
                (seat_model.get("relative_risk") or seat_model.get("probs") or {}).get(tile)
            )
            alpha = min(1.0, max(0.0, _as_float(alphas.get(seat), 1.0)))
            loss = estimate_loss(
                self_wind,
                seat,
                kind,
                honba=honba,
            )
            cost = relative_risk * alpha * loss
            risk_cost += cost
            components[seat] = {
                "relative_risk": relative_risk,
                "alpha": alpha,
                "expected_loss": loss,
                "cost": cost,
            }
            seat_confidence = seat_model.get("confidence")
            if seat_confidence == "low":
                confidence = "low"
            elif seat_confidence == "medium" and confidence != "low":
                confidence = "medium"

        combined = min(1.0, max(0.0, _as_float(combined_all.get(tile))))
        scaled_risk = risk_cost * risk_scale
        utility = (1.0 - combined) * offense - scaled_risk
        keep_penalty = yakuhai_keep.get(tile, 0.0)
        if keep_penalty:
            utility -= keep_penalty
        # 罚符信用加项（仿 yakuhai_keep 模式）：保听有奖、弃听有罚；
        # 未启用时恒为 0.0，保证报告列稳定
        if noten_packed is not None:
            noten_credit, noten_components = candidate_noten_credit(
                candidate, noten_packed, _N
            )
        else:
            noten_credit, noten_components = 0.0, None
        utility += noten_credit
        candidate["offense_ev"] = offense
        candidate["offense_fallback"] = group_fallback
        candidate["combined_risk"] = combined
        candidate["risk_components"] = components
        candidate["risk_cost"] = risk_cost
        candidate["risk_cost_scaled"] = scaled_risk
        candidate["yakuhai_keep_penalty"] = keep_penalty
        candidate["noten_credit"] = noten_credit
        candidate["noten_components"] = noten_components
        candidate["adjusted_utility"] = utility
        candidate["model_confidence"] = confidence
        candidate["offensive_desire"] = desire
        utilities.append(utility)

    weights = softmax_weights(utilities, temperature)
    for candidate, weight in zip(candidates, weights):
        candidate["recommendation_weight"] = weight
    return candidates


def best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the highest-weight discard with deterministic efficiency ties."""
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            -_as_float(c.get("recommendation_weight")),
            1 if c.get("furiten") else 0,
            _as_int(c.get("shanten"), 99),
            -_as_float(c.get("win_prob")),
            -_as_float(c.get("tenpai_prob")),
            -_as_float(c.get("exp_score")),
            -_as_int(c.get("uke")),
            str(c.get("tile") or ""),
        ),
    )


def equivalent_best_tiles(
    candidates: List[Dict[str, Any]],
    *,
    epsilon: float = EQUIVALENT_UTILITY_EPSILON,
) -> List[str]:
    """Return candidates whose utility is indistinguishable from the maximum."""
    if not candidates:
        return []
    tolerance = max(0.0, _as_float(epsilon, EQUIVALENT_UTILITY_EPSILON))
    best_utility = max(_as_float(c.get("adjusted_utility")) for c in candidates)
    return [
        str(c.get("tile"))
        for c in candidates
        if c.get("tile") is not None
        and best_utility - _as_float(c.get("adjusted_utility")) <= tolerance
    ]


def candidate_weight(candidates: List[Dict[str, Any]], tile: Any) -> Optional[float]:
    key = normalize_tile(tile)
    for candidate in candidates:
        if normalize_tile(candidate.get("tile")) == key:
            return _as_float(candidate.get("recommendation_weight"))
    return None


def is_acceptable_choice(
    candidates: List[Dict[str, Any]],
    actual: Any,
    best: Any,
    *,
    ratio: float = ACCEPTABLE_WEIGHT_RATIO,
) -> bool:
    """True when actual's recommendation weight is close enough to the best."""
    if not candidates or actual is None or best is None:
        return False
    if normalize_tile(actual) == normalize_tile(best):
        return True
    best_weight = candidate_weight(candidates, best)
    actual_weight = candidate_weight(candidates, actual)
    if best_weight is None or actual_weight is None or best_weight <= 0:
        return False
    threshold = min(1.0, max(0.0, _as_float(ratio, ACCEPTABLE_WEIGHT_RATIO)))
    return actual_weight >= best_weight * threshold


def classify_discard_match(
    candidates: List[Dict[str, Any]],
    actual: Any,
    best: Any,
    *,
    utility_epsilon: float = EQUIVALENT_UTILITY_EPSILON,
    weight_ratio: float = ACCEPTABLE_WEIGHT_RATIO,
) -> Dict[str, Any]:
    """Classify actual discard as best / equivalent / acceptable / different."""
    equivalent = equivalent_best_tiles(candidates, epsilon=utility_epsilon)
    actual_key = normalize_tile(actual) if actual is not None else ""
    best_key = normalize_tile(best) if best is not None else ""
    if not actual_key or not best_key:
        return {
            "match": None,
            "match_kind": None,
            "equivalent_best": equivalent,
        }
    if actual_key == best_key:
        kind = "best"
    elif actual_key in {normalize_tile(tile) for tile in equivalent}:
        kind = "equivalent"
    elif is_acceptable_choice(
        candidates,
        actual_key,
        best_key,
        ratio=weight_ratio,
    ):
        kind = "acceptable"
    else:
        kind = "different"
    return {
        "match": kind in ("best", "equivalent"),
        "match_kind": kind,
        "equivalent_best": equivalent,
    }


def order_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order all candidates by recommendation weight and stable tie-breaks."""
    return sorted(
        candidates,
        key=lambda c: (
            -_as_float(c.get("recommendation_weight")),
            1 if c.get("furiten") else 0,
            _as_int(c.get("shanten"), 99),
            -_as_int(c.get("uke")),
            -_as_float(c.get("tenpai_prob")),
            str(c.get("tile") or ""),
        ),
    )
