"""攻防姿态（押し引き）状态机：全牌效 → 兜牌 → 形听 → 全弃（单局内单调不退）。

无威胁时按 nanikiru 建议全速做牌；威胁分两级：

- 硬威胁：立直，或 ≥3 副露（≈立直）——按手牌强度分派兜牌/形听/全弃；
- 软威胁：仅 2 副露（任意数量、任意组合）——地板为兜牌，不直接全弃。

分派规则：
- 例外（保持全牌效）：手牌期望值极高，或自家 4 位且大比分落后；
- 兜牌：0 向听，或 1 向听且强度不错（EV 不低且进张多）——只打低风险牌维持向听
  （0 向听时允许退向至 1 向听，但拆听须过门控：和了率不降 / EV 增益足够 /
  显著更安全三选一，防止为微薄 EV 放弃听牌）；
- 形听：晚巡（≥keiten_min_turn）1 向听面对硬威胁时，若有低危切牌保持向听
  （摸到即听牌，含无役形听/振听——流局罚符对待牌质量免疫）且罚符信用达标，
  则保向听/保听收罚符；
- 全弃：仅当存在硬威胁、手牌不达标且安全存量足够（≥1 张完全安牌，或
  ≥2 张极低危牌）时 —— 按危险度升序切牌；存量不足则留兜牌。
  全弃状态下不再考虑形听（形听保听池为空时也降级为全弃）。

单局内状态只能 全牌效→兜牌→形听→全弃 单调前进，绝不回退
（review.review_paipu 负责携带状态）。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional

from .params import PARAMS as _P
from .scoring import (
    DEFAULT_TEMPERATURE,
    EQUIVALENT_UTILITY_EPSILON,
    MIN_RECOMMENDATION_WEIGHT,
    normalize_tile,
    softmax_weights,
)


class Posture(IntEnum):
    FULL_EFFICIENCY = 0  # 全牌效
    MANEUVER = 1  # 兜牌
    KEITEN = 2  # 形听
    FOLD = 3  # 全弃


POSTURE_LABELS = {
    Posture.FULL_EFFICIENCY: "全牌效",
    Posture.MANEUVER: "兜牌",
    Posture.KEITEN: "形听",
    Posture.FOLD: "全弃",
}

# 经验参数统一存于 params.py（此处仅为兼容别名）
POSTURE_CONFIG = _P["posture"]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v


def _cand_shanten(c: Dict[str, Any]) -> int:
    s = c.get("shanten")
    try:
        return int(s)
    except (TypeError, ValueError):
        return 99


def _cand_uke(c: Dict[str, Any]) -> int:
    try:
        return int(c.get("uke") or 0)
    except (TypeError, ValueError):
        return 0


def _cand_ev(c: Dict[str, Any]) -> float:
    v = c.get("offense_ev")
    if v is None:
        v = c.get("exp_score")
    return _as_float(v, 0.0)


def _cand_utility(c: Dict[str, Any]) -> float:
    return _as_float(c.get("adjusted_utility"), _cand_ev(c))


def _cand_risk(c: Dict[str, Any]) -> float:
    """综合危险度；无威胁时视为 0。"""
    v = c.get("combined_risk")
    if v is None:
        v = (c.get("deal_in") or {}).get("combined")
    return max(0.0, min(1.0, _as_float(v, 0.0)))


def _cand_winp(c: Dict[str, Any]) -> float:
    v = c.get("win_prob")
    return _as_float(v, 0.0)


# 全弃：综合效用改为纯安全分 = −危险度×该系数（0% → 0，1.61% → −161）
FOLD_SAFETY_SCALE = _P["posture"]["fold_safety_scale"]


def _rescore_fold(ordered: List[Dict[str, Any]]) -> None:
    """全弃口径重算综合效用与推荐权重，使显示数字与危险度排序一致。"""
    for c in ordered:
        c["adjusted_utility"] = -_cand_risk(c) * FOLD_SAFETY_SCALE
    weights = softmax_weights(
        [c["adjusted_utility"] for c in ordered], DEFAULT_TEMPERATURE
    )
    for c, w in zip(ordered, weights):
        c["recommendation_weight"] = w


def _rescore_maneuver(
    candidates: List[Dict[str, Any]], eligible: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """兜牌口径：安全线内按综合效用降序并重新 softmax；线外权重压底并排末。"""
    eligible.sort(
        key=lambda c: (_cand_utility(c), _cand_ev(c), _cand_uke(c)), reverse=True
    )
    weights = softmax_weights(
        [_cand_utility(c) for c in eligible], DEFAULT_TEMPERATURE
    )
    for c, w in zip(eligible, weights):
        c["recommendation_weight"] = w
    elig_ids = {id(c) for c in eligible}
    rest = [c for c in candidates if id(c) not in elig_ids]
    rest.sort(key=lambda c: -_as_float(c.get("recommendation_weight")))
    for c in rest:
        c["recommendation_weight"] = MIN_RECOMMENDATION_WEIGHT
    return eligible + rest


def _rescore_keiten(
    candidates: List[Dict[str, Any]], eligible: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """形听口径：保听池内 综合效用 = 罚符信用 − 缩放后风险成本。

    罚符信用（noten_credit）即「打到流局」的罚符期望，风险成本用 scoring
    已写好的 risk_cost_scaled（含 risk_scale 缩放）；缺该字段时退化为未缩放
    risk_cost（相当于 risk_scale≈1 的近似）。池内重新 softmax，池外压底。
    best 为池内调整后效用最大者（即罚符收益口径下最低危的保听切）。
    """
    for c in eligible:
        credit = _as_float(c.get("noten_credit"), 0.0)
        scaled = c.get("risk_cost_scaled")
        if scaled is None:
            scaled = c.get("risk_cost")
        c["adjusted_utility"] = credit - _as_float(scaled, 0.0)
    eligible.sort(
        key=lambda c: (_cand_utility(c), -_cand_risk(c)), reverse=True
    )
    weights = softmax_weights(
        [_cand_utility(c) for c in eligible], DEFAULT_TEMPERATURE
    )
    for c, w in zip(eligible, weights):
        c["recommendation_weight"] = w
    elig_ids = {id(c) for c in eligible}
    rest = [c for c in candidates if id(c) not in elig_ids]
    rest.sort(key=lambda c: -_as_float(c.get("recommendation_weight")))
    for c in rest:
        c["recommendation_weight"] = MIN_RECOMMENDATION_WEIGHT
    return eligible + rest


def _furo_count(t: Dict[str, Any]) -> int:
    try:
        return int(t.get("furo_count") or 1)
    except (TypeError, ValueError):
        return 1


def is_hard_threat(t: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
    """硬威胁：立直，或 ≥hard_furo_count 副露（≈立直）。仅 2 副露为软威胁。"""
    cfg = config or POSTURE_CONFIG
    if t.get("kind") == "riichi":
        return True
    return _furo_count(t) >= int(cfg["hard_furo_count"])


def trigger_threats(defense: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """立直或 ≥min_furo_trigger 副露的威胁才触发姿态评估。"""
    cfg = config or POSTURE_CONFIG
    out = []
    for t in (defense.get("threats") or []):
        if t.get("kind") == "riichi":
            out.append(t)
        elif _furo_count(t) >= int(cfg["min_furo_trigger"]):
            out.append(t)
    return out


def _hand_strength(candidates: List[Dict[str, Any]]) -> tuple:
    """(最低向听, 该向听层最优候选的 EV, 其进张)。手牌强度的代表。"""
    min_s = min(_cand_shanten(c) for c in candidates)
    pool = [c for c in candidates if _cand_shanten(c) == min_s and not c.get("furiten")]
    if not pool:
        pool = [c for c in candidates if _cand_shanten(c) == min_s]
    best = max(pool, key=_cand_ev)
    return min_s, _cand_ev(best), _cand_uke(best)


def evaluate_posture(
    analysis: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Posture:
    """根据威胁与手牌强度给出本巡的建议姿态（不含单调约束）。"""
    cfg = config or POSTURE_CONFIG
    defense = analysis.get("defense") or {}
    threats = trigger_threats(defense, cfg)
    if not threats:
        return Posture.FULL_EFFICIENCY

    candidates = analysis.get("candidates") or []
    if not candidates:
        return Posture.FULL_EFFICIENCY

    desire = (analysis.get("recommendation_model") or {}).get("offensive_desire") or {}
    rank = _as_float(desire.get("average_rank"), 2.5)
    margin = _as_float(desire.get("margin"), 0.0)

    min_s, hand_ev, hand_uke = _hand_strength(candidates)

    # 例外：手牌期望值极高（罕见强攻）
    if hand_ev >= float(cfg["high_ev"]):
        return Posture.FULL_EFFICIENCY
    # 例外：4 位且大比分落后，不写死防守
    if rank >= 3.5 and margin <= float(cfg["despair_margin"]):
        return Posture.FULL_EFFICIENCY

    # 软威胁（仅 2 副露，任意数量组合）：地板为兜牌，不直接全弃
    if not any(is_hard_threat(t, cfg) for t in threats):
        return Posture.MANEUVER

    multi = len(threats) >= 2
    # 0 向听（含多威胁）：维持听牌，只打低风险牌
    if min_s == 0:
        return Posture.MANEUVER
    # 1 向听且强度不错：兜牌（多威胁时要求听牌才兜）
    if min_s == 1 and not multi and hand_uke >= int(cfg["maneuver_min_uke"]):
        if hand_ev >= float(cfg["maneuver_min_ev"]):
            return Posture.MANEUVER
        # EV 1500~2000 强度稍欠：存在 ≤2% 的低风险维持牌才放行兜牌
        if hand_ev >= float(cfg["maneuver_min_ev_low_risk"]):
            keep = [
                c
                for c in candidates
                if _cand_shanten(c) <= 1 and not c.get("furiten")
            ]
            if keep and min(_cand_risk(c) for c in keep) <= float(
                cfg["maneuver_low_risk_cap"]
            ):
                return Posture.MANEUVER
    # 形听（KEITEN）：晚巡 1 向听面对硬威胁（上文已确认存在硬威胁）时，
    # 若存在能保持 1 向听的低危切牌（下巡摸到即听牌；听牌认定含无役形听/
    # 振听/空听——荒牌流局罚符对待牌质量免疫）且罚符信用达标 → 形听路线。
    # 优先级在兜牌之后、全弃之前：能按兜牌强度做牌的先兜牌；
    # 0 向听已由上文兜牌保听分支截获；advance 单调性保证进入形听后
    # 不会因下一巡摸到 0 向听回退。
    # 形听路线的向听口径：1 向听手不存在直接切到 0 向听的候选（最低向听
    # 为 1 时任何切牌均保持 1 向听），故路线取「保持最低向听」的候选；
    # 进入形听后摸到听牌，apply_posture 的保听池（shanten ≤ 最低向听）
    # 自然退化为 shanten==0 保听切。
    if min_s == 1 and _as_float(analysis.get("stat_turn")) >= float(
        cfg["keiten_min_turn"]
    ):
        k_cap = (
            float(cfg["keiten_danger_cap_multi"])
            if multi
            else float(cfg["keiten_danger_cap"])
        )
        keiten_route = [
            c
            for c in candidates
            if _cand_shanten(c) <= min_s
            and _cand_risk(c) <= k_cap
            and _as_float(c.get("noten_credit")) >= float(cfg["keiten_min_credit"])
        ]
        if keiten_route:
            return Posture.KEITEN
    # 2 向听及以上面对硬威胁（立直/≥3副露）：果断弃和。
    # 但全弃以「有牌可弃」为前提：≥1 张完全安牌（危险度=0，即对所有威胁家
    # 均为现物/绝对安全），或 ≥2 张极低危牌才允许全弃；安全存量不足时留在
    # 兜牌（盲弃等于闭着眼睛打牌，不如维持向听打最低危险牌）。
    zero_risk = [c for c in candidates if _cand_risk(c) <= 1e-9]
    if zero_risk:
        return Posture.FOLD
    low_risk = [
        c for c in candidates if _cand_risk(c) <= float(cfg["fold_low_risk_tile_cap"])
    ]
    if len(low_risk) >= 2:
        return Posture.FOLD
    return Posture.MANEUVER


def advance(current: Posture, proposed: Posture) -> Posture:
    """单局内只能 全牌效→兜牌→全弃，绝不回退。"""
    return Posture(max(int(current), int(proposed)))


def _classify_maneuver(
    eligible: List[Dict[str, Any]],
    pick: Dict[str, Any],
    actual: Optional[str],
) -> Dict[str, Any]:
    actual_key = normalize_tile(actual) if actual else ""
    pick_key = normalize_tile(pick.get("tile"))
    pick_u = _cand_utility(pick)
    equivalent = [
        str(c.get("tile"))
        for c in eligible
        if c.get("tile") is not None and abs(_cand_utility(c) - pick_u) <= EQUIVALENT_UTILITY_EPSILON
    ]
    if actual_key and actual_key == pick_key:
        kind = "best"
    elif actual_key and actual_key in {normalize_tile(t) for t in equivalent}:
        kind = "equivalent"
    elif actual_key and actual_key in {normalize_tile(c.get("tile")) for c in eligible}:
        kind = "acceptable"
    else:
        kind = "different"
    return {"match": kind in ("best", "equivalent"), "match_kind": kind, "equivalent_best": equivalent}


def _classify_fold(
    ordered: List[Dict[str, Any]],
    pick: Dict[str, Any],
    actual: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    actual_key = normalize_tile(actual) if actual else ""
    pick_key = normalize_tile(pick.get("tile"))
    min_risk = _cand_risk(pick)
    tier = [
        c for c in ordered if _cand_risk(c) <= min_risk + float(config["fold_danger_tier"])
    ]
    equivalent = [str(c.get("tile")) for c in tier if c.get("tile") is not None]
    if actual_key and actual_key == pick_key:
        kind = "best"
    elif actual_key and actual_key in {normalize_tile(t) for t in equivalent}:
        kind = "equivalent"
    elif actual_key:
        actual_cand = next(
            (c for c in ordered if normalize_tile(c.get("tile")) == actual_key), None
        )
        if actual_cand is not None and _cand_risk(actual_cand) <= min_risk + float(
            config["fold_accept_slack"]
        ):
            kind = "acceptable"
        else:
            kind = "different"
    else:
        kind = "different"
    return {"match": kind in ("best", "equivalent"), "match_kind": kind, "equivalent_best": equivalent}


def apply_posture(
    analysis: Dict[str, Any],
    posture: Posture,
    config: Optional[Dict[str, Any]] = None,
) -> Posture:
    """按姿态调整推荐与候选排序；返回实际使用的姿态（兜牌可降级为全弃）。"""
    cfg = config or POSTURE_CONFIG
    candidates = analysis.get("candidates") or []
    if not candidates or posture is Posture.FULL_EFFICIENCY:
        analysis["posture"] = POSTURE_LABELS[posture]
        analysis["posture_code"] = int(posture)
        return posture

    actual = analysis.get("actual")
    final = posture

    if posture is Posture.MANEUVER:
        threats = trigger_threats(analysis.get("defense") or {}, cfg)
        cap = (
            float(cfg["maneuver_danger_cap_multi"])
            if len(threats) >= 2
            else float(cfg["maneuver_danger_cap"])
        )
        min_s = min(_cand_shanten(c) for c in candidates)
        # 已听牌（0 向听）允许退向到 1 向听，但拆听是质变，须过门控：
        # 和了率不降（真改良）/ EV 增益足够 / 显著更安全（安全阀）三选一。
        # 弱听死守不如退向好形的情形由 EV 增益或安全阀覆盖。
        max_s = min_s + 1 if min_s == 0 else min_s
        eligible = [
            c
            for c in candidates
            if _cand_shanten(c) <= max_s
            and not c.get("furiten")
            and _cand_risk(c) <= cap
        ]
        if min_s == 0:
            anchor_pool = [c for c in eligible if _cand_shanten(c) == 0]
            if anchor_pool:
                anchor = max(
                    anchor_pool, key=lambda c: (_cand_utility(c), _cand_ev(c))
                )
                a_winp, a_ev, a_risk = (
                    _cand_winp(anchor),
                    _cand_ev(anchor),
                    _cand_risk(anchor),
                )
                ev_min = float(cfg["retreat_from_tenpai_ev_min"])
                margin = float(cfg["retreat_safety_margin"])

                def _retreat_ok(c: Dict[str, Any]) -> bool:
                    if _cand_shanten(c) == 0:
                        return True
                    if 0.0 < a_winp <= _cand_winp(c):
                        return True
                    if _cand_ev(c) - a_ev >= ev_min:
                        return True
                    return _cand_risk(c) <= a_risk - margin

                eligible = [c for c in eligible if _retreat_ok(c)]
        if not eligible:
            # 维持向听的牌全部超危险上限 → 降级全弃（单调前进，允许）
            final = Posture.FOLD
        else:
            ordered = _rescore_maneuver(candidates, eligible)
            pick = ordered[0]
            analysis["best"] = pick.get("tile")
            analysis["candidates"] = ordered
            analysis.update(_classify_maneuver(eligible, pick, actual))

    if posture is Posture.KEITEN:
        # 形听保听池：保持当前最低向听且低危的候选。
        # 1 向听入池 = 保向听（下巡摸到即听牌）；已摸到听牌后最低向听为 0，
        # 池内即 shanten==0 保听切。
        threats = trigger_threats(analysis.get("defense") or {}, cfg)
        k_cap = (
            float(cfg["keiten_danger_cap_multi"])
            if len(threats) >= 2
            else float(cfg["keiten_danger_cap"])
        )
        min_s = min(_cand_shanten(c) for c in candidates)
        eligible = [
            c
            for c in candidates
            if _cand_shanten(c) <= min_s and _cand_risk(c) <= k_cap
        ]
        if not eligible:
            # 保听池为空 → 降级全弃（单调前进，允许；全弃覆盖逻辑照常生效，
            # 符合「全弃状态下不考虑形听」）
            final = Posture.FOLD
        else:
            ordered = _rescore_keiten(candidates, eligible)
            pick = ordered[0]
            analysis["best"] = pick.get("tile")
            analysis["candidates"] = ordered
            # 分类口径仿兜牌：实切在保听池内即「尚可」，最优/等价按效用 ε 判定
            analysis.update(_classify_maneuver(eligible, pick, actual))

    if final is Posture.FOLD:
        ordered = sorted(
            candidates,
            key=lambda c: (
                _cand_risk(c),
                _cand_shanten(c),
                -_cand_uke(c),
                -_cand_ev(c),
                str(c.get("tile") or ""),
            ),
        )
        _rescore_fold(ordered)
        pick = ordered[0]
        analysis["best"] = pick.get("tile")
        analysis["candidates"] = ordered
        analysis.update(_classify_fold(ordered, pick, actual, cfg))

    analysis["posture"] = POSTURE_LABELS[final]
    analysis["posture_code"] = int(final)
    return final
