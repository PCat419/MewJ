"""荒牌流局罚符（不听罚符）的期望模型。

规则背景（日麻）：
- 荒牌流局时未听牌者向听牌者支付罚符，场上总量恒定 3000 点：
  1 听 3 未听 → 未听各付 1000 / 听收 3000；
  2 听 2 未听 → 各 ±1500；
  3 听 1 未听 → 未听付 3000 / 听各收 1000；
  全听或全未听 → 无收付。
- 听牌认定极宽：形式听牌（无役）、振听听牌、空听（待牌耗尽）全部算听牌，
  罚符价值与待牌质量无关 —— 因此本模型的听牌概率绝不用 win_prob 折减。
- 对手听牌率推断：立直者规则上必听牌；四副露 = 单骑必听；三副露按听牌
  处理；其余对手（含 1~2 副露）视为已弃和（未听）。
- nanikiru 引擎的 exp_score 不含流局不听罚符价值，本模块把该缺口
  以 noten_credit 形式注入切牌综合效用（scoring.py）。

信用是双向的：流局时仍听牌 → 收取罚符（正）；未听 → 支付罚符（负）。
低听牌率候选的信用为负，这是有意设计，不可改成只奖不罚。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .defense import THREAT_SEATS
from .params import PARAMS as _P

# 经验参数统一存于 params.py（此处仅为兼容别名）
NOTEN_CONFIG = _P["noten"]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _furo_count(t: Dict[str, Any]) -> int:
    try:
        return int(t.get("furo_count") or 1)
    except (TypeError, ValueError):
        return 1


def is_pusher(t: Dict[str, Any]) -> bool:
    """推进者（硬威胁）：立直，或 ≥3 副露 —— 与 posture.is_hard_threat 同口径。"""
    if t.get("kind") == "riichi":
        return True
    return _furo_count(t) >= 3


def opponent_tenpai_probs(
    threats: Optional[List[Dict[str, Any]]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """推断 3 个对手（相对座位）各自的听牌率。

    立直 → p_riichi；≥4 副露 → p_furo4；3 副露 → p_furo3；
    不在 threats 里的对手（及 1~2 副露）→ p_other（视为已弃和）。
    """
    cfg = cfg or NOTEN_CONFIG
    by_seat = {t.get("seat"): t for t in (threats or [])}
    probs: Dict[str, float] = {}
    for seat in THREAT_SEATS:
        t = by_seat.get(seat)
        if t is None:
            probs[seat] = float(cfg["p_other"])
        elif t.get("kind") == "riichi":
            probs[seat] = float(cfg["p_riichi"])
        else:
            fc = _furo_count(t)
            if fc >= 4:
                probs[seat] = float(cfg["p_furo4"])
            elif fc == 3:
                probs[seat] = float(cfg["p_furo3"])
            else:
                # 1~2 副露不按听牌处理（软威胁，视为未听）
                probs[seat] = float(cfg["p_other"])
    return probs


def payoff_table(
    probs: Dict[str, float],
    pool: float = 3000.0,
) -> Tuple[float, float]:
    """对 3 家对手听牌状态做 2^3 枚举，返回 (E_tenpai, E_noten)。

    组合内 k 个对手听牌时：我听牌收 pool/(k+1)（k=3 全听则无收付）；
    我未听付 −pool/(4−k)（k=0 全未听则无收付）。E_noten 为负值。
    """
    seats = list(probs)
    e_tenpai = 0.0
    e_noten = 0.0
    for mask in range(1 << len(seats)):
        p = 1.0
        k = 0
        for i, seat in enumerate(seats):
            pi = min(1.0, max(0.0, _as_float(probs.get(seat))))
            if (mask >> i) & 1:
                p *= pi
                k += 1
            else:
                p *= 1.0 - pi
        if p <= 0.0:
            continue
        if k < 3:
            e_tenpai += p * pool / (k + 1)
        if k > 0:
            e_noten += p * (-pool / (4 - k))
    return e_tenpai, e_noten


def ryukyoku_prob(
    turn: Optional[int],
    n_pushers: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """荒牌流局率：按巡目基底表查 p0，再乘 pusher_decay**n_pushers。"""
    cfg = cfg or NOTEN_CONFIG
    t = int(turn) if turn is not None else 1
    base = float(cfg["ryukyoku_base"][-1][1])
    for max_turn, p in cfg["ryukyoku_base"]:
        if t <= int(max_turn):
            base = float(p)
            break
    return base * float(cfg["pusher_decay"]) ** max(0, int(n_pushers))


def prepare_noten_ctx(
    threats: Optional[List[Dict[str, Any]]],
    turn: Optional[int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把每个决策点只需算一次的部分（罚符期望/流局率/推进者数）预打包。"""
    cfg = cfg or NOTEN_CONFIG
    threats = threats or []
    n_pushers = sum(1 for t in threats if is_pusher(t))
    e_tenpai, e_noten = payoff_table(
        opponent_tenpai_probs(threats, cfg), pool=float(cfg["pool"])
    )
    return {
        "threats": threats,
        "turn": turn,
        "e_tenpai": e_tenpai,
        "e_noten": e_noten,
        "p_ryukyoku": ryukyoku_prob(turn, n_pushers, cfg),
        "n_pushers": n_pushers,
    }


def _tenpai_by_end(cand: Dict[str, Any]) -> float:
    """「打到流局时已听牌」的原始概率：取听牌率数组末段（最后一个非 None）。

    候选上优先读 review.py 附带的按巡数组 tenpai_prob_arr；缺失时退化为
    标量 tenpai_prob（当前巡值，低估流局时听牌率，是保守近似）；都没有则 0。
    """
    arr = cand.get("tenpai_prob_arr")
    if isinstance(arr, (list, tuple)) and arr:
        for v in reversed(arr):
            if v is not None:
                return min(1.0, max(0.0, _as_float(v)))
        return 0.0
    v = cand.get("tenpai_prob")
    if isinstance(v, (list, tuple)):
        for x in reversed(v):
            if x is not None:
                return min(1.0, max(0.0, _as_float(x)))
        return 0.0
    return min(1.0, max(0.0, _as_float(v)))


def candidate_noten_credit(
    cand: Dict[str, Any],
    ctx: Optional[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """单个切牌候选的罚符期望信用（点，可正可负）。

    p_end = 本切牌后「流局时自家仍听牌」的概率：
    - 已听牌（shanten==0）候选：keep_base**n_hard_threats。
      振听候选与无役听牌（win_prob==0）照拿全额保有系数 —— 听牌认定不含
      待牌质量（形式听牌/振听/空听均算听牌），不得用 win_prob 折减。
    - 未听牌（shanten>=1）候选：流局时听牌率（数组末段）× 同样的保有折扣。

    credit = 流局率 × (p_end×E_tenpai + (1−p_end)×E_noten)。E_noten<0，
    故低听牌率候选信用为负（双向设计：保听有奖、弃听有罚）。
    """
    cfg = cfg or NOTEN_CONFIG
    if not ctx:
        return 0.0, {}
    # ctx 可能是 prepare_noten_ctx 的预打包结果，也可能是原始 {threats, turn}
    if "e_tenpai" in ctx:
        e_tenpai = _as_float(ctx.get("e_tenpai"))
        e_noten = _as_float(ctx.get("e_noten"))
        p_ryukyoku = _as_float(ctx.get("p_ryukyoku"))
        n_pushers = int(_as_float(ctx.get("n_pushers")))
    else:
        packed = prepare_noten_ctx(ctx.get("threats"), ctx.get("turn"), cfg)
        e_tenpai = packed["e_tenpai"]
        e_noten = packed["e_noten"]
        p_ryukyoku = packed["p_ryukyoku"]
        n_pushers = packed["n_pushers"]

    keep = float(cfg["keep_base"]) ** n_pushers
    try:
        shanten = int(cand.get("shanten"))
    except (TypeError, ValueError):
        shanten = 99
    if shanten == 0:
        p_end = keep
    else:
        p_end = _tenpai_by_end(cand) * keep
    credit = p_ryukyoku * (p_end * e_tenpai + (1.0 - p_end) * e_noten)
    components = {
        "p_end": p_end,
        "e_tenpai": e_tenpai,
        "e_noten": e_noten,
        "p_ryukyoku": p_ryukyoku,
        "n_pushers": n_pushers,
    }
    return credit, components
