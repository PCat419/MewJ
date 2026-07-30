"""先制立直判断（立直 vs 默听）：《统计学麻雀战术》局收支简化阈值。

在门前听牌切牌点上，附加「是否宣言立直」的检讨轴（两阶段，对标 Mortal）：
1. ``line_options``：立直元动作与各默听切牌共用 Softmax 权重；
2. ``riichi_cuts``：选定立直线后，听牌切的 Softmax 权重。

SMS 阈值决定立直线效用相对最优默听切的偏置（±margin），不单独改切牌 Softmax。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .call_eval import form_shanten_parts, hand_without_tile
from .params import PARAMS as _P
from .posture import Posture
from .scoring import DEFAULT_TEMPERATURE, normalize_tile, softmax_weights

_WAIT_LABEL = {"ryanmen": "良形", "bad": "愚形", "honor": "字牌待"}
_REC_LABEL = {"riichi": "立直", "dama": "默听"}
_P_RD = _P["riichi_declare"]
_P_SC = _P["scoring"]


def _norm(tile: Any) -> str:
    return normalize_tile(tile)


def _is_honor(tile: str) -> bool:
    t = _norm(tile)
    return len(t) >= 2 and t[-1] == "z"


def _is_terminal_or_honor(tile: str) -> bool:
    t = _norm(tile)
    if not t or len(t) < 2:
        return True
    if t[-1] == "z":
        return True
    try:
        n = int(t[0])
    except ValueError:
        return True
    return n in (1, 9)


def _dora_from_indicator(ind: str) -> str:
    """宝牌指示牌 → 宝牌（红五指示按普通 5 处理）。"""
    t = _norm(ind)
    if len(t) < 2:
        return t
    suit = t[-1]
    try:
        n = int(t[0])
    except ValueError:
        return t
    if suit == "z":
        if 1 <= n <= 4:
            return f"{n % 4 + 1}z"
        if 5 <= n <= 7:
            return f"{(n - 5 + 1) % 3 + 5}z"
        return t
    return f"{n % 9 + 1}{suit}"


def count_aka(tiles: Sequence[str]) -> int:
    """赤五张数（保留 0m/0p/0s 字面；normalize 后无法区分时不计）。"""
    n = 0
    for raw in tiles:
        s = str(raw or "").strip()
        if s.lower().endswith("r"):
            s = s[:-1]
        if s in ("0m", "0p", "0s"):
            n += 1
    return n


def count_dora(tiles: Sequence[str], indicators: Sequence[str]) -> int:
    """手牌中宝牌枚数（红五按普通 5 计入对应宝牌）。"""
    if not indicators:
        return 0
    dora_tiles = [_dora_from_indicator(i) for i in indicators]
    counts = Counter(_norm(t) for t in tiles)
    return sum(counts.get(d, 0) for d in dora_tiles)


def _ceil_100(x: int) -> int:
    return ((x + 99) // 100) * 100


def basic_points(han: int, fu: int) -> int:
    """符翻基本点（满贯以上返回满贯档基本点 2000/3000/…）。"""
    if han <= 0:
        return 0
    if han >= 13:
        return 8000
    if han >= 11:
        return 6000
    if han >= 8:
        return 4000
    if han >= 6:
        return 3000
    if han >= 5 or (han == 4 and fu >= 40) or (han == 3 and fu >= 70):
        return 2000
    return min(fu * (2 ** (han + 2)), 2000)


def ron_points(han: int, fu: int, *, dealer: bool) -> int:
    """荣和得点（不计本场/供托）。"""
    if han <= 0:
        return 0
    bp = basic_points(han, fu)
    if bp >= 2000:
        table = (
            {2000: 12000, 3000: 18000, 4000: 24000, 6000: 36000, 8000: 48000}
            if dealer
            else {2000: 8000, 3000: 12000, 4000: 16000, 6000: 24000, 8000: 32000}
        )
        return int(table.get(bp, 8000 if not dealer else 12000))
    return _ceil_100((6 if dealer else 4) * bp)


def classify_wait(hand13: Sequence[str], waits: Sequence[Any]) -> str:
    """待牌分类：ryanmen（良形）/ honor（字牌）/ bad（愚形）。

    ``waits`` 可为必要牌 dict（含 tile）或 mpsz 字符串。
    """
    tiles: List[str] = []
    for w in waits or []:
        if isinstance(w, dict):
            t = _norm(w.get("tile"))
        else:
            t = _norm(w)
        if t:
            tiles.append(t)
    # 去重保序
    seen = set()
    uniq: List[str] = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return "bad"
    if all(_is_honor(t) for t in uniq):
        return "honor"

    nums: List[Tuple[int, str]] = []
    for t in uniq:
        if _is_honor(t):
            continue
        try:
            nums.append((int(t[0]), t[1]))
        except (ValueError, IndexError):
            continue

    # 两面/两门差 3（含延贝等良形）
    for i, (n1, s1) in enumerate(nums):
        for n2, s2 in nums[i + 1 :]:
            if s1 == s2 and abs(n1 - n2) == 3:
                return "ryanmen"

    # 三面以上数牌待 → 良形
    if len(nums) >= 3:
        return "ryanmen"

    return "bad"


def _yakuhai_tiles(round_wind: str, seat_wind: str) -> set:
    wind_map = {
        "east": "1z",
        "东": "1z",
        "東": "1z",
        "south": "2z",
        "南": "2z",
        "west": "3z",
        "西": "3z",
        "north": "4z",
        "北": "4z",
    }
    out = {"5z", "6z", "7z"}
    rw = wind_map.get(str(round_wind or "").strip().lower()) or wind_map.get(
        str(round_wind or "").strip()
    )
    sw = wind_map.get(str(seat_wind or "").strip().lower()) or wind_map.get(
        str(seat_wind or "").strip()
    )
    if rw:
        out.add(rw)
    if sw:
        out.add(sw)
    return out


def _has_triplet(counts: Counter, tile: str) -> bool:
    return counts.get(_norm(tile), 0) >= 3


def estimate_dama_value(
    hand13: Sequence[str],
    waits: Sequence[Any],
    *,
    dora_indicators: Optional[Sequence[str]] = None,
    round_wind: str = "east",
    seat_wind: str = "east",
    win_prob: Optional[float] = None,
    dealer: bool = False,
    hand14_raw: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """估计默听荣和翻符与得点。

    ``hand14_raw`` 用于赤五计数（normalize 前）；缺省则从 hand13 字面统计。
    ``win_prob≈0`` 时直接判无役。
    """
    indicators = list(dora_indicators or [])
    wait_tiles: List[str] = []
    for w in waits or []:
        if isinstance(w, dict):
            t = _norm(w.get("tile"))
        else:
            t = _norm(w)
        if t:
            wait_tiles.append(t)

    if win_prob is not None and float(win_prob) <= 1e-9:
        return {
            "han": 0,
            "fu": 0,
            "points": 0,
            "yaku": [],
            "dora": 0,
            "aka": 0,
            "no_yaku": True,
            "notes": "引擎和率≈0，视为无役",
        }

    hand_n = [_norm(t) for t in hand13]
    counts = Counter(hand_n)
    aka_src = list(hand14_raw) if hand14_raw is not None else list(hand13)
    aka = count_aka(aka_src)
    dora = count_dora(hand_n, indicators)

    yaku: List[str] = []
    yaku_han = 0

    # 七对（门前听牌：七对向听 0）
    _reg, sp, _orph = form_shanten_parts(list(hand13), 0)
    is_chiitoitsu = sp == 0

    if is_chiitoitsu:
        yaku.append("七对子")
        yaku_han += 2
    else:
        yakuhai_set = _yakuhai_tiles(round_wind, seat_wind)
        # 役牌暗刻（手牌中已成刻）
        wind_map = {
            "east": "1z",
            "south": "2z",
            "west": "3z",
            "north": "4z",
        }
        rw = wind_map.get(str(round_wind or "").strip().lower())
        sw = wind_map.get(str(seat_wind or "").strip().lower())
        for tile in ("5z", "6z", "7z"):
            if _has_triplet(counts, tile):
                yaku.append({"5z": "白", "6z": "发", "7z": "中"}[tile])
                yaku_han += 1
        if rw and _has_triplet(counts, rw):
            yaku.append("场风")
            yaku_han += 1
        if sw and _has_triplet(counts, sw):
            # 连风额外 +1（场风=自风同一张时已计场风，再加自风）
            if sw == rw:
                yaku.append("自风")
                yaku_han += 1
            else:
                yaku.append("自风")
                yaku_han += 1

        # 断幺：手牌与所有待牌均非幺九字
        if hand_n and all(not _is_terminal_or_honor(t) for t in hand_n):
            if wait_tiles and all(not _is_terminal_or_honor(t) for t in wait_tiles):
                yaku.append("断幺")
                yaku_han += 1

        wait_class = classify_wait(hand13, waits)
        # 平和粗判：良形待 + 无役牌刻子 + 手牌无字牌刻/对役牌倾向
        has_yakuhai_anko = any(
            _has_triplet(counts, t) for t in yakuhai_set
        )
        if (
            wait_class == "ryanmen"
            and not has_yakuhai_anko
            and yaku_han == (1 if "断幺" in yaku else 0)
        ):
            # 无暗刻役牌；若还有其他刻子则可能非平和，用「无 3 张同数牌」近似
            has_anko = any(c >= 3 for c in counts.values())
            if not has_anko:
                yaku.append("平和")
                yaku_han += 1

    if yaku_han <= 0:
        return {
            "han": 0,
            "fu": 0,
            "points": 0,
            "yaku": [],
            "dora": dora,
            "aka": aka,
            "no_yaku": True,
            "notes": "未检出门前役（宝牌不独立成役）",
        }

    han = yaku_han + dora + aka
    wait_class = classify_wait(hand13, waits)

    if is_chiitoitsu:
        fu = 25
    elif "平和" in yaku:
        fu = 30
    elif wait_class == "ryanmen":
        fu = 40  # 有暗刻等非平和良形
    else:
        fu = 40  # 愚形/字牌待闭门荣和常见 40

    points = ron_points(han, fu, dealer=dealer)
    return {
        "han": han,
        "fu": fu,
        "points": points,
        "yaku": yaku,
        "dora": dora,
        "aka": aka,
        "no_yaku": False,
        "notes": "",
    }


def _cand_utility(c: Dict[str, Any]) -> float:
    u = c.get("adjusted_utility")
    if u is None:
        u = c.get("exp_score")
    try:
        return float(u) if u is not None else -1e18
    except (TypeError, ValueError):
        return -1e18


def _cand_riichi_utility(c: Dict[str, Any]) -> float:
    """立直线锚点：在已含危险/罚符的默听效用上，把进攻 EV 换成立直续航。"""
    base = _cand_utility(c)
    try:
        dama_ev = c.get("exp_score")
        riichi_ev = c.get("exp_score_riichi")
        if dama_ev is None or riichi_ev is None:
            return base
        return base - float(dama_ev) + float(riichi_ev)
    except (TypeError, ValueError):
        return base


def _all_cands(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """全部可展示切牌（仅排除食替）；含姿态否决项，便于牌效表完整列出。"""
    return [
        c
        for c in (analysis.get("candidates") or [])
        if not c.get("kuikae")
    ]


def _policy_cands(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        c
        for c in _all_cands(analysis)
        if not c.get("policy_rejected")
    ]


def _tenpai_cands(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """立直切候选：切后 0 向听（含振听；立直后仍可自摸，不受姿态否决影响）。"""
    return [
        c
        for c in _all_cands(analysis)
        if int(c.get("shanten") if c.get("shanten") is not None else 99) == 0
    ]


def build_riichi_cuts(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """立直线下的切牌选项：切后 0 向听（含振听），按综合效用 Softmax 归一。"""
    cands = _tenpai_cands(analysis)
    if not cands:
        return []
    temperature = float(_P_SC.get("temperature", DEFAULT_TEMPERATURE))
    utils = [_cand_utility(c) for c in cands]
    weights = softmax_weights(utils, temperature)
    out: List[Dict[str, Any]] = []
    for c, w, u in zip(cands, weights, utils):
        out.append(
            {
                "action": "riichi_cut",
                "tile": c.get("tile"),
                "shanten": c.get("shanten"),
                "uke": c.get("uke"),
                "exp_score": c.get("exp_score"),
                "win_prob": c.get("win_prob"),
                "deal_in": c.get("deal_in"),
                "adjusted_utility": u,
                "recommendation_weight": w,
                "necessary_tiles": c.get("necessary_tiles"),
                "furiten": bool(c.get("furiten")),
                "policy_rejected": bool(c.get("policy_rejected")),
            }
        )
    out.sort(
        key=lambda x: (
            -(x.get("recommendation_weight") or 0.0),
            str(x.get("tile") or ""),
        )
    )
    return out


def build_line_options(
    analysis: Dict[str, Any],
    *,
    prefer_riichi: bool,
    no_yaku: bool = False,
) -> List[Dict[str, Any]]:
    """立直元动作与各切牌共用 Softmax 权重（含姿态否决切，表中标 rejected）。

    立直线效用 = 最优听牌切的立直续航效用 ± margin；无听牌切时回退到
    最优未否决切。默听切仍用默听综合效用（含手替改听的 exp_score）。
    """
    dama_cands = _all_cands(analysis)
    temperature = float(_P_SC.get("temperature", DEFAULT_TEMPERATURE))
    margin = float(
        _P_RD["line_margin_no_yaku"] if no_yaku else _P_RD["line_margin"]
    )

    dama_utils = [_cand_utility(c) for c in dama_cands]
    # 锚点：宣言切用立直续航效用（有 exp_score_riichi 时）
    tenpai_cands = _tenpai_cands(analysis)
    anchor_tile: Optional[str] = None
    if tenpai_cands:
        anchor = max(
            tenpai_cands,
            key=lambda c: (
                _cand_riichi_utility(c),
                str(c.get("tile") or ""),
            ),
        )
        anchor_u = _cand_riichi_utility(anchor)
        anchor_tile = anchor.get("tile")
    else:
        policy_utils = [
            _cand_utility(c) for c in dama_cands if not c.get("policy_rejected")
        ]
        anchor_u = (
            max(policy_utils)
            if policy_utils
            else (max(dama_utils) if dama_utils else 0.0)
        )
    if prefer_riichi:
        riichi_u = anchor_u + margin
    else:
        riichi_u = anchor_u - margin

    all_utils = [riichi_u] + dama_utils
    weights = softmax_weights(all_utils, temperature)

    options: List[Dict[str, Any]] = [
        {
            "action": "riichi",
            "tile": anchor_tile,
            "adjusted_utility": riichi_u,
            "recommendation_weight": weights[0],
        }
    ]
    for c, w, u in zip(dama_cands, weights[1:], dama_utils):
        options.append(
            {
                "action": "dama",
                "tile": c.get("tile"),
                "shanten": c.get("shanten"),
                "uke": c.get("uke"),
                "exp_score": c.get("exp_score"),
                "win_prob": c.get("win_prob"),
                "adjusted_utility": u,
                "recommendation_weight": w,
                "furiten": bool(c.get("furiten")),
                "policy_rejected": bool(c.get("policy_rejected")),
            }
        )
    options.sort(
        key=lambda x: (
            -(x.get("recommendation_weight") or 0.0),
            0 if x.get("action") == "riichi" else 1,
            str(x.get("tile") or ""),
        )
    )
    return options


def _pick_tenpai_candidate(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """选取用于打点估计的听牌切：优先推荐切（若听牌非振听），否则效用最优听牌切。"""
    cands = _tenpai_cands(analysis)
    if not cands:
        return None
    best_tile = analysis.get("best")
    for c in cands:
        if c.get("tile") == best_tile:
            return c
    return max(cands, key=_cand_utility)


def _meld_type(m: Any) -> str:
    return str((m or {}).get("type") or "").strip().lower()


def melds_allow_riichi(melds: Optional[Sequence[Any]]) -> bool:
    """门前或仅暗杠可立直；吃碰明杠不可。"""
    for m in melds or []:
        t = _meld_type(m)
        if t not in ("ankan", "暗杠", "a"):
            return False
    return True


def eligible(dp: Any, analysis: Dict[str, Any]) -> Tuple[bool, str]:
    """是否具备立直权重/判断条件（0 向听门前或仅暗杠）。"""
    if not analysis.get("ok"):
        return False, "分析未成功"
    melds = getattr(dp, "melds", None) or []
    if not melds_allow_riichi(melds):
        return False, "非门前（有副露）"
    scores = list(getattr(dp, "scores", None) or [])
    seat = int(getattr(dp, "seat", 0) or 0)
    if seat < len(scores):
        try:
            if int(scores[seat]) < 1000:
                return False, "点数不足 1000（无法立直）"
        except (TypeError, ValueError):
            pass
    # 只要存在切后 0 向听（含振听/姿态否决）即给立直权重；合法立直切在 riichi_cuts 再滤
    has_zero = any(
        int(c.get("shanten") if c.get("shanten") is not None else 99) == 0
        for c in _all_cands(analysis)
    )
    if not has_zero:
        return False, "无 0 向听切"
    return True, ""


def is_head_start(analysis: Dict[str, Any]) -> bool:
    threats = (analysis.get("defense") or {}).get("threats") or []
    return not any(t.get("kind") == "riichi" for t in threats)


def sms_decide(
    *,
    wait_class: str,
    dama_points: int,
    han: int,
    turn: int,
    no_yaku: bool,
    head_start: bool,
) -> Dict[str, Any]:
    """SMS 先制阈值树（纯函数，便于单测）。"""
    p = _P["riichi_declare"]
    ryanmen_dama = int(p["ryanmen_dama_points"])
    ryanmen_late_pts, ryanmen_late_turn = p["ryanmen_late_dama"]
    bad_dama = int(p["bad_dama_points"])
    always_dama_han = int(p["always_dama_han"])

    if not head_start:
        return {
            "recommend": "dama",
            "basis": "非先制，倾向默听（仍保留立直权重）",
            "rule": "not_head_start",
        }
    if no_yaku or dama_points <= 0 or han <= 0:
        return {
            "recommend": "riichi",
            "basis": "无役听牌，必须立直",
            "rule": "no_yaku",
        }
    if han >= always_dama_han:
        return {
            "recommend": "dama",
            "basis": f"确定 ≥{always_dama_han} 翻，默听保和率",
            "rule": "haneman_plus",
        }

    wc = wait_class if wait_class in ("ryanmen", "bad", "honor") else "bad"
    label = _WAIT_LABEL.get(wc, wc)

    if wc == "honor":
        return {
            "recommend": "riichi",
            "basis": f"{label}・默听约 {dama_points} 点，先制立直",
            "rule": "honor_riichi",
        }

    if wc == "ryanmen":
        if dama_points >= ryanmen_dama:
            return {
                "recommend": "dama",
                "basis": f"{label}・默听 {dama_points}≥{ryanmen_dama}，默听",
                "rule": "ryanmen_high",
            }
        if dama_points >= int(ryanmen_late_pts) and turn >= int(ryanmen_late_turn):
            return {
                "recommend": "dama",
                "basis": (
                    f"{label}・默听 {dama_points}≥{int(ryanmen_late_pts)} "
                    f"且巡目 {turn}≥{int(ryanmen_late_turn)}，默听"
                ),
                "rule": "ryanmen_late",
            }
        return {
            "recommend": "riichi",
            "basis": f"{label}・默听约 {dama_points} 点・{turn} 巡，立直",
            "rule": "ryanmen_riichi",
        }

    # bad
    if dama_points >= bad_dama:
        return {
            "recommend": "dama",
            "basis": f"{label}・默听 {dama_points}≥{bad_dama}，默听",
            "rule": "bad_high",
        }
    return {
        "recommend": "riichi",
        "basis": f"{label}・默听约 {dama_points} 点・{turn} 巡，立直",
        "rule": "bad_riichi",
    }


def decide(
    dp: Any,
    analysis: Dict[str, Any],
    *,
    posture: Optional[Any] = None,
) -> Dict[str, Any]:
    """对门前听牌点给出立直/默听推荐（两阶段权重，对标 Mortal）。

    1. ``line_options``：立直元动作 + 各默听切，Softmax 权重；
    2. ``riichi_cuts``：立直线下听牌切，Softmax 权重。
    """
    out: Dict[str, Any] = {
        "ok": True,
        "skipped": False,
        "recommend": None,
        "wait_class": None,
        "dama_value": None,
        "han": None,
        "fu": None,
        "turn": getattr(dp, "turn", None),
        "head_start": None,
        "cut_tile": None,
        "riichi_tile": None,
        "dama_tile": None,
        "basis": "",
        "rule": None,
        "yaku": [],
        "line_options": [],
        "riichi_cuts": [],
        "match": None,
        "tile_match": None,
    }

    ok, reason = eligible(dp, analysis)
    if not ok:
        out["skipped"] = True
        out["basis"] = reason
        return out

    # 全弃姿态仍给出立直权重（不跳过）；SMS 偏置改为倾向默听
    fold_bias = posture is not None and int(posture) >= int(Posture.FOLD)

    cand = _pick_tenpai_candidate(analysis)
    # 仅有振听 0 向听时仍可展示立直元动作；打点用任意 0 向听切
    if cand is None:
        zero_cands = [
            c
            for c in _all_cands(analysis)
            if int(c.get("shanten") if c.get("shanten") is not None else 99) == 0
        ]
        cand = max(zero_cands, key=_cand_utility) if zero_cands else None
    if cand is None:
        out["skipped"] = True
        out["basis"] = "无 0 向听切"
        return out
    cut = cand.get("tile")

    hand14 = list(getattr(dp, "hand", None) or [])
    hand13 = hand_without_tile(hand14, cut) if cut else None
    if hand13 is None:
        out["ok"] = False
        out["basis"] = f"无法从手牌移除切牌 {cut}"
        return out

    waits = cand.get("necessary_tiles") or []
    wait_class = classify_wait(hand13, waits)
    out["wait_class"] = wait_class

    seat_wind = str(getattr(dp, "seat_wind", "east") or "east")
    dealer = seat_wind.strip().lower() in ("east", "东", "東")
    win_prob = cand.get("win_prob")
    try:
        wp = float(win_prob) if win_prob is not None else None
    except (TypeError, ValueError):
        wp = None

    val = estimate_dama_value(
        hand13,
        waits,
        dora_indicators=getattr(dp, "dora_indicators", None)
        or analysis.get("dora")
        or getattr(dp, "dora", None)
        or [],
        round_wind=str(getattr(dp, "round_wind", "east") or "east"),
        seat_wind=seat_wind,
        win_prob=wp,
        dealer=dealer,
        hand14_raw=hand14,
    )
    out["dama_value"] = val.get("points")
    out["han"] = val.get("han")
    out["fu"] = val.get("fu")
    out["yaku"] = list(val.get("yaku") or [])
    out["dora"] = val.get("dora")
    out["aka"] = val.get("aka")
    out["no_yaku"] = bool(val.get("no_yaku"))
    if val.get("notes"):
        out["value_notes"] = val["notes"]

    try:
        exp = cand.get("exp_score")
        if wp and wp > 1e-9 and exp is not None:
            out["ev_implied_points"] = round(float(exp) / wp)
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    head_start = is_head_start(analysis)
    out["head_start"] = head_start
    turn = int(getattr(dp, "turn", 1) or 1)

    decision = sms_decide(
        wait_class=wait_class,
        dama_points=int(val.get("points") or 0),
        han=int(val.get("han") or 0),
        turn=turn,
        no_yaku=bool(val.get("no_yaku")),
        head_start=head_start,
    )
    out["basis"] = decision.get("basis") or ""
    out["rule"] = decision.get("rule")

    prefer_riichi = decision.get("recommend") == "riichi" and not fold_bias
    # 无役强制立直：仅先制；非先制尊重 SMS(not_head_start→dama)
    if bool(val.get("no_yaku")) and head_start and not fold_bias:
        prefer_riichi = True
    # 无役大 margin 只用于「先制强制立直」方向；非先制走普通 line_margin
    use_no_yaku_margin = bool(val.get("no_yaku")) and prefer_riichi

    line_options = build_line_options(
        analysis, prefer_riichi=prefer_riichi, no_yaku=use_no_yaku_margin
    )
    riichi_cuts = build_riichi_cuts(analysis)
    out["line_options"] = line_options
    out["riichi_cuts"] = riichi_cuts

    # 推荐线 = 权重最高的 line_options（与 SMS 偏置一致）
    best_line = line_options[0] if line_options else None
    if best_line and best_line.get("action") == "riichi":
        out["recommend"] = "riichi"
    else:
        out["recommend"] = "dama"

    riichi_tile = riichi_cuts[0].get("tile") if riichi_cuts else None
    dama_tile = None
    for opt in line_options:
        if (
            opt.get("action") == "dama"
            and opt.get("tile")
            and not opt.get("policy_rejected")
        ):
            dama_tile = opt.get("tile")
            break
    if dama_tile is None:
        for opt in line_options:
            if opt.get("action") == "dama" and opt.get("tile"):
                dama_tile = opt.get("tile")
                break
    out["riichi_tile"] = riichi_tile
    out["dama_tile"] = dama_tile
    out["cut_tile"] = (
        riichi_tile if out["recommend"] == "riichi" else dama_tile
    )

    is_riichi = bool(getattr(dp, "is_riichi_discard", False))
    actual_tile = getattr(dp, "actual_discard", None)
    out["match"] = match_actual(is_riichi, out)
    # 切牌对错与立直线无关：已立直则比立直切；未立直则比默听切
    if is_riichi:
        out["tile_match"] = (
            normalize_tile(actual_tile) == normalize_tile(riichi_tile)
            if actual_tile and riichi_tile
            else None
        )
    elif out["recommend"] == "dama":
        out["tile_match"] = (
            normalize_tile(actual_tile) == normalize_tile(dama_tile)
            if actual_tile and dama_tile
            else None
        )
    else:
        out["tile_match"] = None
    return out


def match_actual(is_riichi_discard: bool, decision: Dict[str, Any]) -> Optional[bool]:
    """实战是否宣言立直 与 推荐是否一致。跳过评估时返回 None。"""
    if decision.get("skipped") or not decision.get("ok", True):
        return None
    rec = decision.get("recommend")
    if rec is None:
        return None
    if rec == "riichi":
        return bool(is_riichi_discard)
    if rec == "dama":
        return not bool(is_riichi_discard)
    return None


def evaluate_declare(
    dp: Any,
    analysis: Dict[str, Any],
    *,
    posture: Optional[Any] = None,
) -> Dict[str, Any]:
    """review 挂载入口。"""
    try:
        return decide(dp, analysis, posture=posture)
    except Exception as exc:
        return {
            "ok": False,
            "skipped": True,
            "recommend": None,
            "basis": f"立直判断失败：{exc}",
            "match": None,
            "error": str(exc),
        }
