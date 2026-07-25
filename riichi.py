# -*- coding: utf-8 -*-
"""Riichi Mahjong 放铳概率计算（Python 移植版）。

本模块将原本位于前端 index.html 中的铳率分析逻辑移植到后端，
内部统一使用 mpsz 字符串表示牌（如 "1m"、"5z"），与前端保持一致。
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------------
# 常量区
# ---------------------------------------------------------------------------

RIICHI_SUITS = ["m", "p", "s"]
RIICHI_HONORS = ["1z", "2z", "3z", "4z", "5z", "6z", "7z"]
RIICHI_SEATS = ["自家", "下家", "对家", "上家"]
RIICHI_SEAT_POS = {"自家": 0, "下家": 1, "对家": 2, "上家": 3}

# 34 种普通牌（红五已归一化为普通 5）
RIICHI_ALL_TILES: List[str] = []
for suit in RIICHI_SUITS:
    for n in range(1, 10):
        RIICHI_ALL_TILES.append(f"{n}{suit}")
for h in RIICHI_HONORS:
    RIICHI_ALL_TILES.append(h)

# 待牌型定义：key 为铳牌，value 为该铳牌对应的所有听牌型
RIICHI_TILE_WAITS: Dict[str, List[Dict[str, Any]]] = {t: [] for t in RIICHI_ALL_TILES}

for suit in RIICHI_SUITS:
    for n in range(1, 10):
        tile = f"{n}{suit}"
        if n + 3 <= 9:
            RIICHI_TILE_WAITS[tile].append({
                "type": "ryanmen",
                "shape": [n + 1, n + 2],
                "required": [f"{n + 1}{suit}", f"{n + 2}{suit}"],
                "name": f"{n + 1}{n + 2}{suit}两面",
            })
        if n - 3 >= 1:
            RIICHI_TILE_WAITS[tile].append({
                "type": "ryanmen",
                "shape": [n - 2, n - 1],
                "required": [f"{n - 2}{suit}", f"{n - 1}{suit}"],
                "name": f"{n - 2}{n - 1}{suit}两面",
            })
        if n == 3:
            RIICHI_TILE_WAITS[tile].append({
                "type": "penchan",
                "shape": [1, 2],
                "required": [f"1{suit}", f"2{suit}"],
                "name": f"12{suit}边张",
            })
        if n == 7:
            RIICHI_TILE_WAITS[tile].append({
                "type": "penchan",
                "shape": [8, 9],
                "required": [f"8{suit}", f"9{suit}"],
                "name": f"89{suit}边张",
            })
        if n - 1 >= 1 and n + 1 <= 9:
            RIICHI_TILE_WAITS[tile].append({
                "type": "kanchan",
                "shape": [n - 1, n + 1],
                "required": [f"{n - 1}{suit}", f"{n + 1}{suit}"],
                "name": f"{n - 1}{n + 1}{suit}坎张",
            })
        RIICHI_TILE_WAITS[tile].append({
            "type": "shanpon",
            "required": [tile, tile],
            "name": f"{tile[0]}{tile}双碰",
        })
        RIICHI_TILE_WAITS[tile].append({
            "type": "tanki",
            "required": [tile],
            "name": f"{tile}单骑",
        })

for tile in RIICHI_HONORS:
    RIICHI_TILE_WAITS[tile].append({
        "type": "shanpon",
        "required": [tile, tile],
        "name": f"{tile[0]}{tile}双碰",
    })
    RIICHI_TILE_WAITS[tile].append({
        "type": "tanki",
        "required": [tile],
        "name": f"{tile}单骑",
    })

RIICHI_WAIT_TYPE_PRIOR = {"ryanmen": 1.00, "kanchan": 0.40, "penchan": 0.20}
RIICHI_CATEGORY_PRIOR = {
    "shanpon": {"honor": 1.90, "terminal": 1.00, "middle": 0.55},
    "tanki": {"honor": 1.30, "terminal": 0.70, "middle": 0.50},
}

# 双碰修正用的常数：C(4,2)=6，33 种非字牌？前端为 33 * 6
FULL_PARTNER_COMBOS = 33 * 6


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

def normalize_tile(tile: str) -> str:
    """红五归一化为普通 5，例如 '0m' -> '5m'。"""
    if tile and tile[0] == "0":
        return "5" + tile[1:]
    return tile


def parse_riichi_tile_string(text: Optional[str], preserve_marker: bool = False) -> List[str]:
    """解析牌字符串，返回 mpsz 列表。

    支持逗号、空格、换行分隔；识别末尾 'r' 或 'R' 作为立直宣言标记。
    红五会被归一化为普通 5（铳率分析不区分红五）。
    """
    if not text:
        return []
    tokens = re.sub(r"[,\s]+", " ", text.strip()).split(" ")
    tiles: List[str] = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        match = re.match(r"^(\d)([mpsz])r?$", t, re.IGNORECASE)
        if not match:
            continue
        num = match.group(1)
        suit = match.group(2).lower()
        is_riichi = t.lower().endswith("r")
        if suit == "z" and (num < "1" or num > "7"):
            continue
        tile = ("5" if num == "0" else num) + suit
        tiles.append(tile + "r" if is_riichi and preserve_marker else tile)
    return tiles


def count_tiles(tiles: List[str]) -> Dict[str, int]:
    """统计 34 种牌各自出现次数。"""
    counts = {t: 0 for t in RIICHI_ALL_TILES}
    for t in tiles:
        if t in counts:
            counts[t] += 1
    return counts


def binom(n: int, k: int) -> int:
    """组合数 C(n,k)，结果与前端 Math.round 一致（四舍五入）。"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1.0
    for i in range(k):
        res = res * (n - i) / (i + 1)
    # 与 JS Math.round 保持一致：正数四舍五入
    return int(res + 0.5)


def raw_combination_count(wait: Dict[str, Any], r: Dict[str, int]) -> int:
    """计算某听牌型在剩余牌山中的原始组合数。"""
    groups: Dict[str, int] = {}
    for t in wait["required"]:
        groups[t] = groups.get(t, 0) + 1
    result = 1
    for t, need in groups.items():
        if r[t] < need:
            return 0
        result *= binom(r[t], need)
    return result


def group_count(wait: Dict[str, Any], r: Dict[str, int]) -> int:
    """计算某听牌型的最大可能组数（用于显示）。"""
    groups: Dict[str, int] = {}
    for t in wait["required"]:
        groups[t] = groups.get(t, 0) + 1
    min_val = float("inf")
    for t, need in groups.items():
        if r[t] < need:
            return 0
        min_val = min(min_val, r[t] // need)
    return 0 if min_val == float("inf") else min_val


def tile_category(tile: str) -> str:
    """牌分类：字牌 / 老头牌 / 中张牌。"""
    if tile.endswith("z"):
        return "honor"
    return "terminal" if tile[0] in ("1", "9") else "middle"


def wait_prior(wait: Dict[str, Any], tile: str) -> float:
    """听牌型先验权重。"""
    wtype = wait["type"]
    if wtype in ("shanpon", "tanki"):
        return RIICHI_CATEGORY_PRIOR[wtype][tile_category(tile)]
    return RIICHI_WAIT_TYPE_PRIOR.get(wtype, 1.0)


def arrays_equal(a: List[Any], b: List[Any]) -> bool:
    """判断两个列表是否完全相等。"""
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))


def compute_wait_risk(
    remaining: Dict[str, int],
    genbutsu: Optional[List[str]] = None,
    discards_by_seat: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """按可见牌、现物、筋和待牌形计算 34 种牌的相对危险度。

    返回值中的 ``probs`` 是总和为 1 的相对危险度分布，不是由历史
    样本校准出的真实放铳概率。此纯函数既供旧 ``compute_riichi``
    使用，也允许 MewJ 针对每个威胁家独立计算。
    """
    r = {t: max(0, int(remaining.get(t, 0))) for t in RIICHI_ALL_TILES}
    safe = {normalize_tile(t) for t in (genbutsu or [])}
    discards = {
        seat: [normalize_tile(t) for t in tiles]
        for seat, tiles in (discards_by_seat or {}).items()
    }

    cannot_have_by_seat: Dict[str, set] = {}
    for seat, seat_discards in discards.items():
        cannot = set()
        for n_tile in seat_discards:
            if n_tile.endswith("z"):
                continue
            suit = n_tile[-1]
            n = int(n_tile[0])

            if n + 3 <= 9:
                target = f"{n + 3}{suit}"
                shape = [n + 1, n + 2]
                for idx, wait in enumerate(RIICHI_TILE_WAITS.get(target, [])):
                    if wait["type"] == "ryanmen" and arrays_equal(wait["shape"], shape):
                        cannot.add(f"{target}:{idx}")

            if n - 3 >= 1:
                target = f"{n - 3}{suit}"
                shape = [n - 2, n - 1]
                for idx, wait in enumerate(RIICHI_TILE_WAITS.get(target, [])):
                    if wait["type"] == "ryanmen" and arrays_equal(wait["shape"], shape):
                        cannot.add(f"{target}:{idx}")
        cannot_have_by_seat[seat] = cannot

    excluded = set()
    if cannot_have_by_seat:
        for tile in RIICHI_ALL_TILES:
            for idx, wait in enumerate(RIICHI_TILE_WAITS.get(tile, [])):
                if wait["type"] != "ryanmen":
                    continue
                key = f"{tile}:{idx}"
                if all(key in cannot for cannot in cannot_have_by_seat.values()):
                    excluded.add(key)

    total_pair_combos = sum(binom(r[t], 2) for t in RIICHI_ALL_TILES)
    weights: Dict[str, float] = {}
    details: Dict[str, List[Dict[str, Any]]] = {}
    for tile in RIICHI_ALL_TILES:
        weights[tile] = 0.0
        details[tile] = []
        if tile in safe:
            continue
        for idx, wait in enumerate(RIICHI_TILE_WAITS.get(tile, [])):
            if f"{tile}:{idx}" in excluded:
                continue
            count = raw_combination_count(wait, r)
            if count <= 0:
                continue
            prior = wait_prior(wait, tile)
            weighted = count * prior
            if wait["type"] == "shanpon":
                partner_combos = total_pair_combos - binom(r[tile], 2)
                weighted *= partner_combos / FULL_PARTNER_COMBOS
            if weighted <= 0:
                continue
            group = group_count(wait, r)
            weights[tile] += weighted
            details[tile].append({
                "name": wait["name"],
                "type": wait["type"],
                "required": wait["required"],
                "count": count,
                "group": group,
                "prior": prior,
                "weighted": weighted,
            })

    total_weight = sum(weights.values())
    probs = {
        t: (weights[t] / total_weight if total_weight > 0 else 0.0)
        for t in RIICHI_ALL_TILES
    }
    return {
        "weights": weights,
        "probs": probs,
        "details": details,
        "totalWeight": total_weight,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# 副露解析（兼容前端简写格式）
# ---------------------------------------------------------------------------

def normalize_digit(d: int) -> int:
    return 5 if d == 0 else d


def to_tile_name(digit: int, suit: str) -> str:
    return f"{digit}{suit}"


def extract_digits_and_suit(inner: str) -> Dict[str, Any]:
    """从简写字符串中提取数字与花色位置，例如 '234m' -> {'digits':[2,3,4], 'suit':'m', 'suitPos':3}。"""
    digits: List[int] = []
    suit = None
    suit_pos = -1
    for i, ch in enumerate(inner):
        if re.match(r"[mpsz]", ch, re.IGNORECASE):
            if suit is not None:
                raise ValueError(f"副露中只能有一个花色：{inner}")
            suit = ch.lower()
            suit_pos = len(digits)
        elif ch.isdigit():
            digits.append(int(ch))
        else:
            raise ValueError(f"副露包含非法字符：{inner}")
    if suit is None:
        raise ValueError(f"副露缺少花色：{inner}")
    if suit_pos <= 0:
        raise ValueError(f"花色位置非法：{inner}")
    return {"digits": digits, "suit": suit, "suitPos": suit_pos}


def suit_pos_to_relative_source(num_tiles: int, suit_pos: int) -> str:
    if suit_pos == 1:
        return "上家"
    if suit_pos == num_tiles:
        return "下家"
    return "对家"


def relative_source_to_absolute_seat(caller_seat: str, relative: Optional[str]) -> Optional[str]:
    caller_idx = RIICHI_SEAT_POS.get(caller_seat)
    if caller_idx is None or not relative:
        return None
    offset = {"上家": -1, "对家": 2, "下家": 1}.get(relative)
    if offset is None:
        return None
    return RIICHI_SEATS[(caller_idx + offset + 4) % 4]


def is_pon(digits: List[int]) -> bool:
    norm = [normalize_digit(d) for d in digits]
    return all(d == norm[0] for d in norm)


def is_chii(digits: List[int], suit: str) -> bool:
    if suit == "z":
        return False
    norm = sorted(normalize_digit(d) for d in digits)
    return norm[0] + 1 == norm[1] and norm[1] + 1 == norm[2]


def sort_chii_tiles(digits: List[int], suit: str) -> List[str]:
    pairs = [(d, normalize_digit(d)) for d in digits]
    pairs.sort(key=lambda x: x[1])
    return [to_tile_name(d, suit) for d, _ in pairs]


def parse_open_meld(inner: str, caller_seat: str) -> Dict[str, Any]:
    # 末尾 'k' 标记加杠，如 (99s99k)
    is_kakan = inner.strip().lower().endswith("k")
    if is_kakan:
        inner = inner.strip()[:-1]
    parsed = extract_digits_and_suit(inner)
    digits, suit, suit_pos = parsed["digits"], parsed["suit"], parsed["suitPos"]
    relative_source = suit_pos_to_relative_source(len(digits), suit_pos)
    source_seat = relative_source_to_absolute_seat(caller_seat, relative_source)
    called_value = digits[suit_pos - 1]
    called_tile = to_tile_name(called_value, suit)
    n = len(digits)
    if is_kakan:
        if n != 4 or not is_pon(digits):
            raise ValueError(f"加杠必须是 4 张相同的牌：({inner}k)")
        return {
            "type": "kakan",
            "tiles": [to_tile_name(d, suit) for d in digits],
            "sourceSeat": source_seat,
            "calledTile": called_tile,
        }
    if n == 3:
        if is_pon(digits):
            return {
                "type": "pon",
                "tiles": [to_tile_name(d, suit) for d in digits],
                "sourceSeat": source_seat,
                "calledTile": called_tile,
            }
        if is_chii(digits, suit):
            return {
                "type": "chii",
                "tiles": sort_chii_tiles(digits, suit),
                "sourceSeat": source_seat,
                "calledTile": called_tile,
            }
        raise ValueError(f"既不是碰也不是吃的非法副露：({inner})")
    if n == 4:
        if not is_pon(digits):
            raise ValueError(f"大明杠必须是 4 张相同的牌：({inner})")
        return {
            "type": "daiminkan",
            "tiles": [to_tile_name(d, suit) for d in digits],
            "sourceSeat": source_seat,
            "calledTile": called_tile,
        }
    raise ValueError(f"副露牌张数错误（应为 3 或 4 张）：({inner})")


def parse_ankan(inner: str, caller_seat: str) -> Dict[str, Any]:
    parsed = extract_digits_and_suit(inner)
    digits, suit, suit_pos = parsed["digits"], parsed["suit"], parsed["suitPos"]
    if len(digits) != 4:
        raise ValueError(f"暗杠必须是 4 张牌：[{inner}]")
    if suit_pos != 4:
        raise ValueError(f"暗杠花色必须记在第 4 张牌后：[{inner}]")
    if not is_pon(digits):
        raise ValueError(f"暗杠必须是 4 张相同的牌：[{inner}]")
    tiles = [to_tile_name(d, suit) for d in digits]
    return {"type": "ankan", "tiles": tiles, "sourceSeat": None, "calledTile": None}


def parse_meld_token(token: str, caller_seat: str) -> Optional[Dict[str, Any]]:
    token = token.strip()
    if not token:
        return None
    if token.startswith("(") and token.endswith(")"):
        return parse_open_meld(token[1:-1], caller_seat)
    if token.startswith("[") and token.endswith("]"):
        return parse_ankan(token[1:-1], caller_seat)
    raise ValueError(f"无法识别的副露格式：{token}")


def parse_melds_input(text: Optional[str], caller_seat: str) -> List[Dict[str, Any]]:
    """将前端简写副露输入解析为标准对象列表。

    例如：
        "(234m) (555m) [6666p]" ->
        [
          {"type":"chii", "tiles":["2m","3m","4m"], "sourceSeat":"上家", "calledTile":"2m"},
          ...
        ]
    """
    if not text or not text.strip():
        return []
    tokens = text.strip().split()
    result: List[Dict[str, Any]] = []
    for token in tokens:
        m = parse_meld_token(token.strip(), caller_seat)
        if m:
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# 立直检测与安全牌分类
# ---------------------------------------------------------------------------

def strip_riichi_marker(tile: str) -> str:
    return re.sub(r"r$", "", tile, flags=re.IGNORECASE)


def has_riichi_marker(tile: str) -> bool:
    return bool(re.search(r"r$", tile, re.IGNORECASE))


def riichi_dealer_pos(seat_wind: str) -> int:
    wind_idx = {"east": 0, "south": 1, "west": 2, "north": 3}
    return (4 - wind_idx[seat_wind.lower()]) % 4


def riichi_global_index(seat: str, k: int, seat_wind: str) -> int:
    p = RIICHI_SEAT_POS[seat]
    d = riichi_dealer_pos(seat_wind)
    return 4 * (k - 1) + ((p - d + 4) % 4) + 1


def classify_from_river(
    seat_wind: str,
    hand: List[str],
    rivers: Dict[str, List[str]],
    melds: List[Dict[str, Any]],
    dora_indicators: List[str],
) -> Dict[str, Any]:
    """根据牌河、副露、手牌分类出现物与已知牌，并检测立直。

    Args:
        seat_wind: 自风，如 'east'
        hand: 自家手牌列表（已归一化）
        rivers: 四家牌河，key 为座位名，value 为带/不带 'r' 标记的出牌列表
        melds: 四家副露对象列表（带 sourceSeat / calledTile）
        dora_indicators: 宝牌指示牌列表（已归一化）

    Returns:
        {
            "safe": List[str],
            "known": List[str],
            "riichiList": List[Dict],
            "riichiDiscards": Dict[str, List[str]],
        }
    """
    # 解析四家牌河
    table: Dict[str, List[str]] = {}
    for seat in RIICHI_SEATS:
        table[seat] = list(rivers.get(seat, []))

    # 红五归一化副露牌
    all_meld_tiles = [normalize_tile(t) for m in melds for t in m.get("tiles", [])]

    # 检测对手立直
    riichi_list: List[Dict[str, Any]] = []
    for seat in RIICHI_SEATS:
        if seat == "自家":
            continue
        river = table.get(seat, [])
        for idx, tile in enumerate(river):
            if has_riichi_marker(tile):
                riichi_list.append({
                    "seat": seat,
                    "tile": strip_riichi_marker(tile),
                    "globalIdx": riichi_global_index(seat, idx + 1, seat_wind),
                })

    safe: List[str] = []
    known: List[str] = []
    riichi_discards: Dict[str, List[str]] = {}

    if not riichi_list:
        # 无立直：所有牌河进 known
        for seat in RIICHI_SEATS:
            for tile in table.get(seat, []):
                known.append(strip_riichi_marker(tile))
    else:
        cutoff = max(r["globalIdx"] for r in riichi_list)
        riichi_seats = {r["seat"] for r in riichi_list}

        # 按 cutoff 划分各家牌河
        for seat in RIICHI_SEATS:
            river = table.get(seat, [])
            for idx, tile in enumerate(river):
                clean_tile = strip_riichi_marker(tile)
                g_idx = riichi_global_index(seat, idx + 1, seat_wind)
                if g_idx >= cutoff:
                    safe.append(clean_tile)
                else:
                    known.append(clean_tile)

        # 收集立直家所有打出的牌（含宣言牌之前），用于筋牌排除
        for r in riichi_list:
            seat = r["seat"]
            river = table.get(seat, [])
            riichi_discards[seat] = [strip_riichi_marker(t) for t in river]

        def is_genbutsu_for_seat(seat: str, tile: str) -> bool:
            river = table.get(seat, [])
            for idx, t in enumerate(river):
                clean_tile = strip_riichi_marker(t)
                g_idx = riichi_global_index(seat, idx + 1, seat_wind)
                if clean_tile == tile and g_idx >= cutoff:
                    return True
            for m in melds:
                if m.get("sourceSeat") == seat and normalize_tile(m.get("calledTile", "")) == tile:
                    return True
            return False

        def remove_one_from_known(tile: str) -> None:
            if tile in known:
                known.remove(tile)

        # 根据副露来源补充现物
        for m in melds:
            source_seat = m.get("sourceSeat")
            called_tile = m.get("calledTile")
            if not source_seat or not called_tile:
                continue
            called_tile = normalize_tile(called_tile)
            if len(riichi_seats) == 1:
                only_seat = list(riichi_seats)[0]
                if source_seat == only_seat:
                    remove_one_from_known(called_tile)
                    safe.append(called_tile)
            else:
                all_genbutsu = all(is_genbutsu_for_seat(seat, called_tile) for seat in riichi_seats)
                if all_genbutsu:
                    remove_one_from_known(called_tile)
                    safe.append(called_tile)

    # 宝牌、手牌、所有副露牌都进 known
    known.extend(dora_indicators)
    known.extend(hand)
    known.extend(all_meld_tiles)

    return {
        "safe": safe,
        "known": known,
        "riichiList": riichi_list,
        "riichiDiscards": riichi_discards,
    }


# ---------------------------------------------------------------------------
# 主计算函数
# ---------------------------------------------------------------------------

def compute_riichi(
    seat_wind: str,
    hand: List[str],
    rivers: Dict[str, List[str]],
    melds: List[Dict[str, Any]],
    dora_indicators: List[str],
) -> Dict[str, Any]:
    """计算放铳概率。

    Args 与 classify_from_river 相同。

    Returns:
        与前端 riichiCompute() 输出同构的字典，额外包含 riichiList / riichiDiscards：
        {
            "r": Dict[str, int],
            "weights": Dict[str, float],
            "probs": Dict[str, float],
            "details": Dict[str, List[Dict]],
            "totalWeight": float,
            "countsSafe": Dict[str, int],
            "countsKnown": Dict[str, int],
            "riichiList": List[Dict],
            "riichiDiscards": Dict[str, List[str]],
        }
    """
    # 归一化输入
    hand = [normalize_tile(t) for t in hand]
    dora_indicators = [normalize_tile(t) for t in dora_indicators]
    normalized_rivers: Dict[str, List[str]] = {}
    for seat, tiles in rivers.items():
        normalized_rivers[seat] = [normalize_tile(t) for t in tiles]

    classified = classify_from_river(seat_wind, hand, normalized_rivers, melds, dora_indicators)
    safe_tiles = classified["safe"]
    known_tiles = classified["known"]
    riichi_list = classified["riichiList"]
    riichi_discards = classified["riichiDiscards"]

    counts_safe = count_tiles(safe_tiles)
    counts_known = count_tiles(known_tiles)

    # 校验每种牌 visible 总数不超过 4
    for t in RIICHI_ALL_TILES:
        total = counts_safe[t] + counts_known[t]
        if total > 4:
            raise ValueError(f"牌 {t} 输入总数为 {total} 张，超过 4 张上限。")

    # 剩余牌山
    r = {t: 4 - counts_safe[t] - counts_known[t] for t in RIICHI_ALL_TILES}
    n_unseen = sum(r.values())

    risk = compute_wait_risk(r, safe_tiles, riichi_discards)
    weights = risk["weights"]
    probs = risk["probs"]
    details = risk["details"]
    total_weight = risk["totalWeight"]

    return {
        "r": r,
        "weights": weights,
        "probs": probs,
        "details": details,
        "totalWeight": total_weight,
        "countsSafe": counts_safe,
        "countsKnown": counts_known,
        "riichiList": riichi_list,
        "riichiDiscards": riichi_discards,
        "rivers": normalized_rivers,
    }


# ---------------------------------------------------------------------------
# 便捷函数：从后端常见输入格式直接计算
# ---------------------------------------------------------------------------

def compute_riichi_from_form_data(
    seat_wind: str,
    hand: List[str],
    seen_self: Optional[List[str]] = None,
    seen_next: Optional[List[str]] = None,
    seen_across: Optional[List[str]] = None,
    seen_prev: Optional[List[str]] = None,
    melds_self: Optional[List[Dict[str, Any]]] = None,
    melds_next: Optional[List[Dict[str, Any]]] = None,
    melds_across: Optional[List[Dict[str, Any]]] = None,
    melds_prev: Optional[List[Dict[str, Any]]] = None,
    dora_indicators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """从后端 /api/analyze 的表单字段格式直接计算铳率。"""
    rivers = {
        "自家": list(seen_self or []),
        "下家": list(seen_next or []),
        "对家": list(seen_across or []),
        "上家": list(seen_prev or []),
    }
    melds = (
        list(melds_self or [])
        + list(melds_next or [])
        + list(melds_across or [])
        + list(melds_prev or [])
    )
    return compute_riichi(
        seat_wind=seat_wind,
        hand=hand,
        rivers=rivers,
        melds=melds,
        dora_indicators=dora_indicators or [],
    )


if __name__ == "__main__":
    # 简单自测：无立直时应为均匀分布（34 种牌等概率）
    result = compute_riichi(
        seat_wind="east",
        hand=["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1p"],
        rivers={"自家": [], "下家": [], "对家": [], "上家": []},
        melds=[],
        dora_indicators=[],
    )
    print("totalWeight:", result["totalWeight"])
    print("1m prob:", result["probs"]["1m"])
    print("5z prob:", result["probs"]["5z"])
