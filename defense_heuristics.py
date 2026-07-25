"""Explainable river and open-meld danger modifiers.

The rules in this module adjust relative danger weights.  They are deliberately
small, bounded heuristics rather than calibrated deal-in probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .params import PARAMS as _P

_D = _P["danger"]

SUITS = ("m", "p", "s")
WINDS = ("east", "south", "west", "north")
SEAT_OFFSETS = {"自家": 0, "下家": 1, "对家": 2, "上家": 3}
WIND_TILES = {"east": "1z", "south": "2z", "west": "3z", "north": "4z"}
OPEN_TYPES = frozenset({"chii", "pon", "daiminkan", "kakan"})
TRIPLET_TYPES = frozenset({"pon", "daiminkan", "kakan"})

ALL_TILES: List[str] = (
    [f"{n}{suit}" for suit in SUITS for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
)

# 经验系数统一存于 params.py（此处仅为兼容别名）
RED_FIVE_37_FACTOR = _D["red_five_37"]
EARLY_OUTSIDE_FACTOR = _D["early_outside"]
FLUSH_SUIT_FACTOR = _D["flush_suit"]
HONITSU_HONOR_FACTOR = _D["honitsu_honor"]
YAKU_RELATED_FACTOR = _D["yaku_related"]
TERMINAL_YAKU_FACTOR = _D["terminal_yaku"]
CHANTA_HONOR_FACTOR = _D["chanta_honor"]
DORA_SELF_FACTOR = _D["dora_self"]
DORA_NEAR1_FACTOR = _D["dora_near1"]
DORA_NEAR2_FACTOR = _D["dora_near2"]
RIICHI_DECL_NEAR_FACTOR = _D["riichi_decl_near"]
MIN_AGGREGATE_FACTOR = _D["aggregate_min"]
MAX_AGGREGATE_FACTOR = _D["aggregate_max"]
EARLY_OUTSIDE_DISCARDS = _D["early_outside_discards"]
FLUSH_READ_DISCARDS = _D["flush_read_discards"]

EARLY_OUTSIDE_MAP = {
    2: (1,),
    3: (1, 2),
    7: (8, 9),
    8: (9,),
}


def normalize_tile(tile: Any) -> str:
    value = str(tile or "").strip().lower()
    if value.endswith("r"):
        value = value[:-1]
    if value.startswith("0"):
        value = "5" + value[1:]
    return value


def _tile_sort_key(tile: str) -> Tuple[int, int]:
    return ({"m": 0, "p": 1, "s": 2, "z": 3}.get(tile[-1:], 9), int(tile[0]))


def _sorted_tiles(tiles: Iterable[str]) -> List[str]:
    return sorted({normalize_tile(tile) for tile in tiles if normalize_tile(tile)}, key=_tile_sort_key)


def _open_melds(melds: Sequence[dict]) -> List[dict]:
    return [meld for meld in melds if meld.get("type") in OPEN_TYPES]


def _meld_tiles(meld: dict) -> List[str]:
    return [normalize_tile(tile) for tile in (meld.get("tiles") or [])]


def _triplet_tile(meld: dict) -> Optional[str]:
    if meld.get("type") not in TRIPLET_TYPES:
        return None
    tiles = _meld_tiles(meld)
    if not tiles or any(tile != tiles[0] for tile in tiles):
        return None
    return tiles[0]


def _chii_shape(meld: dict) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    if meld.get("type") != "chii":
        return None
    tiles = _meld_tiles(meld)
    if len(tiles) != 3 or any(tile[-1:] not in SUITS for tile in tiles):
        return None
    suit = tiles[0][-1]
    if any(tile[-1] != suit for tile in tiles):
        return None
    ranks = tuple(sorted(int(tile[0]) for tile in tiles))
    if ranks[0] + 1 != ranks[1] or ranks[1] + 1 != ranks[2]:
        return None
    return suit, ranks


def _target_wind(self_wind: str, relative_seat: str) -> str:
    try:
        index = WINDS.index(str(self_wind).strip().lower())
    except ValueError:
        index = 0
    return WINDS[(index + SEAT_OFFSETS.get(relative_seat, 0)) % 4]


def _yakuhai_tiles(dp: Any, seat: str) -> Set[str]:
    round_tile = WIND_TILES.get(str(getattr(dp, "round_wind", "")).lower())
    seat_tile = WIND_TILES[_target_wind(getattr(dp, "seat_wind", "east"), seat)]
    return {tile for tile in (round_tile, seat_tile, "5z", "6z", "7z") if tile}


def _has_open_yakuhai(dp: Any, seat: str, melds: Sequence[dict]) -> bool:
    yakuhai = _yakuhai_tiles(dp, seat)
    return any(_triplet_tile(meld) in yakuhai for meld in melds)


def _guest_wind_pons(dp: Any, seat: str, melds: Sequence[dict]) -> List[str]:
    yakuhai = _yakuhai_tiles(dp, seat)
    return [
        tile
        for meld in melds
        for tile in [_triplet_tile(meld)]
        if tile in {"1z", "2z", "3z", "4z"} and tile not in yakuhai
    ]


def _add_signal(
    factors: Dict[str, float],
    signals: List[Dict[str, Any]],
    *,
    rule_id: str,
    label: str,
    factor: float,
    evidence: Iterable[str],
    tiles: Iterable[str],
) -> None:
    affected = _sorted_tiles(tile for tile in tiles if normalize_tile(tile) in factors)
    if not affected:
        return
    for tile in affected:
        factors[tile] *= factor
    signals.append(
        {
            "id": rule_id,
            "label": label,
            "direction": "up" if factor > 1.0 else "down",
            "factor": factor,
            "evidence": list(dict.fromkeys(str(item) for item in evidence if item)),
            "tiles": affected,
        }
    )


def _river_safety_signals(
    events: Sequence[dict],
    factors: Dict[str, float],
    signals: List[Dict[str, Any]],
) -> None:
    for event in events:
        raw = str(event.get("tile_raw") or "")
        is_red = bool(event.get("is_red")) or raw.startswith("0")
        if not is_red or event.get("tedashi") is not True:
            continue
        suit = (raw or str(event.get("tile") or ""))[-1:].lower()
        if suit not in SUITS:
            continue
        _add_signal(
            factors,
            signals,
            rule_id=f"red_five_37_{suit}",
            label=f"手切红5过同色3、7",
            factor=RED_FIVE_37_FACTOR,
            evidence=[raw or f"0{suit}"],
            tiles=[f"3{suit}", f"7{suit}"],
        )

    early = list(events[:EARLY_OUTSIDE_DISCARDS])
    for index, event in enumerate(early, 1):
        if event.get("tedashi") is not True:
            continue
        tile = normalize_tile(event.get("tile"))
        if len(tile) != 2 or tile[-1] not in SUITS:
            continue
        rank = int(tile[0])
        outside = EARLY_OUTSIDE_MAP.get(rank)
        if not outside:
            continue
        targets = [f"{number}{tile[-1]}" for number in outside]
        _add_signal(
            factors,
            signals,
            rule_id=f"early_outside_{index}_{tile}",
            label=f"前6巡手切{tile}的早外",
            factor=EARLY_OUTSIDE_FACTOR,
            evidence=[tile],
            tiles=targets,
        )


def _flush_pattern_candidate(events: Sequence[dict]) -> Tuple[Optional[str], List[str]]:
    early = list(events[:FLUSH_READ_DISCARDS])
    for index, event in enumerate(early):
        tile = normalize_tile(event.get("tile"))
        if event.get("tedashi") is not True or not tile.endswith("z"):
            continue
        prior_suits = {
            prior_tile[-1]
            for prior in early[:index]
            for prior_tile in [normalize_tile(prior.get("tile"))]
            if prior.get("tedashi") is True and prior_tile[-1:] in SUITS
        }
        if len(prior_suits) == 2:
            candidate = next(suit for suit in SUITS if suit not in prior_suits)
            evidence = [
                normalize_tile(prior.get("tile"))
                for prior in early[:index]
                if prior.get("tedashi") is True
                and normalize_tile(prior.get("tile"))[-1:] in prior_suits
            ]
            evidence.append(tile)
            return candidate, evidence
    return None, []


def _flush_signals(
    dp: Any,
    seat: str,
    events: Sequence[dict],
    melds: Sequence[dict],
    factors: Dict[str, float],
    signals: List[Dict[str, Any]],
) -> None:
    suited_meld_suits = {
        tile[-1]
        for meld in melds
        for tile in _meld_tiles(meld)
        if tile[-1:] in SUITS
    }
    if len(suited_meld_suits) > 1:
        return

    guest_winds = _guest_wind_pons(dp, seat, melds)
    pattern_suit, pattern_evidence = _flush_pattern_candidate(events)
    meld_suit = next(iter(suited_meld_suits), None)
    candidate = pattern_suit or meld_suit

    if pattern_suit and meld_suit and pattern_suit != meld_suit:
        return
    if candidate is None and guest_winds:
        early_suits = {
            normalize_tile(event.get("tile"))[-1]
            for event in events[:FLUSH_READ_DISCARDS]
            if event.get("tedashi") is True
            and normalize_tile(event.get("tile"))[-1:] in SUITS
        }
        if len(early_suits) == 2:
            candidate = next(suit for suit in SUITS if suit not in early_suits)

    # A river pattern is independently meaningful.  A guest-wind call can use
    # either a consistent suited meld or the two early discarded suits to
    # identify the likely flush suit.
    if candidate is None or (not pattern_suit and not guest_winds):
        return

    evidence = pattern_evidence + guest_winds
    for meld in melds:
        if candidate in {tile[-1:] for tile in _meld_tiles(meld)}:
            evidence.extend(_meld_tiles(meld))
    _add_signal(
        factors,
        signals,
        rule_id=f"flush_tendency_{candidate}",
        label=f"{candidate}花色染手倾向",
        factor=FLUSH_SUIT_FACTOR,
        evidence=evidence,
        tiles=[f"{number}{candidate}" for number in range(1, 10)],
    )
    if guest_winds:
        _add_signal(
            factors,
            signals,
            rule_id=f"honitsu_honors_{candidate}",
            label="客风副露下的混一色字牌倾向",
            factor=HONITSU_HONOR_FACTOR,
            evidence=guest_winds,
            tiles=[f"{number}z" for number in range(1, 8)],
        )


def _terminal_yaku_compatible(meld: dict) -> bool:
    chii = _chii_shape(meld)
    if chii:
        return chii[1] in ((1, 2, 3), (7, 8, 9))
    triplet = _triplet_tile(meld)
    return bool(triplet and (triplet.endswith("z") or triplet[0] in ("1", "9")))


def _yaku_signals(
    dp: Any,
    seat: str,
    melds: Sequence[dict],
    factors: Dict[str, float],
    signals: List[Dict[str, Any]],
) -> None:
    if not melds or _has_open_yakuhai(dp, seat, melds):
        return

    chiis: Dict[Tuple[int, int, int], Set[str]] = {}
    runs_by_suit: Dict[str, Set[Tuple[int, int, int]]] = {}
    terminal_triplets: Dict[int, Set[str]] = {1: set(), 9: set()}
    for meld in melds:
        chii = _chii_shape(meld)
        if chii:
            suit, ranks = chii
            runs_by_suit.setdefault(suit, set()).add(ranks)
            if ranks in ((1, 2, 3), (7, 8, 9)):
                chiis.setdefault(ranks, set()).add(suit)
        triplet = _triplet_tile(meld)
        if triplet and triplet[-1:] in SUITS and int(triplet[0]) in (1, 9):
            terminal_triplets[int(triplet[0])].add(triplet[-1])

    for ranks, represented_suits in sorted(chiis.items()):
        missing_suits = [suit for suit in SUITS if suit not in represented_suits]
        _add_signal(
            factors,
            signals,
            rule_id=f"sanshoku_doujun_{ranks[0]}",
            label=f"{ranks[0]}{ranks[1]}{ranks[2]}三色同顺候选",
            factor=YAKU_RELATED_FACTOR,
            evidence=[f"{rank}{suit}" for suit in represented_suits for rank in ranks],
            tiles=[f"{rank}{suit}" for suit in missing_suits for rank in ranks],
        )

    all_runs = {(1, 2, 3), (4, 5, 6), (7, 8, 9)}
    for suit, represented_runs in sorted(runs_by_suit.items()):
        if not represented_runs.intersection({(1, 2, 3), (7, 8, 9)}):
            continue
        missing_runs = all_runs - represented_runs
        _add_signal(
            factors,
            signals,
            rule_id=f"ittsu_{suit}",
            label=f"{suit}花色一气通贯候选",
            factor=YAKU_RELATED_FACTOR,
            evidence=[
                f"{rank}{suit}"
                for ranks in represented_runs
                for rank in ranks
            ],
            tiles=[f"{rank}{suit}" for ranks in missing_runs for rank in ranks],
        )

    for rank, represented_suits in terminal_triplets.items():
        if not represented_suits:
            continue
        missing_suits = [suit for suit in SUITS if suit not in represented_suits]
        _add_signal(
            factors,
            signals,
            rule_id=f"sanshoku_doukou_{rank}",
            label=f"{rank}三色同刻候选",
            factor=YAKU_RELATED_FACTOR,
            evidence=[f"{rank}{suit}" for suit in represented_suits],
            tiles=[f"{rank}{suit}" for suit in missing_suits],
        )

    if all(_terminal_yaku_compatible(meld) for meld in melds):
        _add_signal(
            factors,
            signals,
            rule_id="chanta_junchan_numeric",
            label="混全/纯全带幺九候选",
            factor=TERMINAL_YAKU_FACTOR,
            evidence=[tile for meld in melds for tile in _meld_tiles(meld)],
            tiles=[f"{rank}{suit}" for suit in SUITS for rank in (1, 2, 3, 7, 8, 9)],
        )
        _add_signal(
            factors,
            signals,
            rule_id="chanta_honors",
            label="混全带幺九字牌候选",
            factor=CHANTA_HONOR_FACTOR,
            evidence=[tile for meld in melds for tile in _meld_tiles(meld)],
            tiles=[f"{rank}z" for rank in range(1, 8)],
        )


def _dora_from_indicator(indicator: str) -> Optional[str]:
    """指示牌 → 宝牌（数牌 9→1 循环，风牌 4z→1z，三元 7z→5z）。"""
    tile = normalize_tile(indicator)
    if len(tile) != 2:
        return None
    suit = tile[-1]
    try:
        rank = int(tile[0])
    except ValueError:
        return None
    if suit in SUITS:
        return f"{rank % 9 + 1}{suit}"
    if suit == "z":
        if 1 <= rank <= 4:
            return f"{rank % 4 + 1}z"
        if 5 <= rank <= 7:
            return f"{(rank - 5 + 1) % 3 + 5}z"
    return None


def _near_tiles(tile: str, distances: Set[int]) -> List[str]:
    """同花色 ±distances 的邻牌（字牌无邻牌）。"""
    if len(tile) != 2 or tile[-1] not in SUITS:
        return []
    suit = tile[-1]
    rank = int(tile[0])
    out = []
    for d in distances:
        for r in (rank - d, rank + d):
            if 1 <= r <= 9:
                out.append(f"{r}{suit}")
    return out


def _red_dora_tiles(dp: Any) -> Set[str]:
    """场上可见的赤宝牌（手牌 0 记法 + 牌河 is_red），只加成牌种本身。"""
    reds: Set[str] = set()
    for tile in getattr(dp, "hand", None) or []:
        value = str(tile or "")
        if value.startswith("0") and value[-1:] in SUITS:
            reds.add(f"5{value[-1]}")
    for event in getattr(dp, "discards_log", None) or []:
        if event.get("is_red"):
            tile = normalize_tile(event.get("tile") or "")
            if tile and tile[-1] in SUITS and tile[0] == "5":
                reds.add(tile)
    return reds


def _dora_signals(
    dp: Any,
    factors: Dict[str, float],
    signals: List[Dict[str, Any]],
) -> None:
    doras: List[str] = []
    for ind in getattr(dp, "dora_indicators", None) or []:
        dora = _dora_from_indicator(str(ind))
        if dora and dora not in doras:
            doras.append(dora)
    for dora in doras:
        _add_signal(
            factors,
            signals,
            rule_id=f"dora_self_{dora}",
            label=f"宝牌{dora}本身",
            factor=DORA_SELF_FACTOR,
            evidence=[dora],
            tiles=[dora],
        )
        near1 = _near_tiles(dora, {1})
        _add_signal(
            factors,
            signals,
            rule_id=f"dora_near1_{dora}",
            label=f"宝牌{dora}邻牌",
            factor=DORA_NEAR1_FACTOR,
            evidence=[dora],
            tiles=near1,
        )
        near2 = _near_tiles(dora, {2})
        _add_signal(
            factors,
            signals,
            rule_id=f"dora_near2_{dora}",
            label=f"宝牌{dora}间牌",
            factor=DORA_NEAR2_FACTOR,
            evidence=[dora],
            tiles=near2,
        )
    for red in sorted(_red_dora_tiles(dp)):
        _add_signal(
            factors,
            signals,
            rule_id=f"red_dora_self_{red}",
            label=f"赤宝牌{red}本身",
            factor=DORA_SELF_FACTOR,
            evidence=[red],
            tiles=[red],
        )


def _riichi_declaration_signals(
    events: Sequence[dict],
    factors: Dict[str, float],
    signals: List[Dict[str, Any]],
) -> None:
    """立直宣言牌跨筋（±1/±2）中幅加成；仅对立直家。"""
    riichi_ev = next((e for e in events if e.get("riichi")), None)
    if not riichi_ev:
        return
    decl = normalize_tile(riichi_ev.get("tile") or "")
    if not decl or decl[-1] not in SUITS:
        return
    near = _near_tiles(decl, {1, 2})
    _add_signal(
        factors,
        signals,
        rule_id=f"riichi_decl_near_{decl}",
        label=f"立直宣言牌{decl}跨筋",
        factor=RIICHI_DECL_NEAR_FACTOR,
        evidence=[decl],
        tiles=near,
    )


def analyze_danger_signals(dp: Any, seat: str) -> Dict[str, Any]:
    """Return bounded per-tile factors and an explanation trace for one target."""
    factors = {tile: 1.0 for tile in ALL_TILES}
    signals: List[Dict[str, Any]] = []
    events = [
        event
        for event in (getattr(dp, "discards_log", None) or [])
        if event.get("rel") == seat
    ]
    melds = _open_melds(
        (getattr(dp, "melds_by_rel", None) or {}).get(seat) or []
    )

    _river_safety_signals(events, factors, signals)
    _flush_signals(dp, seat, events, melds, factors, signals)
    _yaku_signals(dp, seat, melds, factors, signals)
    _dora_signals(dp, factors, signals)
    _riichi_declaration_signals(events, factors, signals)

    bounded = {
        tile: min(MAX_AGGREGATE_FACTOR, max(MIN_AGGREGATE_FACTOR, value))
        for tile, value in factors.items()
    }
    return {
        "tile_factors": bounded,
        "modifiers": {
            tile: value for tile, value in bounded.items() if abs(value - 1.0) > 1e-12
        },
        "signals": signals,
        "bounds": {
            "minimum": MIN_AGGREGATE_FACTOR,
            "maximum": MAX_AGGREGATE_FACTOR,
        },
    }
