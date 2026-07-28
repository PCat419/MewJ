"""Tenhou.net/6 JSON → per-seat decision-point snapshots (Classic review style)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Union, Tuple

from .converter import meld_tile_sort_key

# Tenhou JSON tile int → mpsz
def tenhou_to_mpsz(tile: int) -> str:
    if tile == 51:
        return "0m"
    if tile == 52:
        return "0p"
    if tile == 53:
        return "0s"
    suit = tile // 10
    num = tile % 10
    if suit == 1:
        return f"{num}m"
    if suit == 2:
        return f"{num}p"
    if suit == 3:
        return f"{num}s"
    if suit == 4:
        return f"{num}z"
    raise ValueError(f"bad tenhou tile: {tile}")


def mpsz_list(tiles: List[int]) -> List[str]:
    return [tenhou_to_mpsz(t) for t in tiles]


def mpsz_sort_key(tile: str):
    """man → pin → sou → honor; aka 5 sorts just before plain 5."""
    t = str(tile).strip().lower()
    if len(t) < 2:
        return (9, 99, 9, t)
    suit = t[-1]
    raw = t[0]
    suit_rank = {"m": 0, "p": 1, "s": 2, "z": 3}.get(suit, 9)
    try:
        num = 5 if raw == "0" else int(raw)
    except ValueError:
        num = 99
    aka_rank = 0 if raw == "0" else 1
    return (suit_rank, num, aka_rank, t)


def sort_hand_mpsz(tiles: List[str]) -> List[str]:
    return sorted(tiles, key=mpsz_sort_key)


_WIND_NAMES = ("east", "south", "west", "north")
_KYOKU_LABEL = (
    [f"东{i}" for i in range(1, 5)]
    + [f"南{i}" for i in range(1, 5)]
    + [f"西{i}" for i in range(1, 5)]
    + [f"北{i}" for i in range(1, 5)]
)


def kyoku_label(kyoku_idx: int, honba: int) -> str:
    base = _KYOKU_LABEL[kyoku_idx] if 0 <= kyoku_idx < len(_KYOKU_LABEL) else f"局{kyoku_idx}"
    return f"{base}-{honba}本场"


def round_and_seat_wind(kyoku_idx: int, seat: int) -> tuple[str, str]:
    """Return (round_wind, seat_wind) for player `seat` in this kyoku."""
    round_wind = _WIND_NAMES[kyoku_idx // 4]
    dealer = kyoku_idx % 4
    seat_wind = _WIND_NAMES[(seat - dealer) % 4]
    return round_wind, seat_wind


def _split_tile_groups(token: str) -> List[Union[str, int]]:
    """Split '1212p12' into [12, 12, 'p12'] style groups."""
    parts: List[Union[str, int]] = []
    i = 0
    while i < len(token):
        if token[i] in "cpmka":
            marker = token[i]
            i += 1
            if i + 1 >= len(token) or not token[i : i + 2].isdigit():
                raise ValueError(f"bad meld token: {token}")
            parts.append(marker + token[i : i + 2])
            i += 2
        elif token[i].isdigit():
            if i + 1 >= len(token) or not token[i + 1].isdigit():
                raise ValueError(f"bad meld token: {token}")
            parts.append(int(token[i : i + 2]))
            i += 2
        else:
            raise ValueError(f"bad meld token: {token}")
    return parts


@dataclass
class ParsedMeld:
    type: str  # chii | pon | daiminkan | ankan | kakan
    tiles: List[int]  # all tiles in the meld (incl. called)
    called: Optional[int]
    hand_tiles: List[int]  # tiles removed from concealed hand


def parse_meld_token(token: str) -> ParsedMeld:
    token = str(token)
    if "c" in token:
        # c{called}{a}{b}
        m = re.fullmatch(r"c(\d{2})(\d{2})(\d{2})", token)
        if not m:
            raise ValueError(f"bad chi: {token}")
        called, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return ParsedMeld("chii", [called, a, b], called, [a, b])

    if "a" in token:
        # ankan: ...aXX
        groups = _split_tile_groups(token)
        tiles: List[int] = []
        for g in groups:
            if isinstance(g, int):
                tiles.append(g)
            elif isinstance(g, str) and g.startswith("a"):
                tiles.append(int(g[1:]))
            else:
                raise ValueError(f"bad ankan: {token}")
        if len(tiles) != 4:
            raise ValueError(f"ankan needs 4 tiles: {token}")
        return ParsedMeld("ankan", tiles, None, list(tiles))

    if "k" in token:
        groups = _split_tile_groups(token)
        tiles = []
        called = None
        for g in groups:
            if isinstance(g, int):
                tiles.append(g)
            elif isinstance(g, str) and g.startswith("k"):
                called = int(g[1:])
                tiles.append(called)
            else:
                raise ValueError(f"bad kakan: {token}")
        if called is None or len(tiles) != 4:
            raise ValueError(f"bad kakan: {token}")
        # kakan: only the added tile leaves the hand (the other 3 already in pon)
        return ParsedMeld("kakan", tiles, called, [called])

    if "m" in token:
        groups = _split_tile_groups(token)
        tiles = []
        called = None
        for g in groups:
            if isinstance(g, int):
                tiles.append(g)
            elif isinstance(g, str) and g.startswith("m"):
                called = int(g[1:])
                tiles.append(called)
            else:
                raise ValueError(f"bad daiminkan: {token}")
        if called is None:
            raise ValueError(f"bad daiminkan: {token}")
        hand_tiles = [t for t in tiles if True]
        # remove 3 from hand (not the called one) — count carefully with duplicates
        hand_remove = list(tiles)
        hand_remove.remove(called)
        return ParsedMeld("daiminkan", tiles, called, hand_remove)

    if "p" in token:
        groups = _split_tile_groups(token)
        tiles = []
        called = None
        for g in groups:
            if isinstance(g, int):
                tiles.append(g)
            elif isinstance(g, str) and g.startswith("p"):
                called = int(g[1:])
                tiles.append(called)
            else:
                raise ValueError(f"bad pon: {token}")
        if called is None or len(tiles) != 3:
            raise ValueError(f"bad pon: {token}")
        hand_remove = list(tiles)
        hand_remove.remove(called)
        return ParsedMeld("pon", tiles, called, hand_remove)

    raise ValueError(f"unknown meld token: {token}")


def _remove_tiles(hand: Counter, tiles: List[int]) -> None:
    for t in tiles:
        if hand[t] <= 0:
            # aka / normal 5 interchange: 51↔15 style not in tenhou ints for non-aka
            # try aka equivalents for 5s
            alt = None
            if t in (15, 25, 35):
                alt = {15: 51, 25: 52, 35: 53}[t]
            elif t in (51, 52, 53):
                alt = {51: 15, 52: 25, 53: 35}[t]
            if alt is not None and hand[alt] > 0:
                hand[alt] -= 1
                continue
            raise ValueError(f"cannot remove {t} from hand {dict(hand)}")
        hand[t] -= 1
        if hand[t] == 0:
            del hand[t]


def _hand_list(hand: Counter) -> List[int]:
    out: List[int] = []
    for t in sorted(hand.keys()):
        out.extend([t] * hand[t])
    return out


_REL_SEATS = ("自家", "下家", "对家", "上家")


def abs_to_rel_seat(abs_seat: int, self_seat: int) -> str:
    return _REL_SEATS[(abs_seat - self_seat) % 4]


@dataclass
class DecisionPoint:
    kyoku_index: int
    kyoku_meta: List[int]  # [kyoku, honba, riichi_sticks]
    seat: int
    turn: int  # 1-based draw count for this seat
    hand: List[str]  # mpsz, sorted concealed tiles at decision
    drawn_tile: Optional[str]  # just-drawn tile (mpsz); None after chi/pon
    melds: List[dict]  # analyze-api style (自家)
    dora_indicators: List[str]
    round_wind: str
    seat_wind: str
    actual_discard: Optional[str]  # mpsz；实加杠巡为 None
    actual_discard_raw: Optional[Union[int, str]]
    is_riichi_discard: bool
    is_tsumogiri: bool
    scores: List[int]
    label: str  # e.g. 东1 Turn 3
    # 防守分析：相对自家的四家牌河（可含末尾 r 立直标记）与副露
    rivers: dict = field(default_factory=dict)  # 自家/下家/对家/上家 → [mpsz...]
    melds_by_rel: dict = field(default_factory=dict)  # 同上 → [meld...]
    # 全局打出记录（本决策之前），seq 从 1 起
    # {seq, abs, rel, tile, tile_raw, is_red, tedashi, riichi}
    # ``tile`` is normalized (0m -> 5m); ``tile_raw`` preserves red fives.
    discards_log: list = field(default_factory=list)
    # 各家首次副露完成时，被吃碰杠的那张牌的全局 seq（副露后切牌 seq 必大于此）
    open_after_seq: dict = field(default_factory=dict)  # rel → int
    # 加杠：合法选项 [{tile, pon_index}, ...]；实加杠时 actual_kakan 为消耗牌
    legal_kakans: list = field(default_factory=list)
    actual_kakan: Optional[str] = None
    # 暗杠：合法选项 [{tiles, kind}, ...]；实暗杠时 actual_ankan 为代表牌
    legal_ankans: list = field(default_factory=list)
    actual_ankan: Optional[str] = None
    # 立直后仅暗杠窗口产点（强制摸切不检讨，只对照杠）
    is_riichi_post: bool = False


@dataclass
class CallOpportunity:
    """他家舍牌时自家的副露机会点（碰/吃 vs 跳过的反事实评估输入）。"""

    kyoku_index: int
    kyoku_meta: List[int]  # [kyoku, honba, riichi_sticks]
    seat: int
    turn: int  # 自家即将进行的巡 = 自家已摸次数 + 1
    hand: List[str]  # mpsz，副露前 13 张（排序）
    melds: List[dict]  # analyze-api style（自家已有副露）
    dora_indicators: List[str]
    round_wind: str
    seat_wind: str
    scores: List[int]
    discarder: int  # 打出被副露牌的绝对座位
    discarder_rel: str  # 相对自家：上家/对家/下家
    disc_tile: str  # 被副露牌 mpsz
    legal: dict  # legal_calls 结果：{"pon": [...], "chii": [...], "daiminkan": [...]}
    actual: str  # pon / chii / daiminkan / skip
    actual_tiles: List[str]  # 实际副露消耗的手牌 mpsz（skip 为 []）
    label: str
    # 牌桌快照（含刚打出、等待副露的这张牌）
    rivers: dict = field(default_factory=dict)  # 自家/下家/对家/上家 → [mpsz...]
    melds_by_rel: dict = field(default_factory=dict)  # 同上 → [meld...]
    discards_log: list = field(default_factory=list)
    open_after_seq: dict = field(default_factory=dict)  # rel → int
    # 自家已舍牌（归一化 mpsz，0m→5m），跳过侧振听置 0 用
    self_discards: List[str] = field(default_factory=list)
    # 实际副露后的实切 mpsz（actual 为 pon/chii 且可解析时；大明杠/加杠 token 记 None）
    actual_cut: Optional[str] = None


@dataclass
class KyokuView:
    index: int
    label: str
    meta: List[int]
    scores: List[int]
    result: Any
    dora_indicators: List[str] = field(default_factory=list)
    seat_wind: str = "east"
    decisions: List[Union[DecisionPoint, CallOpportunity]] = field(default_factory=list)


def _meld_to_api(parsed: ParsedMeld, source_seat: Optional[str] = None) -> dict:
    type_map = {
        "chii": "chii",
        "pon": "pon",
        "daiminkan": "daiminkan",
        "ankan": "ankan",
        "kakan": "kakan",
    }
    tiles = list(parsed.tiles)
    # 吃张按序位升序（与 converter.parse_melds / nanikiru 一致）；乱序会算错役
    if parsed.type == "chii":
        tiles = sorted(tiles, key=meld_tile_sort_key)
    return {
        "type": type_map[parsed.type],
        "tiles": mpsz_list(tiles),
        "calledTile": tenhou_to_mpsz(parsed.called) if parsed.called is not None else None,
        "sourceSeat": source_seat,
    }


def _tile_id_eq(a: int, b: int) -> bool:
    if a == b:
        return True
    aka = {15: 51, 25: 52, 35: 53, 51: 15, 52: 25, 53: 35}
    return aka.get(a) == b


def _source_abs_from_token(token: str, caller_abs: int) -> Optional[int]:
    """Infer absolute seat of the discarder from a Tenhou meld token."""
    tok = str(token)
    if "c" in tok:
        return (caller_abs + 3) % 4  # chi always from 上家
    if "a" in tok:
        return None
    groups = _split_tile_groups(tok)
    slot = next((i for i, g in enumerate(groups) if isinstance(g, str)), None)
    if slot is None:
        return None
    n = len(groups)
    if slot == 0:
        offset = -1  # 上家
    elif slot == n - 1:
        offset = 1  # 下家
    else:
        offset = 2  # 对家
    return (caller_abs + offset) % 4


def _is_ankan_or_kakan_token(tok: Any) -> bool:
    if not isinstance(tok, str):
        return False
    try:
        parsed = parse_meld_token(tok)
    except ValueError:
        return False
    return parsed.type in ("ankan", "kakan")


def _find_caller(
    draws: List[List[Any]],
    i_d: List[int],
    discarder: int,
    disc_tile: int,
) -> Optional[int]:
    """Who (if anyone) calls the just-discarded tile next."""
    candidates: List[tuple] = []
    priority = {"daiminkan": 3, "pon": 2, "chii": 1}
    for a in range(4):
        if a == discarder or i_d[a] >= len(draws[a]):
            continue
        tok = draws[a][i_d[a]]
        if not isinstance(tok, str) or "a" in tok:
            continue
        try:
            parsed = parse_meld_token(tok)
        except ValueError:
            continue
        if parsed.called is None or not _tile_id_eq(parsed.called, disc_tile):
            continue
        src = _source_abs_from_token(tok, a)
        if src != discarder:
            continue
        candidates.append((priority.get(parsed.type, 0), a))
    if not candidates:
        return None
    return max(candidates)[1]


def _parse_discard_token(
    disc_tok: Any, last_drawn: Optional[int]
) -> tuple:
    """Return (tile_id, is_riichi, is_tsumogiri)."""
    is_riichi = False
    if isinstance(disc_tok, str) and disc_tok.startswith("r"):
        is_riichi = True
        inner = disc_tok[1:]
        if inner == "60":
            if last_drawn is None:
                raise ValueError("r60 without last_drawn")
            return last_drawn, True, True
        disc_tile = int(inner)
        return disc_tile, True, disc_tile == last_drawn
    if disc_tok == 60:
        if last_drawn is None:
            raise ValueError("60 without last_drawn")
        return last_drawn, False, True
    disc_tile = int(disc_tok)
    return disc_tile, is_riichi, disc_tile == last_drawn


def _rel_table_snapshot(
    rivers: List[List[str]],
    melds: List[List[dict]],
    self_seat: int,
) -> tuple:
    rivers_rel = {
        abs_to_rel_seat(s, self_seat): list(rivers[s]) for s in range(4)
    }
    melds_rel = {
        abs_to_rel_seat(s, self_seat): [dict(m) for m in melds[s]] for s in range(4)
    }
    return rivers_rel, melds_rel


def iter_seat_decisions(
    kyoku: list,
    kyoku_index: int,
    seat: int,
    include_calls: bool = False,
) -> Iterator[Union[DecisionPoint, CallOpportunity]]:
    """Yield a DecisionPoint after each acquire, before the matching discard.

    Walks all four seats in dealer order so rivers / melds are accurate for danger analysis.
    include_calls=True 时，在他家舍牌且自家有合法碰/吃/杠窗口处，按时间顺序
    额外 yield CallOpportunity（位于两个自家 DecisionPoint 之间）。
    """
    # 延迟 import：call_eval 依赖本模块，避免循环
    from .call_eval import legal_ankans, legal_kakans

    if include_calls:
        from .call_eval import legal_calls
    else:
        legal_calls = None

    meta = kyoku[0]
    scores = kyoku[1]
    doras = list(kyoku[2])
    dealer = meta[0] % 4

    draws = [list(kyoku[4 + s * 3 + 1]) for s in range(4)]
    discards = [list(kyoku[4 + s * 3 + 2]) for s in range(4)]
    hands: List[Counter] = [Counter(kyoku[4 + s * 3]) for s in range(4)]
    melds: List[List[dict]] = [[] for _ in range(4)]
    rivers: List[List[str]] = [[] for _ in range(4)]
    i_d = [0, 0, 0, 0]
    i_c = [0, 0, 0, 0]
    turns = [0, 0, 0, 0]
    last_drawn: List[Optional[int]] = [None, None, None, None]
    global_seq = 0
    discards_log: List[dict] = []
    # 自家立直后：普通摸切不产点；仅暗杠窗口产决策点（待牌不变闸在评估侧）
    self_riichi = False
    # abs seat → seq of the discard that was called into their first open meld
    first_open_after: List[Optional[int]] = [None, None, None, None]
    # 自家已舍牌（归一化 mpsz），副露机会点跳过侧振听置 0 用
    self_discards: List[str] = []

    round_wind, seat_wind = round_and_seat_wind(meta[0], seat)
    label_base = kyoku_label(meta[0], meta[1])

    def _mark_open(caller: int) -> None:
        if first_open_after[caller] is None:
            first_open_after[caller] = global_seq

    def _open_after_rel() -> dict:
        return {
            abs_to_rel_seat(s, seat): first_open_after[s]
            for s in range(4)
            if first_open_after[s] is not None
        }

    actor = dealer
    for _ in range(2000):
        if i_d[actor] >= len(draws[actor]):
            break

        tok = draws[actor][i_d[actor]]
        i_d[actor] += 1

        if isinstance(tok, int):
            hands[actor][tok] += 1
            last_drawn[actor] = tok
            turns[actor] += 1
        else:
            parsed = parse_meld_token(str(tok))
            src_abs = _source_abs_from_token(str(tok), actor)
            src_rel = abs_to_rel_seat(src_abs, seat) if src_abs is not None else None
            if parsed.type == "daiminkan":
                _remove_tiles(hands[actor], parsed.hand_tiles)
                melds[actor].append(_meld_to_api(parsed, src_rel))
                _mark_open(actor)
                last_drawn[actor] = None
                if i_c[actor] < len(discards[actor]) and discards[actor][i_c[actor]] == 0:
                    i_c[actor] += 1
                continue
            if parsed.type in ("chii", "pon"):
                _remove_tiles(hands[actor], parsed.hand_tiles)
                melds[actor].append(_meld_to_api(parsed, src_rel))
                _mark_open(actor)
                last_drawn[actor] = None
                turns[actor] += 1
            else:
                raise ValueError(f"unexpected draw-side token: {tok}")

        if i_c[actor] >= len(discards[actor]):
            break
        peek = discards[actor][i_c[actor]]

        if peek == 0:
            i_c[actor] += 1
            continue

        if _is_ankan_or_kakan_token(peek):
            i_c[actor] += 1
            parsed = parse_meld_token(str(peek))
            if parsed.type == "ankan":
                # 实暗杠前产出决策点（对照切牌/暗杠；立直后亦产）
                if actor == seat:
                    hand_now = _hand_list(hands[actor])
                    hand_mpsz = sort_hand_mpsz(mpsz_list(hand_now))
                    melds_now = [dict(m) for m in melds[actor]]
                    ankans = legal_ankans(hand_mpsz, melds_now)
                    ankan_tiles = mpsz_list(parsed.hand_tiles)
                    ankan_tile = ankan_tiles[0] if ankan_tiles else None
                    rivers_rel, melds_rel = _rel_table_snapshot(rivers, melds, seat)
                    yield DecisionPoint(
                        kyoku_index=kyoku_index,
                        kyoku_meta=list(meta),
                        seat=seat,
                        turn=turns[actor],
                        hand=hand_mpsz,
                        drawn_tile=tenhou_to_mpsz(last_drawn[actor])
                        if last_drawn[actor] is not None
                        else None,
                        melds=melds_now,
                        dora_indicators=mpsz_list(doras),
                        round_wind=round_wind,
                        seat_wind=seat_wind,
                        actual_discard=None,
                        actual_discard_raw=None,
                        is_riichi_discard=False,
                        is_tsumogiri=False,
                        scores=list(scores),
                        label=f"{label_base} 第{turns[actor]}巡",
                        rivers=rivers_rel,
                        melds_by_rel=melds_rel,
                        discards_log=[dict(x) for x in discards_log],
                        open_after_seq=_open_after_rel(),
                        legal_kakans=[]
                        if self_riichi
                        else legal_kakans(hand_mpsz, melds_now),
                        actual_kakan=None,
                        legal_ankans=ankans,
                        actual_ankan=ankan_tile,
                        is_riichi_post=self_riichi,
                    )
                _remove_tiles(hands[actor], parsed.hand_tiles)
                melds[actor].append(_meld_to_api(parsed, None))
            elif parsed.type == "kakan":
                # 实加杠前产出决策点（对照切牌/加杠）
                if actor == seat and not self_riichi:
                    hand_now = _hand_list(hands[actor])
                    hand_mpsz = sort_hand_mpsz(mpsz_list(hand_now))
                    melds_now = [dict(m) for m in melds[actor]]
                    kakans = legal_kakans(hand_mpsz, melds_now)
                    kakan_tile = (
                        tenhou_to_mpsz(parsed.hand_tiles[0])
                        if parsed.hand_tiles
                        else tenhou_to_mpsz(parsed.called)
                        if parsed.called is not None
                        else None
                    )
                    rivers_rel, melds_rel = _rel_table_snapshot(rivers, melds, seat)
                    yield DecisionPoint(
                        kyoku_index=kyoku_index,
                        kyoku_meta=list(meta),
                        seat=seat,
                        turn=turns[actor],
                        hand=hand_mpsz,
                        drawn_tile=tenhou_to_mpsz(last_drawn[actor])
                        if last_drawn[actor] is not None
                        else None,
                        melds=melds_now,
                        dora_indicators=mpsz_list(doras),
                        round_wind=round_wind,
                        seat_wind=seat_wind,
                        actual_discard=None,
                        actual_discard_raw=None,
                        is_riichi_discard=False,
                        is_tsumogiri=False,
                        scores=list(scores),
                        label=f"{label_base} 第{turns[actor]}巡",
                        rivers=rivers_rel,
                        melds_by_rel=melds_rel,
                        discards_log=[dict(x) for x in discards_log],
                        open_after_seq=_open_after_rel(),
                        legal_kakans=kakans,
                        actual_kakan=kakan_tile,
                        legal_ankans=legal_ankans(hand_mpsz, melds_now),
                        actual_ankan=None,
                        is_riichi_post=False,
                    )
                _remove_tiles(hands[actor], parsed.hand_tiles)
                upgraded = False
                for m in melds[actor]:
                    if m["type"] != "pon":
                        continue
                    if Counter(m["tiles"]) == Counter(
                        mpsz_list(
                            [t for t in parsed.tiles if t != parsed.called][:3]
                            or parsed.tiles[:3]
                        )
                    ):
                        m["type"] = "kakan"
                        m["tiles"] = mpsz_list(parsed.tiles)
                        upgraded = True
                        break
                if not upgraded:
                    melds[actor].append(_meld_to_api(parsed, None))
            last_drawn[actor] = None
            continue

        disc_tok = discards[actor][i_c[actor]]
        i_c[actor] += 1
        disc_tile, is_riichi, is_tsumogiri = _parse_discard_token(
            disc_tok, last_drawn[actor]
        )

        if actor == seat:
            hand_now = _hand_list(hands[actor])
            hand_mpsz = sort_hand_mpsz(mpsz_list(hand_now))
            melds_now = [dict(m) for m in melds[actor]]
            rivers_rel, melds_rel = _rel_table_snapshot(rivers, melds, seat)
            if self_riichi:
                # 立直后：仅可暗杠窗口产决策点；普通摸切不检讨
                ankans = legal_ankans(hand_mpsz, melds_now)
                if ankans:
                    yield DecisionPoint(
                        kyoku_index=kyoku_index,
                        kyoku_meta=list(meta),
                        seat=seat,
                        turn=turns[actor],
                        hand=hand_mpsz,
                        drawn_tile=tenhou_to_mpsz(last_drawn[actor])
                        if last_drawn[actor] is not None
                        else None,
                        melds=melds_now,
                        dora_indicators=mpsz_list(doras),
                        round_wind=round_wind,
                        seat_wind=seat_wind,
                        actual_discard=tenhou_to_mpsz(disc_tile),
                        actual_discard_raw=disc_tok,
                        is_riichi_discard=False,
                        is_tsumogiri=is_tsumogiri,
                        scores=list(scores),
                        label=f"{label_base} 第{turns[actor]}巡",
                        rivers=rivers_rel,
                        melds_by_rel=melds_rel,
                        discards_log=[dict(x) for x in discards_log],
                        open_after_seq=_open_after_rel(),
                        legal_kakans=[],
                        actual_kakan=None,
                        legal_ankans=ankans,
                        actual_ankan=None,
                        is_riichi_post=True,
                    )
            else:
                yield DecisionPoint(
                    kyoku_index=kyoku_index,
                    kyoku_meta=list(meta),
                    seat=seat,
                    turn=turns[actor],
                    hand=hand_mpsz,
                    drawn_tile=tenhou_to_mpsz(last_drawn[actor])
                    if last_drawn[actor] is not None
                    else None,
                    melds=melds_now,
                    dora_indicators=mpsz_list(doras),
                    round_wind=round_wind,
                    seat_wind=seat_wind,
                    actual_discard=tenhou_to_mpsz(disc_tile),
                    actual_discard_raw=disc_tok,
                    is_riichi_discard=is_riichi,
                    is_tsumogiri=is_tsumogiri,
                    scores=list(scores),
                    label=f"{label_base} 第{turns[actor]}巡",
                    rivers=rivers_rel,
                    melds_by_rel=melds_rel,
                    discards_log=[dict(x) for x in discards_log],
                    open_after_seq=_open_after_rel(),
                    legal_kakans=legal_kakans(hand_mpsz, melds_now),
                    actual_kakan=None,
                    legal_ankans=legal_ankans(hand_mpsz, melds_now),
                    actual_ankan=None,
                    is_riichi_post=False,
                )

        if actor == seat and is_riichi:
            self_riichi = True

        tile_name = tenhou_to_mpsz(disc_tile)
        if tile_name.startswith("0"):
            bare = "5" + tile_name[1:]
        else:
            bare = tile_name
        global_seq += 1
        discards_log.append(
            {
                "seq": global_seq,
                "abs": actor,
                "rel": abs_to_rel_seat(actor, seat),
                "tile": bare,
                "tile_raw": tile_name,
                "is_red": tile_name.startswith("0"),
                "tedashi": not is_tsumogiri,
                "riichi": is_riichi,
            }
        )
        _remove_tiles(hands[actor], [disc_tile])
        rivers[actor].append(tile_name + ("r" if is_riichi else ""))
        last_drawn[actor] = None
        if actor == seat:
            self_discards.append(bare)

        # 该舍牌是否被某家副露（提前算出，循环末尾推进时复用）
        caller = _find_caller(draws, i_d, actor, disc_tile)

        # 副露机会点：他家舍牌、自家未立直、且自家有合法碰/吃/杠窗口。
        # 在该舍牌入河、从手牌移除之后 yield（seen 含这张牌），actor 推进之前。
        if include_calls and actor != seat and not self_riichi:
            # caller 为别家 → 自家无窗口；为自家 → 实际副露；为 None → 实际跳过
            actual: Optional[str] = None
            actual_tiles: List[str] = []
            actual_cut: Optional[str] = None
            if caller == seat:
                meld_tok = str(draws[seat][i_d[seat]])
                parsed_call = parse_meld_token(meld_tok)
                actual = parsed_call.type
                actual_tiles = mpsz_list(parsed_call.hand_tiles)
                if actual in ("pon", "chii"):
                    # 副露后的实切 token：必为手切（可能是 r<tile> 立直宣言）；
                    # 0 占位/加杠暗杠 token/60 系（理论上不出现）记 None
                    if i_c[seat] < len(discards[seat]):
                        cut_tok = discards[seat][i_c[seat]]
                        if (
                            cut_tok not in (0, 60, "r60")
                            and not _is_ankan_or_kakan_token(cut_tok)
                        ):
                            cut_tile, _, _ = _parse_discard_token(cut_tok, None)
                            actual_cut = tenhou_to_mpsz(cut_tile)
            elif caller is None:
                # 终局舍牌无副露窗口：摸牌顺序下一家 draw 流已耗尽，说明该舍牌
                # 之后事件流终止（被荣和或荒牌/中途流局）；河底牌规则上也不可被吃碰
                if i_d[(actor + 1) % 4] < len(draws[(actor + 1) % 4]):
                    actual = "skip"
            if actual is not None:
                legal = legal_calls(
                    hands[seat], disc_tile, from_kamicha=actor == (seat + 3) % 4
                )
                if legal["pon"] or legal["chii"] or legal["daiminkan"]:
                    rivers_rel, melds_rel = _rel_table_snapshot(rivers, melds, seat)
                    turn_next = turns[seat] + 1
                    discarder_rel = abs_to_rel_seat(actor, seat)
                    yield CallOpportunity(
                        kyoku_index=kyoku_index,
                        kyoku_meta=list(meta),
                        seat=seat,
                        turn=turn_next,
                        hand=sort_hand_mpsz(mpsz_list(_hand_list(hands[seat]))),
                        melds=[dict(m) for m in melds[seat]],
                        dora_indicators=mpsz_list(doras),
                        round_wind=round_wind,
                        seat_wind=seat_wind,
                        scores=list(scores),
                        discarder=actor,
                        discarder_rel=discarder_rel,
                        disc_tile=tile_name,
                        legal=legal,
                        actual=actual,
                        actual_tiles=actual_tiles,
                        actual_cut=actual_cut,
                        label=(
                            f"{label_base} 第{turn_next}巡·"
                            f"{discarder_rel}打{tile_name}"
                        ),
                        rivers=rivers_rel,
                        melds_by_rel=melds_rel,
                        discards_log=[dict(x) for x in discards_log],
                        open_after_seq=_open_after_rel(),
                        self_discards=list(self_discards),
                    )

        actor = caller if caller is not None else (actor + 1) % 4


def extract_kyoku_views(
    paipu: dict, seat: int, include_calls: bool = False
) -> List[KyokuView]:
    views: List[KyokuView] = []
    for ki, kyoku in enumerate(paipu.get("log") or []):
        meta = kyoku[0]
        _, seat_wind = round_and_seat_wind(meta[0], seat)
        view = KyokuView(
            index=ki,
            label=kyoku_label(meta[0], meta[1]),
            meta=list(meta),
            scores=list(kyoku[1]),
            result=kyoku[-1],
            dora_indicators=mpsz_list(list(kyoku[2] or [])),
            seat_wind=seat_wind,
        )
        try:
            view.decisions = list(
                iter_seat_decisions(kyoku, ki, seat, include_calls=include_calls)
            )
        except Exception as exc:
            view.decisions = []
            view.result = {"error": str(exc), "raw": kyoku[-1]}
        views.append(view)
    return views
