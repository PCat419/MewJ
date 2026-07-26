"""cvmaj / mpsz 牌名与 mahjong-cpp tile ID 之间的转换"""

# mahjong-cpp tile ID 映射
# 0-8 万子 1m-9m，9-17 筒子 1p-9p，18-26 索子 1s-9s
# 27-33 字牌 1z-7z（东南西北白发中）
# 34-36 红五 0m/0p/0s
_TILE_NAME_TO_ID = {
    **{"%dm" % i: i - 1 for i in range(1, 10)},   # 1m..9m -> 0..8
    **{"%dp" % i: i + 8 for i in range(1, 10)},   # 1p..9p -> 9..17
    **{"%ds" % i: i + 17 for i in range(1, 10)},  # 1s..9s -> 18..26
    **{"%dz" % i: i + 26 for i in range(1, 8)},   # 1z..7z -> 27..33
    "0m": 34, "0p": 35, "0s": 36,
}

_ID_TO_TILE_NAME = {v: k for k, v in _TILE_NAME_TO_ID.items()}

_WIND_NAME_TO_ID = {
    "east": 27, "东": 27, "東": 27,
    "south": 28, "南": 28,
    "west": 29, "西": 29,
    "north": 30, "北": 30,
}

_MELD_TYPE_TO_CODE = {
    "pon": 0, "碰": 0, "p": 0,
    "chii": 1, "chi": 1, "吃": 1, "c": 1,
    "ankan": 2, "暗杠": 2, "a": 2,
    "daiminkan": 3, "大明杠": 3, "d": 3, "minkan": 3,
    "kakan": 4, "加杠": 4, "k": 4,
}


def tile_name_to_id(name):
    """把 '1m' / '0p' / '7z' 等转换为 mahjong-cpp tile ID。"""
    if isinstance(name, int):
        if 0 <= name <= 36:
            return name
        raise ValueError("tile ID 必须在 0-36 之间: %s" % name)
    name = str(name).strip().lower()
    if name in _TILE_NAME_TO_ID:
        return _TILE_NAME_TO_ID[name]
    raise ValueError("无法识别的牌名: %s" % name)


def id_to_tile_name(tile_id):
    """mahjong-cpp tile ID 转 mpsz 字符串。"""
    return _ID_TO_TILE_NAME.get(tile_id, str(tile_id))


def parse_wind(value):
    """解析风：'East'/'东'/27 -> 27。"""
    if isinstance(value, int):
        if value in (27, 28, 29, 30):
            return value
        raise ValueError("场风/自风 ID 必须是 27-30: %s" % value)
    s = str(value).strip().lower()
    if s.isdigit():
        return parse_wind(int(s))
    if s in _WIND_NAME_TO_ID:
        return _WIND_NAME_TO_ID[s]
    raise ValueError("无法识别的风名: %s" % value)


def parse_hand(hand):
    """把手牌列表（字符串或整数）转成 tile ID 列表。"""
    if hand is None:
        return []
    return [tile_name_to_id(t) for t in hand]


def _parse_meld_type(mtype):
    if isinstance(mtype, int):
        if 0 <= mtype <= 4:
            return mtype
        raise ValueError("副露类型代码必须在 0-4 之间: %s" % mtype)
    s = str(mtype).strip().lower()
    if s.isdigit():
        return _parse_meld_type(int(s))
    if s in _MELD_TYPE_TO_CODE:
        return _MELD_TYPE_TO_CODE[s]
    raise ValueError("无法识别的副露类型: %s" % mtype)


# 红五按普通 5 的序位排序（吃张必须升序，否则 nanikiru 役种/EV 会错）
_AKA_SORT_AS = {34: 4, 35: 13, 36: 22}


def meld_tile_sort_key(tile) -> int:
    """副露牌排序键：红五视作对应普通 5，其余用 mahjong-cpp tile ID。"""
    tid = tile_name_to_id(tile) if not isinstance(tile, int) else tile
    return _AKA_SORT_AS.get(tid, tid)


def parse_melds(melds):
    """把 [{'type': 'pon', 'tiles': ['2m','2m','2m']}, ...] 转成 mahjong-cpp 格式。

    吃（type=1）的 tiles 按序位升序排列。天凤 token / 牌谱里常见
    ``[被鸣牌, a, b]``（如 7m,6m,8m），乱序传入会让引擎认不出顺子，
    进而算错三色等役、EV 倒置。
    """
    if not melds:
        return []
    result = []
    for i, m in enumerate(melds):
        if not isinstance(m, dict) or "type" not in m or "tiles" not in m:
            raise ValueError("第 %d 个副露必须是包含 type 和 tiles 的对象" % (i + 1))
        mtype = _parse_meld_type(m["type"])
        tiles = [tile_name_to_id(t) for t in m["tiles"]]
        if mtype == 1:  # chii：必须升序
            tiles = sorted(tiles, key=meld_tile_sort_key)
        result.append({
            "type": mtype,
            "tiles": tiles,
        })
    return result


def build_wall(hand, melds, dora_indicators, seen=None, game_mode=1, other_melds=None):
    """根据手牌、副露、指示牌和已出现的牌（牌河）计算剩余牌山。

    标准牌山：34 种普通牌各 4 张，红五（0m/0p/0s）各 1 张。
    从牌山中扣掉：手牌、副露、宝牌指示牌、牌河中已出现的牌。
    注意：红五同时占用对应普通 5 的槽位，与 mahjong-cpp 保持一致。
    """
    # 初始牌山：0-33 各 4 张；34-36（红五）各 1 张
    wall = [4] * 34 + [1, 1, 1]

    # 红五 ID 到对应普通 5 ID 的映射
    _RED_FIVE_MAP = {34: 4, 35: 13, 36: 22}  # 0m->5m, 0p->5p, 0s->5s

    def subtract(tile_id, source):
        if tile_id < 0 or tile_id > 36:
            raise ValueError("非法牌 ID: %s" % tile_id)
        if wall[tile_id] <= 0:
            raise ValueError(
                "牌山计数不足：%s（%s）。请检查手牌、副露、指示牌或已出现牌是否重复或超出 4 张。"
                % (id_to_tile_name(tile_id), source)
            )
        wall[tile_id] -= 1
        # 红五同时占用普通 5 的槽位
        if tile_id in _RED_FIVE_MAP:
            normal_id = _RED_FIVE_MAP[tile_id]
            if wall[normal_id] <= 0:
                raise ValueError(
                    "牌山计数不足：%s（%s）。红五对应的普通 5 已被用完。"
                    % (id_to_tile_name(normal_id), source)
                )
            wall[normal_id] -= 1

    # 扣掉手牌
    for t in hand:
        subtract(t, "手牌")

    # 扣掉副露
    for m in melds:
        for t in m["tiles"]:
            subtract(t, "副露")

    # 扣掉他家副露（仅用于牌山计数，不影响牌效分析）
    if other_melds:
        for m in other_melds:
            for t in m["tiles"]:
                subtract(t, "他家副露")

    # 扣掉宝牌指示牌
    for t in dora_indicators:
        subtract(t, "宝牌指示牌")

    # 扣掉牌河/已出现的牌
    if seen:
        for name in seen:
            subtract(tile_name_to_id(name), "已出现的牌")

    # 三人麻将：万子 2-8 与红五万不可用
    if game_mode == 0:
        for i in range(1, 8):   # 2m-8m -> ID 1-7
            wall[i] = 0
        wall[34] = 0             # 红五万

    return wall


def build_request(game_mode=1, round_wind="east", seat_wind="east",
                  dora_indicators=None, hand=None, melds=None, other_melds=None, seen=None,
                  t_min=None, t_max=None,
                  enable_reddora=True, enable_uradora=True,
                  enable_shanten_down=True, enable_tegawari=True,
                  version="0.9.8"):
    """构造符合 mahjong-cpp 请求 schema 的 JSON 字典。"""
    hand_ids = parse_hand(hand)
    meld_objs = parse_melds(melds)
    other_meld_objs = parse_melds(other_melds)
    dora_ids = [tile_name_to_id(t) for t in (dora_indicators or [])]
    wall = build_wall(hand_ids, meld_objs, dora_ids, seen, game_mode, other_melds=other_meld_objs)

    req = {
        "game_mode": int(game_mode),
        "round_wind": parse_wind(round_wind),
        "seat_wind": parse_wind(seat_wind),
        "dora_indicators": dora_ids,
        "hand": hand_ids,
        "melds": meld_objs,
        "wall": wall,
        "enable_reddora": bool(enable_reddora),
        "enable_uradora": bool(enable_uradora),
        "enable_shanten_down": bool(enable_shanten_down),
        "enable_tegawari": bool(enable_tegawari),
        "version": version,
    }
    if t_min is not None:
        req["t_min"] = int(t_min)
    if t_max is not None:
        req["t_max"] = int(t_max)
    return req
