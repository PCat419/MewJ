"""副露判断（碰/吃 vs 跳过）的反事实期望评估。

对同一个副露机会点：
- 跳过侧期望：EV_skip = Σ_t P(摸t) × EV(手牌13张+t)，WP_skip 同理（和率）；
  每个摸牌假设的 14 张手牌按与切牌卡相同的综合选切（门控 + 防守调整效用
  + 罚符信用）取值，再按巡目取 exp_score / win_prob。
- 副露侧：每个合法碰/吃变体（消耗手牌 2 张 + 被副露牌成面子，余 11 张）
  查引擎后同样按切牌卡口径选切，取该切的 EV/和率作为「副露后期望」。
  副露卡本身不推荐切哪张——碰后怎么切交给随后的切牌决策卡。
- 每次查询同时缓存裸 EV top3 候选（含门控标记）：形听轴等内部逻辑使用。

判定（自杀层 + 双轴，阈值见 params.PARAMS["call"]）：
- 自杀层：WP_call == 0 → 该变体否决（EV 比值否决默认禁用，suicide_ev_ratio>0 才启用）；
- 双轴：存活变体中 Δ和率 ≥ win_prob_delta_min（速度轴绝对差），或
  和率比 ≥ win_prob_ratio_min 且 Δ和率 ≥ win_prob_delta_floor（速度轴比例旁路），或
  ΔEV > ev_margin（收益轴）→ 合格；合格者取 EV 最高推荐副露；
  无合格者 → 推荐跳过。

形听轴（第三轴，axis="form_tenpai"）：晚巡（≥form_tenpai_turn_min）跳过侧
1 向听且存在副露即听牌（cut_shanten==0）变体时，win==0 的无役形听变体豁免
自杀层，改按 罚符价值 = ΔP听 × 流局率 × (E听 − E未听)（复用 noten.py）与
保听切危险度判定；全弃（posture==FOLD）或无危险查询能力（risk_lookup=None）
时形听轴整体关闭，保持旧行为。win>0 的真听牌变体照旧走双轴，不被形听轴截胡。

报告权重：判定完成后对每个选项（跳过 + 全部变体）做 softmax 归一化；
显示效用差 d（相对跳过基准 0）与判定构造性一致：速度轴合格 = Δ和率 ×
win_prob_weight_scale，EV 轴合格 = ΔEV − ev_margin，形听轴合格 =
罚符价值 − form_tenpai_value_min，未合格 = min(两者)（恒 ≤ 0，
跳过权重必然不低于任何未合格变体）。仅显示校准，不参与判定。
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

from .converter import (
    build_request,
    build_wall,
    id_to_tile_name,
    parse_hand,
    parse_melds,
    tile_name_to_id,
)
from .nanikiru_pool import pick_url, resolve_workers
from .noten import is_pusher, opponent_tenpai_probs, payoff_table, ryukyoku_prob
from .params import PARAMS as _P
from .posture import Posture
from .replay import CallOpportunity, tenhou_to_mpsz
from .review import (
    _NANIKIRU_TRANSPORT_ERRORS,
    _at_turn,
    _cand_ev,
    _cand_uke,
    _cand_utility,
    _filter_furiten_waits,
    _norm_tile_name,
    _policy_valid_candidates,
    restart_nanikiru,
)
from .scoring import offensive_desire_from_dp, score_candidates

_AKA = {15: 51, 25: 52, 35: 53, 51: 15, 52: 25, 53: 35}
_FIVE_RED_PAIR = {4: 34, 13: 35, 22: 36, 34: 4, 35: 13, 36: 22}  # 5m/5p/5s <-> 红五槽

# 引擎查询韧性：复杂手形双开 flag 会栈溢出/挂起（已知行为），按序降级重试；
# 最后一档双关是保底，避免副露跳过侧摸牌枚举把整局拖死。
_FLAG_ATTEMPTS = ((True, True), (True, False), (False, True), (False, False))


def legal_calls(hand_counter: Counter, disc_tile: int, from_kamicha: bool) -> dict:
    """返回 {'pon': [...variants], 'chii': [...], 'daiminkan': bool}。

    每个 variant 是需从手牌消耗的 tenhou int 列表（赤五保留实体）。
    """
    out = {"pon": [], "chii": [], "daiminkan": False}
    kind = disc_tile if disc_tile not in (51, 52, 53) else {51: 15, 52: 25, 53: 35}[disc_tile]
    aka = _AKA.get(kind)  # 普通5 <-> 赤五

    # 手牌中该牌种的实体牌
    same = []
    if hand_counter[kind] > 0:
        same += [kind] * hand_counter[kind]
    if aka and hand_counter[aka] > 0:
        same += [aka] * hand_counter[aka]

    if len(same) >= 2:
        # 碰：枚举不同的 2 张消耗组合（实体去重）
        seen = set()
        for i in range(len(same)):
            for j in range(i + 1, len(same)):
                key = tuple(sorted((same[i], same[j])))
                if key not in seen:
                    seen.add(key)
                    out["pon"].append(list(key))
    if len(same) >= 3:
        out["daiminkan"] = True

    if from_kamicha and disc_tile < 40:  # 数牌才可吃
        suit = disc_tile // 10
        num = disc_tile % 10
        if suit in (1, 2, 3):
            for lo, hi in ((num - 2, num - 1), (num - 1, num + 1), (num + 1, num + 2)):
                if not (1 <= lo <= 9 and 1 <= hi <= 9):
                    continue
                t_lo, t_hi = suit * 10 + lo, suit * 10 + hi
                # 每个需求牌种可匹配实体（普通或赤五）
                def _variants(t):
                    vs = []
                    if hand_counter[t] > 0:
                        vs.append(t)
                    a = _AKA.get(t)
                    if a and hand_counter[a] > 0:
                        vs.append(a)
                    return vs

                for v_lo in _variants(t_lo):
                    for v_hi in _variants(t_hi):
                        # 实体牌不能重复使用同一张（Counter 计数约束）
                        need = Counter([v_lo, v_hi])
                        if all(hand_counter[k] >= n for k, n in need.items()):
                            out["chii"].append([v_lo, v_hi])
        # 吃组合去重
        uniq = []
        seen = set()
        for v in out["chii"]:
            key = tuple(sorted(v))
            if key not in seen:
                seen.add(key)
                uniq.append(v)
        out["chii"] = uniq
    return out


def _strip_river_marker(tile: str) -> str:
    """去掉牌河条目末尾的 r 立直标记（保留赤五 0m/0p/0s）。"""
    t = str(tile).strip()
    if t.lower().endswith("r"):
        t = t[:-1]
    return t


def _norm_kind(tile: str) -> str:
    """赤五归一化为普通 5。"""
    t = str(tile)
    if t and t[0] == "0":
        return "5" + t[1:]
    return t


def _build_seen_other(
    opp: CallOpportunity, extra_self_meld: Optional[dict] = None
):
    """由牌河+副露构造引擎 seen / other_melds / 自家副露列表。

    牌河中保留被副露的牌；每个 open meld 的 calledTile 从 seen 删一份
    （与 review._wall_inputs_from_dp 同口径）。extra_self_meld 为反事实
    新增副露（其 calledTile 即被副露牌，也在河中，同样删一份）。
    """
    seen: List[str] = []
    for river in (opp.rivers or {}).values():
        for t in river or []:
            name = _strip_river_marker(t)
            if name:
                seen.append(name)

    self_melds = list(opp.melds) + ([extra_self_meld] if extra_self_meld else [])

    def remove_one(tile: str) -> None:
        key = _norm_kind(tile)
        for i, t in enumerate(seen):
            if _norm_kind(t) == key:
                del seen[i]
                return

    other_melds: List[dict] = []
    for rel, melds in (opp.melds_by_rel or {}).items():
        for m in melds or []:
            if (
                m.get("type") in ("chii", "pon", "daiminkan", "kakan")
                and m.get("calledTile")
                and m.get("sourceSeat")
            ):
                remove_one(str(m["calledTile"]))
            if rel == "自家":
                continue
            tiles = m.get("tiles") or []
            if tiles:
                other_melds.append({"type": m.get("type"), "tiles": list(tiles)})
    if extra_self_meld is not None:
        remove_one(str(extra_self_meld["calledTile"]))
    return seen, other_melds, self_melds


def _furiten_wall_zero(wall: List[int], self_discards: List[str]) -> None:
    """自家舍牌牌种（含对应红五槽）剩余置 0（与 review._apply_furiten_wall_zero 同口径）。"""
    n = len(wall)
    for name in self_discards or []:
        try:
            tid = tile_name_to_id(name)
        except ValueError:
            continue
        for t in (tid, _FIVE_RED_PAIR.get(tid)):
            if t is not None and 0 <= t < n:
                wall[t] = 0


# ---------------------------------------------------------------------------
# 本地向听数计算（标准形 / ≤2向听的七对与国士），用于补齐/门控引擎 shanten 字段
# ---------------------------------------------------------------------------

_ORPHAN_IDX34 = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def _tile_name_to_idx34(name: str) -> int:
    """'1m'..'9m' / '0m'(赤5) / '1z'..'7z' → 0-33。"""
    n, suit = str(name)[0], str(name)[1]
    d = 5 if n == "0" else int(n)
    return {"m": 0, "p": 9, "s": 18, "z": 27}[suit] + d - 1


def _regular_shanten(cnt: List[int], n_meld: int) -> int:
    """标准形向听：DFS 拆面子/搭子/雀头；n_meld 为已副露组数（杠按 1 组计）。"""
    best = [8]

    def dfs(i: int, m: int, t: int, pair: int) -> None:
        while i < 34 and cnt[i] == 0:
            i += 1
        if i >= 34:
            m2 = m + n_meld
            t2 = t
            if m2 + t2 > 4:
                t2 = 4 - m2
            sh = 8 - 2 * m2 - t2 - pair
            if sh < best[0]:
                best[0] = sh
            return
        if cnt[i] >= 3:  # 刻子
            cnt[i] -= 3
            dfs(i, m + 1, t, pair)
            cnt[i] += 3
        if i < 27 and i % 9 <= 6 and cnt[i + 1] and cnt[i + 2]:  # 顺子
            cnt[i] -= 1
            cnt[i + 1] -= 1
            cnt[i + 2] -= 1
            dfs(i, m + 1, t, pair)
            cnt[i] += 1
            cnt[i + 1] += 1
            cnt[i + 2] += 1
        if pair == 0 and cnt[i] >= 2:  # 雀头
            cnt[i] -= 2
            dfs(i, m, t, 1)
            cnt[i] += 2
        if cnt[i] >= 2:  # 对子搭子
            cnt[i] -= 2
            dfs(i, m, t + 1, pair)
            cnt[i] += 2
        if i < 27 and i % 9 <= 7 and cnt[i + 1]:  # 两面/边张搭子
            cnt[i] -= 1
            cnt[i + 1] -= 1
            dfs(i, m, t + 1, pair)
            cnt[i] += 1
            cnt[i + 1] += 1
        if i < 27 and i % 9 <= 6 and cnt[i + 2]:  # 嵌张搭子
            cnt[i] -= 1
            cnt[i + 2] -= 1
            dfs(i, m, t + 1, pair)
            cnt[i] += 1
            cnt[i + 2] += 1
        dfs(i + 1, m, t, pair)  # 孤张

    dfs(0, 0, 0, 0)
    return best[0]


def _seven_pairs_shanten(cnt: List[int]) -> int:
    pairs = sum(1 for c in cnt if c >= 2)
    uniq = sum(1 for c in cnt if c >= 1)
    return 6 - pairs + max(0, 7 - uniq)


def _thirteen_orphans_shanten(cnt: List[int]) -> int:
    uniq = sum(1 for i in _ORPHAN_IDX34 if cnt[i] >= 1)
    pair = any(cnt[i] >= 2 for i in _ORPHAN_IDX34)
    return 13 - uniq - (1 if pair else 0)


# 七对 / 国士仅在此向听及以内才并入综合向听；更远时只走一般型。
SPECIAL_HAND_SHANTEN_MAX = 2

_AKA_TILE_ALT = {
    "0m": "5m",
    "0p": "5p",
    "0s": "5s",
    "5m": "0m",
    "5p": "0p",
    "5s": "0s",
}


def _fold_special_shanten(regular: int, seven_pairs: int, thirteen_orphans: int) -> int:
    """一般型向听 + 门清七对/国士（仅 ≤SPECIAL_HAND_SHANTEN_MAX）。"""
    sh = int(regular)
    if seven_pairs <= SPECIAL_HAND_SHANTEN_MAX:
        sh = min(sh, int(seven_pairs))
    if thirteen_orphans <= SPECIAL_HAND_SHANTEN_MAX:
        sh = min(sh, int(thirteen_orphans))
    return sh


def form_shanten_parts(
    hand_tiles: List[str], n_meld: int = 0
) -> Tuple[int, int, int]:
    """分项向听 ``(regular, seven_pairs, thirteen_orphans)``。

    有副露时七对/国士无意义，后两项返回 99（不参与比较）。
    """
    cnt = [0] * 34
    for t in hand_tiles:
        cnt[_tile_name_to_idx34(t)] += 1
    regular = _regular_shanten(cnt, n_meld)
    if n_meld > 0:
        return regular, 99, 99
    return regular, _seven_pairs_shanten(cnt), _thirteen_orphans_shanten(cnt)


def local_shanten(hand_tiles: List[str], n_meld: int = 0) -> int:
    """手牌向听数：有副露仅一般型；门清时一般型与（≤2 向听的）七对/国士取 min。"""
    regular, seven_pairs, thirteen_orphans = form_shanten_parts(hand_tiles, n_meld)
    if n_meld > 0:
        return regular
    return _fold_special_shanten(regular, seven_pairs, thirteen_orphans)


def gated_shanten_all(shanten_info: dict, n_meld: int = 0) -> Optional[int]:
    """按七对/国士 ≤2 门控重算引擎 ``shanten.all``；缺分项时回退原 ``all``。"""
    if not shanten_info:
        return None
    reg = shanten_info.get("regular")
    if reg is None:
        return shanten_info.get("all")
    if n_meld > 0:
        return int(reg)
    sp = shanten_info.get("seven_pairs")
    to = shanten_info.get("thirteen_orphans")
    if sp is None or to is None:
        return shanten_info.get("all")
    return _fold_special_shanten(int(reg), int(sp), int(to))


def apply_shanten_gate(shanten_info: dict, n_meld: int = 0) -> dict:
    """返回新 dict，``all`` 已按七对/国士 ≤2 门控重写。"""
    out = dict(shanten_info or {})
    gated = gated_shanten_all(out, n_meld)
    if gated is not None:
        out["all"] = gated
    return out


def hand_without_tile(hand: List[str], tile: str) -> Optional[List[str]]:
    """从手牌移除一张（普通/赤五可互换兜底）；失败返回 None。"""
    after = list(hand)
    if tile in after:
        after.remove(tile)
        return after
    alt = _AKA_TILE_ALT.get(tile)
    if alt and alt in after:
        after.remove(alt)
        return after
    return None


def _build_cut_candidates(
    raw: dict,
    turn: int,
    self_discards: List[str],
    *,
    hand_tiles: Optional[List[str]] = None,
    n_meld: int = 0,
) -> List[dict]:
    """引擎 stats → 与切牌分析（review._parse_response）同形的候选列表。

    字段：tile / exp_score / win_prob / tenpai_prob / tenpai_prob_arr /
    shanten（切后向听）/ uke / furiten——可直接喂给门控与 score_candidates。
    ``tenpai_prob_arr`` 必须保留：罚符模型取数组末段估计流局时听牌率，
    缺省会误用当前巡标量，把拆听高听牌率候选错当成接近保听。

    传入 ``hand_tiles`` 时，切后向听按本地口径重算（七对/国士仅 ≤2 才计入）。
    """
    own = {t for t in (_norm_tile_name(x) for x in (self_discards or [])) if t}
    cands: List[dict] = []
    for st in raw.get("stats") or []:
        tid = st.get("tile")
        if tid is None or tid == -1:
            continue
        exp_arr = st.get("exp_score") or []
        exp = _at_turn(exp_arr, turn) if exp_arr else None
        if exp is None:
            continue
        win_arr = st.get("win_prob") or []
        ten_arr = st.get("tenpai_prob") or []
        has_probs = bool(ten_arr or win_arr or exp_arr)
        necessary = []
        for nt in st.get("necessary_tiles") or []:
            nt_id = nt.get("tile")
            if nt_id is None:
                continue
            necessary.append({"tile": id_to_tile_name(nt_id), "count": nt.get("count")})
        name = id_to_tile_name(tid)
        shanten = st.get("shanten")
        if hand_tiles is not None:
            after = hand_without_tile(hand_tiles, name)
            if after is not None:
                shanten = local_shanten(after, n_meld)
        furiten = False
        if own and shanten == 0:
            necessary, uke, furiten = _filter_furiten_waits(necessary, own)
        else:
            uke = 0
            for nt in necessary:
                try:
                    uke += int(nt.get("count") or 0)
                except (TypeError, ValueError):
                    pass
        cands.append(
            {
                "tile": name,
                "exp_score": exp,
                "win_prob": _at_turn(win_arr, turn) if win_arr else None,
                "tenpai_prob": _at_turn(ten_arr, turn) if ten_arr else None,
                "tenpai_prob_arr": ten_arr if has_probs else None,
                "shanten": shanten,
                "uke": uke,
                "furiten": furiten,
            }
        )
    return cands


def policy_best_cut(
    cands: List[dict],
    turn: int,
    *,
    opp: Optional[CallOpportunity] = None,
    defense: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    """与切牌卡同一套选切：门控后按综合效用（含防守）取最优。

    若传入 ``defense``，先 ``score_candidates`` 写入 ``adjusted_utility``
    （危险成本、罚符信用等，与 review._attach_defense 同口径）；无防守数据
    时效用退化为裸 EV。再经 ``_policy_valid_candidates`` 过滤后按
    (效用, 进张, EV) 取最优。
    """
    if not cands:
        return None
    if defense is not None and opp is not None:
        desire_info = offensive_desire_from_dp(opp)
        score_candidates(
            cands,
            defense,
            opp,
            offensive_desire=desire_info["offensive_desire"],
            noten_ctx={
                "threats": defense.get("threats") or [],
                "turn": turn,
            },
        )
    valid = _policy_valid_candidates(cands, turn)
    pool = valid or list(cands)
    return max(pool, key=lambda c: (_cand_utility(c), _cand_uke(c), _cand_ev(c)))


def _top3_cache(cands: List[dict], turn: int) -> List[dict]:
    """裸 EV 降序 top3 候选（含门控标记）入缓存：离线重算时 top3 内即可
    判定合格最优（合格者中 EV 最高者必在裸 EV 最前）；仅当 top3 全部
    被拦才需实况重查。"""
    ranked = sorted(cands, key=lambda c: -float(c.get("exp_score") or -1e18))
    valid = _policy_valid_candidates(cands, turn)
    valid_ids = {id(c) for c in valid}
    return [
        {
            "tile": c.get("tile"),
            "exp_score": c.get("exp_score"),
            "win_prob": c.get("win_prob"),
            "tenpai_prob": c.get("tenpai_prob"),
            "shanten": c.get("shanten"),
            "uke": c.get("uke"),
            "furiten": c.get("furiten"),
            "policy_rejected": id(c) not in valid_ids,
        }
        for c in ranked[:3]
    ]


def _query_best(
    hand_tiles: List[str],
    melds: List[dict],
    other_melds: List[dict],
    seen: List[str],
    self_discards: List[str],
    dora: List[str],
    round_wind: str,
    seat_wind: str,
    turn: int,
    nanikiru_url: str,
    timeout: float,
    *,
    opp: Optional[CallOpportunity] = None,
    defense: Optional[Dict[str, Any]] = None,
) -> dict:
    """一次引擎查询，按切牌卡综合口径取最优切，返回 {ev, win, best_tile,
    shanten, cut_shanten, top3}；失败 {"error": ...}。

    连接断开/超时时重启引擎并按 (手替, 向听回退) 降级重试；
    success:false + err_msg 视为该次失败，不重试。
    ``defense`` 非空时选切计入危险度与罚符信用（与切牌卡一致）。
    """
    req = build_request(
        game_mode=1,
        round_wind=round_wind,
        seat_wind=seat_wind,
        dora_indicators=dora,
        hand=hand_tiles,
        melds=[{"type": m["type"], "tiles": m["tiles"]} for m in melds],
        other_melds=other_melds,
        seen=seen,
        t_min=1,
        t_max=18,
        version="0.9.8",
    )
    wall = req.get("wall")
    if isinstance(wall, list):
        _furiten_wall_zero(wall, self_discards)
    last_exc: Optional[BaseException] = None
    n_flags = len(_FLAG_ATTEMPTS)
    url = pick_url(nanikiru_url)
    for i, (teg, sd) in enumerate(_FLAG_ATTEMPTS):
        req["enable_tegawari"] = teg
        req["enable_shanten_down"] = sd
        try:
            resp = requests.post(url, json=req, timeout=timeout)
            raw = resp.json()
        except _NANIKIRU_TRANSPORT_ERRORS as exc:
            last_exc = exc
            kind = "超时" if isinstance(exc, requests.exceptions.Timeout) else "断开"
            print(f"    nanikiru {kind}（手替={teg} 向听回退={sd}），重启…", flush=True)
            if i + 1 < n_flags:
                restart_nanikiru(url)
            continue
        except Exception as exc:
            last_exc = exc
            continue
        if raw.get("success", True) is False and raw.get("err_msg"):
            return {"error": raw.get("err_msg")}
        n_meld = len(melds)
        cands = _build_cut_candidates(
            raw, turn, self_discards, hand_tiles=hand_tiles, n_meld=n_meld
        )
        if not cands:
            return {"error": "no candidates"}
        # 与切牌卡同口径：门控 + 防守调整效用后的最优切（决定本假设的 EV/和率）
        picked = policy_best_cut(cands, turn, opp=opp, defense=defense)
        if picked is None:
            return {"error": "no candidates"}
        shanten = gated_shanten_all(raw.get("shanten") or {}, n_meld)
        if shanten is None:
            # 旧引擎响应缺 shanten：本地按同口径补齐
            shanten = local_shanten(hand_tiles, n_meld)
        return {
            "ev": picked["exp_score"],
            "win": picked["win_prob"],
            "best_tile": picked["tile"],
            "shanten": shanten,
            "cut_shanten": picked["shanten"],
            "top3": _top3_cache(cands, turn),
        }
    return {"error": f"engine unreachable: {last_exc}"}


def evaluate_opportunity(
    opp: CallOpportunity,
    nanikiru_url: str,
    timeout: float = 30.0,
    *,
    posture: Optional[Any] = None,
    threats: Optional[List[Dict[str, Any]]] = None,
    defense: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """对一个副露机会点求 EV(跳过) 与 EV(各碰/吃变体)。

    跳过/副露两侧的选切均走切牌卡综合口径（``defense`` 非空时计入防守）。
    ``posture`` / ``threats`` 为形听轴语境；``defense`` 为完整防守结果
    （review 传入）。危险查询 callable 不可序列化，故 risk_lookup 不经本
    函数而直接传给 decide。

    摸牌假设与副露变体的引擎查询可按 ``workers`` 并行（依赖活跃 nanikiru 池）。
    """
    turn = opp.turn  # 副露后切牌 / 下次摸牌均按「自家即将进行的巡」取值
    dora = opp.dora_indicators
    errors: List[str] = []
    # threats 可单独传入；有完整 defense 时以其 threats 为准
    if defense is not None and threats is None:
        threats = defense.get("threats") or []
    n_workers = resolve_workers(workers)

    # ---- 跳过侧：本地 wall 算摸牌分布，逐假设查引擎加权 ----
    seen0, om0, _ = _build_seen_other(opp)
    hand_ids = parse_hand(opp.hand)
    wall = build_wall(
        hand_ids,
        parse_melds([{"type": m["type"], "tiles": m["tiles"]} for m in opp.melds]),
        [tile_name_to_id(t) for t in dora],
        [tile_name_to_id(t) for t in seen0],
        1,
        other_melds=parse_melds(om0),
    )
    _furiten_wall_zero(wall, opp.self_discards)

    # sum(wall[0:34]) = 实体牌总数（红五已并入普通 5 槽，与引擎 sum 口径一致）
    total_w = sum(wall[:34])
    draw_kinds = []
    for tid in range(34):
        n_normal = wall[tid]
        n_red = wall[_FIVE_RED_PAIR[tid]] if tid in _FIVE_RED_PAIR else 0
        if n_normal - n_red > 0:
            draw_kinds.append((id_to_tile_name(tid), n_normal - n_red))
        if n_red > 0:
            draw_kinds.append((id_to_tile_name(_FIVE_RED_PAIR[tid]), n_red))

    # 相关牌种剪枝：与手牌同花色±2 以外、非手牌字牌、非宝牌/赤五的牌种，
    # 摸到后基本必然摸切、EV 近似相同，只查一个代表并按计数加权
    hand_ids_set = set(hand_ids)
    dora_kinds = set()
    for d in dora:
        try:
            did = tile_name_to_id(d)
        except ValueError:
            continue
        # 指示牌 -> 宝牌（下一张；字牌 1z-4z / 5z-7z 各自循环）
        if did < 27:
            dora_kinds.add((did // 9) * 9 + (did % 9 + 1) % 9)
        elif did < 31:
            dora_kinds.add(27 + (did - 27 + 1) % 4)
        else:
            dora_kinds.add(31 + (did - 31 + 1) % 3)

    def is_relevant(tid: int) -> bool:
        """tid: 0-33 普通槽或 34-36 红五槽。"""
        if tid >= 34:
            return True  # 赤五必查
        if tid in dora_kinds:
            return True
        if tid in hand_ids_set or tid in (4, 13, 22) and (
            {4: 34, 13: 35, 22: 36}[tid] in hand_ids_set
        ):
            return True  # 手牌已有该牌种（含手牌持赤五对应的普通5）
        if tid >= 27:
            return False  # 字牌无邻牌
        suit, num = tid // 9, tid % 9
        for h in hand_ids_set:
            if h >= 27:
                continue
            hs, hn = h // 9, h % 9
            if hs == suit and abs(hn - num) <= 2:
                return True
            # 手牌赤五视同普通5
            if h in (34, 35, 36):
                rn = {34: (0, 4), 35: (1, 4), 36: (2, 4)}[h]
                if rn[0] == suit and abs(rn[1] - num) <= 2:
                    return True
        return False

    relevant = [(t, c) for t, c in draw_kinds if is_relevant(tile_name_to_id(t))]
    irrelevant = [(t, c) for t, c in draw_kinds if not is_relevant(tile_name_to_id(t))]

    def query_draw(tname: str) -> dict:
        return _query_best(
            opp.hand + [tname],
            opp.melds,
            om0,
            seen0,
            opp.self_discards,
            dora,
            opp.round_wind,
            opp.seat_wind,
            turn,
            nanikiru_url,
            timeout,
            opp=opp,
            defense=defense,
        )

    # 并行查询：相关摸牌 + 无关牌代表
    draw_jobs: List[Tuple[str, int, bool]] = [
        (tname, cnt, False) for tname, cnt in relevant
    ]
    if irrelevant:
        draw_jobs.append(
            (irrelevant[0][0], sum(c for _, c in irrelevant), True)
        )

    draw_results: Dict[int, dict] = {}
    if draw_jobs:
        with ThreadPoolExecutor(
            max_workers=max(1, min(n_workers, len(draw_jobs)))
        ) as ex:
            fut_map = {
                ex.submit(query_draw, tname): idx
                for idx, (tname, _cnt, _lump) in enumerate(draw_jobs)
            }
            for fut in as_completed(fut_map):
                draw_results[fut_map[fut]] = fut.result()

    skip_detail: List[dict] = []
    ev_skip = 0.0
    wp_skip = 0.0
    for idx, (tname, cnt, is_lump) in enumerate(draw_jobs):
        r = draw_results.get(idx) or {}
        label = (
            f"无关牌x{len(irrelevant)}(代表{tname})"
            if is_lump
            else tname
        )
        if r.get("ev") is None:
            err_label = f"无关牌代表{tname}" if is_lump else f"摸{tname}"
            errors.append(f"{err_label}: {r.get('error')}")
            skip_detail.append(
                {
                    "tile": (
                        f"无关牌x{len(irrelevant)}" if is_lump else tname
                    ),
                    "cnt": cnt,
                    "error": r.get("error"),
                }
            )
            continue
        p = cnt / total_w
        ev_skip += p * r["ev"]
        wp_skip += p * (r["win"] or 0.0)
        skip_detail.append(
            {
                "tile": label,
                "cnt": cnt,
                "p": round(p, 4),
                "ev": round(r["ev"], 1),
                "win": r["win"],
                "cut": r["best_tile"],
                "shanten": r.get("shanten"),
                "cut_shanten": r.get("cut_shanten"),
                "top3": r.get("top3"),
            }
        )

    # 相关牌种全部失败 → 跳过侧期望无法计算，整点失败
    if not any(d.get("ev") is not None for d in skip_detail):
        return {
            "ok": False,
            "turn": turn,
            "errors": errors,
            "error": "跳过侧期望无法计算：" + "; ".join(errors or ["无可用摸牌假设"]),
        }

    # 当前 13 张手牌的向听：摸牌后 14 张向听 ∈ {n−1, n}，取各假设向听最大值
    _shs = [d["shanten"] for d in skip_detail if d.get("shanten") is not None]
    skip_shanten = max(_shs) if _shs else None

    # ---- 副露侧：每个合法碰/吃变体（消耗 2 张成面子，余 11 张查引擎） ----
    variant_specs: List[dict] = []
    for action in ("pon", "chii"):
        for consume in (opp.legal or {}).get(action) or []:
            consume_names = [tenhou_to_mpsz(t) for t in consume]
            new_hand = list(opp.hand)
            ok = True
            for cn in consume_names:
                # 实体移除；赤五精确匹配，普通/赤五互换兜底
                if cn in new_hand:
                    new_hand.remove(cn)
                else:
                    alt = {"0m": "5m", "0p": "5p", "0s": "5s"}.get(cn)
                    if alt and alt in new_hand:
                        new_hand.remove(alt)
                    else:
                        ok = False
                        break
            if not ok:
                continue
            meld_tiles = sorted(consume_names + [opp.disc_tile])
            new_meld = {
                "type": action,
                "tiles": meld_tiles,
                "calledTile": opp.disc_tile,
                "sourceSeat": opp.discarder,
            }
            seen1, om1, self_melds1 = _build_seen_other(opp, extra_self_meld=new_meld)
            variant_specs.append(
                {
                    "action": action,
                    "consume_names": consume_names,
                    "new_hand": new_hand,
                    "self_melds1": self_melds1,
                    "seen1": seen1,
                    "om1": om1,
                }
            )

    def query_variant(spec: dict) -> dict:
        return _query_best(
            spec["new_hand"],
            spec["self_melds1"],
            spec["om1"],
            spec["seen1"],
            opp.self_discards,
            dora,
            opp.round_wind,
            opp.seat_wind,
            turn,
            nanikiru_url,
            timeout,
            opp=opp,
            defense=defense,
        )

    variant_results: Dict[int, dict] = {}
    if variant_specs:
        with ThreadPoolExecutor(
            max_workers=max(1, min(n_workers, len(variant_specs)))
        ) as ex:
            fut_map = {
                ex.submit(query_variant, spec): idx
                for idx, spec in enumerate(variant_specs)
            }
            for fut in as_completed(fut_map):
                variant_results[fut_map[fut]] = fut.result()

    variants: List[dict] = []
    for idx, spec in enumerate(variant_specs):
        action = spec["action"]
        consume_names = spec["consume_names"]
        new_hand = spec["new_hand"]
        self_melds1 = spec["self_melds1"]
        r = variant_results.get(idx) or {}
        if r.get("ev") is None:
            errors.append(f"{action}{consume_names}: {r.get('error')}")
            continue
        # 候选级向听：切完最优切之后的向听。优先引擎 per-candidate 值，
        # 缺失（旧缓存条目）时本地计算「副露后手牌 − 所切牌」的向听
        cut_sh = r.get("cut_shanten")
        if cut_sh is None and r.get("best_tile"):
            try:
                after_cut: Optional[List[str]] = list(new_hand)
                bt = r["best_tile"]
                if bt in after_cut:
                    after_cut.remove(bt)
                else:
                    alt = {"0m": "5m", "0p": "5p", "0s": "5s"}.get(bt)
                    if alt and alt in after_cut:
                        after_cut.remove(alt)
                    else:
                        after_cut = None
                if after_cut is not None:
                    cut_sh = local_shanten(after_cut, len(self_melds1))
            except Exception:
                cut_sh = None
        variants.append(
            {
                "action": action,
                "consume": consume_names,
                "ev": r["ev"],
                "win": r["win"],
                "cut": r["best_tile"],
                "shanten": r.get("shanten"),
                "cut_shanten": cut_sh,
                "cut_top3": r.get("top3"),
                "delta_ev": r["ev"] - ev_skip,
                "delta_win": (r["win"] or 0.0) - wp_skip,
                "rejected": None,
            }
        )

    return {
        "ok": True,
        "turn": turn,
        "ev_skip": ev_skip,
        "win_skip": wp_skip,
        "skip_shanten": skip_shanten,
        "skip_detail": skip_detail,
        "variants": variants,
        # 形听轴语境（decide 用）：当前姿态（int，FOLD 时形听轴关闭）与威胁列表
        "posture": int(posture) if posture is not None else None,
        "threats": threats or [],
        "errors": errors,
    }


def _form_tenpai_context(
    eval_result: Dict[str, Any],
    risk_lookup: Any,
    p_call: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """形听语境判定：全部条件满足返回上下文 dict，否则返回 None（形听轴关闭）。

    条件：危险查询能力可用（risk_lookup 非 None）；姿态不为全弃（全弃下
    形听轴关闭）；巡目 ≥ form_tenpai_turn_min；跳过侧当前 1 向听；
    存在副露即听牌（cut_shanten==0）的变体。上下文含 ΔP听 / 罚符价值 /
    危险上限，供自杀层豁免、形听轴判定与权重构造共用。
    """
    if risk_lookup is None or not eval_result.get("ok"):
        return None
    # 全弃（posture==FOLD）时形听轴关闭；posture 缺失（旧缓存/其他调用方）按非全弃
    if eval_result.get("posture") == int(Posture.FOLD):
        return None
    turn = eval_result.get("turn")
    if turn is None or int(turn) < int(p_call["form_tenpai_turn_min"]):
        return None
    if eval_result.get("skip_shanten") != 1:
        return None
    variants = eval_result.get("variants") or []
    if not any(v.get("cut_shanten") == 0 for v in variants):
        return None

    # P(跳过侧终局前听牌)：按摸牌概率加权聚合（摸到后 14 张或切后 13 张听牌即计）。
    # 数据口径说明：skip_detail 只覆盖下一巡摸牌（引擎数据所限），故 P听 为下界、
    # ΔP听 为上界，门槛参数已按此口径标定；无数据的摸牌假设不计入。
    p_skip = 0.0
    n_data = 0
    for d in eval_result.get("skip_detail") or []:
        pi = d.get("p")
        sh, csh = d.get("shanten"), d.get("cut_shanten")
        if pi is None or (sh is None and csh is None):
            continue
        n_data += 1
        if sh == 0 or csh == 0:
            p_skip += float(pi)
    if n_data == 0:
        # 数据不足退化：用 win_skip 作 P听 下界（和率 ≤ 听牌率，ΔP 略高估，
        # 仅在 skip_detail 完全无向听数据时启用）
        p_skip = max(0.0, float(eval_result.get("win_skip") or 0.0))
    delta_p = 1.0 - p_skip

    cfg_n = _P["noten"]
    threats = eval_result.get("threats") or []
    # 推进者（立直或 ≥3 副露）数量与对手罚符期望：复用 noten 模型（不新造规则）
    n_pushers = sum(1 for t in threats if is_pusher(t))
    e_tenpai, e_noten = payoff_table(
        opponent_tenpai_probs(threats, cfg_n), pool=float(cfg_n["pool"])
    )
    p_ryu = ryukyoku_prob(turn, n_pushers, cfg_n)
    # 罚符价值 = ΔP听 × 流局率 × (听牌与未听的流局收支摆幅)
    value = delta_p * p_ryu * (e_tenpai - e_noten)
    cap = (
        float(p_call["form_tenpai_danger_cap_multi"])
        if len(threats) >= 2
        else float(p_call["form_tenpai_danger_cap"])
    )
    return {
        "delta_p": delta_p,
        "p_skip_tenpai": p_skip,
        "value": value,
        "p_ryukyoku": p_ryu,
        "e_tenpai": e_tenpai,
        "e_noten": e_noten,
        "n_pushers": n_pushers,
        "danger_cap": cap,
        "risk_lookup": risk_lookup,
    }


def _form_tenpai_cut(v: Dict[str, Any], risk_lookup: Any) -> tuple:
    """形听语境的副露后选切：cut_top3 内保听（shanten==0）且未被门控的候选中
    取 combined 危险最小者；缓存无保听切时退化为变体既定最优切
    （其 cut_shanten==0 已保证保听）。返回 (tile, combined_danger)。"""
    pool = [
        c
        for c in (v.get("cut_top3") or [])
        if c.get("shanten") == 0 and not c.get("policy_rejected")
    ]
    if pool:
        tile = min(pool, key=lambda c: risk_lookup(c.get("tile"))).get("tile")
    else:
        tile = v.get("cut")
    return tile, risk_lookup(tile)


def decide(
    eval_result: Dict[str, Any],
    *,
    risk_lookup: Any = None,
) -> Dict[str, Any]:
    """自杀层 + 双轴 + 形听轴判定：返回推荐结论与人类可读依据。

    ``risk_lookup`` 为危险查询能力（(tile) -> combined 危险指数），由 review
    调用处基于 defense 上下文构造；为 None 时形听轴自动禁用，保持旧行为。
    """
    p = _P["call"]
    out: Dict[str, Any] = {
        "recommend": None,
        "variant": None,
        "axis": None,
        "basis": "",
        "rejected_variants": [],
    }
    if not eval_result.get("ok"):
        out["basis"] = f"评估失败：{eval_result.get('error') or '未知错误'}"
        return out
    ev_skip = eval_result.get("ev_skip") or 0.0
    wp_skip = eval_result.get("win_skip") or 0.0

    delta_min = float(p["win_prob_delta_min"])
    ratio_min = float(p["win_prob_ratio_min"])
    delta_floor = float(p["win_prob_delta_floor"])
    ev_margin = float(p["ev_margin"])
    suicide_ratio = float(p["suicide_ev_ratio"])

    def _speed_ok(v: Dict[str, Any]) -> bool:
        """速度轴合格：绝对差路径，或比例旁路（和率比≥阈值 且 Δ和率≥floor）。

        WP_skip==0 时比例视为无穷大：Δ和率 ≥ floor 即达标（从死牌复活是实质增益）。
        """
        dw = v.get("delta_win") or 0.0
        if dw >= delta_min:
            return True
        if dw < delta_floor:
            return False
        if wp_skip > 0:
            return (v.get("win") or 0.0) / wp_skip >= ratio_min
        return True

    alive = []
    # 形听语境（须先于自杀层判定：语境决定 win==0 变体是否豁免）
    ft = _form_tenpai_context(eval_result, risk_lookup, p)
    ft_candidates = []  # 被形听豁免、改走形听轴的 win==0 变体（不进双轴）
    for v in eval_result.get("variants") or []:
        win = v.get("win") or 0.0
        ev = v.get("ev") or 0.0
        # 自杀层：副露后和率归零 → 否决；EV 比值否决仅在参数 >0 时启用
        if win == 0:
            # 形听豁免：形听语境下副露即听牌（cut_shanten==0）的变体不因和率 0
            # 否决——无役形听/振听/空听对荒牌流局罚符均合规（罚符免疫待牌质量）
            if ft is not None and v.get("cut_shanten") == 0:
                v["form_tenpai_candidate"] = True
                ft_candidates.append(v)
            else:
                v["rejected"] = "suicide"
                out["rejected_variants"].append({**v, "reason": "自杀层：副露后和率为 0"})
        elif suicide_ratio > 0 and ev < suicide_ratio * ev_skip:
            v["rejected"] = "suicide"
            ratio = ev / ev_skip if ev_skip else 0.0
            out["rejected_variants"].append({**v, "reason": f"自杀层：EV 退化至 {ratio:.1f}×"})
        else:
            alive.append(v)

    qualified = []  # (variant, axis)
    for v in alive:
        if _speed_ok(v):
            qualified.append((v, "speed"))
        elif (v.get("delta_ev") or 0.0) > ev_margin:
            qualified.append((v, "ev"))

    # 形听轴：豁免变体中 ΔP听 / 罚符价值 / 保听切危险三闸全过者合格
    ft_qualified = []
    if ft is not None:
        ft_delta_min = float(p["form_tenpai_delta_min"])
        ft_value_min = float(p["form_tenpai_value_min"])
        for v in ft_candidates:
            tile, danger = _form_tenpai_cut(v, ft["risk_lookup"])
            ok = (
                ft["delta_p"] >= ft_delta_min
                and ft["value"] >= ft_value_min
                and danger <= ft["danger_cap"]
            )
            v["form_tenpai"] = {
                "delta_p": ft["delta_p"],
                "value": ft["value"],
                "p_ryukyoku": ft["p_ryukyoku"],
                "cut": tile,
                "danger": danger,
                "danger_cap": ft["danger_cap"],
                "qualified": ok,
            }
            if ok:
                # 形听语境下选切改为保听切中 combined 危险最小者
                v["cut"] = tile
                ft_qualified.append(v)

    if qualified:
        # 双轴优先：win>0 的真听牌变体照旧走原逻辑，不被形听轴截胡
        best_v, axis = max(qualified, key=lambda q: q[0].get("ev") or 0.0)
        out["recommend"] = "call"
        out["variant"] = best_v
        out["axis"] = axis
        if axis == "speed":
            dw = best_v.get("delta_win") or 0.0
            if dw >= delta_min:
                out["basis"] = f"速度轴：Δ和率{dw:+.3f} ≥ {delta_min:.2f}"
            else:
                ratio = (best_v.get("win") or 0.0) / wp_skip if wp_skip > 0 else None
                ratio_txt = f"{ratio:.2f}" if ratio is not None else "∞"
                out["basis"] = (
                    f"速度轴（比例旁路）：和率比{ratio_txt} ≥ {ratio_min:.2f}"
                    f" 且 Δ和率{dw:+.3f} ≥ {delta_floor:.2f}"
                )
        else:
            out["basis"] = (
                f"收益轴：ΔEV{best_v.get('delta_ev') or 0.0:+.0f}"
                f" > {ev_margin:.0f}"
            )
    elif ft_qualified:
        # 合格后按罚符价值最高推荐
        best_v = max(
            ft_qualified, key=lambda v: v["form_tenpai"]["value"]
        )
        fti = best_v["form_tenpai"]
        out["recommend"] = "call"
        out["variant"] = best_v
        out["axis"] = "form_tenpai"
        out["basis"] = (
            f"形听：副露后打低危牌保听避罚符"
            f"（ΔP听{fti['delta_p']:.2f} ≥ {float(p['form_tenpai_delta_min']):.2f}，"
            f"罚符价值{fti['value']:.0f} ≥ {float(p['form_tenpai_value_min']):.0f}，"
            f"危险{fti['danger']:.3f} ≤ {fti['danger_cap']:.2f}）"
        )
    # 双轴/形听轴均未过时不写提示（报告侧不展示“依据：双轴均未过”）

    # ---- 报告权重（softmax 显示校准，不参与判定） ----
    # 显示效用差 d（相对跳过基准 0），构造性与判定一致：
    # - 速度轴合格（绝对差或比例旁路）：d = Δ和率 × win_prob_weight_scale；
    # - 否则 EV 轴合格：d = ΔEV − ev_margin；
    # - 形听轴合格：d = 罚符价值 − form_tenpai_value_min（对齐收益轴构造）；
    # - 否则（未合格/被否决）：d = min(Δ和率 × scale, ΔEV − ev_margin)，恒 ≤ 0，
    #   保证跳过权重必然不低于任何未合格变体（形听候选 win==0 → Δ和率 ≤ 0，
    #   min 同样恒 ≤ 0）。
    # 权重 = softmax({0, d_1, d_2, ...})，温度用 scoring.temperature。
    temp = float(_P["scoring"]["temperature"])
    scale = float(p["win_prob_weight_scale"])
    ft_value_min = float(p["form_tenpai_value_min"])
    utils: List[float] = [0.0]  # 第 0 项为跳过（基准 0）
    for v in eval_result.get("variants") or []:
        d_win = (v.get("delta_win") or 0.0) * scale
        d_ev = (v.get("delta_ev") or 0.0) - ev_margin
        fti = v.get("form_tenpai")
        if fti and fti.get("qualified"):
            d = fti["value"] - ft_value_min
        elif v.get("form_tenpai_candidate"):
            d = min(d_win, d_ev)
        elif not v.get("rejected") and _speed_ok(v):
            d = d_win
        elif not v.get("rejected") and (v.get("delta_ev") or 0.0) > ev_margin:
            d = d_ev
        else:
            d = min(d_win, d_ev)
        utils.append(d)
    peak = max(utils)
    exps = [math.exp((u - peak) / temp) for u in utils]
    z = sum(exps)
    out["skip"] = {"recommendation_weight": exps[0] / z}
    for v, e in zip(eval_result.get("variants") or [], exps[1:]):
        v["recommendation_weight"] = e / z
    return out


def match_actual(actual: str, decision: Dict[str, Any]) -> Optional[bool]:
    """实际行动与推荐是否一致。大明杠 v1 不评估，返回 None（不计入统计）。"""
    if actual == "daiminkan":
        return None
    if actual == "skip":
        return decision.get("recommend") is None
    variant = decision.get("variant") or {}
    return decision.get("recommend") == "call" and variant.get("action") == actual
