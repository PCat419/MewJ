"""副露判断（碰/吃/大明杠 vs 跳过）的反事实期望评估。

对同一个副露机会点：
- 跳过侧期望：EV_skip = Σ_t P(摸t) × EV(手牌13张+t)，WP_skip 同理（和率）；
  每个摸牌假设的 14 张手牌按与切牌卡相同的综合选切（门控 + 防守调整效用
  + 罚符信用）取值，再按巡目取 exp_score / win_prob。
- 碰/吃变体：消耗手牌 2 张 + 被副露牌成面子，余 11 张查引擎后按切牌卡
  口径选切，取该切的 EV/和率。副露卡不推荐切哪张——交给随后切牌卡。
- 大明杠变体（params.call.eval_daiminkan）：消耗 3 张成杠后手牌 10 张，
  对剩余牌墙做岭上摸牌期望（不翻新宝牌指示），再选切聚合 EV/和率。
  情境偏置（门清 / 听牌 / 他家立直）乘到合格判定与报告权重上；门清与
  「未听+有立直」硬门控不推荐杠。

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
跳过权重必然不低于任何未合格变体）。大明杠再乘情境倍率。仅显示校准，
不参与判定（合格判定另用倍率后的 Δ）。
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import requests

from .converter import (
    build_request,
    build_wall,
    id_to_tile_name,
    meld_tile_sort_key,
    parse_hand,
    parse_melds,
    tile_name_to_id,
)
from .nanikiru_pool import pick_url, resolve_workers
from .noten import is_pusher, opponent_tenpai_probs, payoff_table, ryukyoku_prob
from .params import PARAMS as _P
from .posture import Posture
from .replay import CallOpportunity, DecisionPoint, tenhou_to_mpsz
from .review import (
    _NANIKIRU_TRANSPORT_ERRORS,
    _at_turn,
    _cand_ev,
    _cand_uke,
    _cand_utility,
    _filter_furiten_waits,
    _norm_tile_name,
    _policy_valid_candidates,
    _wall_inputs_from_dp,
    restart_nanikiru,
)
from .scoring import offensive_desire_from_dp, score_candidates, softmax_weights

_AKA = {15: 51, 25: 52, 35: 53, 51: 15, 52: 25, 53: 35}
_FIVE_RED_PAIR = {4: 34, 13: 35, 22: 36, 34: 4, 35: 13, 36: 22}  # 5m/5p/5s <-> 红五槽

# 引擎查询韧性：复杂手形双开 flag 会栈溢出/挂起（已知行为），按序降级重试；
# 最后一档双关是保底，避免副露跳过侧摸牌枚举把整局拖死。
# 副露检讨查询量大（见 _query_best）：
#   切后向听 ≥2 → 关退向（引擎本身 ≥3 才关）；
#   切后向听 ≥3 → 连手替也关（3 向听散牌开手替易把进程打崩，对副露相对判定影响很小）。
_FLAG_ATTEMPTS = ((True, True), (True, False), (False, True), (False, False))
_FLAG_ATTEMPTS_NO_SHANTEN_DOWN = ((True, False), (False, False))
_FLAG_ATTEMPTS_NO_TEGAWARI = ((False, False),)


def legal_calls(hand_counter: Counter, disc_tile: int, from_kamicha: bool) -> dict:
    """返回 {'pon': [...], 'chii': [...], 'daiminkan': [...]}。

    每个 variant 是需从手牌消耗的 tenhou int 列表（赤五保留实体）。
    碰消耗 2 张，大明杠消耗 3 张，吃消耗 2 张。
    """
    out: Dict[str, list] = {"pon": [], "chii": [], "daiminkan": []}
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
        # 大明杠：枚举不同的 3 张消耗组合（含赤五混搭）
        seen_kan = set()
        for i in range(len(same)):
            for j in range(i + 1, len(same)):
                for k in range(j + 1, len(same)):
                    key = tuple(sorted((same[i], same[j], same[k])))
                    if key not in seen_kan:
                        seen_kan.add(key)
                        out["daiminkan"].append(list(key))

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


def legal_kakans(hand: List[str], melds: Optional[List[dict]]) -> List[dict]:
    """可加杠选项：已有碰 + 手牌持有第 4 张。

    返回 ``[{"tile": mpsz, "pon_index": int}, ...]``；同一碰若手牌有
    普通/赤五两种实体则各列一条。
    """
    out: List[dict] = []
    for i, m in enumerate(melds or []):
        t = str((m or {}).get("type") or "").strip().lower()
        if t not in ("pon", "碰", "p"):
            continue
        mt = (m or {}).get("tiles") or []
        if not mt:
            continue
        kind = _norm_kind(str(mt[0]))
        seen_ent: set = set()
        for ht in hand or []:
            if _norm_kind(str(ht)) != kind:
                continue
            if ht in seen_ent:
                continue
            seen_ent.add(ht)
            out.append({"tile": ht, "pon_index": i})
    return out


def legal_ankans(hand: List[str], melds: Optional[List[dict]] = None) -> List[dict]:
    """可暗杠选项：手牌同种（赤五归一）≥4 张。

    返回 ``[{"tiles": [4 mpsz], "kind": str}, ...]``；普通/赤五实体组合去重。
    ``melds`` 保留以与 ``legal_kakans`` 签名对齐（暗杠不依赖已有碰）。
    """
    _ = melds
    by_kind: Dict[str, List[str]] = {}
    for ht in hand or []:
        k = _norm_kind(str(ht))
        by_kind.setdefault(k, []).append(str(ht))
    out: List[dict] = []
    seen: set = set()
    for kind, ents in by_kind.items():
        if len(ents) < 4:
            continue
        for combo in combinations(range(len(ents)), 4):
            tiles = [ents[i] for i in combo]
            key = tuple(sorted(tiles))
            if key in seen:
                continue
            seen.add(key)
            out.append({"tiles": list(tiles), "kind": kind})
    return out


_WAIT_TILE_KINDS: List[str] = (
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
)


def _winning_tile_kinds(hand: List[str], n_meld: int) -> set:
    """听牌形下可和牌种集合（归一化 mpsz）；非听牌返回空集。"""
    if local_shanten(list(hand), n_meld) != 0:
        return set()
    waits: set = set()
    for t in _WAIT_TILE_KINDS:
        if local_shanten(list(hand) + [t], n_meld) == -1:
            waits.add(t)
    return waits


def ankan_preserves_waits(
    hand14: List[str],
    melds: Optional[List[dict]],
    ankan_tiles: List[str],
    drawn_tile: Optional[str],
) -> bool:
    """暗杠前后可和牌集合是否相同。

    未听牌时不设闸，返回 True。立直后 / 立直前已听时用于合法性硬闸。
    """
    n_meld = len(melds or [])
    if drawn_tile:
        before = _remove_consume_from_hand(list(hand14), [str(drawn_tile)])
    else:
        before = list(hand14)
        if len(before) % 3 == 2 and before:
            # 14 张无 drawn 标记：去掉一张与暗杠同种的牌作为「摸入」近似
            kind = _norm_kind(str(ankan_tiles[0])) if ankan_tiles else ""
            removed = False
            for i, t in enumerate(before):
                if _norm_kind(str(t)) == kind:
                    before = before[:i] + before[i + 1 :]
                    removed = True
                    break
            if not removed:
                before = before[:-1]
    if before is None:
        return False
    if local_shanten(before, n_meld) != 0:
        return True
    waits_before = _winning_tile_kinds(before, n_meld)
    after_hand = _remove_consume_from_hand(list(hand14), list(ankan_tiles))
    if after_hand is None:
        return False
    waits_after = _winning_tile_kinds(after_hand, n_meld + 1)
    return waits_before == waits_after


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


def kuikae_forbidden_tiles(meld: Optional[dict]) -> set:
    """鸣牌后立刻禁切的牌种（現物食替 + 吃的筋食替），归一化 mpsz。

    规则（天凤/雀魂同）：
    - 碰/吃：不可切与被鸣牌同种（赤五≡普通 5）；
    - 吃：以升序顺子 [L,M,R] 计，鸣 R 禁 L−1，鸣 L 禁 R+1，鸣 M 两侧都禁。
    """
    if not meld:
        return set()
    mtype = str(meld.get("type") or "").lower()
    if mtype not in ("chii", "chi", "pon", "碰", "吃"):
        return set()
    called = meld.get("calledTile")
    if not called:
        return set()
    forbidden = {_norm_kind(str(called))}
    if mtype in ("pon", "碰"):
        return forbidden

    ranks: List[int] = []
    suit: Optional[str] = None
    for raw in meld.get("tiles") or []:
        t = _norm_kind(str(raw))
        if len(t) < 2 or t[1] not in "mps" or not t[0].isdigit():
            return forbidden
        ranks.append(int(t[0]))
        suit = t[1]
    if suit is None or len(ranks) != 3:
        return forbidden
    ranks.sort()
    lo, mid, hi = ranks[0], ranks[1], ranks[2]
    if hi - lo != 2 or mid != lo + 1:
        return forbidden
    c = int(_norm_kind(str(called))[0])
    if c == hi and lo > 1:
        forbidden.add(f"{lo - 1}{suit}")
    if c == lo and hi < 9:
        forbidden.add(f"{hi + 1}{suit}")
    if c == mid:
        if lo > 1:
            forbidden.add(f"{lo - 1}{suit}")
        if hi < 9:
            forbidden.add(f"{hi + 1}{suit}")
    return forbidden


def apply_kuikae_marks(cands: List[dict], meld: Optional[dict]) -> set:
    """给候选打上 kuikae 标记；返回禁切牌种集合。"""
    banned = kuikae_forbidden_tiles(meld)
    if not banned:
        return banned
    for c in cands:
        if _norm_kind(str(c.get("tile") or "")) in banned:
            c["kuikae"] = True
    return banned


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
    pool = [c for c in (valid or list(cands)) if not c.get("kuikae")]
    if not pool:
        pool = [c for c in cands if not c.get("kuikae")] or list(cands)
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
            "kuikae": bool(c.get("kuikae")),
            "policy_rejected": id(c) not in valid_ids or bool(c.get("kuikae")),
        }
        for c in ranked[:3]
    ]


def _min_cut_shanten(hand_tiles: List[str], n_meld: int = 0) -> int:
    """查询手牌的最低切后向听（14/11 张需切一张；13/10 张直接算）。"""
    if len(hand_tiles) % 3 != 2:
        return local_shanten(hand_tiles, n_meld)
    best = 99
    for i in range(len(hand_tiles)):
        after = hand_tiles[:i] + hand_tiles[i + 1 :]
        best = min(best, local_shanten(after, n_meld))
    return best


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
    kuikae_meld: Optional[dict] = None,
) -> dict:
    """一次引擎查询，按切牌卡综合口径取最优切，返回 {ev, win, best_tile,
    shanten, cut_shanten, top3}；失败 {"error": ...}。

    连接断开/超时时重启引擎并按 (手替, 向听回退) 降级重试；
    success:false + err_msg 视为该次失败，不重试。
    ``defense`` 非空时选切计入危险度与罚符信用（与切牌卡一致）。
    ``kuikae_meld`` 为刚鸣上的面子时，禁切食替牌后再选最优。
    切后向听 ≥2 时关闭退向；≥3 时连手替也关（防散牌手替栈溢出）。
    """
    n_meld = len(melds)
    cut_sh = _min_cut_shanten(hand_tiles, n_meld)
    # ≥2 关退向；≥3 再关手替（副露查询多，优先稳与相对判定）
    if cut_sh >= 3:
        flag_attempts = _FLAG_ATTEMPTS_NO_TEGAWARI
    elif cut_sh >= 2:
        flag_attempts = _FLAG_ATTEMPTS_NO_SHANTEN_DOWN
    else:
        flag_attempts = _FLAG_ATTEMPTS

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
    n_flags = len(flag_attempts)
    url = pick_url(nanikiru_url)
    for i, (teg, sd) in enumerate(flag_attempts):
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
        cands = _build_cut_candidates(
            raw, turn, self_discards, hand_tiles=hand_tiles, n_meld=n_meld
        )
        if not cands:
            return {"error": "no candidates"}
        if kuikae_meld is not None:
            apply_kuikae_marks(cands, kuikae_meld)
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


_OPEN_MELD_TYPES = frozenset({"chii", "pon", "daiminkan", "kakan", "吃", "碰", "大明杠", "加杠"})


def _is_menzen(melds: Optional[List[dict]]) -> bool:
    """无吃碰明杠加杠则门清（仅暗杠仍算门清）。"""
    for m in melds or []:
        t = str((m or {}).get("type") or "").strip().lower()
        if t in _OPEN_MELD_TYPES:
            return False
    return True


def _daiminkan_bias(menzen: bool, self_tenpai: bool, opp_riichi: bool) -> float:
    """按情境返回大明杠倍率。"""
    p = _P["call"]
    if menzen:
        return float(p["daiminkan_bias_menzen"])
    if self_tenpai and not opp_riichi:
        return float(p["daiminkan_bias_tenpai_safe"])
    if self_tenpai and opp_riichi:
        return float(p["daiminkan_bias_tenpai_riichi"])
    if (not self_tenpai) and opp_riichi:
        return float(p["daiminkan_bias_noten_riichi"])
    return float(p["daiminkan_bias_default"])


def _daiminkan_hard_block(menzen: bool, self_tenpai: bool, opp_riichi: bool) -> bool:
    """门清，或未听且有他家立直 → 硬门控不推荐大明杠。"""
    return menzen or ((not self_tenpai) and opp_riichi)


def _remove_consume_from_hand(
    hand: List[str], consume_names: List[str]
) -> Optional[List[str]]:
    """从手牌移除消耗牌（赤五精确匹配，普通/赤五互换兜底）；失败返回 None。"""
    new_hand = list(hand)
    for cn in consume_names:
        if cn in new_hand:
            new_hand.remove(cn)
            continue
        alt = {"0m": "5m", "0p": "5p", "0s": "5s", "5m": "0m", "5p": "0p", "5s": "0s"}.get(
            cn
        )
        if alt and alt in new_hand:
            new_hand.remove(alt)
        else:
            return None
    return new_hand


def _build_draw_jobs(
    hand_tiles: List[str],
    melds: List[dict],
    other_melds: List[dict],
    seen: List[str],
    self_discards: List[str],
    dora: List[str],
) -> Tuple[float, List[Tuple[str, int, bool]], List[Tuple[str, int]]]:
    """构造剩余牌墙摸牌假设。

    返回 ``(total_w, draw_jobs, irrelevant)``。
    ``draw_jobs`` 元素为 ``(tname, cnt, is_lump)``；无关牌打包为一项。
    """
    hand_ids = parse_hand(hand_tiles)
    wall = build_wall(
        hand_ids,
        parse_melds([{"type": m["type"], "tiles": m["tiles"]} for m in melds]),
        [tile_name_to_id(t) for t in dora],
        [tile_name_to_id(t) for t in seen],
        1,
        other_melds=parse_melds(other_melds),
    )
    _furiten_wall_zero(wall, self_discards)

    total_w = float(sum(wall[:34]))
    draw_kinds: List[Tuple[str, int]] = []
    for tid in range(34):
        n_normal = wall[tid]
        n_red = wall[_FIVE_RED_PAIR[tid]] if tid in _FIVE_RED_PAIR else 0
        if n_normal - n_red > 0:
            draw_kinds.append((id_to_tile_name(tid), n_normal - n_red))
        if n_red > 0:
            draw_kinds.append((id_to_tile_name(_FIVE_RED_PAIR[tid]), n_red))

    hand_ids_set = set(hand_ids)
    dora_kinds = set()
    for d in dora:
        try:
            did = tile_name_to_id(d)
        except ValueError:
            continue
        if did < 27:
            dora_kinds.add((did // 9) * 9 + (did % 9 + 1) % 9)
        elif did < 31:
            dora_kinds.add(27 + (did - 27 + 1) % 4)
        else:
            dora_kinds.add(31 + (did - 31 + 1) % 3)

    def is_relevant(tid: int) -> bool:
        if tid >= 34:
            return True
        if tid in dora_kinds:
            return True
        if tid in hand_ids_set or tid in (4, 13, 22) and (
            {4: 34, 13: 35, 22: 36}[tid] in hand_ids_set
        ):
            return True
        if tid >= 27:
            return False
        suit, num = tid // 9, tid % 9
        for h in hand_ids_set:
            if h >= 27:
                continue
            hs, hn = h // 9, h % 9
            if hs == suit and abs(hn - num) <= 2:
                return True
            if h in (34, 35, 36):
                rn = {34: (0, 4), 35: (1, 4), 36: (2, 4)}[h]
                if rn[0] == suit and abs(rn[1] - num) <= 2:
                    return True
        return False

    relevant = [(t, c) for t, c in draw_kinds if is_relevant(tile_name_to_id(t))]
    irrelevant = [(t, c) for t, c in draw_kinds if not is_relevant(tile_name_to_id(t))]
    draw_jobs: List[Tuple[str, int, bool]] = [
        (tname, cnt, False) for tname, cnt in relevant
    ]
    if irrelevant:
        draw_jobs.append(
            (irrelevant[0][0], sum(c for _, c in irrelevant), True)
        )
    return total_w, draw_jobs, irrelevant


def _rinshan_expectation(
    hand10: List[str],
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
    workers: int = 1,
) -> dict:
    """杠后 10 张手牌：对岭上摸牌取期望（当前 dora，不翻新指示牌）。

    返回 ``{ev, win, best_tile, cut_shanten, shanten, top3}`` 或 ``{error}``。
    ``cut_shanten`` / ``best_tile`` 取概率最大的成功摸牌路径。
    """
    total_w, draw_jobs, irrelevant = _build_draw_jobs(
        hand10, melds, other_melds, seen, self_discards, dora
    )
    if total_w <= 0 or not draw_jobs:
        return {"error": "岭上牌墙为空"}

    def query_draw(tname: str) -> dict:
        return _query_best(
            hand10 + [tname],
            melds,
            other_melds,
            seen,
            self_discards,
            dora,
            round_wind,
            seat_wind,
            turn,
            nanikiru_url,
            timeout,
            opp=opp,
            defense=defense,
        )

    draw_results: Dict[int, dict] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(workers, len(draw_jobs)))
    ) as ex:
        fut_map = {
            ex.submit(query_draw, tname): idx
            for idx, (tname, _cnt, _lump) in enumerate(draw_jobs)
        }
        for fut in as_completed(fut_map):
            draw_results[fut_map[fut]] = fut.result()

    ev_sum = 0.0
    wp_sum = 0.0
    ok_weight = 0.0
    best_path: Optional[dict] = None
    best_p = -1.0
    errors: List[str] = []
    for idx, (tname, cnt, is_lump) in enumerate(draw_jobs):
        r = draw_results.get(idx) or {}
        if r.get("ev") is None:
            label = f"无关牌代表{tname}" if is_lump else f"岭上{tname}"
            errors.append(f"{label}: {r.get('error')}")
            continue
        p = cnt / total_w
        ev_sum += p * r["ev"]
        wp_sum += p * (r["win"] or 0.0)
        ok_weight += p
        if p > best_p:
            best_p = p
            best_path = r

    if ok_weight <= 0 or best_path is None:
        return {
            "error": "岭上期望无法计算："
            + "; ".join(errors or ["无可用摸牌假设"]),
        }
    # 成功路径未覆盖全分布时按条件期望归一（失败假设不计入）
    scale = 1.0 / ok_weight if ok_weight < 1.0 - 1e-9 else 1.0
    return {
        "ev": ev_sum * scale,
        "win": wp_sum * scale,
        "best_tile": best_path.get("best_tile"),
        "cut_shanten": best_path.get("cut_shanten"),
        "shanten": best_path.get("shanten"),
        "top3": best_path.get("top3"),
    }


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
    """对一个副露机会点求 EV(跳过) 与 EV(各碰/吃/大明杠变体)。

    跳过/副露两侧的选切均走切牌卡综合口径（``defense`` 非空时计入防守）。
    ``posture`` / ``threats`` 为形听轴语境；``defense`` 为完整防守结果
    （review 传入）。危险查询 callable 不可序列化，故 risk_lookup 不经本
    函数而直接传给 decide。

    摸牌假设与副露变体的引擎查询可按 ``workers`` 并行（依赖活跃 nanikiru 池）。
    大明杠走岭上期望（不翻新宝牌），受 ``params.call.eval_daiminkan`` 开关控制。
    """
    turn = opp.turn  # 副露后切牌 / 下次摸牌均按「自家即将进行的巡」取值
    dora = opp.dora_indicators
    errors: List[str] = []
    # threats 可单独传入；有完整 defense 时以其 threats 为准
    if defense is not None and threats is None:
        threats = defense.get("threats") or []
    n_workers = resolve_workers(workers)

    # ---- 情境：门清 / 听牌 / 他家立直（大明杠偏置） ----
    menzen = _is_menzen(opp.melds)
    try:
        self_tenpai = local_shanten(opp.hand, len(opp.melds or [])) == 0
    except Exception:
        self_tenpai = False
    opp_riichi = any(
        (t or {}).get("kind") == "riichi" for t in (threats or [])
    )
    kan_bias = _daiminkan_bias(menzen, self_tenpai, opp_riichi)
    kan_hard_block = _daiminkan_hard_block(menzen, self_tenpai, opp_riichi)
    kan_context = {
        "menzen": menzen,
        "self_tenpai": self_tenpai,
        "opp_riichi": opp_riichi,
        "bias": kan_bias,
        "hard_block": kan_hard_block,
    }

    # ---- 跳过侧：本地 wall 算摸牌分布，逐假设查引擎加权 ----
    seen0, om0, _ = _build_seen_other(opp)
    total_w, draw_jobs, irrelevant = _build_draw_jobs(
        opp.hand, opp.melds, om0, seen0, opp.self_discards, dora
    )

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

    draw_results: Dict[int, dict] = {}
    if draw_jobs and total_w > 0:
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
            "kan_context": kan_context,
            "error": "跳过侧期望无法计算：" + "; ".join(errors or ["无可用摸牌假设"]),
        }

    # 当前 13 张手牌的向听：摸牌后 14 张向听 ∈ {n−1, n}，取各假设向听最大值
    _shs = [d["shanten"] for d in skip_detail if d.get("shanten") is not None]
    skip_shanten = max(_shs) if _shs else None

    # ---- 副露侧：碰/吃（立即切）+ 大明杠（岭上期望） ----
    variant_specs: List[dict] = []
    for action in ("pon", "chii"):
        for consume in (opp.legal or {}).get(action) or []:
            consume_names = [tenhou_to_mpsz(t) for t in consume]
            new_hand = _remove_consume_from_hand(opp.hand, consume_names)
            if new_hand is None:
                continue
            meld_tiles = sorted(
                consume_names + [opp.disc_tile], key=meld_tile_sort_key
            )
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
                    "new_meld": new_meld,
                    "self_melds1": self_melds1,
                    "seen1": seen1,
                    "om1": om1,
                    "mode": "immediate",
                }
            )

    if bool(_P["call"].get("eval_daiminkan")):
        raw_kan = (opp.legal or {}).get("daiminkan")
        if isinstance(raw_kan, list):
            kan_consumes = raw_kan
        elif raw_kan:
            # 旧快照 bool：按手牌实体重建 3 张消耗（忽略赤五细组合，用同种三张）
            kan_consumes = []
            kind = _norm_kind(str(opp.disc_tile or ""))
            pool = [t for t in opp.hand if _norm_kind(t) == kind]
            if len(pool) >= 3:
                # 唯一实体组合：排序后取三张的多重集去重由 legal 保证；这里只取一种
                consume_names = sorted(pool[:3], key=meld_tile_sort_key)
                # 伪 tenhou 列表：后续用 consume_names 分支
                kan_consumes = [consume_names]  # mark as already mpsz
        else:
            kan_consumes = []
        for consume in kan_consumes:
            if consume and isinstance(consume[0], str):
                consume_names = list(consume)
            else:
                consume_names = [tenhou_to_mpsz(t) for t in consume]
            new_hand = _remove_consume_from_hand(opp.hand, consume_names)
            if new_hand is None:
                continue
            meld_tiles = sorted(
                consume_names + [opp.disc_tile], key=meld_tile_sort_key
            )
            new_meld = {
                "type": "daiminkan",
                "tiles": meld_tiles,
                "calledTile": opp.disc_tile,
                "sourceSeat": opp.discarder,
            }
            seen1, om1, self_melds1 = _build_seen_other(
                opp, extra_self_meld=new_meld
            )
            variant_specs.append(
                {
                    "action": "daiminkan",
                    "consume_names": consume_names,
                    "new_hand": new_hand,
                    "new_meld": new_meld,
                    "self_melds1": self_melds1,
                    "seen1": seen1,
                    "om1": om1,
                    "mode": "rinshan",
                }
            )

    def query_variant(spec: dict) -> dict:
        if spec.get("mode") == "rinshan":
            return _rinshan_expectation(
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
                workers=n_workers,
            )
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
            kuikae_meld=spec.get("new_meld"),
        )

    # 立即切变体外层并行；岭上变体串行（内部已并行摸牌），避免嵌套线程池过载
    variant_results: Dict[int, dict] = {}
    immediate_idx = [
        i for i, s in enumerate(variant_specs) if s.get("mode") != "rinshan"
    ]
    rinshan_idx = [
        i for i, s in enumerate(variant_specs) if s.get("mode") == "rinshan"
    ]
    if immediate_idx:
        with ThreadPoolExecutor(
            max_workers=max(1, min(n_workers, len(immediate_idx)))
        ) as ex:
            fut_map = {
                ex.submit(query_variant, variant_specs[i]): i for i in immediate_idx
            }
            for fut in as_completed(fut_map):
                variant_results[fut_map[fut]] = fut.result()
    for i in rinshan_idx:
        variant_results[i] = query_variant(variant_specs[i])

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
        cut_sh = r.get("cut_shanten")
        if cut_sh is None and r.get("best_tile") and spec.get("mode") != "rinshan":
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
        entry = {
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
        if action == "daiminkan":
            entry["daiminkan_bias"] = kan_bias
            entry["daiminkan_hard_block"] = kan_hard_block
        variants.append(entry)

    return {
        "ok": True,
        "turn": turn,
        "ev_skip": ev_skip,
        "win_skip": wp_skip,
        "skip_shanten": skip_shanten,
        "skip_detail": skip_detail,
        "variants": variants,
        "kan_context": kan_context,
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
    大明杠另受情境硬门控与 bias 倍率（合格判定与报告权重）。
    """
    p = _P["call"]
    out: Dict[str, Any] = {
        "recommend": None,
        "variant": None,
        "axis": None,
        "basis": "",
        "rejected_variants": [],
        "kan_context": eval_result.get("kan_context"),
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

    def _kan_bias(v: Dict[str, Any]) -> float:
        if v.get("action") != "daiminkan":
            return 1.0
        b = v.get("daiminkan_bias")
        if b is None:
            ctx = eval_result.get("kan_context") or {}
            b = ctx.get("bias", 1.0)
        return float(b) if b is not None else 1.0

    def _eff_delta_win(v: Dict[str, Any]) -> float:
        return (v.get("delta_win") or 0.0) * _kan_bias(v)

    def _eff_delta_ev(v: Dict[str, Any]) -> float:
        return (v.get("delta_ev") or 0.0) * _kan_bias(v)

    def _speed_ok(v: Dict[str, Any]) -> bool:
        """速度轴合格：绝对差路径，或比例旁路（和率比≥阈值 且 Δ和率≥floor）。

        WP_skip==0 时比例视为无穷大：Δ和率 ≥ floor 即达标（从死牌复活是实质增益）。
        大明杠用 bias 放大/缩小有效 Δ和率。
        """
        dw = _eff_delta_win(v)
        raw_dw = v.get("delta_win") or 0.0
        if dw >= delta_min:
            return True
        if dw < delta_floor:
            return False
        if wp_skip > 0:
            # 比例旁路仍看原始和率比，但要求有效 Δ 过 floor
            return (v.get("win") or 0.0) / wp_skip >= ratio_min
        return raw_dw >= 0.0

    alive = []
    # 形听语境（须先于自杀层判定：语境决定 win==0 变体是否豁免）
    ft = _form_tenpai_context(eval_result, risk_lookup, p)
    ft_candidates = []  # 被形听豁免、改走形听轴的 win==0 变体（不进双轴）
    for v in eval_result.get("variants") or []:
        win = v.get("win") or 0.0
        ev = v.get("ev") or 0.0
        # 大明杠硬门控：门清 / 未听+有立直 → 不推荐（仍参与权重展示）
        if v.get("action") == "daiminkan" and (
            v.get("daiminkan_hard_block")
            or (eval_result.get("kan_context") or {}).get("hard_block")
        ):
            v["rejected"] = "kan_context"
            out["rejected_variants"].append(
                {**v, "reason": "大明杠情境硬门控（门清或未听+他家立直）"}
            )
            continue
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
        elif _eff_delta_ev(v) > ev_margin:
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
        # 双轴优先：合格者按 bias 后 EV 取最高（杠与碰并存时情境可抬杠）
        best_v, axis = max(
            qualified,
            key=lambda q: (q[0].get("ev") or 0.0) * _kan_bias(q[0]),
        )
        out["recommend"] = "call"
        out["variant"] = best_v
        out["axis"] = axis
        if axis == "speed":
            dw = best_v.get("delta_win") or 0.0
            if _eff_delta_win(best_v) >= delta_min:
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
    # 双轴/形听轴均未过时不写提示（报告侧不展示依据）

    # ---- 报告权重（softmax 显示校准，不参与判定） ----
    temp = float(_P["scoring"]["temperature"])
    scale = float(p["win_prob_weight_scale"])
    ft_value_min = float(p["form_tenpai_value_min"])
    utils: List[float] = [0.0]  # 第 0 项为跳过（基准 0）
    for v in eval_result.get("variants") or []:
        d_win = _eff_delta_win(v) * scale
        d_ev = _eff_delta_ev(v) - ev_margin
        fti = v.get("form_tenpai")
        if fti and fti.get("qualified"):
            d = (fti["value"] - ft_value_min) * _kan_bias(v)
        elif v.get("form_tenpai_candidate"):
            d = min(d_win, d_ev)
        elif not v.get("rejected") and _speed_ok(v):
            d = d_win
        elif not v.get("rejected") and _eff_delta_ev(v) > ev_margin:
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
    """实际行动与推荐是否一致（含大明杠）。"""
    if actual == "skip":
        return decision.get("recommend") is None
    variant = decision.get("variant") or {}
    return decision.get("recommend") == "call" and variant.get("action") == actual


def _kakan_bias(self_tenpai: bool, opp_riichi: bool) -> float:
    """加杠情境倍率（无门清档：可加杠必已有碰）。"""
    p = _P["call"]
    if self_tenpai and not opp_riichi:
        return float(p["kakan_bias_tenpai_safe"])
    if self_tenpai and opp_riichi:
        return float(p["kakan_bias_tenpai_riichi"])
    if (not self_tenpai) and opp_riichi:
        return float(p["kakan_bias_noten_riichi"])
    return float(p["kakan_bias_default"])


def _self_discards_from_dp(dp: Any) -> List[str]:
    out: List[str] = []
    for t in (getattr(dp, "rivers", None) or {}).get("自家") or []:
        name = _strip_river_marker(t)
        if name.startswith("0"):
            name = "5" + name[1:]
        if name:
            out.append(name)
    return out


def _upgrade_pon_to_kakan(
    melds: List[dict], pon_index: int, tile: str
) -> List[dict]:
    new_melds = [dict(m) for m in melds]
    if not (0 <= pon_index < len(new_melds)):
        return new_melds
    m = dict(new_melds[pon_index])
    tiles = list(m.get("tiles") or []) + [tile]
    m["type"] = "kakan"
    m["tiles"] = sorted(tiles, key=meld_tile_sort_key)
    new_melds[pon_index] = m
    return new_melds


def attach_kakan_to_analysis(
    dp: DecisionPoint,
    analysis: Dict[str, Any],
    nanikiru_url: str,
    *,
    defense: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """在切牌分析结果上挂接加杠岭上期望，并与切牌候选合并 softmax 权重。

    修改 ``analysis`` 原地：增加 ``kakan_options`` / ``recommend_action``，
    重写各候选与杠行的 ``recommendation_weight``，更新 ``best`` / ``match``。
    """
    if not analysis.get("ok"):
        return analysis
    if not bool(_P["call"].get("eval_kakan")):
        return analysis

    legal = list(getattr(dp, "legal_kakans", None) or [])
    if not legal:
        legal = legal_kakans(dp.hand, dp.melds)
    if not legal:
        return analysis

    cands = list(analysis.get("candidates") or [])
    if not cands:
        return analysis

    # 基线：综合效用最高的切（与推荐口径一致）
    best_c = max(
        cands,
        key=lambda c: float(c.get("adjusted_utility") or c.get("exp_score") or -1e18),
    )
    ev_base = float(best_c.get("exp_score") or 0.0)
    wp_base = float(best_c.get("win_prob") or 0.0)
    util_base = float(best_c.get("adjusted_utility") or ev_base)

    self_tenpai = any(int(c.get("shanten") or 99) == 0 for c in cands)
    threats = []
    if defense is not None:
        threats = defense.get("threats") or []
    elif isinstance(analysis.get("defense"), dict):
        threats = analysis["defense"].get("threats") or []
    opp_riichi = any((t or {}).get("kind") == "riichi" for t in threats)
    bias = _kakan_bias(self_tenpai, opp_riichi)
    danger_cap = float(_P["call"]["kakan_chankan_danger_cap"])

    # 抢杠简版：有立直，或加杠牌危险过高 → 硬门控
    def _chankan_hard(tile: str) -> bool:
        if opp_riichi:
            return True
        if defense is None:
            return False
        try:
            from .defense import deal_in_for_tile

            v = deal_in_for_tile(defense, tile or "").get("combined")
            return v is not None and float(v) >= danger_cap
        except Exception:
            return False

    seen0, om0 = _wall_inputs_from_dp(dp)
    self_disc = _self_discards_from_dp(dp)
    n_workers = resolve_workers(workers)
    turn = int(getattr(dp, "turn", 1) or 1)
    dora = list(getattr(dp, "dora_indicators", None) or [])

    p_call = _P["call"]
    delta_min = float(p_call["win_prob_delta_min"])
    ratio_min = float(p_call["win_prob_ratio_min"])
    delta_floor = float(p_call["win_prob_delta_floor"])
    ev_margin = float(p_call["ev_margin"])
    scale = float(p_call["win_prob_weight_scale"])
    temp = float(_P["scoring"]["temperature"])

    kakan_options: List[dict] = []
    for spec in legal:
        tile = spec.get("tile")
        pon_index = int(spec.get("pon_index", -1))
        if not tile or pon_index < 0:
            continue
        new_hand = _remove_consume_from_hand(list(dp.hand), [tile])
        if new_hand is None:
            continue
        new_melds = _upgrade_pon_to_kakan(list(dp.melds or []), pon_index, tile)
        r = _rinshan_expectation(
            new_hand,
            new_melds,
            om0,
            seen0,
            self_disc,
            dora,
            dp.round_wind,
            dp.seat_wind,
            turn,
            nanikiru_url,
            timeout,
            defense=defense,
            workers=n_workers,
        )
        if r.get("ev") is None:
            kakan_options.append(
                {
                    "action": "kakan",
                    "tile": tile,
                    "pon_index": pon_index,
                    "error": r.get("error"),
                    "rejected": "error",
                    "bias": bias,
                    "hard_block": True,
                }
            )
            continue
        ev = float(r["ev"])
        win = float(r["win"] or 0.0)
        delta_ev = ev - ev_base
        delta_win = win - wp_base
        hard = _chankan_hard(str(tile)) or (
            (not self_tenpai) and opp_riichi
        )
        # 未听+立直已在 hard；听牌+立直用低 bias，另若危险高也 hard
        entry = {
            "action": "kakan",
            "tile": tile,
            "pon_index": pon_index,
            "ev": ev,
            "win": win,
            "exp_score": ev,
            "win_prob": win,
            "cut": r.get("best_tile"),
            "cut_shanten": r.get("cut_shanten"),
            "shanten": r.get("cut_shanten"),
            "delta_ev": delta_ev,
            "delta_win": delta_win,
            "bias": bias,
            "hard_block": hard,
            "rejected": "chankan" if hard else None,
            "deal_in": None,
        }
        if defense is not None:
            try:
                from .defense import deal_in_for_tile

                entry["deal_in"] = deal_in_for_tile(defense, str(tile))
            except Exception:
                pass
        kakan_options.append(entry)

    if not kakan_options:
        return analysis

    def _eff_dw(opt: dict) -> float:
        return float(opt.get("delta_win") or 0.0) * float(opt.get("bias") or 1.0)

    def _eff_de(opt: dict) -> float:
        return float(opt.get("delta_ev") or 0.0) * float(opt.get("bias") or 1.0)

    def _speed_ok(opt: dict) -> bool:
        dw = _eff_dw(opt)
        if dw >= delta_min:
            return True
        if dw < delta_floor:
            return False
        if wp_base > 0:
            return (float(opt.get("win") or 0.0) / wp_base) >= ratio_min
        return True

    def _kakan_d(opt: dict) -> float:
        if opt.get("rejected") or opt.get("hard_block") or opt.get("ev") is None:
            d_win = _eff_dw(opt) * scale
            d_ev = _eff_de(opt) - ev_margin
            return min(d_win, d_ev, -1.0)
        d_win = _eff_dw(opt) * scale
        d_ev = _eff_de(opt) - ev_margin
        if _speed_ok(opt):
            return d_win
        if _eff_de(opt) > ev_margin:
            return d_ev
        return min(d_win, d_ev)

    # 合格杠（用于推荐）
    qualified = [
        o
        for o in kakan_options
        if o.get("ev") is not None
        and not o.get("hard_block")
        and not o.get("rejected")
        and (_speed_ok(o) or _eff_de(o) > ev_margin)
    ]

    # softmax：切牌相对 util_base + 杠侧 d
    utils: List[float] = [
        float(c.get("adjusted_utility") or 0.0) - util_base for c in cands
    ]
    for o in kakan_options:
        utils.append(_kakan_d(o))
    weights = softmax_weights(utils, temp)
    for c, w in zip(cands, weights[: len(cands)]):
        c["recommendation_weight"] = w
    for o, w in zip(kakan_options, weights[len(cands) :]):
        o["recommendation_weight"] = w
        # 报告用综合效用：基线 + d
        o["adjusted_utility"] = util_base + _kakan_d(o)

    # 推荐：合格杠中权重最高，否则最高权切牌
    recommend_action = "discard"
    recommend_kakan = None
    if qualified:
        best_k = max(
            qualified, key=lambda o: float(o.get("recommendation_weight") or 0.0)
        )
        best_disc_w = max(float(c.get("recommendation_weight") or 0.0) for c in cands)
        if float(best_k.get("recommendation_weight") or 0.0) >= best_disc_w:
            recommend_action = "kakan"
            recommend_kakan = best_k

    if recommend_action == "kakan" and recommend_kakan is not None:
        analysis["best"] = None
        analysis["best_kakan"] = recommend_kakan.get("tile")
        analysis["recommend_action"] = "kakan"
    else:
        # 保持切牌 best（按新权重重选）
        from .scoring import best_candidate

        bc = best_candidate(cands)
        analysis["best"] = bc.get("tile") if bc else analysis.get("best")
        analysis["best_kakan"] = None
        analysis["recommend_action"] = "discard"

    analysis["kakan_options"] = kakan_options
    analysis["kakan_context"] = {
        "self_tenpai": self_tenpai,
        "opp_riichi": opp_riichi,
        "bias": bias,
    }

    # match
    actual_kakan = getattr(dp, "actual_kakan", None)
    actual_disc = getattr(dp, "actual_discard", None)
    if actual_kakan:
        match = (
            analysis.get("recommend_action") == "kakan"
            and _norm_kind(str(analysis.get("best_kakan") or ""))
            == _norm_kind(str(actual_kakan))
        )
        analysis["match"] = match
        analysis["match_kind"] = "same" if match else "different"
        analysis["actual"] = actual_kakan
        analysis["actual_action"] = "kakan"
    elif actual_disc:
        analysis["actual_action"] = "discard"
        if analysis.get("recommend_action") == "kakan":
            analysis["match"] = False
            analysis["match_kind"] = "different"
        else:
            from .scoring import classify_discard_match

            classified = classify_discard_match(
                cands, actual_disc, analysis.get("best")
            )
            analysis["match"] = classified["match"]
            analysis["match_kind"] = classified["match_kind"]
            analysis["equivalent_best"] = classified.get("equivalent_best")

    return analysis


def _ankan_bias(self_tenpai: bool, opp_riichi: bool) -> float:
    """暗杠情境倍率（无门清否决：暗杠后仍可门清立直）。"""
    p = _P["call"]
    if self_tenpai and not opp_riichi:
        return float(p["ankan_bias_tenpai_safe"])
    if self_tenpai and opp_riichi:
        return float(p["ankan_bias_tenpai_riichi"])
    if (not self_tenpai) and opp_riichi:
        return float(p["ankan_bias_noten_riichi"])
    return float(p["ankan_bias_default"])


def _make_ankan_meld(tiles: List[str]) -> dict:
    return {
        "type": "ankan",
        "tiles": sorted(list(tiles), key=meld_tile_sort_key),
        "calledTile": None,
        "sourceSeat": None,
    }


def attach_ankan_to_analysis(
    dp: DecisionPoint,
    analysis: Dict[str, Any],
    nanikiru_url: str,
    *,
    defense: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """在切牌（及已有加杠）分析结果上挂接暗杠岭上期望，合并 softmax。

    修改 ``analysis`` 原地：增加 ``ankan_options`` / 可能改写 ``recommend_action``。
    立直后/已听牌时待牌集合改变 → ``rejected=wait_change`` 硬门控。
    """
    if not analysis.get("ok"):
        return analysis
    if not bool(_P["call"].get("eval_ankan")):
        return analysis

    legal = list(getattr(dp, "legal_ankans", None) or [])
    if not legal:
        legal = legal_ankans(dp.hand, dp.melds)
    if not legal and not getattr(dp, "actual_ankan", None):
        return analysis

    cands = list(analysis.get("candidates") or [])
    if not cands:
        return analysis

    best_c = max(
        cands,
        key=lambda c: float(c.get("adjusted_utility") or c.get("exp_score") or -1e18),
    )
    ev_base = float(best_c.get("exp_score") or 0.0)
    wp_base = float(best_c.get("win_prob") or 0.0)
    util_base = float(best_c.get("adjusted_utility") or ev_base)

    self_tenpai = any(int(c.get("shanten") or 99) == 0 for c in cands)
    # 立直后决策点必听
    if getattr(dp, "is_riichi_post", False):
        self_tenpai = True
    threats = []
    if defense is not None:
        threats = defense.get("threats") or []
    elif isinstance(analysis.get("defense"), dict):
        threats = analysis["defense"].get("threats") or []
    opp_riichi = any((t or {}).get("kind") == "riichi" for t in threats)
    bias = _ankan_bias(self_tenpai, opp_riichi)

    seen0, om0 = _wall_inputs_from_dp(dp)
    self_disc = _self_discards_from_dp(dp)
    n_workers = resolve_workers(workers)
    turn = int(getattr(dp, "turn", 1) or 1)
    dora = list(getattr(dp, "dora_indicators", None) or [])
    drawn = getattr(dp, "drawn_tile", None)

    p_call = _P["call"]
    delta_min = float(p_call["win_prob_delta_min"])
    ratio_min = float(p_call["win_prob_ratio_min"])
    delta_floor = float(p_call["win_prob_delta_floor"])
    ev_margin = float(p_call["ev_margin"])
    scale = float(p_call["win_prob_weight_scale"])
    temp = float(_P["scoring"]["temperature"])

    # 听牌时才跑待牌闸（立直后强制；立直前 cut 后听才闸）
    gate_waits = self_tenpai or bool(getattr(dp, "is_riichi_post", False))

    ankan_options: List[dict] = []
    for spec in legal:
        tiles = list(spec.get("tiles") or [])
        kind = str(spec.get("kind") or (_norm_kind(tiles[0]) if tiles else ""))
        if len(tiles) != 4 or not kind:
            continue
        new_hand = _remove_consume_from_hand(list(dp.hand), tiles)
        if new_hand is None:
            continue

        wait_ok = True
        if gate_waits:
            wait_ok = ankan_preserves_waits(
                list(dp.hand), list(dp.melds or []), tiles, drawn
            )

        new_melds = [dict(m) for m in (dp.melds or [])] + [_make_ankan_meld(tiles)]
        if not wait_ok:
            ankan_options.append(
                {
                    "action": "ankan",
                    "tile": tiles[0],
                    "tiles": tiles,
                    "kind": kind,
                    "ev": None,
                    "win": None,
                    "exp_score": None,
                    "win_prob": None,
                    "delta_ev": None,
                    "delta_win": None,
                    "bias": bias,
                    "hard_block": True,
                    "rejected": "wait_change",
                    "deal_in": None,
                }
            )
            continue

        r = _rinshan_expectation(
            new_hand,
            new_melds,
            om0,
            seen0,
            self_disc,
            dora,
            dp.round_wind,
            dp.seat_wind,
            turn,
            nanikiru_url,
            timeout,
            defense=defense,
            workers=n_workers,
        )
        if r.get("ev") is None:
            ankan_options.append(
                {
                    "action": "ankan",
                    "tile": tiles[0],
                    "tiles": tiles,
                    "kind": kind,
                    "error": r.get("error"),
                    "rejected": "error",
                    "bias": bias,
                    "hard_block": True,
                }
            )
            continue
        ev = float(r["ev"])
        win = float(r["win"] or 0.0)
        delta_ev = ev - ev_base
        delta_win = win - wp_base
        hard = (not self_tenpai) and opp_riichi
        entry = {
            "action": "ankan",
            "tile": tiles[0],
            "tiles": tiles,
            "kind": kind,
            "ev": ev,
            "win": win,
            "exp_score": ev,
            "win_prob": win,
            "cut": r.get("best_tile"),
            "cut_shanten": r.get("cut_shanten"),
            "shanten": r.get("cut_shanten"),
            "delta_ev": delta_ev,
            "delta_win": delta_win,
            "bias": bias,
            "hard_block": hard,
            "rejected": "noten_riichi" if hard else None,
            "deal_in": None,
        }
        if defense is not None:
            try:
                from .defense import deal_in_for_tile

                entry["deal_in"] = deal_in_for_tile(defense, str(tiles[0]))
            except Exception:
                pass
        ankan_options.append(entry)

    if not ankan_options:
        return analysis

    def _eff_dw(opt: dict) -> float:
        return float(opt.get("delta_win") or 0.0) * float(opt.get("bias") or 1.0)

    def _eff_de(opt: dict) -> float:
        return float(opt.get("delta_ev") or 0.0) * float(opt.get("bias") or 1.0)

    def _speed_ok(opt: dict) -> bool:
        dw = _eff_dw(opt)
        if dw >= delta_min:
            return True
        if dw < delta_floor:
            return False
        if wp_base > 0:
            return (float(opt.get("win") or 0.0) / wp_base) >= ratio_min
        return True

    def _kan_d(opt: dict) -> float:
        if opt.get("rejected") or opt.get("hard_block") or opt.get("ev") is None:
            d_win = _eff_dw(opt) * scale
            d_ev = _eff_de(opt) - ev_margin
            return min(d_win, d_ev, -1.0)
        d_win = _eff_dw(opt) * scale
        d_ev = _eff_de(opt) - ev_margin
        if _speed_ok(opt):
            return d_win
        if _eff_de(opt) > ev_margin:
            return d_ev
        return min(d_win, d_ev)

    kakan_options = list(analysis.get("kakan_options") or [])

    def _qual(o: dict) -> bool:
        return (
            o.get("ev") is not None
            and not o.get("hard_block")
            and not o.get("rejected")
            and (_speed_ok(o) or _eff_de(o) > ev_margin)
        )

    qualified_ankan = [o for o in ankan_options if _qual(o)]
    qualified_kakan = [o for o in kakan_options if _qual(o)]

    # 统一 softmax：切 + 已有加杠 + 暗杠
    utils: List[float] = [
        float(c.get("adjusted_utility") or 0.0) - util_base for c in cands
    ]
    for o in kakan_options:
        utils.append(_kan_d(o))
    for o in ankan_options:
        utils.append(_kan_d(o))
    weights = softmax_weights(utils, temp)
    i = 0
    for c in cands:
        c["recommendation_weight"] = weights[i]
        i += 1
    for o in kakan_options:
        o["recommendation_weight"] = weights[i]
        o["adjusted_utility"] = util_base + _kan_d(o)
        i += 1
    for o in ankan_options:
        o["recommendation_weight"] = weights[i]
        o["adjusted_utility"] = util_base + _kan_d(o)
        i += 1

    best_disc_w = max(
        (float(c.get("recommendation_weight") or 0.0) for c in cands), default=0.0
    )
    recommend_action = "discard"
    recommend_ankan = None
    recommend_kakan = None

    best_kan_w = -1.0
    best_kan_action = None
    best_kan_opt = None
    for o in qualified_ankan:
        w = float(o.get("recommendation_weight") or 0.0)
        if w > best_kan_w:
            best_kan_w = w
            best_kan_action = "ankan"
            best_kan_opt = o
    for o in qualified_kakan:
        w = float(o.get("recommendation_weight") or 0.0)
        if w > best_kan_w:
            best_kan_w = w
            best_kan_action = "kakan"
            best_kan_opt = o

    if best_kan_opt is not None and best_kan_w >= best_disc_w:
        recommend_action = best_kan_action or "discard"
        if recommend_action == "ankan":
            recommend_ankan = best_kan_opt
        else:
            recommend_kakan = best_kan_opt

    if recommend_action == "ankan" and recommend_ankan is not None:
        analysis["best"] = None
        analysis["best_ankan"] = recommend_ankan.get("tile")
        analysis["best_ankan_kind"] = recommend_ankan.get("kind")
        analysis["best_kakan"] = None
        analysis["recommend_action"] = "ankan"
    elif recommend_action == "kakan" and recommend_kakan is not None:
        analysis["best"] = None
        analysis["best_kakan"] = recommend_kakan.get("tile")
        analysis["best_ankan"] = None
        analysis["recommend_action"] = "kakan"
    else:
        from .scoring import best_candidate

        bc = best_candidate(cands)
        analysis["best"] = bc.get("tile") if bc else analysis.get("best")
        analysis["best_ankan"] = None
        if not kakan_options:
            analysis["best_kakan"] = None
        elif recommend_action == "discard":
            # 保持/清除加杠推荐
            analysis["best_kakan"] = None
        analysis["recommend_action"] = "discard"

    analysis["ankan_options"] = ankan_options
    analysis["ankan_context"] = {
        "self_tenpai": self_tenpai,
        "opp_riichi": opp_riichi,
        "bias": bias,
    }
    if kakan_options:
        analysis["kakan_options"] = kakan_options

    # match
    actual_ankan = getattr(dp, "actual_ankan", None)
    actual_kakan = getattr(dp, "actual_kakan", None)
    actual_disc = getattr(dp, "actual_discard", None)
    if actual_ankan:
        match = (
            analysis.get("recommend_action") == "ankan"
            and _norm_kind(str(analysis.get("best_ankan") or analysis.get("best_ankan_kind") or ""))
            == _norm_kind(str(actual_ankan))
        )
        analysis["match"] = match
        analysis["match_kind"] = "same" if match else "different"
        analysis["actual"] = actual_ankan
        analysis["actual_action"] = "ankan"
    elif actual_kakan:
        # attach_kakan 已写过；若本函数改写了推荐则重算
        match = (
            analysis.get("recommend_action") == "kakan"
            and _norm_kind(str(analysis.get("best_kakan") or ""))
            == _norm_kind(str(actual_kakan))
        )
        analysis["match"] = match
        analysis["match_kind"] = "same" if match else "different"
        analysis["actual"] = actual_kakan
        analysis["actual_action"] = "kakan"
    elif actual_disc:
        analysis["actual_action"] = "discard"
        if analysis.get("recommend_action") in ("ankan", "kakan"):
            analysis["match"] = False
            analysis["match_kind"] = "different"
        else:
            from .scoring import classify_discard_match

            classified = classify_discard_match(
                cands, actual_disc, analysis.get("best")
            )
            analysis["match"] = classified["match"]
            analysis["match_kind"] = classified["match_kind"]
            analysis["equivalent_best"] = classified.get("equivalent_best")

    return analysis
