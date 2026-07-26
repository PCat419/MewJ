"""Analyze decision points via mahjong-cpp (nanikiru) and build Classic report data."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .converter import build_request, id_to_tile_name, tile_name_to_id  # noqa: E402
from .defense import compute_defense, deal_in_for_tile  # noqa: E402
from .nanikiru_pool import (  # noqa: E402
    DEFAULT_NANIKIRU,
    NanikiruPool,
    default_nanikiru_exe,
    nanikiru_port,
    nanikiru_reachable,
    pick_url,
    resolve_workers,
    restart_worker,
    set_active_pool,
)
from .params import PARAMS as _P  # noqa: E402
from .posture import Posture, advance, apply_posture, evaluate_posture  # noqa: E402
from .replay import CallOpportunity, DecisionPoint, extract_kyoku_views  # noqa: E402
from .scoring import (  # noqa: E402
    ACCEPTABLE_WEIGHT_RATIO,
    DEFAULT_TEMPERATURE,
    EQUIVALENT_UTILITY_EPSILON,
    MIN_RECOMMENDATION_WEIGHT,
    classify_discard_match,
    offensive_desire_from_dp,
    order_candidates,
    score_candidates,
    softmax_weights,
)

_MEWJ_ROOT = Path(__file__).resolve().parent

MAHJONG_CPP_VERSION = "0.9.8"

DEFAULT_NANIKIRU_EXE = default_nanikiru_exe()

# Preferred flag order for client requests. Server also forces shanten_down=off
# when shanten >= 3; this fallback covers stack-overflow on awkward 0-2 shanten hands.
# Final (False, False) is last-resort when even single-flag modes still crash/hang.
_FLAG_ATTEMPTS: Tuple[Tuple[bool, bool], ...] = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)

# Transport failures that usually mean nanikiru died or hung mid-request.
_NANIKIRU_TRANSPORT_ERRORS: Tuple[type, ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _at_turn(arr: Any, turn: int) -> Any:
    """Pick stats for absolute turn ``turn``.

    Arrays are indexed 0..t_max. Index 0 is the post-discard / pre-draw slot
    (usually 0). Index t is the value when the game clock is turn t — same as
    setting 巡目=t in backend/templates/index.html (t_min=1 → idx=t).
    """
    if not arr:
        return None
    if turn is None or turn < 0:
        turn = 1
    idx = min(max(int(turn), 0), len(arr) - 1)
    if idx == 0 and len(arr) > 1:
        idx = 1
    return arr[idx]


def _uke_count(stat: dict) -> int:
    total = 0
    for nt in stat.get("necessary_tiles") or []:
        try:
            total += int(nt.get("count") or 0)
        except (TypeError, ValueError):
            pass
    return total


def _norm_tile_name(tile: Optional[str]) -> str:
    if not tile:
        return ""
    t = str(tile)
    if t.lower().endswith("r"):
        t = t[:-1]
    if t and t[0] == "0":
        return "5" + t[1:]
    return t


def _strip_river_marker(tile: str) -> str:
    """Drop trailing 'r' riichi marker; keep aka (0m/0p/0s)."""
    t = str(tile).strip()
    if t.lower().endswith("r"):
        t = t[:-1]
    return t


def _aka_equiv_key(tile: str) -> str:
    t = _strip_river_marker(tile)
    if t and t[0] == "0":
        return "5" + t[1:]
    return t


def _remove_one_seen(seen: List[str], tile: str) -> bool:
    """Remove one river copy that matches ``tile`` (aka ↔ 5)."""
    key = _aka_equiv_key(tile)
    for i, t in enumerate(seen):
        if _aka_equiv_key(t) == key:
            del seen[i]
            return True
    return False


def _wall_inputs_from_dp(dp: DecisionPoint) -> Tuple[List[str], List[dict]]:
    """Build ``seen`` + ``other_melds`` for nanikiru wall counting.

    Replay keeps called tiles in the discarder's river, while melds also list
    them — drop those river copies so they are not subtracted twice.
    """
    seen: List[str] = []
    for river in (dp.rivers or {}).values():
        for t in river or []:
            name = _strip_river_marker(t)
            if name:
                seen.append(name)

    other_melds: List[dict] = []
    for rel, melds in (dp.melds_by_rel or {}).items():
        for m in melds or []:
            mtype = m.get("type")
            # Open-call tile stays in our simulated river; melds also list it.
            if (
                mtype in ("chii", "pon", "daiminkan", "kakan")
                and m.get("calledTile")
                and m.get("sourceSeat")
            ):
                _remove_one_seen(seen, str(m["calledTile"]))
            if rel == "自家":
                continue
            tiles = m.get("tiles") or []
            if tiles:
                other_melds.append({"type": mtype, "tiles": list(tiles)})
    return seen, other_melds


def _own_discard_set(dp: DecisionPoint) -> set:
    """自家现物：牌河 + 被他家副露走的自家打出牌（与旧前端 computeOwnDiscards 一致）。"""
    out = set()
    for t in (dp.rivers or {}).get("自家") or []:
        key = _norm_tile_name(t)
        if key:
            out.add(key)
    for rel, melds in (dp.melds_by_rel or {}).items():
        if rel == "自家":
            continue
        for m in melds or []:
            if m.get("sourceSeat") == "自家" and m.get("calledTile"):
                key = _norm_tile_name(m.get("calledTile"))
                if key:
                    out.add(key)
    return out


def _filter_furiten_waits(
    necessary: List[dict], own_discards: set
) -> tuple:
    """0向听：保留自摸进张，只标记振听。"""
    out = []
    uke = 0
    hit = False
    for nt in necessary:
        item = dict(nt)
        key = _norm_tile_name(item.get("tile"))
        if key and key in own_discards:
            hit = True
            item["furiten_wait"] = True
        else:
            item["furiten_wait"] = False
        try:
            uke += int(item.get("count") or 0)
        except (TypeError, ValueError):
            pass
        out.append(item)
    return out, uke, hit


def _mark_uke_furiten_risk(
    necessary: List[dict], own_discards: set
) -> tuple:
    """非0向听：牌河同种进张仅作提示，不从自摸进张中扣除。"""
    out = []
    uke = 0
    has_risk = False
    for nt in necessary:
        item = dict(nt)
        key = _norm_tile_name(item.get("tile"))
        try:
            cnt = int(item.get("count") or 0)
        except (TypeError, ValueError):
            cnt = 0
        if key and key in own_discards:
            item["risk"] = True
            has_risk = True
        else:
            item["risk"] = False
        uke += cnt
        out.append(item)
    return out, uke, has_risk


def _cand_uke(c: dict) -> int:
    try:
        return int(c.get("uke") or 0)
    except (TypeError, ValueError):
        return 0


def _delta_ev_min(turn: int, shanten_gap: int) -> float:
    """Minimum EV gain required to accept a higher-shanten (退向) cut."""
    r = _P["review"]["retreat"]
    t = max(1, int(turn or 1))
    base = min(
        float(r["delta_ev_cap"]),
        float(r["delta_ev_base"]) + float(r["delta_ev_per_turn"]) * max(0, t - 5),
    )
    gap = max(1, int(shanten_gap))
    # Deeper retreats need more EV compensation.
    return base * (1.0 + float(r["delta_ev_gap_coef"]) * (gap - 1))


def _tenpai_rate_min(turn: int, shanten_gap: int) -> float:
    """Minimum tenpai-rate for the 退向 candidate."""
    r = _P["review"]["retreat"]
    t = max(1, int(turn or 1))
    base = min(
        float(r["tenpai_rate_cap"]),
        float(r["tenpai_rate_base"])
        + float(r["tenpai_rate_per_turn"]) * max(0, t - 5),
    )
    gap = max(1, int(shanten_gap))
    return min(
        float(r["tenpai_rate_hard_cap"]),
        base + float(r["tenpai_rate_per_gap"]) * (gap - 1),
    )


def _cand_ev(c: dict) -> float:
    v = c.get("exp_score")
    return float(v) if v is not None else -1e18


def _cand_tenpai(c: dict) -> float:
    v = c.get("tenpai_prob")
    return float(v) if v is not None else 0.0


def _cand_winp(c: dict) -> float:
    v = c.get("win_prob")
    return float(v) if v is not None else 0.0


def _cand_shanten(c: dict) -> int:
    s = c.get("shanten")
    return int(s) if s is not None else 99


# 拆听门控：听牌先制/罚符价值不体现在纯 EV 差中。
# 和率不降 → 放行；和率下降 → ΔEV 与 Δ和率按交换率结算；无和率统计 → EV 下限。
_RETREAT = _P["review"]["retreat"]
RETREAT_FROM_TENPAI_EV_MIN = float(_RETREAT["from_tenpai_ev_min"])
RETREAT_FROM_TENPAI_WIN_SCALE = float(_RETREAT["from_tenpai_win_scale"])
RETREAT_FROM_TENPAI_TENPAI_SCALE = float(_RETREAT["from_tenpai_tenpai_scale"])


def _allows_retreat(
    advance: dict,
    candidate: dict,
    turn: int,
) -> bool:
    """Whether ``candidate`` (higher shanten) may beat ``advance`` (最低向听).

    拆听（0→1）：和率不降视为真改良直接放行；和率下降时要求
    ``ΔEV + Δ和率×win_scale + Δ听牌率×tenpai_scale ≥ 0``（役满肥尾抬 EV
    但和率崩盘时过不了闸）。无和率统计时回退 ``ΔEV ≥ from_tenpai_ev_min``。
    其余退向沿用按巡目/深度递进的 EV 与听牌率阈值。
    """
    gap = _cand_shanten(candidate) - _cand_shanten(advance)
    if gap <= 0:
        return True
    delta = _cand_ev(candidate) - _cand_ev(advance)
    if _cand_shanten(advance) == 0:
        win_a = _cand_winp(advance)
        win_c = _cand_winp(candidate)
        if 0.0 < win_a <= win_c:
            return True
        if win_a <= 0.0 and win_c <= 0.0:
            return delta >= RETREAT_FROM_TENPAI_EV_MIN
        net = (
            delta
            + (win_c - win_a) * RETREAT_FROM_TENPAI_WIN_SCALE
            + (_cand_tenpai(candidate) - _cand_tenpai(advance))
            * RETREAT_FROM_TENPAI_TENPAI_SCALE
        )
        return net >= 0.0
    need_ev = _delta_ev_min(turn, gap)
    need_ten = _tenpai_rate_min(turn, gap)
    return delta >= need_ev and _cand_tenpai(candidate) >= need_ten


def pick_recommended(
    candidates: List[dict],
    turn: int,
    *,
    use_ev: bool = True,
) -> Optional[str]:
    """Choose suggested discard among 进取 + allowed 退向 cuts.

    Take the best EV among: minimum-shanten cuts, plus any higher-shanten cut
    that clears the turn/gap 退向 bar against the best 进取 cut.
    """
    if not candidates:
        return None
    if not use_ev:
        return candidates[0].get("tile")

    min_s = min(_cand_shanten(c) for c in candidates)
    advance_pool = [c for c in candidates if _cand_shanten(c) == min_s]
    advance_pool.sort(
        key=lambda c: (
            1 if c.get("furiten") else 0,
            -_cand_ev(c),
            -_cand_winp(c),
            -_cand_tenpai(c),
            -_cand_uke(c),
            c.get("tile") or "",
        )
    )
    a = advance_pool[0]

    allowed = [c for c in advance_pool if not c.get("furiten")] or list(advance_pool)
    for c in candidates:
        if _cand_shanten(c) <= min_s:
            continue
        if _allows_retreat(a, c, turn):
            allowed.append(c)

    allowed.sort(
        key=lambda c: (
            1 if c.get("furiten") else 0,
            -_cand_ev(c),
            -_cand_winp(c),
            -_cand_tenpai(c),
            -_cand_uke(c),
            _cand_shanten(c),
            c.get("tile") or "",
        )
    )
    return allowed[0].get("tile")


def _cand_utility(c: dict) -> float:
    try:
        return float(c.get("adjusted_utility"))
    except (TypeError, ValueError):
        v = c.get("exp_score")
        return float(v) if v is not None else -1e18


def _policy_valid_candidates(candidates: List[dict], turn: int) -> List[dict]:
    """进取（最低向听）候选 + 过闸退向候选。

    与 pick_recommended 同一套门控：拆听须过和率交换率（或无统计时 EV
    下限），其余退向须过巡目/深度阈值。供 _attach_defense 在
    集合内按调整后效用选最优，避免门控被 argmax 架空（原 R1 死代码）。
    """
    if not candidates:
        return []
    min_s = min(_cand_shanten(c) for c in candidates)
    advance_pool = [c for c in candidates if _cand_shanten(c) == min_s]
    advance_pool.sort(
        key=lambda c: (
            1 if c.get("furiten") else 0,
            -_cand_ev(c),
            -_cand_winp(c),
            -_cand_tenpai(c),
            -_cand_uke(c),
            c.get("tile") or "",
        )
    )
    a = advance_pool[0]
    valid = [c for c in advance_pool if not c.get("furiten")] or list(advance_pool)
    for c in candidates:
        if _cand_shanten(c) <= min_s:
            continue
        if _allows_retreat(a, c, turn):
            valid.append(c)
    return valid


def _order_candidates_with_best(
    candidates: List[dict],
    best: Optional[str],
    turn: int = 1,
    *,
    use_ev: bool = True,
) -> List[dict]:
    """Rank list: recommended first, then policy-valid cuts, then the rest.

    Within each group: lower shanten, then higher EV, win rate, tenpai, uke.
    Rejected 退向 cuts sink below same/lower-shanten alternatives.
    """
    if not candidates:
        return candidates

    if not use_ev:
        ordered = list(candidates)
        if best:
            ordered = [c for c in ordered if c.get("tile") == best] + [
                c for c in ordered if c.get("tile") != best
            ]
        return ordered

    min_s = min(_cand_shanten(c) for c in candidates)
    advance_pool = [c for c in candidates if _cand_shanten(c) == min_s]
    advance_pool.sort(
        key=lambda c: (
            -_cand_ev(c),
            -_cand_winp(c),
            -_cand_tenpai(c),
            -_cand_uke(c),
            c.get("tile") or "",
        )
    )
    a = advance_pool[0]

    def is_policy_valid(c: dict) -> bool:
        if _cand_shanten(c) <= min_s:
            return True
        return _allows_retreat(a, c, turn)

    def sort_key(c: dict):
        tile = c.get("tile")
        return (
            0 if best and tile == best else 1,
            0 if is_policy_valid(c) else 1,
            _cand_shanten(c),
            -_cand_ev(c),
            -_cand_winp(c),
            -_cand_tenpai(c),
            -_cand_uke(c),
            tile or "",
        )

    return sorted(candidates, key=sort_key)


def _retreat_exchange_net(advance: dict, candidate: dict) -> float:
    """ΔEV + Δ和率×Sw + Δ听牌率×St（与拆听门控同一交换率）。"""
    return (
        (_cand_ev(candidate) - _cand_ev(advance))
        + (_cand_winp(candidate) - _cand_winp(advance))
        * RETREAT_FROM_TENPAI_WIN_SCALE
        + (_cand_tenpai(candidate) - _cand_tenpai(advance))
        * RETREAT_FROM_TENPAI_TENPAI_SCALE
    )


def _rescore_policy_pool(cands: List[dict]) -> None:
    """显示权重与综合效用：全体进 softmax（和为 100%），判定资格不变。

    对齐副露报告口径（call_eval）：判定用门控选 best；权重/综合效用仅作显示校准。
    - 合格候选：保留 ``adjusted_utility``；
    - 未过闸：相对最优合格切的显示差 d ≤ 0，写回 ``adjusted_utility`` 再 softmax
      （拆听用交换率净额；其它退向用效用差，若已不低于最优则压
      ``from_tenpai_ev_min``），保证推荐项综合效用排第一，且避免 100%/0% 失真。
    牌效 EV（``exp_score``）仍显示引擎原值，不受影响。
    """
    eligible = [c for c in cands if not c.get("policy_rejected")]
    if not eligible:
        return
    best_elig = max(
        eligible,
        key=lambda c: (
            _cand_utility(c),
            _cand_winp(c),
            _cand_tenpai(c),
            _cand_ev(c),
        ),
    )
    best_u = _cand_utility(best_elig)
    display: List[float] = []
    for c in cands:
        if not c.get("policy_rejected"):
            display.append(_cand_utility(c))
            continue
        # 未合格：构造 ≤0 的显示差（相对 best_elig），与副露 skip 基准 0 同构
        if _cand_shanten(best_elig) == 0 and _cand_shanten(c) > 0:
            d = min(0.0, _retreat_exchange_net(best_elig, c))
        else:
            d = min(0.0, _cand_utility(c) - best_u)
            if d >= 0.0:
                d = -RETREAT_FROM_TENPAI_EV_MIN
        display.append(best_u + d)
    weights = softmax_weights(display, DEFAULT_TEMPERATURE)
    for c, u, w in zip(cands, display, weights):
        c["adjusted_utility"] = u
        c["recommendation_weight"] = w


# 单实例回退：无活跃池时仍可启动/重启一个 nanikiru（兼容旧调用）
_NANIKIRU_PROC: Optional[subprocess.Popen] = None
_NANIKIRU_LOG_HANDLE = None
_ATEXIT_REGISTERED = False


def _nanikiru_log_path() -> Path:
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_nanikiru.log"


def _shutdown_owned_nanikiru() -> None:
    """atexit: terminate the nanikiru process started by this process."""
    global _NANIKIRU_PROC, _NANIKIRU_LOG_HANDLE
    proc = _NANIKIRU_PROC
    _NANIKIRU_PROC = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if _NANIKIRU_LOG_HANDLE is not None:
        try:
            _NANIKIRU_LOG_HANDLE.close()
        except Exception:
            pass
        _NANIKIRU_LOG_HANDLE = None


def _nanikiru_child_env(exe_path: Path) -> dict:
    """Child env with MSYS2 runtime DLL dirs prepended to PATH (libspdlog etc.)."""
    env = dict(os.environ)
    candidates = [
        exe_path.parent,
        Path(r"C:\msys64\mingw64\bin"),
        Path(r"C:\msys64\ucrt64\bin"),
    ]
    prepend = [str(p) for p in candidates if p.is_dir()]
    env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    return env


def _legacy_restart_nanikiru(
    nanikiru_url: str = DEFAULT_NANIKIRU,
    exe: Optional[Path] = None,
) -> bool:
    """Single-instance restart (PID of owned proc, else taskkill /IM)."""
    global _NANIKIRU_PROC, _NANIKIRU_LOG_HANDLE, _ATEXIT_REGISTERED
    path = Path(exe) if exe else DEFAULT_NANIKIRU_EXE
    port = nanikiru_port(nanikiru_url)
    if not path.is_file():
        print(f"  [warn] nanikiru.exe not found: {path}", flush=True)
        return False

    if _NANIKIRU_PROC is not None and _NANIKIRU_PROC.poll() is None:
        try:
            _NANIKIRU_PROC.terminate()
            _NANIKIRU_PROC.wait(timeout=3)
        except Exception:
            try:
                _NANIKIRU_PROC.kill()
            except Exception:
                pass
    elif sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/IM", "nanikiru.exe", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        subprocess.run(["pkill", "-f", "nanikiru"], capture_output=True, check=False)

    if _NANIKIRU_LOG_HANDLE is not None:
        try:
            _NANIKIRU_LOG_HANDLE.close()
        except Exception:
            pass
    _NANIKIRU_LOG_HANDLE = open(_nanikiru_log_path(), "a", encoding="utf-8")

    time.sleep(0.4)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    _NANIKIRU_PROC = subprocess.Popen(
        [str(path), str(port)],
        cwd=str(path.parent),
        stdout=_NANIKIRU_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        env=_nanikiru_child_env(path),
        creationflags=creationflags,
    )
    if not _ATEXIT_REGISTERED:
        atexit.register(_shutdown_owned_nanikiru)
        _ATEXIT_REGISTERED = True
    for _ in range(40):
        time.sleep(0.15)
        if nanikiru_reachable(nanikiru_url):
            return True
    return False


def restart_nanikiru(
    nanikiru_url: str = DEFAULT_NANIKIRU,
    exe: Optional[Path] = None,
) -> bool:
    """Restart the worker for ``nanikiru_url`` (pool-aware when active)."""
    return restart_worker(
        nanikiru_url,
        legacy_restart=lambda u: _legacy_restart_nanikiru(u, exe=exe),
    )


def _parse_response(
    dp: DecisionPoint,
    raw: dict,
    *,
    enable_tegawari: bool,
    enable_shanten_down: bool,
) -> Dict[str, Any]:
    if not raw.get("success", True) and raw.get("err_msg"):
        return {"ok": False, "error": raw.get("err_msg"), "raw": raw}

    from .call_eval import apply_shanten_gate, hand_without_tile, local_shanten

    n_meld = len(dp.melds or [])
    shanten_info = apply_shanten_gate(raw.get("shanten") or {}, n_meld)
    calc_stats = bool((raw.get("config") or {}).get("calc_stats", True))
    cfg = raw.get("config") or {}
    server_teg = cfg.get("enable_tegawari", enable_tegawari)
    server_sd = cfg.get("enable_shanten_down", enable_shanten_down)

    tile_map = {str(i): id_to_tile_name(i) for i in range(37)}
    own_discards = _own_discard_set(dp)
    ranked = []
    for st in raw.get("stats") or []:
        tid = st.get("tile")
        if tid is None or tid == -1:
            continue
        name = tile_map.get(str(tid), str(tid))
        tenpai_arr = st.get("tenpai_prob") or []
        win_arr = st.get("win_prob") or []
        exp_arr = st.get("exp_score") or []
        has_probs = bool(tenpai_arr or win_arr or exp_arr)
        tenpai = _at_turn(tenpai_arr, dp.turn) if has_probs else None
        win = _at_turn(win_arr, dp.turn) if has_probs else None
        exp = _at_turn(exp_arr, dp.turn) if has_probs else None
        necessary = []
        for nt in st.get("necessary_tiles") or []:
            nt_id = nt.get("tile")
            necessary.append(
                {
                    "tile": tile_map.get(str(nt_id), str(nt_id)),
                    "tile_id": nt_id,
                    "count": nt.get("count"),
                }
            )
        # 切后向听按本地口径（七对/国士仅 ≤2）；失败时回退引擎值
        shanten = st.get("shanten")
        after = hand_without_tile(dp.hand or [], name)
        if after is not None:
            shanten = local_shanten(after, n_meld)
        is_furiten = False
        has_uke_risk = False
        if own_discards and shanten == 0:
            necessary, uke, is_furiten = _filter_furiten_waits(
                necessary, own_discards
            )
        elif own_discards and shanten is not None and shanten != 0:
            necessary, uke, has_uke_risk = _mark_uke_furiten_risk(
                necessary, own_discards
            )
        else:
            uke = 0
            for nt in necessary:
                try:
                    uke += int(nt.get("count") or 0)
                except (TypeError, ValueError):
                    pass
        ranked.append(
            {
                "tile": name,
                "tile_id": tid,
                "shanten": shanten,
                "tenpai_prob": tenpai,
                # 按巡完整听牌率数组：罚符模型（noten.py）取末段估计流局时听牌率
                "tenpai_prob_arr": tenpai_arr if has_probs else None,
                "win_prob": win,
                "exp_score": exp,
                "uke": uke,
                "necessary_tiles": necessary[:14],
                "furiten": is_furiten,
                "uke_risk": has_uke_risk,
            }
        )

    if calc_stats and any(c.get("exp_score") is not None for c in ranked):
        ranked.sort(
            key=lambda x: (
                -(x["exp_score"] if x["exp_score"] is not None else -1e18),
                -(x["tenpai_prob"] if x["tenpai_prob"] is not None else -1),
                x["shanten"] if x["shanten"] is not None else 99,
            )
        )
        use_ev = True
    else:
        ranked.sort(
            key=lambda x: (
                x["shanten"] if x["shanten"] is not None else 99,
                -(x["uke"] or 0),
            )
        )
        use_ev = False

    best = pick_recommended(ranked, dp.turn, use_ev=use_ev)
    ranked = _order_candidates_with_best(
        ranked, best, dp.turn, use_ev=use_ev
    )
    return {
        "ok": True,
        "shanten": shanten_info,
        "calc_stats": calc_stats,
        "stat_turn": dp.turn,
        "enable_tegawari": bool(server_teg),
        "enable_shanten_down": bool(server_sd),
        "best": best,
        "actual": dp.actual_discard,
        "match": best == dp.actual_discard if best else None,
        "candidates": ranked,
        "hand": dp.hand,
        "melds": dp.melds,
    }


def _attach_defense(dp: DecisionPoint, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Attach relative danger, adjusted utility and normalized weights."""
    try:
        defense = compute_defense(dp)
        signals_by_seat = {
            seat: list(model.get("signals") or [])
            for seat, model in (defense.get("per_seat") or {}).items()
            if model.get("signals")
        }
        analysis["defense"] = {
            "has_threat": defense["has_threat"],
            "threats": defense["threats"],
            "alphas": defense["alphas"],
            "metric": defense["metric"],
            "calibrated": defense["calibrated"],
            "signals_by_seat": signals_by_seat,
        }
        for c in analysis.get("candidates") or []:
            row = deal_in_for_tile(defense, c.get("tile") or "")
            c["deal_in"] = row
            c["relative_danger"] = row
            # Compatibility alias. The value is a relative index, not a
            # calibrated probability; new code should use relative_danger.
            c["deal_in_prob"] = row.get("combined")
            tile = _norm_tile_name(c.get("tile") or "")
            reasons = []
            for seat, model in (defense.get("per_seat") or {}).items():
                factor = (model.get("modifiers") or {}).get(tile)
                if factor is None:
                    continue
                matched = [
                    signal
                    for signal in (model.get("signals") or [])
                    if tile in (signal.get("tiles") or [])
                ]
                reasons.append(
                    {
                        "seat": seat,
                        "factor": factor,
                        "direction": "up" if factor > 1.0 else "down",
                        "rules": [
                            {
                                "id": signal.get("id"),
                                "label": signal.get("label"),
                                "factor": signal.get("factor"),
                            }
                            for signal in matched
                        ],
                    }
                )
            c["danger_reasons"] = reasons
    except Exception as exc:
        analysis["defense_error"] = str(exc)
        defense = {
            "has_threat": False,
            "threats": [],
            "alphas": {},
            "per_seat": {},
            "combined": {},
        }

    cands = analysis.get("candidates") or []
    if analysis.get("ok") and cands:
        desire_info = offensive_desire_from_dp(dp)
        score_candidates(
            cands,
            defense,
            dp,
            offensive_desire=desire_info["offensive_desire"],
            # 罚符模型上下文：威胁列表 + 当前巡目（stat_turn 在 analysis 里）。
            # 用局部 defense（异常回退时 threats=[]），不读 analysis["defense"]
            noten_ctx={
                "threats": defense.get("threats") or [],
                "turn": analysis.get("stat_turn"),
            },
        )
        # 门控生效（修复 R1 死代码）：best 只在 policy-valid 集合内按调整后
        # 效用 argmax；被门控拒绝的退向候选标记 policy_rejected 并沉底展示。
        valid = _policy_valid_candidates(cands, dp.turn)
        valid_ids = {id(c) for c in valid}
        for c in cands:
            c["policy_rejected"] = id(c) not in valid_ids
        picked = max(
            valid,
            key=lambda c: (
                _cand_utility(c),
                _cand_winp(c),
                _cand_tenpai(c),
                _cand_ev(c),
            ),
        )
        best = picked.get("tile") if picked else None
        classified = classify_discard_match(cands, dp.actual_discard, best)
        # 一致/尚可分类完成后重算权重池：被门控/振听过滤的候选权重压至 ≈0，
        # 保证任何卡片中 推荐项 == 权重最高候选（统计口径不变）
        _rescore_policy_pool(cands)
        analysis["best"] = best
        analysis["equivalent_best"] = classified["equivalent_best"]
        analysis["match"] = classified["match"]
        analysis["match_kind"] = classified["match_kind"]
        ordered = order_candidates(cands)
        ordered.sort(key=lambda c: 1 if c.get("policy_rejected") else 0)
        analysis["candidates"] = ordered
        analysis["recommendation_model"] = {
            "name": "risk-adjusted-softmax",
            "danger_metric": "relative_danger",
            "calibrated": False,
            "temperature": DEFAULT_TEMPERATURE,
            "minimum_weight": MIN_RECOMMENDATION_WEIGHT,
            "equivalent_utility_epsilon": EQUIVALENT_UTILITY_EPSILON,
            "acceptable_weight_ratio": ACCEPTABLE_WEIGHT_RATIO,
            # Hidden score-situation knob; not rendered in the Classic report UI.
            "offensive_desire": desire_info,
        }
    return analysis


# 振听补偿：nanikiru 的请求不含牌河信息，引擎无法感知振听，
# 会把自家已舍牌种的剩余枚数照常计入待牌/进张价值（如切过 8m 仍高估 6m7m 搭子）。
# 修正：将自家牌河出现过的牌种剩余枚数一律置 0（含对应红五槽）。
#   - 只处理自家（rel=="自家"）的舍牌；他家舍牌与自家振听无关，不动
#   - 舍牌振听整局永久有效，故每个决策点按当时牌河全量置 0
#   - 已知近似：该牌种的自摸分支一并被抹除（wall 整数计数下的最近似）
_FIVE_RED_PAIR = {4: 34, 13: 35, 22: 36, 34: 4, 35: 13, 36: 22}  # 5m/5p/5s <-> 红五槽


def _apply_furiten_wall_zero(req: Dict[str, Any], dp: DecisionPoint) -> None:
    wall = req.get("wall")
    if not isinstance(wall, list):
        return
    n = len(wall)
    for entry in (dp.discards_log or []):
        if entry.get("rel") != "自家":
            continue
        try:
            tid = tile_name_to_id(entry.get("tile") or "")
        except ValueError:
            continue
        for t in (tid, _FIVE_RED_PAIR.get(tid)):
            if t is not None and 0 <= t < n:
                wall[t] = 0


def analyze_decision(
    dp: DecisionPoint,
    *,
    nanikiru_url: str = DEFAULT_NANIKIRU,
    timeout: float = 30.0,
    enable_tegawari: bool = True,
    enable_shanten_down: bool = True,
) -> Dict[str, Any]:
    """Call nanikiru and return efficiency, relative danger and recommendation."""
    seen, other_melds = _wall_inputs_from_dp(dp)
    req = build_request(
        game_mode=1,
        round_wind=dp.round_wind,
        seat_wind=dp.seat_wind,
        dora_indicators=dp.dora_indicators,
        hand=dp.hand,
        melds=[{"type": m["type"], "tiles": m["tiles"]} for m in dp.melds],
        other_melds=other_melds,
        seen=seen,
        t_min=1,
        t_max=18,
        version=MAHJONG_CPP_VERSION,
        enable_tegawari=enable_tegawari,
        enable_shanten_down=enable_shanten_down,
    )
    _apply_furiten_wall_zero(req, dp)
    resp = requests.post(nanikiru_url, json=req, timeout=timeout)
    resp.raise_for_status()
    analysis = _parse_response(
        dp,
        resp.json(),
        enable_tegawari=enable_tegawari,
        enable_shanten_down=enable_shanten_down,
    )
    if analysis.get("ok"):
        _attach_defense(dp, analysis)
    return analysis



def analyze_decision_resilient(
    dp: DecisionPoint,
    *,
    nanikiru_url: str = DEFAULT_NANIKIRU,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Analyze with old-project defaults; on nanikiru crash/hang, restart and fall back.

    Some hands stack-overflow or hang when tegawari / shanten_down are on.
    Keep 手替 whenever possible (disable 向听回退 first); last resort turns both off.
    Pins one pool worker for the whole retry chain so restart only hits that instance.
    """
    url = pick_url(nanikiru_url)
    last_exc: Optional[BaseException] = None
    for i, (teg, sd) in enumerate(_FLAG_ATTEMPTS):
        try:
            return analyze_decision(
                dp,
                nanikiru_url=url,
                timeout=timeout,
                enable_tegawari=teg,
                enable_shanten_down=sd,
            )
        except _NANIKIRU_TRANSPORT_ERRORS as exc:
            last_exc = exc
            kind = "超时" if isinstance(exc, requests.exceptions.Timeout) else "断开"
            print(
                f"    nanikiru {kind}（手替={teg} 向听回退={sd}），尝试重启…",
                flush=True,
            )
            if i + 1 >= len(_FLAG_ATTEMPTS):
                break
            if not restart_nanikiru(url):
                raise RuntimeError(
                    "nanikiru 崩溃后无法重启。请检查 nanikiru.exe 路径"
                    "（环境变量 MEWJ_NANIKIRU_EXE 或 MewJ/engine/nanikiru.exe）"
                ) from exc
    assert last_exc is not None
    raise RuntimeError(
        "nanikiru 在多种参数下均崩溃/断开，无法分析该手。"
    ) from last_exc


def review_paipu(
    paipu: dict,
    seat: int,
    *,
    kyoku_indices: Optional[List[int]] = None,
    max_turns: Optional[int] = None,
    nanikiru_url: str = DEFAULT_NANIKIRU,
    skip_analyze: bool = False,
    workers: Optional[int] = None,
) -> dict:
    """Build Classic-style review payload for one seat.

    When analyzing, starts a nanikiru worker pool (``workers`` / MEWJ_WORKERS /
    params.runtime.workers). Discard points in each kyoku are analyzed in
    parallel; posture is still applied in chronological order. Call points
    keep sequential posture context but fan out their internal engine queries.
    """
    from . import call_eval  # 延迟 import：call_eval 依赖本模块，避免循环

    views = extract_kyoku_views(paipu, seat, include_calls=True)
    if kyoku_indices is not None:
        selected = set(kyoku_indices)
        views = [v for v in views if v.index in selected]

    total = sum(len(v.decisions) for v in views)
    if max_turns is not None:
        total = min(total, max_turns)

    n_workers = resolve_workers(workers)
    pool: Optional[NanikiruPool] = None
    if not skip_analyze:
        pool = NanikiruPool(base_url=nanikiru_url, workers=n_workers)
        try:
            pool.start()
        except RuntimeError:
            # Fall back to single-instance legacy start on base URL
            if not nanikiru_reachable(nanikiru_url):
                print("nanikiru 未就绪，尝试启动…", flush=True)
                if not _legacy_restart_nanikiru(nanikiru_url):
                    raise
            pool = NanikiruPool(base_url=nanikiru_url, workers=1)
            pool.start()
        set_active_pool(pool)
        print(
            f"共 {total} 个决策点，开始调用 nanikiru"
            f"（workers={len(pool.urls())}，默认手替开）…",
            flush=True,
        )

    try:
        return _review_paipu_body(
            views,
            paipu,
            seat,
            max_turns=max_turns,
            total=total,
            nanikiru_url=nanikiru_url,
            skip_analyze=skip_analyze,
            n_workers=n_workers,
            call_eval=call_eval,
        )
    finally:
        set_active_pool(None)
        if pool is not None:
            # Keep adopted (pre-existing) listeners; only shut down owned children.
            pool.shutdown()


def _review_paipu_body(
    views,
    paipu: dict,
    seat: int,
    *,
    max_turns: Optional[int],
    total: int,
    nanikiru_url: str,
    skip_analyze: bool,
    n_workers: int,
    call_eval: Any,
) -> dict:
    report_kyokus = []
    analyzed = 0
    progress_lock = threading.Lock()

    def bump(label: str) -> int:
        nonlocal analyzed
        with progress_lock:
            analyzed += 1
            n = analyzed
        print(f"  [{n}/{total}] {label} …", flush=True)
        return n

    for view in views:
        # Plan which decisions will be analyzed (max_turns is chronological).
        planned: List[Tuple[Any, bool]] = []
        for dp in view.decisions:
            if max_turns is not None and analyzed + sum(
                1 for _, a in planned if a
            ) >= max_turns:
                planned.append((dp, False))
            elif skip_analyze:
                planned.append((dp, False))
            else:
                planned.append((dp, True))

        # Prefetch discard analyses in parallel (posture applied later in order).
        discard_jobs: Dict[int, Any] = {}
        if not skip_analyze and n_workers >= 1:
            discards_to_run = [
                (i, dp)
                for i, (dp, do) in enumerate(planned)
                if do and not isinstance(dp, CallOpportunity)
            ]
            if discards_to_run:
                with ThreadPoolExecutor(
                    max_workers=max(1, min(n_workers, len(discards_to_run)))
                ) as ex:
                    fut_map = {
                        ex.submit(
                            analyze_decision_resilient, dp, nanikiru_url=nanikiru_url
                        ): i
                        for i, dp in discards_to_run
                    }
                    for fut in as_completed(fut_map):
                        i = fut_map[fut]
                        dp = planned[i][0]
                        bump(dp.label)
                        try:
                            discard_jobs[i] = fut.result()
                        except Exception as exc:
                            discard_jobs[i] = {"ok": False, "error": str(exc)}

        decisions_out = []
        # 攻防姿态：每局从全牌效开始，局内只能 全牌效→兜牌→全弃 单调前进
        posture = Posture.FULL_EFFICIENCY
        for i, (dp, do_analyze) in enumerate(planned):
            if isinstance(dp, CallOpportunity):
                entry = {
                    "kind": "call",
                    "label": dp.label,
                    "turn": dp.turn,
                    "hand": dp.hand,
                    "melds": dp.melds,
                    "disc_tile": dp.disc_tile,
                    "discarder_rel": dp.discarder_rel,
                    "actual": dp.actual,
                    "actual_tiles": dp.actual_tiles,
                    "actual_cut": dp.actual_cut,
                    "legal_summary": {
                        "pon": len((dp.legal or {}).get("pon") or []),
                        "chii": len((dp.legal or {}).get("chii") or []),
                        "daiminkan": bool((dp.legal or {}).get("daiminkan")),
                    },
                    "analysis": None,
                    "skipped": False,
                }
                if not do_analyze:
                    entry["skipped"] = True
                else:
                    bump(dp.label)
                    try:
                        call_threats = None
                        call_risk_lookup = None
                        call_defense = None
                        try:
                            call_defense = compute_defense(dp)
                            call_threats = call_defense.get("threats") or []

                            def call_risk_lookup(tile, _d=call_defense):
                                v = deal_in_for_tile(_d, tile or "").get("combined")
                                return float(v) if v is not None else 0.0

                        except Exception:
                            call_threats, call_risk_lookup, call_defense = (
                                None,
                                None,
                                None,
                            )
                        ev = call_eval.evaluate_opportunity(
                            dp,
                            nanikiru_url,
                            posture=posture,
                            threats=call_threats,
                            defense=call_defense,
                            workers=n_workers,
                        )
                        decision = call_eval.decide(ev, risk_lookup=call_risk_lookup)
                        match = (
                            call_eval.match_actual(dp.actual, decision)
                            if ev.get("ok")
                            else None
                        )
                        entry["analysis"] = {
                            **ev,
                            **decision,
                            "match": match,
                            "match_kind": (
                                "same"
                                if match is True
                                else "different"
                                if match is False
                                else None
                            ),
                        }
                    except Exception as exc:
                        entry["analysis"] = {"ok": False, "error": str(exc)}
                decisions_out.append(entry)
                continue

            if not do_analyze:
                decisions_out.append(
                    {
                        "kind": "discard",
                        "label": dp.label,
                        "turn": dp.turn,
                        "hand": dp.hand,
                        "drawn_tile": dp.drawn_tile,
                        "melds": dp.melds,
                        "actual": dp.actual_discard,
                        "is_riichi": dp.is_riichi_discard,
                        "is_tsumogiri": dp.is_tsumogiri,
                        "analysis": None,
                        "skipped": True,
                    }
                )
                continue

            entry = {
                "kind": "discard",
                "label": dp.label,
                "turn": dp.turn,
                "hand": dp.hand,
                "drawn_tile": dp.drawn_tile,
                "melds": dp.melds,
                "dora": dp.dora_indicators,
                "round_wind": dp.round_wind,
                "seat_wind": dp.seat_wind,
                "actual": dp.actual_discard,
                "is_riichi": dp.is_riichi_discard,
                "is_tsumogiri": dp.is_tsumogiri,
                "analysis": None,
                "skipped": False,
            }
            try:
                if i in discard_jobs:
                    ana = discard_jobs[i]
                else:
                    bump(dp.label)
                    ana = analyze_decision_resilient(dp, nanikiru_url=nanikiru_url)
                entry["analysis"] = ana
                if ana.get("ok"):
                    posture = advance(posture, evaluate_posture(ana))
                    posture = apply_posture(ana, posture)
            except Exception as exc:
                entry["analysis"] = {"ok": False, "error": str(exc)}
            decisions_out.append(entry)

        report_kyokus.append(
            {
                "index": view.index,
                "label": view.label,
                "scores": view.scores,
                "dora": view.dora_indicators,
                "seat_wind": view.seat_wind,
                "result": view.result,
                "decisions": decisions_out,
            }
        )

    return {
        "title": paipu.get("title"),
        "ref": paipu.get("ref"),
        "names": paipu.get("name"),
        "seat": seat,
        "player": (paipu.get("name") or [None] * 4)[seat],
        "kyokus": report_kyokus,
    }
