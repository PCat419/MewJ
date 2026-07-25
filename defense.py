"""Per-opponent relative danger from full river timing.

Threats: riichi and/or open-meld opponents only (no damaten).
Genbutsu cutoff: riichi declaration seq, or last discard after first open meld.
Riichi: visible tiles + wait shapes + suji priors.
Furo: danger rises with the number of open melds; four melds force tanki.
Combined: weighted soft-OR (α_riichi=1, α_furo depends on meld count).

Values are normalized relative-danger indices, not calibrated deal-in rates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .riichi import compute_wait_risk

from .defense_heuristics import analyze_danger_signals
from .params import PARAMS
from .replay import DecisionPoint

FURO_ALPHAS = PARAMS["defense"]["furo_alphas"]
THREAT_SEATS = ("下家", "对家", "上家")
_OPEN_TYPES = frozenset({"chii", "pon", "daiminkan", "kakan"})
ALL_TILES: List[str] = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
)


def normalize_tile(tile: str) -> str:
    t = str(tile)
    if t.lower().endswith("r"):
        t = t[:-1]
    if t and t[0] == "0":
        return "5" + t[1:]
    return t


def _open_melds(melds: List[dict]) -> List[dict]:
    return [m for m in (melds or []) if m.get("type") in _OPEN_TYPES]


def _meld_tiles_hand_side(m: dict) -> List[str]:
    """Tiles contributed from the caller's hand (exclude one called tile)."""
    tiles = [normalize_tile(t) for t in (m.get("tiles") or [])]
    called = m.get("calledTile")
    if not called or m.get("type") == "ankan":
        return tiles
    want = normalize_tile(str(called))
    for i, t in enumerate(tiles):
        if t == want:
            del tiles[i]
            break
    return tiles


def wall_remaining(dp: DecisionPoint) -> Dict[str, int]:
    counts = {t: 0 for t in ALL_TILES}

    def add(tile: str, n: int = 1) -> None:
        key = normalize_tile(tile)
        if key in counts:
            counts[key] += n

    for t in dp.hand or []:
        add(t)
    for t in dp.dora_indicators or []:
        add(t)
    for river in (dp.rivers or {}).values():
        for t in river or []:
            add(t)
    for melds in (dp.melds_by_rel or {}).values():
        for m in melds or []:
            for t in _meld_tiles_hand_side(m):
                add(t)
    return {t: max(0, 4 - counts[t]) for t in ALL_TILES}


def _genbutsu(log: List[dict], seat: str, cutoff: int) -> Set[str]:
    safe: Set[str] = set()
    for e in log:
        tile = normalize_tile(e.get("tile") or "")
        if not tile:
            continue
        if e.get("rel") == seat:
            safe.add(tile)
        elif int(e.get("seq") or 0) > cutoff:
            safe.add(tile)
    return safe


def _uniform_risk(genbutsu: Set[str], remaining: Dict[str, int]) -> Dict[str, float]:
    non_g = [t for t in ALL_TILES if t not in genbutsu]
    probs = {t: 0.0 for t in ALL_TILES}
    if not non_g:
        return probs
    # The discarded copy itself may be the last visible copy and can still
    # complete a sequence wait, so zero unseen copies must not imply safety.
    weights = {t: 1 + max(0, remaining.get(t, 0)) for t in non_g}
    total = sum(weights.values())
    if total <= 0:
        return probs
    for t in non_g:
        probs[t] = weights[t] / total
    return probs


def _tanki_risk(genbutsu: Set[str], remaining: Dict[str, int]) -> Dict[str, float]:
    """Four open melds leave one concealed tile, so the wait is forced tanki."""
    weights = {
        t: (0 if t in genbutsu else max(0, remaining.get(t, 0)))
        for t in ALL_TILES
    }
    total = sum(weights.values())
    if total <= 0:
        return {t: 0.0 for t in ALL_TILES}
    return {t: weights[t] / total for t in ALL_TILES}


def _apply_tile_factors(
    probs: Dict[str, float],
    factors: Dict[str, float],
    genbutsu: Set[str],
) -> Dict[str, float]:
    """Apply bounded heuristic factors while preserving hard-safe tiles."""
    weighted = {
        tile: (
            0.0
            if tile in genbutsu
            else max(0.0, float(probs.get(tile) or 0.0))
            * max(0.0, float(factors.get(tile, 1.0)))
        )
        for tile in ALL_TILES
    }
    total = sum(weighted.values())
    if total <= 0:
        return {tile: 0.0 for tile in ALL_TILES}
    return {tile: weighted[tile] / total for tile in ALL_TILES}


def _riichi_risk(
    genbutsu: Set[str],
    remaining: Dict[str, int],
    seat: str,
    log: List[dict],
) -> Dict[str, Any]:
    discards = [
        normalize_tile(e.get("tile") or "")
        for e in log
        if e.get("rel") == seat and normalize_tile(e.get("tile") or "")
    ]
    return compute_wait_risk(
        remaining,
        list(genbutsu),
        {seat: discards},
    )


def _collect_threats(dp: DecisionPoint) -> List[Dict[str, Any]]:
    log = dp.discards_log or []
    melds_by = dp.melds_by_rel or {}
    open_after = dp.open_after_seq or {}
    threats: List[Dict[str, Any]] = []

    for rel in THREAT_SEATS:
        events = [e for e in log if e.get("rel") == rel]
        riichi_ev = next((e for e in events if e.get("riichi")), None)
        opens = _open_melds(melds_by.get(rel) or [])

        if riichi_ev:
            threats.append(
                {
                    "seat": rel,
                    "kind": "riichi",
                    "cutoff": int(riichi_ev["seq"]),
                }
            )
            continue

        if not opens:
            continue

        after = open_after.get(rel)
        if after is None:
            after = 0
        post = [e for e in events if int(e.get("seq") or 0) > int(after)]
        if not post:
            # 副露后尚无切牌：极少见；用副露完成点作截止
            cutoff = int(after)
        else:
            cutoff = int(post[-1]["seq"])
        threats.append(
            {
                "seat": rel,
                "kind": "furo",
                "cutoff": cutoff,
                "furo_count": min(4, len(opens)),
            }
        )

    return threats


def _alphas(threats: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        t["seat"]: (
            FURO_ALPHAS.get(int(t.get("furo_count") or 1), FURO_ALPHAS[1])
            if t["kind"] == "furo"
            else 1.0
        )
        for t in threats
    }


def compute_defense(dp: DecisionPoint) -> Dict[str, Any]:
    """Return per-seat / combined relative danger for all 34 tiles."""
    threats = _collect_threats(dp)
    remaining = wall_remaining(dp)
    log = dp.discards_log or []

    per_seat: Dict[str, Dict[str, Any]] = {}
    for th in threats:
        gb = _genbutsu(log, th["seat"], th["cutoff"])
        if th["kind"] == "riichi":
            modeled = _riichi_risk(gb, remaining, th["seat"], log)
            probs = modeled["probs"]
            details = modeled["details"]
            model = "wait-shape"
            confidence = "medium"
        else:
            furo_count = int(th.get("furo_count") or 1)
            probs = (
                _tanki_risk(gb, remaining)
                if furo_count >= 4
                else _uniform_risk(gb, remaining)
            )
            details = {}
            model = "forced-tanki" if furo_count >= 4 else "visible-uniform"
            confidence = "high" if furo_count >= 4 else (
                "medium" if furo_count == 3 else "low"
            )
        base_probs = dict(probs)
        heuristic = analyze_danger_signals(dp, th["seat"])
        probs = _apply_tile_factors(
            base_probs,
            heuristic["tile_factors"],
            gb,
        )
        per_seat[th["seat"]] = {
            "kind": th["kind"],
            "cutoff": th["cutoff"],
            "base_probs": base_probs,
            "probs": probs,
            "relative_risk": probs,
            "details": details,
            "model": model,
            "confidence": confidence,
            "genbutsu_count": len(gb),
            "modifiers": heuristic["modifiers"],
            "signals": heuristic["signals"],
            "modifier_bounds": heuristic["bounds"],
        }

    alphas = _alphas(threats)
    combined: Dict[str, Optional[float]] = {}
    if not threats:
        for t in ALL_TILES:
            combined[t] = None
    else:
        for t in ALL_TILES:
            prod = 1.0
            for th in threats:
                p = float(per_seat[th["seat"]]["probs"].get(t) or 0.0)
                a = alphas[th["seat"]]
                prod *= 1.0 - min(1.0, a * p)
            combined[t] = 1.0 - prod

    return {
        "has_threat": bool(threats),
        "threats": threats,
        "alphas": alphas,
        "per_seat": per_seat,
        "combined": combined,
        "relative_risk": combined,
        "remaining": remaining,
        "metric": "relative_danger",
        "calibrated": False,
    }


def deal_in_for_tile(defense: Dict[str, Any], tile: str) -> Dict[str, Optional[float]]:
    """Compatibility row for one tile; values are relative danger indices."""
    key = normalize_tile(tile)
    if not defense.get("has_threat"):
        return {
            "combined": None,
            "上家": None,
            "对家": None,
            "下家": None,
        }
    per = defense.get("per_seat") or {}
    row: Dict[str, Optional[float]] = {
        "combined": (defense.get("combined") or {}).get(key),
    }
    for seat in THREAT_SEATS:
        if seat in per:
            row[seat] = float((per[seat].get("probs") or {}).get(key) or 0.0)
        else:
            row[seat] = None
    return row
