"""Render a Mortal-Classic-style static HTML review report."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Optional

_MEWJ_ROOT = Path(__file__).resolve().parent
_TILE_DIR = _MEWJ_ROOT / "assets" / "tiles"


_WIND_ZH = {
    "east": "东",
    "south": "南",
    "west": "西",
    "north": "北",
}


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _fmt_weight(value) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        percent = v * 100
        if percent <= 0:
            return "0%"
        if 0 < percent < 0.01:
            return "<0.01%"
        if percent < 0.1:
            return f"{percent:.2f}%"
        # 上端与下端对称：余量不足 0.01% 时才标 >99.99%（与 <0.01% 对应），
        # 次高端保留两位小数，保证同表各权重显示值相加为 100%
        if 99.99 < percent < 100:
            return ">99.99%"
        if percent > 99.9:
            return f"{percent:.2f}%"
        return f"{percent:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _fmt_score(s) -> str:
    if s is None:
        return "-"
    try:
        return f"{float(s):.0f}"
    except (TypeError, ValueError):
        return str(s)


def _fmt_deal_in(p) -> str:
    if p is None:
        return "—"
    try:
        return f"{float(p) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(p)


def _deal_in_cell(p) -> str:
    if p is None:
        return "<td class='na'>—</td>"
    try:
        v = float(p)
    except (TypeError, ValueError):
        return f"<td>{_esc(p)}</td>"
    safe = " safe" if v <= 0 else ""
    return f"<td class='deal-in{safe}'>{_fmt_deal_in(v)}</td>"


def _tile_src(name: Optional[str], tile_base: str) -> Optional[str]:
    if not name:
        return None
    key = str(name).strip().lower()
    path = _TILE_DIR / f"{key}.png"
    if not path.is_file():
        return None
    base = tile_base.rstrip("/")
    return f"{base}/{key}.png"


def _tile_img(name: Optional[str], tile_base: str, extra_class: str = "") -> str:
    """Render one tile as <img>, falling back to text if asset missing."""
    if not name:
        return ""
    src = _tile_src(name, tile_base)
    cls = f"tile-img {extra_class}".strip()
    if src:
        return (
            f'<img class="{cls}" src="{_esc(src)}" alt="{_esc(name)}" '
            f'title="{_esc(name)}" width="36" height="50"/>'
        )
    return f'<span class="tile-fallback">{_esc(name)}</span>'


def _mpsz_sort_key(tile: str):
    """Sort key: man → pin → sou → honor; aka 5 before plain 5."""
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


def _sort_hand_mpsz(tiles) -> list:
    return sorted(tiles or [], key=_mpsz_sort_key)


def _hand_span(tiles, tile_base: str, drawn_tile: Optional[str] = None) -> str:
    """Render hand; left 13 in mpsz order, drawn tile separated on the right."""
    tiles = list(tiles or [])
    if drawn_tile and drawn_tile in tiles:
        rest = list(tiles)
        rest.remove(drawn_tile)
        rest = _sort_hand_mpsz(rest)
        closed = "".join(_tile_img(t, tile_base) for t in rest)
        drawn = _tile_img(drawn_tile, tile_base)
        return f'{closed}<span class="draw-gap"></span>{drawn}'
    return "".join(_tile_img(t, tile_base) for t in _sort_hand_mpsz(tiles))


_MELD_TYPE_ZH = {
    "chii": "吃",
    "pon": "碰",
    "daiminkan": "大明杠",
    "ankan": "暗杠",
    "kakan": "加杠",
}


def _melds_html(melds, tile_base: str) -> str:
    if not melds:
        return ""
    parts = []
    for m in melds:
        mtype = m.get("type") or ""
        label = _esc(_MELD_TYPE_ZH.get(mtype, mtype))
        imgs = "".join(_tile_img(t, tile_base) for t in (m.get("tiles") or []))
        parts.append(f'<span class="meld"><span class="meld-type">{label}</span>{imgs}</span>')
    return '<span class="melds">' + "".join(parts) + "</span>"


def _defense_panel_html(analysis: dict, tile_base: str) -> str:
    if analysis.get("defense_error"):
        return f"<div class='err'>危险度分析失败：{_esc(analysis.get('defense_error'))}</div>"
    defense = analysis.get("defense") or {}
    if not defense.get("has_threat"):
        return "<p class='muted'>当前无立直/副露威胁，危险度不计算</p>"

    kind_label = {"riichi": "立直", "furo": "副露"}
    badges = []
    for th in defense.get("threats") or []:
        seat = _esc(th.get("seat"))
        kind = kind_label.get(th.get("kind"), th.get("kind"))
        if th.get("kind") == "furo":
            kind = f"{kind}{_esc(th.get('furo_count') or 1)}"
        badges.append(f"<span class='threat-badge'>{seat}·{kind}</span>")
    badge_html = '<div class="threat-list">' + "".join(badges) + "</div>"

    # Unique hand candidates; prefer analysis.candidates order, else fall back
    rows_src = list(analysis.get("candidates") or [])
    if not rows_src:
        return badge_html + "<p class='muted'>无候选切牌</p>"

    def sort_key(c):
        di = c.get("deal_in") or {}
        comb = di.get("combined")
        return (comb is None, comb if comb is not None else 1.0)

    rows_src = sorted(rows_src, key=sort_key)
    body = []
    for c in rows_src:
        di = c.get("deal_in") or {}
        body.append(
            "<tr>"
            f"<td>{_tile_img(c.get('tile'), tile_base)}</td>"
            f"{_deal_in_cell(di.get('combined'))}"
            f"{_deal_in_cell(di.get('上家'))}"
            f"{_deal_in_cell(di.get('对家'))}"
            f"{_deal_in_cell(di.get('下家'))}"
            "</tr>"
        )
    head = (
        "<th>切</th><th>综合危险度</th><th>上家危险度</th>"
        "<th>对家危险度</th><th>下家危险度</th>"
    )
    return f"""
    {badge_html}
    <div class="table-scroll"><table class="riichi">
      <thead><tr>{head}</tr></thead>
      <tbody>{"".join(body)}</tbody>
    </table></div>"""


def _uke_cell(c: dict, tile_base: str) -> str:
    """进张 count + tiles with corner-count badges."""
    uke = c.get("uke")
    n = "—" if uke is None else _esc(uke)
    furiten = (
        '<span class="furiten-badge">振听</span>' if c.get("furiten") else ""
    )
    tiles = c.get("necessary_tiles") or []
    if not tiles:
        return (
            f"<td class='uke'><span class='uke-n'>{n}</span>{furiten}</td>"
        )
    parts = []
    for nt in tiles:
        t = nt.get("tile")
        cnt = nt.get("count")
        img = _tile_img(t, tile_base)
        if not img:
            continue
        risk = " risk" if nt.get("risk") else ""
        badge = (
            f'<span class="uke-badge{risk}">{_esc(cnt)}</span>'
            if cnt is not None
            else ""
        )
        parts.append(f'<span class="uke-tile">{img}{badge}</span>')
    strip = "".join(parts)
    return (
        f"<td class='uke'><span class='uke-n'>{n}</span>{furiten}"
        f"<div class='uke-tiles'>{strip}</div></td>"
    )


_ACTION_ZH = {"pon": "碰", "chii": "吃", "daiminkan": "大明杠", "skip": "跳过"}


def _fmt_win(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _call_turn_html(d: dict, ky: dict, tile_base: str, seq: int):
    """副露决策点卡片：碰/吃 vs 跳过的反事实对比表。"""
    analysis = d.get("analysis") or {}
    actual = d.get("actual")
    match = analysis.get("match")
    if match is True:
        badge = '<span class="badge ok">✓ 一致</span>'
        status = "s-ok"
    elif match is False:
        badge = '<span class="badge diff">✗ 不一致</span>'
        status = "s-diff"
    elif analysis.get("error") or (analysis and analysis.get("ok") is False):
        badge = (
            f'<span class="badge err">'
            f'{_esc(analysis.get("error") or "评估失败")}</span>'
        )
        status = "s-err"
    else:
        badge = '<span class="badge muted">未分析</span>'
        status = "s-skip"

    # 推荐 chip：合格者 EV 最高的碰/吃变体，否则跳过
    recommend = ""
    if analysis.get("ok"):
        variant = analysis.get("variant") or {}
        if analysis.get("recommend") == "call" and variant:
            action_zh = _ACTION_ZH.get(variant.get("action"), variant.get("action"))
            tiles = "".join(
                _tile_img(t, tile_base) for t in (variant.get("consume") or [])
            )
            recommend = f'<span class="rec">推荐：{_esc(action_zh)} {tiles}</span>'
        else:
            recommend = '<span class="rec">推荐：跳过</span>'

    # 玩家 chip
    actual_zh = _ACTION_ZH.get(str(actual), actual)
    actual_tiles = "".join(
        _tile_img(t, tile_base) for t in (d.get("actual_tiles") or [])
    )
    actual_chip = (
        f'<span class="meta-item">玩家：{_esc(actual_zh)}{actual_tiles}</span>'
    )

    # 对比表：跳过期望行 + 各副露变体行，按推荐权重降序
    # （推荐高亮、玩家标记、自杀层淡色；消耗牌图并入选项列。
    #  不展示「切」——副露后怎么切交给随后的切牌决策卡。）
    detail = ""
    if analysis.get("ok"):
        head = (
            "<th>选项</th><th>向听数</th><th>EV</th>"
            "<th>和率</th><th>推荐权重</th>"
        )
        skip_weight = (analysis.get("skip") or {}).get("recommendation_weight")
        rows = [
            {
                "weight": skip_weight,
                "html": (
                    "<tr>"
                    "<td>跳过</td>"
                    f"<td>{_esc(analysis.get('skip_shanten') if analysis.get('skip_shanten') is not None else '—')}</td>"
                    f"<td>{_fmt_score(analysis.get('ev_skip'))}</td>"
                    f"<td>{_fmt_win(analysis.get('win_skip'))}</td>"
                    f"<td class='weight'>{_fmt_weight(skip_weight)}</td>"
                    "</tr>"
                ),
            }
        ]
        rec_variant = analysis.get("variant")
        for v in analysis.get("variants") or []:
            mark = ""
            if v.get("rejected"):
                mark += " rejected"
            # 推荐行高亮：内存引用相等优先，JSON 中转后按 动作+消耗 值比较
            if analysis.get("recommend") == "call" and rec_variant and (
                v is rec_variant
                or (
                    v.get("action") == rec_variant.get("action")
                    and list(v.get("consume") or [])
                    == list(rec_variant.get("consume") or [])
                )
            ):
                mark += " best"
            is_actual_row = actual in ("pon", "chii") and v.get("action") == actual
            if is_actual_row:
                mark += " actual"
            action_zh = _ACTION_ZH.get(v.get("action"), v.get("action"))
            consume = "".join(
                _tile_img(t, tile_base) for t in (v.get("consume") or [])
            )
            # 「向听数」：综合选切之后的向听（与 EV/和率同一条选切口径）
            cut_sh = v.get("cut_shanten")
            rows.append(
                {
                    "weight": v.get("recommendation_weight"),
                    "html": (
                        f"<tr class='{mark.strip()}'>"
                        f"<td>{_esc(action_zh)} {consume}</td>"
                        f"<td>{_esc(cut_sh if cut_sh is not None else '—')}</td>"
                        f"<td>{_fmt_score(v.get('ev'))}</td>"
                        f"<td>{_fmt_win(v.get('win'))}</td>"
                        f"<td class='weight'>{_fmt_weight(v.get('recommendation_weight'))}</td>"
                        "</tr>"
                    ),
                }
            )
        rows.sort(key=lambda r: -(r["weight"] or 0.0))
        body = "".join(r["html"] for r in rows)
        detail = f"""
                    <div class="legend">
                      <span><i class="lg best-lg"></i>推荐</span>
                      <span><i class="lg actual-lg"></i>玩家</span>
                    </div>
                    <div class="table-scroll"><table class="cand">
                      <thead><tr>{head}</tr></thead>
                      <tbody>{body}</tbody>
                    </table></div>"""
        # 判定依据（速度轴/收益轴/形听轴的中文 basis 文案由 decide 生成）
        basis = str(analysis.get("basis") or "")
        if basis:
            detail += f"<div class='muted call-basis'>依据：{_esc(basis)}</div>"
    elif analysis.get("error") or (analysis and analysis.get("ok") is False):
        detail = (
            f"<div class='err'>{_esc(analysis.get('error') or '评估失败')}</div>"
        )

    tid = f"t{ky.get('index')}_{d.get('turn')}_c{seq}"
    short = str(actual_zh)
    turn_open = " open" if match is False else ""

    row = f"""
                <details class="turn {status}" id="{tid}"{turn_open}>
                  <summary class="turn-head">
                    <strong>{_esc(d.get('label'))}</strong>
                    {badge}
                    <span class="meta">
                      {actual_chip}
                      {recommend}
                    </span>
                  </summary>
                  <div class="turn-body">
                    <div class="handline">
                      <span class="label">手牌</span> {_hand_span(d.get("hand"), tile_base)}
                      {_melds_html(d.get("melds") or [], tile_base)}
                      <span class="call-disc"><span class="label">{_esc(d.get("discarder_rel"))}打</span>{_tile_img(d.get("disc_tile"), tile_base)}</span>
                    </div>
                    {detail}
                  </div>
                </details>"""
    return row, (tid, short, status)


def render_classic_html(report: dict, tile_base: str = "../assets/tiles") -> str:
    sections = []
    side_groups = []
    names = report.get("names") or []
    seat = report.get("seat")
    total = analyzed = matched = blunders = 0
    for ky in report.get("kyokus") or []:
        rows = []
        nav_items = []
        call_seq = 0
        for d in ky.get("decisions") or []:
            analysis = d.get("analysis") or {}
            best = analysis.get("best") if analysis.get("ok") else None
            actual = d.get("actual")
            match = analysis.get("match")
            total += 1
            if analysis.get("ok"):
                analyzed += 1
            if match is True:
                matched += 1
            elif match is False and analysis.get("match_kind") == "different":
                blunders += 1
            if d.get("kind") == "call":
                # 副露决策点：独立卡片（计数逻辑与切牌点共用，见上）
                row_html, nav_item = _call_turn_html(d, ky, tile_base, call_seq)
                rows.append(row_html)
                nav_items.append(nav_item)
                call_seq += 1
                continue
            if match is True:
                badge_text = (
                    "✓ 一致·等价最优"
                    if analysis.get("match_kind") == "equivalent"
                    else "✓ 一致"
                )
                badge = f'<span class="badge ok">{badge_text}</span>'
                status = "s-ok"
            elif analysis.get("match_kind") == "acceptable":
                badge = '<span class="badge fair">△ 尚可</span>'
                status = "s-fair"
            elif match is False:
                badge = '<span class="badge diff">✗ 不一致</span>'
                status = "s-diff"
            elif d.get("skipped"):
                badge = '<span class="badge muted">未分析</span>'
                status = "s-skip"
            elif analysis.get("error"):
                badge = f'<span class="badge err">{_esc(analysis.get("error"))}</span>'
                status = "s-err"
            else:
                badge = ""
                status = ""

            posture_label = analysis.get("posture") if analysis.get("ok") else None
            posture_badge = ""
            if posture_label:
                p_cls = {
                    "全牌效": "p-eff",
                    "兜牌": "p-man",
                    "形听": "p-keit",
                    "全弃": "p-fold",
                }.get(str(posture_label), "p-eff")
                posture_badge = (
                    f'<span class="badge post {p_cls}">{_esc(posture_label)}</span>'
                )

            cand_rows = ""
            show_uke_only = analysis.get("calc_stats") is False
            # 候选按推荐权重降序渲染（权重=综合效用 softmax，与推荐徽标口径一致；
            # 无权重时保持原顺序，即向听数→牌效EV）
            _cands = list(analysis.get("candidates") or [])
            if any(c.get("recommendation_weight") is not None for c in _cands):
                _cands.sort(
                    key=lambda c: c.get("recommendation_weight")
                    if c.get("recommendation_weight") is not None
                    else -1.0,
                    reverse=True,
                )
            for c in _cands:
                mark = ""
                if c.get("tile") == actual:
                    mark += " actual"
                if c.get("tile") == best:
                    mark += " best"
                if c.get("policy_rejected") or c.get("kuikae"):
                    mark += " rejected"
                kuikae_badge = (
                    '<span class="kuikae-badge">食替</span>' if c.get("kuikae") else ""
                )
                uke_cell = _uke_cell(c, tile_base)
                exp_cell = (
                    "<td class='na'>—</td>"
                    if show_uke_only
                    else f"<td>{_fmt_score(c.get('exp_score'))}</td>"
                )
                deal = c.get("deal_in") or {}
                deal_cell = _deal_in_cell(deal.get("combined"))
                utility_cell = f"<td>{_fmt_score(c.get('adjusted_utility'))}</td>"
                weight_cell = (
                    f"<td class='weight'>{_fmt_weight(c.get('recommendation_weight'))}</td>"
                )
                cand_rows += (
                    f"<tr class='{mark}'>"
                    f"<td>{_tile_img(c.get('tile'), tile_base)}{kuikae_badge}</td>"
                    f"<td>{_esc(c.get('shanten'))}</td>"
                    f"{uke_cell}{exp_cell}{deal_cell}{utility_cell}{weight_cell}"
                    f"</tr>"
                )

            detail = ""
            if cand_rows or analysis.get("defense") or analysis.get("defense_error"):
                head = (
                    "<th>切</th><th>向听数</th><th>进张</th><th>牌效EV</th>"
                    "<th>综合危险度</th><th>综合效用</th><th>推荐权重</th>"
                )
                if cand_rows:
                    eff_panel = f"""
                    <div class="legend">
                      <span><i class="lg best-lg"></i>推荐</span>
                      <span><i class="lg actual-lg"></i>玩家</span>
                    </div>
                    <div class="table-scroll"><table class="cand">
                      <thead><tr>{head}</tr></thead>
                      <tbody>{cand_rows}</tbody>
                    </table></div>"""
                elif analysis.get("error"):
                    eff_panel = f"<div class='err'>{_esc(analysis.get('error'))}</div>"
                else:
                    eff_panel = "<p class='muted'>无牌效候选</p>"

                riichi_panel = _defense_panel_html(analysis, tile_base)
                detail = f"""
                <div class="tabs" data-tabs>
                  <div class="tab-bar" role="tablist">
                    <button type="button" class="tab-btn active" data-tab="eff"
                      aria-selected="true">牌效分析</button>
                    <button type="button" class="tab-btn" data-tab="riichi"
                      aria-selected="false">危险度分析</button>
                  </div>
                  <div class="tab-panel active" data-panel="eff">{eff_panel}</div>
                  <div class="tab-panel" data-panel="riichi">{riichi_panel}</div>
                </div>"""
            elif analysis.get("error"):
                detail = f"<div class='err'>{_esc(analysis.get('error'))}</div>"

            recommend = ""
            if best:
                recommend = f'<span class="rec">推荐：{_tile_img(best, tile_base)}</span>'

            tid = f"t{ky.get('index')}_{d.get('turn')}"
            short = str(d.get("label") or "").split()[-1] or f"第{d.get('turn')}巡"
            nav_items.append((tid, short, status))
            turn_open = (
                " open"
                if match is False and analysis.get("match_kind") != "acceptable"
                else ""
            )

            rows.append(
                f"""
                <details class="turn {status}" id="{tid}"{turn_open}>
                  <summary class="turn-head">
                    <strong>{_esc(d.get('label'))}</strong>
                    {badge}
                    {posture_badge}
                    <span class="meta">
                      <span class="meta-item">玩家：{_tile_img(actual, tile_base)}</span>
                      {'<span class="flag">立直</span>' if d.get("is_riichi") else ""}
                      {'<span class="flag">摸切</span>' if d.get("is_tsumogiri") else ""}
                      {recommend}
                    </span>
                  </summary>
                  <div class="turn-body">
                    <div class="handline">
                      <span class="label">手牌</span> {_hand_span(d.get("hand"), tile_base, d.get("drawn_tile"))}
                      {_melds_html(d.get("melds") or [], tile_base)}
                    </div>
                    {detail}
                  </div>
                </details>"""
            )

        scores = [int(x) for x in (ky.get("scores") or [])]
        score_chips = ""
        for i, s in enumerate(scores):
            nm = names[i] if i < len(names) else ("东南西北"[i] if i < 4 else str(i))
            me = " me" if seat is not None and i == seat else ""
            score_chips += (
                f'<span class="score{me}">'
                f'<span class="score-name">{_esc(nm)}</span>{s}</span>'
            )
        dora_tiles = ky.get("dora") or []
        if not dora_tiles and (ky.get("decisions") or []):
            dora_tiles = (ky["decisions"][0].get("dora") or [])
        dora_html = ""
        if dora_tiles:
            dora_imgs = "".join(_tile_img(t, tile_base) for t in dora_tiles)
            dora_html = (
                f'<span class="dora-box">'
                f'<span class="dora-label">宝牌指示牌</span>{dora_imgs}</span>'
            )
        seat_wind = ky.get("seat_wind")
        if not seat_wind and (ky.get("decisions") or []):
            seat_wind = ky["decisions"][0].get("seat_wind")
        wind_zh = _WIND_ZH.get(str(seat_wind or "").lower(), "")
        wind_html = ""
        if wind_zh:
            wind_html = (
                f'<span class="wind-box">'
                f'<span class="wind-label">自风</span>'
                f'<span class="wind-val">{wind_zh}</span></span>'
            )
        sections.append(
            f"""
            <details class="kyoku" id="k{ky.get('index')}" open>
              <summary class="kyoku-head">
                <h2>{_esc(ky.get('label'))}</h2>
                {wind_html}
                {dora_html}
                <div class="scores">{score_chips}</div>
              </summary>
              <div class="kyoku-body">
                {"".join(rows) or "<p class='muted'>无决策点</p>"}
              </div>
            </details>"""
        )

        ky_label = str(ky.get("label") or "")
        main_label, sep, sub_label = ky_label.partition("-")
        turns_nav = "".join(
            f'<a href="#{tid}"><i class="dot d-{st[2:] if st else "skip"}"></i>{_esc(short)}</a>'
            for tid, short, st in nav_items
        )
        side_groups.append(
            f'<details class="side-group" data-kyoku="k{ky.get("index")}">'
            f'<summary class="side-kyoku">'
            f'<span class="side-kyoku-main">{_esc(main_label)}</span>'
            + (f'<span class="side-kyoku-sub">{_esc(sub_label)}</span>' if sep else "")
            + "</summary>"
            f'<div class="side-turns">{turns_nav}</div>'
            "</details>"
        )

    kyoku_count = len(report.get("kyokus") or [])
    rate = f"{matched / analyzed * 100:.0f}%" if analyzed else "-"
    blunder_rate = f"{blunders / analyzed * 100:.0f}%" if analyzed else "-"
    stats_html = (
        '<div class="stats">'
        f'<span class="stat"><b>{kyoku_count}</b> 局</span>'
        f'<span class="stat"><b>{total}</b> 决策点</span>'
        f'<span class="stat stat-ok"><b>{rate}</b> 一致率</span>'
        f'<span class="stat stat-bad"><b>{blunder_rate}</b> 恶手率</span>'
        "</div>"
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MewJ 牌谱检讨</title>
<style>
:root {{
  --bg:#edf3f0; --card:#ffffff; --text:#1c2b27; --muted:#64748b;
  --felt:#1f6b57; --felt2:#185445; --accent:#2a9d7c; --line:#d7e3dd;
  --ok:#0f766e; --ok-bg:#e6f5f0; --diff:#be123c; --diff-bg:#fde8ee;
  --fair:#a16207; --fair-bg:#fef9c3;
  --best:#0f766e; --shadow:0 8px 24px rgba(15,45,38,.08);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg); color:var(--text); line-height:1.45;
}}
header {{
  background:linear-gradient(135deg, #1a5c4a 0%, #247a62 55%, #2f9b7a 100%);
  color:#fff; padding:1.1rem 1.25rem 1.25rem;
  box-shadow:0 4px 18px rgba(15,45,38,.22);
}}
.head-inner {{ max-width:1100px; margin:0 auto; }}
header h1 {{ margin:0 0 .55rem; font-size:1.55rem; letter-spacing:.02em; }}
.stats {{ display:flex; flex-wrap:wrap; gap:.45rem .7rem; }}
.stat {{
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  border-radius:999px; padding:.2rem .7rem; font-size:.86rem;
}}
.stat b {{ font-size:1.02rem; margin-right:.15rem; }}
.stat-ok {{ background:rgba(190,255,230,.22); border-color:rgba(190,255,230,.45); }}
.stat-bad {{ background:rgba(255,210,220,.22); border-color:rgba(255,210,220,.45); }}
main {{ max-width:1100px; margin:0 auto; padding:1rem 1rem 2rem; }}
.layout {{ display:flex; gap:1rem; align-items:flex-start; }}
.side {{
  width:168px; flex-shrink:0; position:sticky; top:.75rem;
  max-height:calc(100vh - 1.5rem); overflow:auto;
  background:var(--card); border-radius:12px; padding:.65rem .55rem;
  box-shadow:var(--shadow); border:1px solid var(--line);
}}
.side-title {{
  font-size:.75rem; color:var(--muted); font-weight:700;
  letter-spacing:.08em; margin:0 .25rem .4rem;
}}
.side-group {{ margin:0 0 .25rem; border:none; }}
.side-group > summary {{
  list-style:none; cursor:pointer; display:flex; flex-direction:column;
  gap:.05rem; padding:.35rem .4rem; border-radius:8px;
}}
.side-group > summary::-webkit-details-marker {{ display:none; }}
.side-group > summary:hover {{ background:#f1f7f4; }}
.side-kyoku-main {{ font-weight:700; font-size:.92rem; }}
.side-kyoku-sub {{ font-size:.75rem; color:var(--muted); }}
.side-turns {{ display:flex; flex-direction:column; gap:.1rem; padding:.15rem 0 .35rem .35rem; }}
.side-turns a {{
  color:var(--text); text-decoration:none; font-size:.82rem;
  padding:.18rem .35rem; border-radius:6px; display:flex; align-items:center; gap:.35rem;
}}
.side-turns a:hover {{ background:#eef6f2; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:#cbd5e1; flex-shrink:0; }}
.d-ok {{ background:var(--ok); }}
.d-fair {{ background:var(--fair); }}
.d-diff {{ background:var(--diff); }}
.d-err {{ background:#f59e0b; }}
.d-skip {{ background:#cbd5e1; }}
.content {{ flex:1; min-width:0; }}
.kyoku {{
  background:var(--card); border-radius:14px; margin:0 0 .9rem;
  border:1px solid var(--line); box-shadow:var(--shadow); overflow:hidden;
}}
.kyoku > summary.kyoku-head {{
  list-style:none; cursor:pointer; display:flex; flex-wrap:wrap;
  align-items:center; gap:.55rem 1rem; padding:.75rem 1rem;
  background:linear-gradient(90deg, #f3faf7, #fff);
  border-bottom:1px solid var(--line);
}}
.kyoku > summary.kyoku-head::-webkit-details-marker {{ display:none; }}
.kyoku > summary.kyoku-head h2 {{ margin:0; font-size:1.1rem; }}
.dora-box {{
  display:inline-flex; align-items:center; gap:2px;
  padding:.15rem .45rem .15rem .35rem; border-radius:8px;
  background:#f8fafc; border:1px solid var(--line);
}}
.dora-label {{
  font-size:.72rem; color:var(--muted); font-weight:600; margin-right:.25rem;
}}
.dora-box .tile-img {{ width:28px; height:39px; }}
.wind-box {{
  display:inline-flex; align-items:center; gap:.25rem;
  padding:.15rem .5rem; border-radius:8px;
  background:#f8fafc; border:1px solid var(--line);
}}
.wind-label {{ font-size:.72rem; color:var(--muted); font-weight:600; }}
.wind-val {{ font-size:.95rem; font-weight:700; color:var(--felt); }}
.scores {{ display:flex; flex-wrap:wrap; gap:.35rem; margin-left:auto; }}
.score {{
  background:#f1f5f4; border-radius:8px; padding:.15rem .5rem; font-size:.82rem;
  font-variant-numeric:tabular-nums;
}}
.score.me {{ background:var(--ok-bg); color:var(--ok); font-weight:700; }}
.score-name {{ color:var(--muted); margin-right:.3rem; font-weight:500; }}
.kyoku-body {{ padding:.35rem .7rem .7rem; }}
.turn {{
  border:1px solid var(--line); border-radius:10px; margin:.45rem 0;
  padding:.45rem .65rem .55rem; background:#fff; border-left:4px solid var(--line);
  scroll-margin-top:1rem;
}}
.turn:hover {{ box-shadow:var(--shadow); }}
.turn.s-ok {{ border-left-color:var(--ok); }}
.turn.s-fair {{ border-left-color:var(--fair); }}
.turn.s-diff {{ border-left-color:var(--diff); }}
.turn.s-err {{ border-left-color:#f59e0b; }}
.turn > summary.turn-head {{
  list-style:none; cursor:pointer; user-select:none;
  display:flex; flex-wrap:wrap; gap:.45rem .6rem; align-items:center;
  scroll-margin-top:1rem;
}}
.turn > summary.turn-head::-webkit-details-marker {{ display:none; }}
.turn > summary.turn-head::before {{
  content:"▸"; color:var(--muted); font-size:.8rem;
  transition:transform .15s; flex-shrink:0;
}}
.turn[open] > summary.turn-head::before {{ transform:rotate(90deg); }}
.turn-body {{ margin-top:.35rem; }}
.turn-head > strong {{ font-size:.98rem; }}
.badge {{
  display:inline-flex; align-items:center; border-radius:999px;
  padding:.12rem .6rem; font-size:.78rem; font-weight:700; white-space:nowrap;
}}
.badge.ok {{ color:var(--ok); background:var(--ok-bg); }}
.badge.fair {{ color:var(--fair); background:var(--fair-bg); }}
.badge.diff {{ color:var(--diff); background:var(--diff-bg); }}
.badge.muted {{ color:var(--muted); background:#eef2f1; }}
.badge.err {{ color:#b45309; background:#fef3e2; }}
.badge.post {{ color:var(--ok); background:var(--ok-bg); }}
.badge.post.p-man {{ color:var(--fair); background:var(--fair-bg); }}
/* 形听：紫色系，区别于兜牌（黄）/全弃（红） */
.badge.post.p-keit {{ color:#6d28d9; background:#ede9fe; }}
.badge.post.p-fold {{ color:var(--diff); background:var(--diff-bg); }}
.meta {{
  margin-left:auto; color:var(--muted); font-size:.9rem;
  display:inline-flex; align-items:center; gap:.4rem .6rem; flex-wrap:wrap;
}}
.meta-item {{ display:inline-flex; align-items:center; gap:.25rem; }}
.flag {{
  background:#eef2ff; color:#4338ca; border-radius:6px;
  font-size:.75rem; font-weight:600; padding:.08rem .45rem;
}}
.rec {{ display:inline-flex; align-items:center; gap:.25rem; color:var(--best); font-weight:600; }}
.handline {{ margin:.55rem 0 .2rem; display:flex; flex-wrap:wrap; align-items:center; gap:0; }}
.handline .label {{
  color:var(--muted); margin-right:.5rem; font-size:.82rem;
  border:1px solid var(--line); border-radius:6px; padding:.05rem .45rem;
}}
.handline .draw-gap {{ display:inline-block; width:18px; flex-shrink:0; }}
.tile-img {{
  width:36px; height:50px; object-fit:contain; vertical-align:middle;
  margin:0 1px 0 0; border-radius:3px; background:#f8fafc;
  box-shadow:0 1px 2px rgba(15,45,38,.18);
}}
.handline .tile-img {{ transition:transform .12s ease; }}
.handline .tile-img:hover {{ transform:translateY(-4px); }}
.tile-fallback {{
  display:inline-block; min-width:1.6em; padding:.1em .25em; margin:0 .08em;
  background:#fff; border:1px solid #cbd5e1; border-radius:4px;
  font-weight:600; font-size:.85rem;
}}
.melds {{ display:inline-flex; gap:.6rem; margin-left:.7rem; align-items:center; flex-wrap:wrap; }}
.meld {{
  display:inline-flex; align-items:center; gap:1px;
  padding:.18rem .45rem .18rem .3rem; border-left:3px solid var(--accent);
  background:#f4f8f6; border-radius:0 6px 6px 0;
}}
.meld-type {{ font-size:.72rem; color:var(--muted); margin-right:.25rem; }}
.err {{ color:var(--diff); }}
.muted {{ color:var(--muted); font-size:.9rem; }}
.cand td.na, .riichi td.na {{ color:var(--muted); }}
.tabs {{ margin-top:.55rem; }}
.tab-bar {{
  display:flex; gap:.35rem; margin-bottom:.55rem;
  border-bottom:1px solid var(--line); padding-bottom:.35rem;
}}
.tab-btn {{
  appearance:none; border:1px solid transparent; background:transparent;
  color:var(--muted); font-weight:600; font-size:.88rem;
  padding:.35rem .85rem; border-radius:8px 8px 0 0; cursor:pointer;
}}
.tab-btn:hover {{ color:var(--text); background:#f1f7f4; }}
.tab-btn.active {{
  color:var(--felt); background:var(--ok-bg); border-color:var(--line);
}}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}
.legend {{
  display:flex; gap:1.1rem; align-items:center; margin:.35rem 0 .1rem;
  color:var(--muted); font-size:.8rem;
}}
.lg {{
  display:inline-block; width:12px; height:12px; border-radius:3px;
  margin-right:.3rem; vertical-align:-1px;
}}
.best-lg {{ background:#d3f2e4; border:1px solid var(--ok); }}
.actual-lg {{ background:#e3edfb; border:1px solid #3b82f6; }}
.table-scroll {{ width:100%; overflow-x:auto; padding:1px; }}
table.cand, table.riichi {{
  width:100%; border-collapse:collapse; margin-top:.45rem; font-size:.9rem;
  background:#fff; border-radius:10px; overflow:hidden;
  box-shadow:0 0 0 1px var(--line);
}}
table.cand th, table.riichi th {{
  background:var(--felt2); color:#fff; font-weight:600; font-size:.85rem;
  padding:.45rem .5rem; text-align:center;
}}
table.cand td, table.riichi td {{
  border-bottom:1px solid var(--line); padding:.38rem .5rem;
  text-align:center; vertical-align:middle;
}}
table.cand tbody tr:last-child td,
table.riichi tbody tr:last-child td {{ border-bottom:none; }}
table.cand td:nth-child(n+3) {{ font-variant-numeric:tabular-nums; }}
table.cand tbody tr:hover td, table.riichi tbody tr:hover td {{ background:#f4faf8; }}
table.cand tr.best td {{ background:var(--ok-bg); font-weight:600; }}
table.cand tr.best td:first-child {{ box-shadow:inset 3px 0 0 var(--ok); }}
table.cand tr.actual td {{ background:#e8f0fd; }}
table.cand tr.actual td:first-child {{ box-shadow:inset 3px 0 0 #3b82f6; }}
table.cand tr.actual.best td {{ background:var(--ok-bg); }}
table.cand tr.actual.best td:first-child {{ box-shadow:inset 3px 0 0 var(--ok); }}
table.cand tr.rejected td {{ color:var(--muted); }}
.call-disc {{ display:inline-flex; align-items:center; gap:.25rem; margin-left:.7rem; }}
table.cand td.weight {{ color:var(--best); font-weight:700; }}
table.cand td.uke {{
  text-align:left; min-width:7rem; max-width:14rem; font-weight:500;
}}
table.cand .uke-n {{
  display:inline-block; font-variant-numeric:tabular-nums; font-weight:700;
  margin-bottom:.15rem;
}}
table.cand .uke-tiles {{
  display:flex; flex-wrap:wrap; gap:4px 6px; align-items:flex-end;
  margin-top:.2rem;
}}
table.cand .uke-tile {{
  position:relative; display:inline-block; line-height:0;
  padding:0 2px 0 0;
}}
table.cand .uke-tile .tile-img {{
  width:26px; height:36px; margin:0; display:block;
}}
table.cand .uke-badge {{
  position:absolute; top:-5px; right:-6px;
  min-width:14px; height:14px; padding:0 3px;
  border-radius:999px; background:#1f6b57; color:#fff;
  font-size:.65rem; font-weight:700; line-height:14px;
  text-align:center; box-shadow:0 0 0 1px #fff;
  pointer-events:none;
}}
table.cand .uke-badge.risk {{
  background:#ca8a04; color:#fff;
}}
table.cand .furiten-badge {{
  display:inline-block; margin-left:.35rem; padding:.05rem .35rem;
  font-size:.68rem; font-weight:700; color:#b91c1c; background:#fee2e2;
  border-radius:4px; vertical-align:middle;
}}
table.cand .kuikae-badge {{
  display:inline-block; margin-left:.35rem; padding:.05rem .35rem;
  font-size:.68rem; font-weight:700; color:#9f1239; background:#ffe4e6;
  border-radius:4px; vertical-align:middle;
}}
table.riichi tr .deal-in.safe {{ color:var(--ok); font-weight:700; }}
.threat-list {{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.2rem 0 .55rem; }}
.threat-badge {{
  font-size:.8rem; font-weight:600; color:#9a3412;
  background:#fff7ed; border:1px solid #fed7aa; border-radius:8px;
  padding:.25rem .55rem;
}}
.foot {{
  text-align:center; color:var(--muted); font-size:.8rem;
  padding:1.5rem 0 .5rem;
}}
.back-top {{
  position:fixed; right:1.1rem; bottom:1.1rem; z-index:40;
  width:2.5rem; height:2.5rem; border:none; border-radius:14px;
  background:linear-gradient(160deg,var(--accent),var(--felt));
  color:#fff; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 6px 16px rgba(15,45,38,.24), inset 0 1px 0 rgba(255,255,255,.25);
  opacity:0; pointer-events:none; transform:translateY(.5rem);
  transition:opacity .2s ease, transform .2s ease, box-shadow .2s ease, filter .2s ease;
}}
.back-top svg {{ width:1.05rem; height:1.05rem; display:block; }}
.back-top:hover {{
  filter:brightness(1.08);
  box-shadow:0 10px 22px rgba(15,45,38,.3), inset 0 1px 0 rgba(255,255,255,.3);
  transform:translateY(-2px);
}}
.back-top:active {{ transform:translateY(0); filter:brightness(.96); }}
.back-top.show {{ opacity:1; pointer-events:auto; }}
.back-top.show:hover {{ transform:translateY(-2px); }}
@media (max-width:860px) {{
  .layout {{ flex-direction:column; }}
  .side {{ position:static; width:100%; max-height:42vh; }}
}}
@media (max-width:640px) {{
  .tile-img {{ width:28px; height:39px; }}
  .meta {{ margin-left:0; }}
  header h1 {{ font-size:1.25rem; }}
  .back-top {{ right:.85rem; bottom:.85rem; }}
}}
</style>
</head>
<body>
<header>
  <div class="head-inner">
    <h1>🀄 MewJ 牌谱检讨</h1>
    {stats_html}
  </div>
</header>
<main>
  <div class="layout">
    <aside class="side">
      <div class="side-title">目录</div>
      {"".join(side_groups)}
    </aside>
    <div class="content">
      {"".join(sections)}
      <div class="foot">MewJ · Classic 牌谱检讨报告</div>
    </div>
  </div>
</main>
<button type="button" class="back-top" id="back-top" aria-label="回到顶部" title="回到顶部"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5.2 20 18H4Z" fill="currentColor"/><path d="M12 5.2 20 18H4Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg></button>
<script type="application/json" id="report-json">{html.escape(json.dumps(report, ensure_ascii=False))}</script>
<script>
document.addEventListener('click', function (e) {{
  var btn = e.target.closest('.tab-btn');
  if (!btn) return;
  var root = btn.closest('[data-tabs]');
  if (!root) return;
  var name = btn.getAttribute('data-tab');
  root.querySelectorAll('.tab-btn').forEach(function (b) {{
    var on = b === btn;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  }});
  root.querySelectorAll('.tab-panel').forEach(function (p) {{
    p.classList.toggle('active', p.getAttribute('data-panel') === name);
  }});
}});

(function () {{
  var backTop = document.getElementById('back-top');
  if (!backTop) return;
  function syncBackTop() {{
    backTop.classList.toggle('show', window.pageYOffset > 320);
  }}
  backTop.addEventListener('click', function () {{
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }});
  window.addEventListener('scroll', syncBackTop, {{ passive: true }});
  syncBackTop();
}})();

function exclusiveDetails(selector, scopeFn) {{
  document.querySelectorAll(selector).forEach(function (el) {{
    el.addEventListener('toggle', function () {{
      if (!el.open) return;
      var scope = scopeFn ? scopeFn(el) : document;
      scope.querySelectorAll(selector).forEach(function (other) {{
        if (other !== el) other.open = false;
      }});
    }});
  }});
}}

function syncSideToKyoku(kyoku) {{
  if (!kyoku || !kyoku.id) return;
  document.querySelectorAll('aside.side details.side-group').forEach(function (g) {{
    g.open = (g.getAttribute('data-kyoku') === kyoku.id);
  }});
}}

function openTurnById(id) {{
  if (!id) return null;
  var turn = document.getElementById(id);
  if (!turn || !turn.classList.contains('turn')) return null;
  var kyoku = turn.closest('details.kyoku');
  if (kyoku) {{
    kyoku.open = true;
    syncSideToKyoku(kyoku);
  }}
  // Close siblings first so later scroll isn't shifted by collapsing content above.
  document.querySelectorAll('details.turn').forEach(function (el) {{
    if (el !== turn) el.open = false;
  }});
  turn.open = true;
  return turn;
}}

function turnScrollAnchor(turn) {{
  return turn.querySelector('summary.turn-head') || turn;
}}

function scrollTurnToTop(turn, smooth) {{
  var anchor = turnScrollAnchor(turn);
  var top = anchor.getBoundingClientRect().top + window.pageYOffset - 8;
  window.scrollTo({{
    top: Math.max(0, top),
    behavior: smooth ? 'smooth' : 'auto',
  }});
}}

function revealTurn(id) {{
  var turn = openTurnById(id);
  if (!turn) return;
  // Wait until open/close layout settles, then pin the turn header to the top.
  // Use an instant scroll first so tall tables don't leave the viewport mid-body.
  requestAnimationFrame(function () {{
    requestAnimationFrame(function () {{
      scrollTurnToTop(turn, false);
      setTimeout(function () {{ scrollTurnToTop(turn, true); }}, 180);
    }});
  }});
}}

// 目录：一次只展开一局
exclusiveDetails('aside.side details.side-group');
// 正文：一次只展开一巡（全局，跨局）；展开巡目时同步左侧目录到当局
document.querySelectorAll('details.turn').forEach(function (el) {{
  el.addEventListener('toggle', function () {{
    if (!el.open) return;
    document.querySelectorAll('details.turn').forEach(function (other) {{
      if (other !== el) other.open = false;
    }});
    var kyoku = el.closest('details.kyoku');
    if (kyoku) {{
      kyoku.open = true;
      syncSideToKyoku(kyoku);
    }}
  }});
}});

// 右侧展开某局时，左侧目录同步切换到当局
document.querySelectorAll('details.kyoku').forEach(function (el) {{
  el.addEventListener('toggle', function () {{
    if (!el.open) return;
    syncSideToKyoku(el);
  }});
}});

// 左侧目录点击：跳转并展开对应巡目
document.querySelectorAll('.side-turns a[href^="#"]').forEach(function (link) {{
  link.addEventListener('click', function (e) {{
    var href = link.getAttribute('href') || '';
    var id = href.charAt(0) === '#' ? href.slice(1) : '';
    if (!id) return;
    e.preventDefault();
    var group = link.closest('details.side-group');
    if (group) group.open = true;
    if (history.replaceState) {{
      history.replaceState(null, '', href);
    }} else {{
      location.hash = href;
    }}
    revealTurn(id);
  }});
}});

window.addEventListener('hashchange', function () {{
  revealTurn((location.hash || '').replace(/^#/, ''));
}});

// 初始若有多巡展开，只保留第一处；若 URL 带巡目锚点则展开该巡
(function () {{
  var opened = Array.prototype.slice.call(document.querySelectorAll('details.turn[open]'));
  opened.slice(1).forEach(function (el) {{ el.open = false; }});
  var sideOpened = Array.prototype.slice.call(document.querySelectorAll('aside.side details.side-group[open]'));
  sideOpened.slice(1).forEach(function (el) {{ el.open = false; }});
  var hashId = (location.hash || '').replace(/^#/, '');
  if (hashId) {{
    var group = document.querySelector('.side-turns a[href="#' + hashId + '"]');
    if (group) {{
      var side = group.closest('details.side-group');
      if (side) side.open = true;
    }}
    revealTurn(hashId);
  }} else {{
    // 无锚点时：左侧目录对齐正文当前已展开的第一局
    var firstKy = document.querySelector('details.kyoku[open]');
    if (firstKy) syncSideToKyoku(firstKy);
  }}
}})();
</script>
</body>
</html>
"""


def _rel_tile_base(html_path: Path) -> str:
    """URL path from the HTML file to MewJ/assets/tiles."""
    html_dir = html_path.resolve().parent
    tile_dir = _TILE_DIR.resolve()
    rel = os.path.relpath(tile_dir, html_dir)
    return rel.replace("\\", "/")


def write_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_base = _rel_tile_base(path)
    path.write_text(render_classic_html(report, tile_base=tile_base), encoding="utf-8")
    return path
