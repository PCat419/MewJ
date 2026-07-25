"""MewJ end-to-end pipeline: link/JSON → download (optional) → review → HTML."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .report import write_report
from .review import DEFAULT_NANIKIRU, review_paipu

MEWJ_ROOT = Path(__file__).resolve().parent
PAIPU_DIR = MEWJ_ROOT / "paipu"
OUT_DIR = MEWJ_ROOT / "out"


def load_dotenv(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_paipu_source(source: str) -> tuple[str, Optional[Path]]:
    """Return ('file', path) or ('link', None) with raw link/uuid string in source."""
    text = source.strip().strip('"').strip("'")
    if not text:
        raise ValueError("empty paipu source")
    path = Path(text)
    if path.is_file():
        return "file", path
    if path.suffix.lower() == ".json":
        raise FileNotFoundError(f"paipu file not found: {path}")
    return "link", None


def load_paipu_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


async def download_paipu_async(
    link_or_uuid: str,
    *,
    cache_dir: Path = PAIPU_DIR,
    force: bool = False,
) -> Path:
    """Download Majsoul paipu via tensoul; cache under MewJ/paipu/<uuid>.json."""
    _ensure_tensoul_importable()
    try:
        from tensoul import (
            MajsoulDownloadError,
            MajsoulLoginError,
            MajsoulPaipuDownloader,
            parse_paipu_uuid,
        )
    except ImportError as exc:
        raise RuntimeError(
            "无法导入雀魂下载组件（vendor/tensoul）或其依赖。"
            "请先在 MewJ 目录执行: pip install -r requirements.txt"
        ) from exc

    record_uuid = parse_paipu_uuid(link_or_uuid)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{record_uuid}.json"
    if out_path.is_file() and not force:
        print(f"使用缓存牌谱: {out_path}", flush=True)
        return out_path

    username = os.environ.get("MAJSOUL_USERNAME", "").strip()
    password = os.environ.get("MAJSOUL_PASSWORD", "").strip()
    access_token = os.environ.get("MAJSOUL_ACCESS_TOKEN", "").strip()
    if not access_token and (not username or not password):
        raise RuntimeError(
            "下载牌谱需要账号。请在 MewJ/.env 填写 MAJSOUL_USERNAME / MAJSOUL_PASSWORD，"
            "或填写 MAJSOUL_ACCESS_TOKEN"
        )

    proxy = os.environ.get("MAJSOUL_PROXY", "").strip() or None
    if proxy:
        print(f"使用代理: {proxy}", flush=True)

    print(f"下载牌谱: {record_uuid}", flush=True)
    try:
        async with MajsoulPaipuDownloader(proxy=proxy) as downloader:
            print("正在登录...", flush=True)
            if access_token:
                oauth_type = int(os.environ.get("MAJSOUL_OAUTH_TYPE", "0") or "0")
                await downloader.login_with_token(access_token, oauth_type=oauth_type)
            else:
                await downloader.login(username, password)
            print("正在下载...", flush=True)
            logs = await downloader.download(record_uuid)
    except MajsoulLoginError as exc:
        code = exc.code
        msg = f"登录失败{f'（错误码 {code}）' if code else ''}。"
        if code == 151:
            msg += (
                "客户端资源版本可能仍不匹配（一般会自动探测）。"
                "也可在 .env 设置 MAJSOUL_RES_VERSION=登录页上的 0.16.xxx。"
            )
        else:
            msg += "请检查账号密码，或改用 MAJSOUL_ACCESS_TOKEN。"
        raise RuntimeError(msg) from exc
    except MajsoulDownloadError as exc:
        raise RuntimeError(f"下载失败，错误码: {exc.code}") from exc

    out_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {out_path}", flush=True)
    return out_path


def _ensure_tensoul_importable() -> None:
    """Locate the tensoul package: bundled vendor copy > installed > repo sibling.

    The distribution bundles tensoul at ``MewJ/vendor/tensoul/``; when present,
    put ``vendor/`` first on sys.path and drop any prior partial import so the
    bundled copy always wins. Otherwise prefer an installed package that
    exposes ``parse_paipu_uuid``; as a last resort (dev monorepo layout
    ``Mahjong/tensoul/tensoul/``, where the outer folder can shadow as a
    namespace package when cwd is the repo root) put ``Mahjong/tensoul``
    first on sys.path and drop any broken partial import.
    """
    import importlib

    def _purge_tensoul_modules() -> None:
        for name in list(sys.modules):
            if name == "tensoul" or name.startswith("tensoul."):
                del sys.modules[name]

    def _put_first(path: Path) -> None:
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    vendor = MEWJ_ROOT / "vendor"
    if (vendor / "tensoul" / "__init__.py").is_file():
        _put_first(vendor)
        _purge_tensoul_modules()
        return

    def _usable() -> bool:
        try:
            mod = importlib.import_module("tensoul")
            return hasattr(mod, "parse_paipu_uuid")
        except ImportError:
            return False

    if _usable():
        return

    sibling = (MEWJ_ROOT.parent / "tensoul").resolve()
    if not (sibling / "tensoul" / "__init__.py").is_file():
        return

    _put_first(sibling)
    _purge_tensoul_modules()


def download_paipu(link_or_uuid: str, *, cache_dir: Path = PAIPU_DIR, force: bool = False) -> Path:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(
        download_paipu_async(link_or_uuid, cache_dir=cache_dir, force=force)
    )


def obtain_paipu(
    source: str,
    *,
    force_download: bool = False,
) -> tuple[dict, Path]:
    """Load paipu from local JSON or Majsoul link/UUID. Returns (paipu, json_path)."""
    kind, path = resolve_paipu_source(source)
    if kind == "file":
        assert path is not None
        print(f"读取本地牌谱: {path}", flush=True)
        return load_paipu_json(path), path

    json_path = download_paipu(source.strip(), force=force_download)
    return load_paipu_json(json_path), json_path


def run_pipeline(
    source: str,
    seat: int = 0,
    *,
    kyoku_indices: Optional[List[int]] = None,
    max_turns: Optional[int] = None,
    nanikiru_url: str = DEFAULT_NANIKIRU,
    structure_only: bool = False,
    force_download: bool = False,
    output: Optional[Path] = None,
) -> Path:
    """
    Full pipeline: source (link or json) → review → HTML report.
    Returns the written HTML path.
    """
    # Prefer MewJ/.env; fall back to tensoul/.env if present
    load_dotenv(MEWJ_ROOT / ".env", MEWJ_ROOT.parent / "tensoul" / ".env")

    paipu, json_path = obtain_paipu(source, force_download=force_download)
    ref = paipu.get("ref") or json_path.stem
    print(f"检讨 {ref} seat={seat} …", flush=True)

    report = review_paipu(
        paipu,
        seat,
        kyoku_indices=kyoku_indices,
        max_turns=max_turns,
        nanikiru_url=nanikiru_url,
        skip_analyze=structure_only,
    )

    out = output
    if out is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{ref}_seat{seat}.html"

    write_report(report, out)
    n_dec = sum(len(k["decisions"]) for k in report["kyokus"])
    n_ok = sum(
        1
        for k in report["kyokus"]
        for d in k["decisions"]
        if (d.get("analysis") or {}).get("ok")
    )
    print(f"已生成: {out}  decisions={n_dec} analyzed_ok={n_ok}", flush=True)
    return Path(out)
