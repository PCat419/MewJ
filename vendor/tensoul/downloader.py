from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import ms.protocol_pb2 as pb
import websockets
from aiohttp import ClientError, ClientResponseError
from ms.base import MSRPCChannel
from ms.rpc import Lobby
from websockets.exceptions import ConnectionClosedError

from .cfg import cfg
from .constants import RUNES, JPNAME
from .parser import MajsoulPaipuParser

# Direct WS endpoints commonly reachable in CN; tried before slow HTTP discovery.
_FALLBACK_ENDPOINTS = (
    "wss://route-5.maj-soul.com/gateway",
    "wss://route-4.maj-soul.com/gateway",
    "wss://route-3.maj-soul.com/gateway",
    "wss://route-2.maj-soul.com/gateway",
    "wss://route-6.maj-soul.com/gateway",
    "wss://gateway-cdn.maj-soul.com/gateway",
    "wss://gateway-vexcdn.maj-soul.com/gateway",
)

_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=8)
_WS_TIMEOUT = 8

# CN Unity WebGL login uses WebGL_2022-{res}. The res number (0.16.xxx) is a Unity
# hot-update version not published in version.json; we cache the last working value
# and auto-probe nearby builds when login returns 151.
_DEFAULT_RES_VERSION = "0.16.255"
_RES_VERSION_CACHE = Path(__file__).resolve().parent.parent / ".res_version"
_RES_PROBE_SPAN = 8

_PAIPU_UUID_RE = re.compile(
    r"\d{6}-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_ROUTE_ID_RE = re.compile(r"route-(\d+)")


class MajsoulLoginError(BaseException):
    def __init__(self, res: Any = None, *, code: int | None = None, detail: str = ""):
        self.res = res
        self.code = code
        if self.code is None and res is not None:
            err = getattr(res, "error", None)
            self.code = int(getattr(err, "code", 0) or 0) or None
        self.detail = detail
        msg = "Majsoul login failed"
        if self.code:
            msg += f" (error {self.code})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class MajsoulDownloadError(BaseException):
    def __init__(self, code: int):
        self.code = code


def parse_paipu_uuid(text: str) -> str:
    """Extract record UUID from a share link or raw UUID string."""
    text = text.strip()
    if not text:
        raise ValueError("empty paipu link")

    if "paipu=" in text or text.startswith("http"):
        query = parse_qs(urlparse(text).query)
        if "paipu" in query:
            text = query["paipu"][0]
        else:
            match = re.search(r"paipu=([^&#]+)", text)
            if match:
                text = match.group(1)

    text = text.split("_a", 1)[0].strip()
    match = _PAIPU_UUID_RE.search(text)
    if not match:
        raise ValueError(f"cannot parse paipu UUID from: {text!r}")
    return match.group(0)


class MajsoulPaipuDownloader:
    MS_HOST = "https://game.maj-soul.com"

    def __init__(
        self,
        host: str | None = None,
        proxy: str | None = None,
        endpoint: str | None = None,
        client_version: str | None = None,
        trust_env: bool = True,
    ):
        if host:
            self.MS_HOST = host.rstrip("/")
        self.proxy = _normalize_proxy(proxy)
        self.endpoint = endpoint
        self.client_version = client_version
        self.trust_env = trust_env
        self.channel = None
        self.lobby = None
        self.version = None  # version.json asset version, e.g. 0.11.252.w
        self.package_version = None  # Unity productVersion, e.g. 4.0.45
        self.res_version = None  # Unity res version, e.g. 0.16.255
        self.version_to_force = None  # used in client_version_string suffix
        self.token = None

    async def start(self):
        await self._connect()

    async def close(self):
        try:
            if self.channel:
                await self.channel.close()
        except ConnectionClosedError:
            pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def client_version_string(self) -> str:
        # CN Unity WebGL sends WebGL_2022-0.16.255 (confirmed via browser capture).
        return f"WebGL_2022-{self.version_to_force}"

    async def _connect(self):
        print("正在获取雀魂版本信息...", flush=True)
        async with aiohttp.ClientSession(trust_env=self.trust_env) as session:
            async with session.get(
                f"{self.MS_HOST}/1/version.json",
                timeout=_HTTP_TIMEOUT,
            ) as res:
                version_res = await res.json()
                self.version = version_res["version"]

            self.package_version = await _detect_product_version(session, self.MS_HOST)
            self.res_version = (
                self.client_version
                or os.environ.get("MAJSOUL_RES_VERSION", "").strip()
                or _load_cached_res_version()
                or _DEFAULT_RES_VERSION
            )
            self.version_to_force = self.res_version
            print(
                f"客户端版本: {self.client_version_string} "
                f"(res={self.res_version}, package={self.package_version}, "
                f"assets={self.version})",
                flush=True,
            )

        await self._open_websocket()
        await self._request_connection()

    async def _discover_endpoint(self, routes: list[str]) -> str | None:
        attempts: list[tuple[aiohttp.ClientSession, str | None]] = []
        try:
            if self.proxy and self._is_socks(self.proxy):
                attempts.append((await self._make_socks_session(), None))
            elif self.proxy:
                attempts.append((aiohttp.ClientSession(trust_env=False), self.proxy))
            attempts.append((aiohttp.ClientSession(trust_env=self.trust_env), None))

            for session, http_proxy in attempts:
                try:
                    return await _detect_endpoint(session, list(routes)[:4], http_proxy)
                except RuntimeError:
                    continue
            return None
        finally:
            for session, _ in attempts:
                await session.close()

    async def _make_socks_session(self) -> aiohttp.ClientSession:
        assert self.proxy
        try:
            from aiohttp_socks import ProxyConnector
        except ImportError as exc:
            raise RuntimeError(
                "SOCKS proxy requires aiohttp-socks. Install with: pip install aiohttp-socks"
            ) from exc
        return aiohttp.ClientSession(
            connector=ProxyConnector.from_url(self.proxy, rdns=True),
            trust_env=False,
        )

    @staticmethod
    def _is_socks(proxy: str | None) -> bool:
        return bool(proxy and proxy.startswith(("socks5://", "socks5h://", "socks4://")))

    async def _open_websocket(self):
        candidates: list[str] = []
        if self.endpoint:
            candidates.append(self.endpoint)
        for endpoint in _FALLBACK_ENDPOINTS:
            if endpoint not in candidates:
                candidates.append(endpoint)

        failures: list[str] = []
        print("正在连接雀魂网关...", flush=True)
        for endpoint in candidates:
            print(f"  尝试 {endpoint}", flush=True)
            channel = MSRPCChannel(endpoint)
            lobby = Lobby(channel)
            try:
                await asyncio.wait_for(self._ws_connect(channel), timeout=_WS_TIMEOUT)
            except Exception as exc:
                failures.append(f"{endpoint}: {type(exc).__name__}: {exc}")
                try:
                    await channel.close()
                except Exception:
                    pass
                continue

            self.endpoint = endpoint
            self.channel = channel
            self.lobby = lobby
            print(f"已连接: {endpoint}", flush=True)
            return

        hint = (
            " 代理已设置但仍失败，请检查代理能否访问雀魂网关。"
            if self.proxy
            else " 可在 .env 中设置 MAJSOUL_PROXY=socks5://127.0.0.1:10808 后重试。"
        )
        raise RuntimeError("无法连接雀魂网关。" + hint + " 详情: " + "; ".join(failures))

    async def _ws_connect(self, channel: MSRPCChannel):
        # Only use an explicit proxy; system HTTP_PROXY often breaks maj-soul WS.
        proxy: Any = self.proxy
        channel._ws = await websockets.connect(
            channel._endpoint,
            origin=self.MS_HOST,
            proxy=proxy,
            open_timeout=_WS_TIMEOUT,
        )
        channel._msg_dispatcher = asyncio.create_task(channel.dispatch_msg())

    async def _request_connection(self) -> None:
        """CN Unity clients send Route.requestConnection before Lobby.login."""
        assert self.channel and self.endpoint
        route_id = _route_id_from_endpoint(self.endpoint)
        req = pb.ReqRequestConnection()
        req.type = 2
        req.route_id = route_id
        req.timestamp = int(time.time())
        # Newer clients append field 6 = "Web"; ms-api protobuf may not define it yet.
        body = req.SerializeToString() + b"\x32\x03Web"
        res_msg = await self.channel.send_request(".lq.Route.requestConnection", body)
        res = pb.ResRequestConnection()
        if res_msg:
            res.ParseFromString(res_msg)
        if res.error.code:
            raise RuntimeError(f"Route.requestConnection failed (error {res.error.code})")
        print(f"网关握手完成: {route_id}", flush=True)

    def _fill_client_version(self, req) -> None:
        req.client_version_string = self.client_version_string
        if hasattr(req, "client_version"):
            req.client_version.resource = self.res_version or self.version_to_force
            if self.package_version and hasattr(req.client_version, "package"):
                req.client_version.package = self.package_version

    def _fill_device(self, req) -> None:
        req.device.platform = "pc"
        req.device.hardware = "pc"
        req.device.os = "windows"
        req.device.os_version = "win10"
        req.device.is_browser = True
        req.device.software = "Chrome"
        req.device.sale_platform = "web"
        req.device.screen_width = 1920
        req.device.screen_height = 1080
        req.device.screen_type = 1
        req.device.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        )

    def _fill_currency_platforms(self, req) -> None:
        req.currency_platforms.extend([1, 2, 5, 6, 8, 10, 11])

    def _set_res_version(self, res_version: str) -> None:
        self.res_version = res_version
        self.version_to_force = res_version

    def _res_version_candidates(self) -> list[str]:
        """Prefer current, then nearby Unity hot-update builds."""
        base = self.res_version or _DEFAULT_RES_VERSION
        parts = base.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            return [base]
        major, minor, patch = (int(p) for p in parts)
        offsets = [0]
        for i in range(1, _RES_PROBE_SPAN + 1):
            offsets.extend((i, -i))
        out: list[str] = []
        for off in offsets:
            p = patch + off
            if p < 0:
                continue
            ver = f"{major}.{minor}.{p}"
            if ver not in out:
                out.append(ver)
        return out

    async def login(self, username, password):
        digest = hmac.new(b"lailai", password.encode(), hashlib.sha256).hexdigest()
        last_res = None
        forced = bool(
            self.client_version or os.environ.get("MAJSOUL_RES_VERSION", "").strip()
        )
        candidates = [self.res_version] if forced else self._res_version_candidates()

        for res_version in candidates:
            self._set_res_version(res_version)
            if res_version != candidates[0]:
                print(f"尝试资源版本: {res_version}", flush=True)

            req = pb.ReqLogin()
            req.account = username
            req.password = digest
            req.reconnect = False
            self._fill_device(req)
            req.random_key = str(uuid.uuid1())
            req.gen_access_token = True
            self._fill_client_version(req)
            self._fill_currency_platforms(req)
            req.type = 0
            req.tag = "cn"

            res = await self.lobby.login(req)
            if res.access_token:
                _save_cached_res_version(res_version)
                if res_version != candidates[0]:
                    print(f"已自动切换资源版本: {res_version}", flush=True)
                self.token = res.access_token
                return

            last_res = res
            err = getattr(res, "error", None)
            code = int(getattr(err, "code", 0) or 0)
            # 151 = client/version rejected. Other codes (e.g. 1003 wrong password)
            # mean this res version is accepted — stop probing.
            if code != 151 or forced:
                break

        raise MajsoulLoginError(
            last_res,
            detail="password login rejected",
        )

    async def login_with_token(self, access_token: str, *, oauth_type: int = 0):
        """Login with a browser access_token (oauth2Login). Useful when password login fails."""
        forced = bool(
            self.client_version or os.environ.get("MAJSOUL_RES_VERSION", "").strip()
        )
        candidates = [self.res_version] if forced else self._res_version_candidates()
        last_res = None

        for res_version in candidates:
            self._set_res_version(res_version)
            if res_version != candidates[0]:
                print(f"尝试资源版本: {res_version}", flush=True)

            req = pb.ReqOauth2Login()
            req.type = oauth_type
            req.access_token = access_token
            req.reconnect = False
            self._fill_device(req)
            req.random_key = str(uuid.uuid1())
            req.gen_access_token = True
            self._fill_client_version(req)
            self._fill_currency_platforms(req)
            req.tag = "cn"

            res = await self.lobby.oauth2_login(req)
            if res.access_token:
                _save_cached_res_version(res_version)
                if res_version != candidates[0]:
                    print(f"已自动切换资源版本: {res_version}", flush=True)
                self.token = res.access_token
                return

            last_res = res
            err = getattr(res, "error", None)
            code = int(getattr(err, "code", 0) or 0)
            if code != 151 or forced:
                break

        raise MajsoulLoginError(
            last_res,
            detail="invalid MAJSOUL_ACCESS_TOKEN or oauth type",
        )

    async def download(self, record_uuid: str, target_account_id: int | None = None):
        req = pb.ReqGameRecord()
        req.game_uuid = record_uuid
        req.client_version_string = self.client_version_string
        res = await self.lobby.fetch_game_record(req)

        if res.error.code:
            raise MajsoulDownloadError(code=res.error.code)

        return self._handle_game_record(res, target_account_id=target_account_id)

    def _handle_game_record(self, record, target_account_id: int | None = None):
        res = {}
        ruledisp = ""
        lobby = ""
        nplayers = len(record.head.result.players)
        nakas = nplayers - 1
        tsumoloss_off = False

        res["ver"] = "2.3"
        res["ref"] = record.head.uuid
        res["ratingc"] = f"PF{nplayers}"

        if nplayers == 3:
            ruledisp += RUNES["sanma"][JPNAME]
        if record.head.config.meta.mode_id:
            ruledisp += cfg["desktop"]["matchmode"]["map_"][str(record.head.config.meta.mode_id)]["room_name_jp"]
        elif record.head.config.meta.room_id:
            lobby = f": {record.head.config.meta.room_id}"
            ruledisp += RUNES["friendly"][JPNAME]
            nakas = record.head.config.mode.detail_rule.dora_count
            tsumoloss_off = nplayers == 3 and not record.head.config.mode.detail_rule.have_zimosun
        elif record.head.config.meta.contest_uid:
            lobby = f": {record.head.config.meta.contest_uid}"
            ruledisp += RUNES["tournament"][JPNAME]
            nakas = record.head.config.mode.detail_rule.dora_count
            tsumoloss_off = nplayers == 3 and not record.head.config.mode.detail_rule.have_zimosun

        if record.head.config.mode.mode == 1:
            ruledisp += RUNES["tonpuu"][JPNAME]
        elif record.head.config.mode.mode == 2:
            ruledisp += RUNES["hanchan"][JPNAME]

        if record.head.config.meta.mode_id == 0 and record.head.config.mode.detail_rule.dora_count == 0:
            res["rule"] = {"disp": ruledisp, "aka53": 0, "aka52": 0, "aka51": 0}
        else:
            res["rule"] = {
                "disp": ruledisp,
                "aka53": 1,
                "aka52": 2 if nakas == 4 else 1,
                "aka51": 1 if nplayers == 4 else 0,
            }

        res["lobby"] = 0
        res["dan"] = [""] * nplayers
        for e in record.head.accounts:
            res["dan"][e.seat] = cfg["level_definition"]["level_definition"]["map_"][str(e.level.id)]["full_name_jp"]

        res["rate"] = [0] * nplayers
        for e in record.head.accounts:
            res["rate"][e.seat] = e.level.score

        res["sx"] = ["C"] * nplayers
        res["name"] = ["AI"] * nplayers
        for e in record.head.accounts:
            res["name"][e.seat] = e.nickname

        # Keep account ids by seat so share-link ``_a…`` can map to actor later.
        res["account_ids"] = [0] * nplayers
        for e in record.head.accounts:
            res["account_ids"][e.seat] = int(getattr(e, "account_id", 0) or 0)
        if target_account_id is not None:
            for e in record.head.accounts:
                if int(getattr(e, "account_id", 0) or 0) == int(target_account_id):
                    res["_target_actor"] = int(e.seat)
                    break

        scores = [[e.seat, e.part_point_1, e.total_point / 1000] for e in record.head.result.players]
        res["sc"] = [0] * nplayers * 2
        for e in scores:
            res["sc"][2 * e[0]] = e[1]
            res["sc"][2 * e[0] + 1] = e[2]

        res["title"] = [ruledisp + lobby, datetime.fromtimestamp(record.head.end_time).strftime("%Y-%m-%d %H:%M:%S")]

        wrapper = pb.Wrapper()
        wrapper.ParseFromString(record.data)
        details = pb.GameDetailRecords()
        details.ParseFromString(wrapper.data)

        converter = MajsoulPaipuParser(tsumoloss_off=tsumoloss_off)
        records = details.records if details.version < 210715 and details.records else None
        if records:
            for rec in records:
                round_record_wrapper = pb.Wrapper()
                round_record_wrapper.ParseFromString(rec)
                log = getattr(pb, round_record_wrapper.name[len(".lq.") :])()
                log.ParseFromString(round_record_wrapper.data)
                converter.feed(log)
                res["log"] = [e.dump() for e in converter.getvalue()]
        else:
            for act in details.actions:
                if act.result:
                    round_record_wrapper = pb.Wrapper()
                    round_record_wrapper.ParseFromString(act.result)
                    log = getattr(pb, round_record_wrapper.name[len(".lq.") :])()
                    log.ParseFromString(round_record_wrapper.data)
                    converter.feed(log)
                    res["log"] = [e.dump() for e in converter.getvalue()]

        return res


def _normalize_proxy(proxy: str | None) -> str | None:
    if not proxy:
        return None
    proxy = proxy.strip()
    if proxy.startswith("socks5h://"):
        return "socks5://" + proxy[len("socks5h://") :]
    return proxy


def _route_urls(config: dict) -> list[str]:
    urls: list[str] = []
    for item in config.get("ip", []):
        for key in ("gateways", "region_urls"):
            for entry in item.get(key) or []:
                url = entry.get("url")
                if url:
                    urls.append(url)
    if not urls:
        raise RuntimeError("Cannot find Mahjong Soul gateway route in config.json")
    return urls


async def _detect_endpoint(
    session: aiohttp.ClientSession,
    route_urls: list[str],
    proxy: str | None = None,
) -> str:
    random.shuffle(route_urls)
    failures: list[str] = []

    for route_url in route_urls:
        endpoint_url = route_url + "?service=ws-gateway&protocol=ws&ssl=true"
        try:
            async with session.get(
                endpoint_url,
                proxy=proxy,
                timeout=_HTTP_TIMEOUT,
            ) as res:
                res.raise_for_status()
                content_type = res.headers.get("Content-Type", "")
                if "json" not in content_type.lower():
                    body = await res.text()
                    failures.append(
                        f"{route_url}: expected JSON, got {content_type}: {body[:120]!r}"
                    )
                    continue
                servers_res = await res.json()
        except (TimeoutError, ClientError, ClientResponseError) as exc:
            failures.append(f"{route_url}: {type(exc).__name__}: {exc}")
            continue

        if servers_res.get("servers"):
            server = random.choice(servers_res["servers"])
            return f"wss://{server}/gateway"

        failures.append(f"{route_url}: missing servers in response {servers_res!r}")

    raise RuntimeError("Cannot detect endpoint. Tried routes: " + "; ".join(failures))


def _route_id_from_endpoint(endpoint: str) -> str:
    match = _ROUTE_ID_RE.search(endpoint)
    if match:
        return f"route-{match.group(1)}"
    return "route-5"


async def _detect_product_version(
    session: aiohttp.ClientSession,
    host: str,
    proxy: str | None = None,
) -> str:
    async with session.get(
        f"{host}/1/",
        proxy=proxy,
        timeout=_HTTP_TIMEOUT,
    ) as res:
        html = await res.text()

    match = re.search(r'productVersion:\s*"([^"]+)"', html)
    if match:
        return match.group(1)

    match = re.search(r"release-([0-9]+(?:\.[0-9]+)+)", html)
    if match:
        return match.group(1)

    return "4.0.45"


def _load_cached_res_version() -> str | None:
    try:
        text = _RES_VERSION_CACHE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if re.fullmatch(r"\d+\.\d+\.\d+", text):
        return text
    return None


def _save_cached_res_version(res_version: str) -> None:
    try:
        _RES_VERSION_CACHE.write_text(res_version + "\n", encoding="utf-8")
    except OSError:
        pass
