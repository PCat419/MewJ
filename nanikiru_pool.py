"""Nanikiru worker pool: multiple local engines + URL round-robin.

Wall-clock review time is dominated by mahjong-cpp (nanikiru). A single
process typically serves one request at a time, so Python threads alone do
not help. This pool starts N engines on consecutive ports and hands out
URLs round-robin. Crash recovery kills only the failing worker PID.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_NANIKIRU = "http://127.0.0.1:50000"


def _mewj_root() -> Path:
    return Path(__file__).resolve().parent


def default_nanikiru_exe() -> Path:
    """Resolve nanikiru.exe for portable MewJ runtime.

    Only ``MEWJ_NANIKIRU_EXE`` or ``MewJ/engine/nanikiru.exe`` — never
    ``../mahjong-cpp/build`` (MewJ must run without sibling source trees).
    """
    env = os.environ.get("MEWJ_NANIKIRU_EXE")
    if env:
        return Path(env)
    return _mewj_root() / "engine" / "nanikiru.exe"


def nanikiru_port(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port:
        return int(parsed.port)
    return 50000


def nanikiru_host(url: str) -> str:
    return urlparse(url).hostname or "127.0.0.1"


def make_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def nanikiru_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        # GET is rejected with 400 by nanikiru; any HTTP response means alive.
        requests.get(url, timeout=timeout)
        return True
    except requests.exceptions.HTTPError:
        return True
    except requests.exceptions.RequestException:
        return False


def resolve_workers(explicit: Optional[int] = None) -> int:
    """CLI/env/params → worker count (clamped to 1..16)."""
    if explicit is not None:
        n = int(explicit)
    else:
        env = os.environ.get("MEWJ_WORKERS", "").strip()
        if env:
            n = int(env)
        else:
            try:
                from .params import PARAMS as _P

                n = int((_P.get("runtime") or {}).get("workers") or 4)
            except Exception:
                n = 4
    return max(1, min(16, n))


def _child_env(exe_path: Path) -> dict:
    env = dict(os.environ)
    candidates = [
        exe_path.parent,
        Path(r"C:\msys64\mingw64\bin"),
        Path(r"C:\msys64\ucrt64\bin"),
    ]
    prepend = [str(p) for p in candidates if p.is_dir()]
    env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    return env


def _pid_listening_on_port(port: int) -> Optional[int]:
    """Best-effort PID lookup for a TCP listen port (Windows / Unix)."""
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
                    f"-ErrorAction SilentlyContinue | Select-Object -First 1 "
                    f"-ExpandProperty OwningProcess)",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            text = (r.stdout or "").strip()
            if text.isdigit():
                return int(text)
        except Exception:
            return None
        return None
    try:
        r = subprocess.run(
            ["lsof", "-ti", f"TCP:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        text = (r.stdout or "").strip().splitlines()
        if text and text[0].isdigit():
            return int(text[0])
    except Exception:
        return None
    return None


def _terminate_pid(pid: int, timeout: float = 3.0) -> None:
    if pid <= 0:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.kill(pid, 15)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    return
                time.sleep(0.1)
            os.kill(pid, 9)
    except Exception:
        pass


@dataclass
class WorkerSlot:
    url: str
    port: int
    proc: Optional[subprocess.Popen] = None
    owned: bool = False
    log_handle: Any = None


class NanikiruPool:
    """Own N nanikiru processes (or adopt already-running listeners)."""

    def __init__(
        self,
        base_url: str = DEFAULT_NANIKIRU,
        workers: int = 1,
        exe: Optional[Path] = None,
    ) -> None:
        self.base_url = base_url
        self.workers = max(1, int(workers))
        self.exe = Path(exe) if exe else default_nanikiru_exe()
        self.host = nanikiru_host(base_url)
        self.base_port = nanikiru_port(base_url)
        self._slots: List[WorkerSlot] = []
        self._rr = 0
        self._lock = threading.Lock()
        self._log_path = _mewj_root() / "out" / "_nanikiru.log"
        self._atexit_registered = False

    def __enter__(self) -> "NanikiruPool":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def start(self) -> None:
        if self._slots:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        pending: List[Tuple[int, str, Optional[WorkerSlot], bool]] = []
        try:
            # Phase 1: adopt live listeners / fire all spawns without waiting.
            for i in range(self.workers):
                port = self.base_port + i
                url = make_url(self.host, port)
                if nanikiru_reachable(url):
                    pending.append(
                        (
                            port,
                            url,
                            WorkerSlot(url=url, port=port, proc=None, owned=False),
                            True,
                        )
                    )
                    continue
                slot = self._spawn_slot(port, url, wait=False)
                if slot is None:
                    raise RuntimeError(
                        f"无法启动 nanikiru worker #{i} @ {url}\n"
                        f"请检查 {self.exe} 或设置 MEWJ_NANIKIRU_EXE"
                    )
                pending.append((port, url, slot, False))

            # Phase 2: wait for freshly spawned workers in parallel.
            to_wait = [
                (p, u, s) for p, u, s, ready in pending if not ready and s is not None
            ]
            if to_wait:
                with ThreadPoolExecutor(max_workers=len(to_wait)) as ex:
                    futs = {
                        ex.submit(self._wait_reachable, s, url): (port, url, s)
                        for port, url, s in to_wait
                    }
                    for fut in futs:
                        port, url, s = futs[fut]
                        if not fut.result():
                            self._kill_slot(s)
                            raise RuntimeError(
                                f"nanikiru worker @ {url} 启动超时\n"
                                f"请检查 {self.exe} 或设置 MEWJ_NANIKIRU_EXE"
                            )

            self._slots = [s for _p, _u, s, _r in pending if s is not None]
        except Exception:
            for _p, _u, s, _r in pending:
                if s is not None and s.owned:
                    self._kill_slot(s)
            self._slots = []
            raise
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True
        alive = sum(1 for s in self._slots if nanikiru_reachable(s.url))
        print(
            f"nanikiru 池: {alive}/{len(self._slots)} workers "
            f"(ports {self.base_port}–{self.base_port + len(self._slots) - 1})",
            flush=True,
        )

    def _wait_reachable(self, slot: WorkerSlot, url: str, attempts: int = 40) -> bool:
        for _ in range(attempts):
            time.sleep(0.15)
            if nanikiru_reachable(url):
                return True
            if slot.proc is not None and slot.proc.poll() is not None:
                return False
        return False

    def _kill_slot(self, slot: WorkerSlot) -> None:
        if slot.proc is not None and slot.proc.poll() is None:
            try:
                slot.proc.terminate()
                slot.proc.wait(timeout=2)
            except Exception:
                try:
                    slot.proc.kill()
                except Exception:
                    pass
        if slot.log_handle is not None:
            try:
                slot.log_handle.close()
            except Exception:
                pass
            slot.log_handle = None
        slot.proc = None
        slot.owned = False

    def _spawn_slot(
        self, port: int, url: str, *, wait: bool = True
    ) -> Optional[WorkerSlot]:
        if not self.exe.is_file():
            print(f"  [warn] nanikiru.exe not found: {self.exe}", flush=True)
            return None
        log_handle = open(self._log_path, "a", encoding="utf-8")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        try:
            proc = subprocess.Popen(
                [str(self.exe), str(port)],
                cwd=str(self.exe.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=_child_env(self.exe),
                creationflags=creationflags,
            )
        except Exception as exc:
            try:
                log_handle.close()
            except Exception:
                pass
            print(f"  [warn] spawn nanikiru :{port} failed: {exc}", flush=True)
            return None
        slot = WorkerSlot(
            url=url, port=port, proc=proc, owned=True, log_handle=log_handle
        )
        if not wait:
            return slot
        if self._wait_reachable(slot, url):
            return slot
        self._kill_slot(slot)
        return None

    def urls(self) -> List[str]:
        return [s.url for s in self._slots]

    def next_url(self) -> str:
        with self._lock:
            if not self._slots:
                return self.base_url
            slot = self._slots[self._rr % len(self._slots)]
            self._rr += 1
            return slot.url

    def restart_url(self, url: str) -> bool:
        """Restart the worker that serves ``url`` (PID-scoped, not taskkill /IM)."""
        with self._lock:
            slot = next((s for s in self._slots if s.url == url), None)
            if slot is None:
                port = nanikiru_port(url)
                pid = _pid_listening_on_port(port)
                if pid:
                    _terminate_pid(pid)
                time.sleep(0.4)
                new = self._spawn_slot(port, url)
                return bool(new and nanikiru_reachable(url))

            if slot.proc is not None and slot.proc.poll() is None:
                try:
                    slot.proc.terminate()
                    slot.proc.wait(timeout=3)
                except Exception:
                    try:
                        slot.proc.kill()
                    except Exception:
                        pass
            else:
                pid = _pid_listening_on_port(slot.port)
                if pid:
                    _terminate_pid(pid)
            if slot.log_handle is not None:
                try:
                    slot.log_handle.close()
                except Exception:
                    pass
                slot.log_handle = None
            slot.proc = None
            slot.owned = False

            time.sleep(0.4)
            new = self._spawn_slot(slot.port, slot.url)
            if new is None:
                return False
            slot.proc = new.proc
            slot.owned = new.owned
            slot.log_handle = new.log_handle
            return nanikiru_reachable(slot.url)

    def ensure_any(self) -> bool:
        return any(nanikiru_reachable(s.url) for s in self._slots)

    def shutdown(self) -> None:
        with self._lock:
            slots = list(self._slots)
            self._slots = []
        for slot in slots:
            if not slot.owned:
                continue
            if slot.proc is not None and slot.proc.poll() is None:
                try:
                    slot.proc.terminate()
                    slot.proc.wait(timeout=3)
                except Exception:
                    try:
                        slot.proc.kill()
                    except Exception:
                        pass
            if slot.log_handle is not None:
                try:
                    slot.log_handle.close()
                except Exception:
                    pass


_ACTIVE: Optional[NanikiruPool] = None
_ACTIVE_LOCK = threading.Lock()


def set_active_pool(pool: Optional[NanikiruPool]) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = pool


def get_active_pool() -> Optional[NanikiruPool]:
    with _ACTIVE_LOCK:
        return _ACTIVE


def pick_url(fallback: str) -> str:
    pool = get_active_pool()
    if pool is None:
        return fallback
    return pool.next_url()


def restart_worker(url: str, *, legacy_restart) -> bool:
    """Restart via active pool when possible; else ``legacy_restart(url)``."""
    pool = get_active_pool()
    if pool is not None:
        return pool.restart_url(url)
    return bool(legacy_restart(url))
