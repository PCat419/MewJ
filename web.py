"""MewJ local web UI — paste Majsoul link → review → HTML report.

  python web.py
  # then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import contextlib
import io
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .pipeline import MEWJ_ROOT, OUT_DIR, extract_paipu_ref, load_dotenv, run_pipeline

ASSETS_DIR = MEWJ_ROOT / "assets"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# 检讨成功并打开报告后，稍等前端收到 done 再退出
SHUTDOWN_AFTER_DONE_SEC = 1.5
# 输入页关闭后多久退出；启动后等待首屏打开的宽限
IDLE_TIMEOUT_SEC = 12.0
START_GRACE_SEC = 45.0

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_JOB_ID: Optional[str] = None
_HTTP_SERVER: Optional[ThreadingHTTPServer] = None
_CLIENT_LOCK = threading.Lock()
_LAST_CLIENT_AT = 0.0
_STARTED_AT = 0.0


def _touch_client() -> None:
    global _LAST_CLIENT_AT
    with _CLIENT_LOCK:
        _LAST_CLIENT_AT = time.time()


def _has_active_job() -> bool:
    with _JOB_LOCK:
        if _ACTIVE_JOB_ID is None:
            return False
        job = _JOBS.get(_ACTIVE_JOB_ID)
        return bool(job and job["status"] in ("queued", "running"))


def _request_shutdown(reason: str) -> None:
    server = _HTTP_SERVER
    if server is None:
        return
    print(reason, flush=True)

    def _stop() -> None:
        try:
            server.shutdown()
        except Exception:
            pass

    threading.Thread(target=_stop, daemon=True).start()


def _watchdog_loop() -> None:
    """Exit when the input page is gone (no heartbeat) and no review is running."""
    while True:
        time.sleep(1.5)
        if _HTTP_SERVER is None:
            return
        if _has_active_job():
            continue
        now = time.time()
        with _CLIENT_LOCK:
            last = _LAST_CLIENT_AT
            started = _STARTED_AT
        if last <= 0:
            if started > 0 and now - started > START_GRACE_SEC:
                _request_shutdown("未打开页面，自动退出。")
                return
            continue
        if now - last > IDLE_TIMEOUT_SEC:
            _request_shutdown("页面已关闭，自动退出。")
            return


def _open_report_and_exit(path: Path) -> None:
    """Open finished HTML via file:// then stop the local server."""
    import webbrowser

    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception as exc:
        print(f"打开报告失败: {exc}", flush=True)

    def _later() -> None:
        time.sleep(SHUTDOWN_AFTER_DONE_SEC)
        _request_shutdown("检讨完成，服务已退出。")

    threading.Thread(target=_later, daemon=True).start()


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]")
_TOTAL_RE = re.compile(r"共\s*(\d+)\s*个决策点")


class _LogCapture(io.TextIOBase):
    """Tee stdout lines into a job log list; parse ``[n/total]`` progress."""

    def __init__(self, original: Any, job: dict[str, Any]) -> None:
        super().__init__()
        self._original = original
        self._job = job
        self._buf = ""

    def writable(self) -> bool:
        return True

    def _ingest_line(self, text: str) -> None:
        job = self._job
        with job["lock"]:
            job["logs"].append(text)
            m = _PROGRESS_RE.search(text)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    job["progress_done"] = done
                    job["progress_total"] = total
                    job["progress"] = min(100, int(round(100.0 * done / total)))
                return
            m2 = _TOTAL_RE.search(text)
            if m2:
                total = int(m2.group(1))
                if total > 0 and not job.get("progress_total"):
                    job["progress_total"] = total
                    job["progress_done"] = 0
                    job["progress"] = 0

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            text = line.rstrip("\r")
            if text:
                self._ingest_line(text)
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        if self._buf.strip():
            self._ingest_line(self._buf.rstrip("\r\n"))
            self._buf = ""


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    with job["lock"]:
        return {
            "id": job["id"],
            "status": job["status"],
            "source": job["source"],
            "seat": job["seat"],
            "logs": list(job["logs"]),
            "error": job["error"],
            "report_url": job["report_url"],
            "progress": job.get("progress"),
            "progress_done": job.get("progress_done"),
            "progress_total": job.get("progress_total"),
            "created_at": job["created_at"],
            "finished_at": job["finished_at"],
        }


def _run_job(job_id: str) -> None:
    global _ACTIVE_JOB_ID
    job = _JOBS[job_id]
    with job["lock"]:
        job["status"] = "running"
        job["logs"].append("开始检讨…")

    source = job["source"]
    seat = job["seat"]
    is_link = bool(re.search(r"https?://|paipu=", source, re.I))

    capture = _LogCapture(sys.stdout, job)
    try:
        with contextlib.redirect_stdout(capture):
            out = run_pipeline(
                source,
                seat,
                local_uuid=not is_link,
            )
        capture.flush()
        rel = out.resolve().relative_to(MEWJ_ROOT.resolve()).as_posix()
        with job["lock"]:
            job["status"] = "done"
            job["report_url"] = "/" + rel
            job["progress"] = 100
            if job.get("progress_total"):
                job["progress_done"] = job["progress_total"]
            job["finished_at"] = time.time()
            job["logs"].append(f"完成: {rel}")
        _open_report_and_exit(out)
    except Exception as exc:
        capture.flush()
        with job["lock"]:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            job["logs"].append(f"错误: {exc}")
            tb = traceback.format_exc().strip()
            if tb:
                for line in tb.splitlines()[-8:]:
                    job["logs"].append(line)
    finally:
        with _JOB_LOCK:
            if _ACTIVE_JOB_ID == job_id:
                _ACTIVE_JOB_ID = None


def _start_job(source: str, seat: Optional[int]) -> tuple[Optional[dict], Optional[str], int]:
    """Return (snapshot, error_message, http_status)."""
    global _ACTIVE_JOB_ID
    source = (source or "").strip()
    if not source:
        return None, "请输入牌谱链接或 UUID", HTTPStatus.BAD_REQUEST

    try:
        extract_paipu_ref(source)
    except ValueError as exc:
        return None, str(exc), HTTPStatus.BAD_REQUEST

    if seat is not None and seat not in (0, 1, 2, 3):
        return None, "seat 必须是 0–3 或留空自动识别", HTTPStatus.BAD_REQUEST

    with _JOB_LOCK:
        if _ACTIVE_JOB_ID is not None:
            active = _JOBS.get(_ACTIVE_JOB_ID)
            if active and active["status"] in ("queued", "running"):
                return None, "已有检讨在进行中，请稍后再试", HTTPStatus.CONFLICT

        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "id": job_id,
            "status": "queued",
            "source": source,
            "seat": seat,
            "logs": [],
            "error": None,
            "report_url": None,
            "progress": None,
            "progress_done": None,
            "progress_total": None,
            "created_at": time.time(),
            "finished_at": None,
            "lock": threading.Lock(),
        }
        _JOBS[job_id] = job
        _ACTIVE_JOB_ID = job_id

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return _job_snapshot(job), None, HTTPStatus.OK


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MewJ 牌谱检讨</title>
<style>
:root {
  --ink:#16352c;
  --muted:#6b7f77;
  --line:rgba(22,53,44,.12);
  --accent:#1f8a6a;
  --accent-2:#2bb888;
  --danger:#b42318;
  --card:rgba(255,255,255,.82);
  --shadow:0 18px 50px rgba(18,48,40,.12);
}
* { box-sizing:border-box; }
html, body { height:100%; }
body {
  margin:0;
  min-height:100%;
  color:var(--ink);
  font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  background:
    radial-gradient(900px 480px at 12% -10%, rgba(43,184,136,.22), transparent 60%),
    radial-gradient(700px 420px at 92% 8%, rgba(31,107,87,.16), transparent 55%),
    linear-gradient(165deg, #f3f8f5 0%, #e7f0eb 48%, #dfeae4 100%);
  display:flex;
  align-items:center;
  justify-content:center;
  padding:1.5rem 1rem;
}
.shell {
  width:min(440px, 100%);
  background:var(--card);
  backdrop-filter:blur(14px);
  -webkit-backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,.7);
  border-radius:22px;
  box-shadow:var(--shadow);
  padding:2rem 1.6rem 1.55rem;
}
.brand {
  margin:0 0 1.45rem;
  text-align:center;
}
.brand h1 {
  margin:0;
  font-size:1.7rem;
  font-weight:750;
  letter-spacing:.04em;
  color:var(--ink);
}
label {
  display:block;
  margin:0 0 .4rem;
  font-size:.82rem;
  font-weight:650;
  color:var(--muted);
  letter-spacing:.02em;
}
.field { margin:0 0 .95rem; }
input[type=text], select {
  width:100%;
  height:2.75rem;
  padding:0 .9rem;
  border:1px solid var(--line);
  border-radius:12px;
  background:#fff;
  color:var(--ink);
  font:inherit;
  outline:none;
  transition:border-color .15s, box-shadow .15s;
}
input[type=text]::placeholder { color:#9aada5; }
input[type=text]:focus, select:focus {
  border-color:rgba(31,138,106,.55);
  box-shadow:0 0 0 3px rgba(43,184,136,.18);
}
.actions { margin-top:1.15rem; }
button {
  appearance:none;
  width:100%;
  height:2.85rem;
  border:none;
  border-radius:12px;
  cursor:pointer;
  font:inherit;
  font-weight:700;
  letter-spacing:.03em;
  color:#fff;
  background:linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow:0 10px 22px rgba(31,138,106,.28);
  transition:transform .12s ease, filter .12s ease, opacity .12s ease;
}
button:hover { filter:brightness(1.04); }
button:active { transform:translateY(1px); }
button:disabled { opacity:.55; cursor:not-allowed; filter:none; transform:none; }
button.ghost {
  margin-top:.65rem;
  background:#fff;
  color:var(--ink);
  border:1px solid var(--line);
  box-shadow:none;
  font-weight:600;
}
#progress {
  display:none;
  margin-top:1.15rem;
}
#progress.show { display:block; }
.progress-meta {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin:0 0 .45rem;
  font-size:.82rem;
  color:var(--muted);
}
.bar {
  height:8px;
  border-radius:999px;
  background:rgba(22,53,44,.08);
  overflow:hidden;
}
.bar > i {
  display:block;
  height:100%;
  width:0%;
  border-radius:inherit;
  background:linear-gradient(90deg, var(--accent), var(--accent-2));
  transition:width .35s ease;
}
.bar.indeterminate > i {
  width:35%;
  transition:none;
  animation:slide 1.15s ease-in-out infinite;
}
.bar.done > i {
  width:100%;
  animation:none;
}
.bar.error > i {
  width:100%;
  animation:none;
  background:#f04438;
}
@keyframes slide {
  0% { transform:translateX(-120%); }
  100% { transform:translateX(320%); }
}
#error {
  display:none;
  margin-top:.75rem;
  padding:.7rem .8rem;
  border-radius:10px;
  background:rgba(180,35,24,.08);
  color:var(--danger);
  font-size:.86rem;
  line-height:1.4;
}
#error.show { display:block; }
@media (max-width:480px) {
  .shell { padding:1.55rem 1.15rem 1.25rem; border-radius:18px; }
  .brand h1 { font-size:1.45rem; }
}
</style>
</head>
<body>
  <div class="shell">
    <div class="brand"><h1>MewJ 牌谱检讨</h1></div>
    <form id="form" autocomplete="off">
      <div class="field">
        <label for="source">牌谱链接</label>
        <input id="source" name="source" type="text" required
          placeholder="雀魂分享链接或 UUID"/>
      </div>
      <div class="field">
        <label for="seat">座位</label>
        <select id="seat" name="seat">
          <option value="">自动识别</option>
          <option value="0">东起</option>
          <option value="1">南起</option>
          <option value="2">西起</option>
          <option value="3">北起</option>
        </select>
      </div>
      <div class="actions">
        <button type="submit" id="submit">开始检讨</button>
        <button type="button" class="ghost" id="reset" hidden>再检讨一盘</button>
      </div>
    </form>
    <div id="progress">
      <div class="progress-meta">
        <span id="progress-label">检讨中</span>
        <span id="progress-pct"></span>
      </div>
      <div class="bar indeterminate" id="bar"><i id="bar-fill"></i></div>
      <div id="error"></div>
    </div>
  </div>
<script>
(function () {
  // 页面存活心跳：关掉输入页后服务自动退出
  function beat() {
    try {
      navigator.sendBeacon('/api/heartbeat');
    } catch (e) {
      fetch('/api/heartbeat', { method: 'POST', keepalive: true }).catch(function () {});
    }
  }
  beat();
  setInterval(beat, 3000);

  var form = document.getElementById('form');
  var sourceEl = document.getElementById('source');
  var seatEl = document.getElementById('seat');
  var submitBtn = document.getElementById('submit');
  var resetBtn = document.getElementById('reset');
  var progress = document.getElementById('progress');
  var bar = document.getElementById('bar');
  var barFill = document.getElementById('bar-fill');
  var label = document.getElementById('progress-label');
  var pctEl = document.getElementById('progress-pct');
  var errEl = document.getElementById('error');
  var pollTimer = null;

  function setBusy(busy) {
    submitBtn.disabled = busy;
    sourceEl.disabled = busy;
    seatEl.disabled = busy;
    resetBtn.hidden = busy;
  }

  function showError(msg) {
    if (!msg) {
      errEl.classList.remove('show');
      errEl.textContent = '';
      return;
    }
    errEl.textContent = msg;
    errEl.classList.add('show');
  }

  function setProgress(job) {
    var state = typeof job === 'string' ? job : job.status;
    var pct = (job && typeof job === 'object') ? job.progress : null;
    progress.classList.add('show');
    bar.className = 'bar';
    if (state === 'running' || state === 'queued') {
      showError('');
      if (pct == null) {
        bar.classList.add('indeterminate');
        barFill.style.width = '';
        label.textContent = '准备中';
        pctEl.textContent = '';
      } else {
        var p = Math.max(0, Math.min(100, Number(pct) || 0));
        barFill.style.width = p + '%';
        label.textContent = '检讨中';
        pctEl.textContent = p + '%';
      }
    } else if (state === 'done') {
      bar.classList.add('done');
      barFill.style.width = '100%';
      label.textContent = '完成，报告已打开';
      pctEl.textContent = '100%';
      showError('');
    } else if (state === 'error') {
      bar.classList.add('error');
      barFill.style.width = '100%';
      label.textContent = '失败';
      pctEl.textContent = '';
    }
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function poll(jobId) {
    stopPoll();
    function tick() {
      fetch('/api/jobs/' + encodeURIComponent(jobId))
        .then(function (r) {
          if (!r.ok) throw new Error('任务不存在');
          return r.json();
        })
        .then(function (job) {
          setProgress(job);
          if (job.status === 'done') {
            stopPoll();
            setBusy(false);
            // 报告已用系统浏览器打开；关掉本输入页
            setTimeout(function () {
              window.close();
              // 部分浏览器禁止脚本关标签：退化为空白提示页
              document.documentElement.innerHTML =
                '<head><meta charset="utf-8"><title>MewJ</title></head>' +
                '<body style="margin:0;min-height:100vh;display:flex;align-items:center;' +
                'justify-content:center;font-family:Segoe UI,PingFang SC,Microsoft YaHei,sans-serif;' +
                'background:#e7f0eb;color:#6b7f77;font-size:.95rem">报告已打开，可关闭此页</body>';
            }, 400);
          } else if (job.status === 'error') {
            stopPoll();
            setBusy(false);
            resetBtn.hidden = false;
            showError(job.error || '检讨失败');
          }
        })
        .catch(function (e) {
          setProgress('error');
          showError(String(e.message || e));
          stopPoll();
          setBusy(false);
          resetBtn.hidden = false;
        });
    }
    tick();
    pollTimer = setInterval(tick, 800);
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var source = sourceEl.value.trim();
    if (!source) return;
    var seatVal = seatEl.value;
    var body = { source: source };
    if (seatVal !== '') body.seat = parseInt(seatVal, 10);

    setBusy(true);
    resetBtn.hidden = true;
    setProgress({ status: 'running', progress: null });

    fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(function (r) {
        return r.json().then(function (data) {
          return { ok: r.ok, status: r.status, data: data };
        });
      })
      .then(function (res) {
        if (!res.ok) {
          throw new Error((res.data && res.data.error) || ('HTTP ' + res.status));
        }
        var job = res.data.job || res.data;
        setProgress(job);
        poll(job.id);
      })
      .catch(function (err) {
        setBusy(false);
        resetBtn.hidden = false;
        setProgress('error');
        showError(String(err.message || err));
      });
  });

  resetBtn.addEventListener('click', function () {
    stopPoll();
    showError('');
    progress.classList.remove('show');
    bar.className = 'bar indeterminate';
    barFill.style.width = '';
    label.textContent = '检讨中';
    pctEl.textContent = '';
    resetBtn.hidden = true;
    setBusy(false);
    sourceEl.focus();
  });
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _safe_under(root: Path, url_path: str) -> Optional[Path]:
    """Resolve url_path relative to root; reject path traversal."""
    rel = unquote(url_path).lstrip("/").replace("\\", "/")
    if ".." in rel.split("/"):
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class Handler(BaseHTTPRequestHandler):
    server_version = "MewJWeb/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str, *, extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj: Any) -> None:
        self._send(code, _json_bytes(obj), "application/json; charset=utf-8")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _serve_file(self, path: Path) -> None:
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix.lower() in (".html", ".htm"):
            ctype = "text/html; charset=utf-8"
        data = path.read_bytes()
        self._send(HTTPStatus.OK, data, ctype)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            _touch_client()
            self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/heartbeat":
            _touch_client()
            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        m = re.fullmatch(r"/api/jobs/([0-9a-fA-F]+)", path)
        if m:
            _touch_client()
            job = _JOBS.get(m.group(1))
            if not job:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            self._send_json(HTTPStatus.OK, _job_snapshot(job))
            return

        if path.startswith("/out/"):
            file_path = _safe_under(OUT_DIR, path[len("/out/") :])
            if file_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._serve_file(file_path)
            return

        if path.startswith("/assets/"):
            file_path = _safe_under(ASSETS_DIR, path[len("/assets/") :])
            if file_path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(file_path)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/heartbeat":
            # sendBeacon may post an empty body
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                self.rfile.read(length)
            _touch_client()
            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        if path != "/api/review":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            payload = self._read_json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return

        _touch_client()
        source = str(payload.get("source") or payload.get("link") or "").strip()
        seat_raw = payload.get("seat", None)
        seat: Optional[int]
        if seat_raw is None or seat_raw == "":
            seat = None
        else:
            try:
                seat = int(seat_raw)
            except (TypeError, ValueError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "seat 必须是整数 0–3"})
                return

        snap, err, code = _start_job(source, seat)
        if err:
            self._send_json(code, {"error": err})
            return
        self._send_json(code, {"job": snap})


def main(argv: Optional[list[str]] = None) -> int:
    global _HTTP_SERVER, _STARTED_AT

    load_dotenv(MEWJ_ROOT / ".env", MEWJ_ROOT.parent / "tensoul" / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    host = os.environ.get("MEWJ_WEB_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port_s = os.environ.get("MEWJ_WEB_PORT", str(DEFAULT_PORT)).strip()
    try:
        port = int(port_s)
    except ValueError:
        print(f"无效 MEWJ_WEB_PORT: {port_s!r}", file=sys.stderr)
        return 2

    server = ThreadingHTTPServer((host, port), Handler)
    _HTTP_SERVER = server
    _STARTED_AT = time.time()
    url = f"http://{host}:{port}/"
    print(f"MewJ Web 已启动: {url}", flush=True)
    print("关闭输入页或检讨完成后将自动退出；也可按 Ctrl+C 停止。", flush=True)

    threading.Thread(target=_watchdog_loop, daemon=True).start()

    def _open_browser() -> None:
        import webbrowser

        time.sleep(0.15)
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"打开浏览器失败: {exc}", flush=True)

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    finally:
        _HTTP_SERVER = None
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
