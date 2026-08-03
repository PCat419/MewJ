"""Select the local or server HTML report renderer.

The review/algorithm pipeline is shared by both platforms. Only presentation and
Tenhou viewer asset handling differ between these backends.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def report_mode() -> str:
    """Return the configured report backend: local or server."""
    mode = os.environ.get("MEWJ_REPORT_MODE", "local").strip().lower()
    if mode not in {"local", "server"}:
        raise ValueError(
            f"Invalid MEWJ_REPORT_MODE={mode!r}; expected 'local' or 'server'"
        )
    return mode


def _backend():
    if report_mode() == "server":
        from . import report_server

        return report_server
    from . import report_local

    return report_local


def render_classic_html(*args: Any, **kwargs: Any) -> str:
    """Render a report with the selected presentation backend."""
    return _backend().render_classic_html(*args, **kwargs)


def write_report(report: dict, path: Path) -> Path:
    """Write a report with the selected presentation backend."""
    return _backend().write_report(report, path)
