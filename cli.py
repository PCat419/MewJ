"""MewJ CLI — Classic-style paipu review (tile-efficiency).

Examples:
  python -m MewJ.cli --link "https://game.maj-soul.com/1/?paipu=...._a123" --seat 0
  python -m MewJ.cli paipu/xxx.json --seat 0 --kyoku 0
  python -m MewJ.cli xxx.json --seat 0 --structure-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # 允许在 MewJ 文件夹内直接 `python cli.py` 运行（包名取实际文件夹名）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .pipeline import MEWJ_ROOT, load_dotenv, run_pipeline
from .review import DEFAULT_NANIKIRU


def main(argv: list[str] | None = None) -> int:
    load_dotenv(MEWJ_ROOT / ".env", MEWJ_ROOT.parent / "tensoul" / ".env")

    parser = argparse.ArgumentParser(
        description="MewJ Classic paipu review (JSON file or Majsoul link)"
    )
    parser.add_argument(
        "paipu",
        nargs="?",
        default=None,
        help="tenhou.net/6 JSON file path (optional if --link is set)",
    )
    parser.add_argument(
        "--link",
        "-l",
        default=None,
        help="Majsoul paipu share URL or UUID (downloads via tensoul, cached in MewJ/paipu/)",
    )
    parser.add_argument("--seat", type=int, default=0, choices=range(4), help="player seat 0-3")
    parser.add_argument(
        "--kyoku",
        type=int,
        action="append",
        dest="kyokus",
        help="kyoku index to review (repeatable); default all",
    )
    parser.add_argument("--max-turns", type=int, default=None, help="cap analyzed turns")
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="only list turns / actual discards, skip nanikiru",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download even if MewJ/paipu/<uuid>.json exists",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("NANIKIRU_URL", DEFAULT_NANIKIRU),
        help="nanikiru URL",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output HTML path (default: MewJ/out/<ref>_seatN.html)",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="open the generated HTML in the default browser",
    )
    args = parser.parse_args(argv)

    if args.link and args.paipu:
        print("请只指定 --link 或 JSON 路径之一", file=sys.stderr)
        return 2
    if not args.link and not args.paipu:
        parser.print_help()
        print("\n需要提供 JSON 路径或 --link", file=sys.stderr)
        return 2

    source = args.link if args.link else str(args.paipu)

    try:
        out = run_pipeline(
            source,
            args.seat,
            kyoku_indices=args.kyokus,
            max_turns=args.max_turns,
            nanikiru_url=args.url,
            structure_only=args.structure_only,
            force_download=args.force_download,
            output=args.output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1

    if args.open_browser:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
