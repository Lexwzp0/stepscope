"""stepscope CLI (blueprint §九): ingest | capture | observe | analyze | report | compare."""
from __future__ import annotations

import argparse
import sys

COMMANDS = ("ingest", "capture", "observe", "analyze", "report", "compare")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stepscope")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="register a training output directory as a Run").add_argument("path")
    sub.add_parser("capture", help="attach capture hooks to a training run (optional)").add_argument("config")
    s = sub.add_parser("observe", help="run probe sets on every checkpoint"); s.add_argument("run"); s.add_argument("--probes", nargs="+", required=True)
    s = sub.add_parser("analyze", help="run a study or named analyses"); s.add_argument("run"); s.add_argument("--study"); s.add_argument("--analyses", nargs="*")
    s = sub.add_parser("report", help="render the static HTML report"); s.add_argument("run"); s.add_argument("--out", default="report.html")
    s = sub.add_parser("compare", help="compare runs on the same probes and criteria"); s.add_argument("runs", nargs="+")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(f"stepscope {args.cmd}: not implemented yet (skeleton)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
