"""Renderiza um frame real gravado da demo para SVG, sem nova inferência."""
import argparse
import json
from pathlib import Path

from rich.console import Console
from snake_linux.ui import BG, compose, layout_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=-1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.record.open()]
    frames = [row for row in rows if row["type"] == "frame"]
    frame = frames[args.frame]
    width, height = layout_size(frame["game"]["width"], frame["game"]["height"])
    console = Console(record=True, width=width, height=height, color_system="truecolor", force_terminal=True)
    stats = {**frame["stats"], "replay": True}
    console.print(compose(frame["game"], frame["decision"], stats).rich_text(), end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(args.output), title="Laya Snake Linux, frame real gravado", clear=False)
    print(args.output)


if __name__ == "__main__":
    main()
