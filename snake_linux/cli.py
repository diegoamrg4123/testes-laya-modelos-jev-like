"""Run the Linux/PyTorch adaptation of the Laya Snake terminal demo."""

import argparse
import json
import select
import sys
import termios
import time
import tty
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live

from .alternatives import DeciderAgent, KevAgent, SemIfAgent
from .game import SnakeGame
from .policy import LayaPolicy
from .ui import BG, compose, layout_size


class Keyboard:
    def __enter__(self):
        self.saved = None
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self):
        import os

        if self.saved and select.select([sys.stdin], [], [], 0)[0]:
            return os.read(self.fd, 128).decode(errors="ignore")
        return ""

    def __exit__(self, *_):
        if self.saved:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def positive(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return result


def load_backend(backend, model, *, guarded, prompt, semif_threads, decider_device="cpu"):
    if backend == "laya":
        return LayaPolicy(model, guarded=guarded, prompt=prompt)
    if backend == "decider":
        if not model:
            raise ValueError("--model is required for the decider backend")
        agent = DeciderAgent.from_checkpoint(model, device=decider_device)
        device = str(agent.device or decider_device)
        decider_config = {}
        config_path = Path(model) / "decider_config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as config_file:
                decider_config = json.load(config_file)
        if device.startswith("cuda"):
            import torch
            hardware = torch.cuda.get_device_name(0)
            engine = "decider-ai PyTorch CUDA"
        else:
            hardware = device.upper()
            engine = "decider-ai PyTorch CPU"
        policy = LayaPolicy(model=model, guarded=guarded, prompt=prompt, agent=agent)
        policy.metadata.update(
            name=f"Mapika {agent.name}", engine=engine, hardware=hardware,
            network="offline during play", checkpoint={
                "path": str(model), "name": agent.name,
                "version": decider_config.get("version"),
                "release_date": decider_config.get("release_date"),
            },
            policy="Option probabilities over planner features; optional cycle safety shield",
        )
        return policy
    if backend == "kev":
        if model:
            raise ValueError("--model is only for Laya directories or SemIf GGUF files")
        agent = KevAgent()
        details = agent.describe()
        name, engine, hardware, network = (
            "Kev-0.8B", f"Kev {details['backend']} {details['dtype']}",
            details["device"], "localhost only"
        )
        checkpoint = details
    elif backend == "semif":
        gguf = model or "models/semif/Qwen_Qwen3.5-4B-Q4_K_M.gguf"
        agent = SemIfAgent.from_checkpoint(gguf, "models/semif-tokenizer", threads=semif_threads)
        name, engine, hardware, network = "SemIf-4B Q4_K_M", "llama.cpp CPU", "CPU", "offline"
        checkpoint = agent.metadata
    else:
        raise ValueError(f"Unknown backend: {backend}")
    policy = LayaPolicy(model=model, guarded=guarded, prompt=prompt, agent=agent)
    policy.metadata.update(name=name, engine=engine, hardware=hardware, network=network,
                           checkpoint=checkpoint,
                           policy="Option probabilities over planner features; optional cycle safety shield")
    return policy

def play(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-snake",
        description=__doc__,

    )
    parser.add_argument("--backend", choices=("laya", "decider", "kev", "semif"), default="laya")
    parser.add_argument(
        "--model", help="Laya or decider directory, or SemIf GGUF file; Kev uses the local server"
    )
    parser.add_argument(
        "--decider-device", choices=("cpu", "cuda"), default="cpu",
        help="Device for the decider backend (default: cpu)",
    )
    parser.add_argument("--semif-threads", type=int, default=4, help="CPU threads for SemIf (default: 4)")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")

    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument(
        "--fps", type=positive, default=12, help="Game decisions per second (default: 12)"
    )
    parser.add_argument(
        "--max-speed", action="store_true", help="One move per completed inference, without pacing"
    )
    parser.add_argument(
        "--duration", type=positive, help="Stop after this many seconds, excluding warmup"
    )
    parser.add_argument("--steps", type=int, help="Stop after this many actual game steps")
    parser.add_argument(
        "--unassisted",
        action="store_true",
        help="Execute raw model top-1; disable the safety shield",
    )
    parser.add_argument(
        "--record", type=Path, help="Write timestamped real decisions and board states to JSONL"
    )
    parser.add_argument("--headless", action="store_true", help="Run without a terminal display")
    parser.add_argument(
        "--no-alt-screen", action="store_true", help="Keep the final frame in terminal scrollback"
    )
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.semif_threads < 1:
        parser.error("--semif-threads must be positive")
    if args.backend == "kev" and args.model:
        parser.error("Kev reads its checkpoint through the localhost server, not --model")
    if args.backend == "decider" and not args.model:
        parser.error("--model is required for the decider backend")
    try:
        game = SnakeGame(args.width, args.height, args.seed, args.initial_length)
    except ValueError as error:
        parser.error(str(error))
    console = Console(style=f"on {BG}", highlight=False)
    if not args.headless and not console.is_terminal:
        parser.error("Interactive display needs a TTY. Use --headless for a non-interactive run.")
    print(f"Loading {args.backend} backend...", file=sys.stderr)
    policy = load_backend(args.backend, args.model, guarded=not args.unassisted,
                          prompt=args.prompt, semif_threads=args.semif_threads,
                          decider_device=args.decider_device)
    warm = SnakeGame(args.width, args.height, args.seed + 10000, args.initial_length)
    for _ in range(6):
        decision = policy.decide(warm)
        warm.step(decision.executed)
        if not warm.alive:
            break
    record = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record = args.record.open("x")
        record.write(
            json.dumps(
                {
                    "type": "metadata",
                    "format": "laya-snake-v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": policy.metadata,
                    "settings": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                    "note": "Real synchronized inference; board is shown before the announced action. Risk is 1 - P(safe route).",
                }
            )
            + "\n"
        )
    started = time.perf_counter()
    stats = {
        "hardware": policy.metadata["hardware"],
        "engine": policy.metadata["engine"],
        "name": policy.metadata["name"],
        "network": policy.metadata["network"],
        "guarded": policy.guarded,
        "interventions": 0,
        "best": 0,
        "round": 1,
        "paused": False,
        "elapsed": 0,
        "steps_per_second": 0,
    }
    calls = total_steps = deaths = 0
    timestamps = deque(maxlen=60)
    inference = []
    displayed_board, displayed_decision = game.snapshot(), {}
    live = (
        Live(
            console=console,
            screen=not args.no_alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        if not args.headless
        else None
    )
    quit_requested = False
    try:
        with Keyboard() as keys, live if live else nullcontext():
            while not quit_requested:
                now = time.perf_counter()
                if (args.duration and now - started >= args.duration) or (
                    args.steps and total_steps >= args.steps
                ):
                    break
                pressed = keys.read().lower()
                if "q" in pressed or "\x03" in pressed:
                    break
                if " " in pressed:
                    stats["paused"] = not stats["paused"]
                if "\x1b[a" in pressed or "+" in pressed:
                    args.fps = min(240, args.fps + 2)
                if "\x1b[b" in pressed or "-" in pressed:
                    args.fps = max(1, args.fps - 2)
                if "r" in pressed:
                    stats["round"] += 1
                    game = SnakeGame(
                        args.width, args.height, args.seed + stats["round"] - 1, args.initial_length
                    )
                    displayed_board, displayed_decision = game.snapshot(), {}
                if stats["paused"]:
                    stats["elapsed"] = now - started
                    if live:
                        live.update(
                            compose(displayed_board, displayed_decision, stats).rich_text(),
                            refresh=True,
                        )
                    time.sleep(0.03)
                    continue
                minimum_width, minimum_height = layout_size(game.width, game.height)
                if live and (console.width < minimum_width or console.height < minimum_height):
                    live.update(
                        f"Resize terminal to at least {minimum_width} columns × {minimum_height} rows.\n"
                        "The game is waiting. Q quits.",
                        refresh=True,
                    )
                    time.sleep(0.1)
                    continue
                decision = policy.decide(game)
                calls += 1
                inference.append(decision.inference_ms)
                stats["interventions"] += decision.intervened
                shown = time.perf_counter()
                timestamps.append(shown)
                stats["elapsed"] = shown - started
                stats["steps_per_second"] = (
                    (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                    if len(timestamps) > 1
                    else 0
                )
                stats["best"] = max(stats["best"], game.score)
                board = game.snapshot()
                displayed_board, displayed_decision = board, decision.to_dict()
                if live:
                    canvas = compose(board, decision.to_dict(), stats)
                    live.update(canvas.rich_text(), refresh=True)
                if record:
                    record.write(
                        json.dumps(
                            {
                                "type": "frame",
                                "at": shown - started,
                                "game": board,
                                "decision": decision.to_dict(),
                                "stats": dict(stats),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                if not args.max_speed:
                    remaining = 1 / args.fps - (time.perf_counter() - now)
                    if remaining > 0:
                        time.sleep(remaining)
                game.step(decision.executed)
                total_steps += 1
                stats["best"] = max(stats["best"], game.score)
                if not game.alive or game.won:
                    deaths += not game.alive
                    if record:
                        record.write(
                            json.dumps(
                                {
                                    "type": "round_end",
                                    "at": time.perf_counter() - started,
                                    "game": game.snapshot(),
                                }
                            )
                            + "\n"
                        )
                    if args.unassisted:
                        break
                    if live:
                        live.update(compose(game.snapshot(), {}, stats).rich_text(), refresh=True)
                        time.sleep(1)
                    stats["round"] += 1
                    game = SnakeGame(
                        args.width, args.height, args.seed + stats["round"] - 1, args.initial_length
                    )
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - started
        summary = {
            "steps": total_steps,
            "inference_calls": calls,
            "seconds": elapsed,
            "steps_per_second": total_steps / elapsed if elapsed else 0,
            "score": game.score,
            "length": len(game.body),
            "best_score": stats["best"],
            "interventions": stats["interventions"],
            "deaths": deaths,
            "guarded": policy.guarded,
            "network": policy.metadata["network"],
            "mean_inference_ms": sum(inference) / len(inference) if inference else None,
        }
        if record:
            record.write(
                json.dumps({"type": "end", "summary": summary, "game": game.snapshot()}) + "\n"
            )
            record.close()
        print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return play(argv)
