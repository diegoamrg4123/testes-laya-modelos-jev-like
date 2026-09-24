"""Linux/PyTorch adapter for the Laya-MLX Snake demo.

Adapted from mizorewww/laya-mlx, commit 0a859518634112655cb97c745dbf04f5191aaf13.
Apache-2.0 license and NOTICE in project root.
"""
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .game import DIRECTIONS


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(self, model=None, *, guarded=True, prompt="compact", optimize=False, agent=None):
        self.guarded = guarded
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        if optimize:
            raise ValueError("--optimize is specific to MLX; not implemented for PyTorch")
        self.prompt = prompt
        self.metadata = {
            "name": str(model or "convaiinnovations/laya:multilingual"),
            "hardware": platform.machine(),
            "platform": platform.platform(),
            "network": "offline during play",
            "policy": "Laya probabilities over planner features; optional cycle safety shield",
            "prompt": prompt,
        }
        if agent is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["USE_TF"] = "0"
            import laya
            import torch
            location = Path(model or "models/laya").expanduser()
            if not location.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint local ausente: {location}. Faça o download antes de jogar."
                )
            if (location / "multilingual" / "rl_agent_config.json").exists():
                self.agent = laya.load(str(location), subfolder="multilingual")
            else:
                self.agent = laya.load(str(location))
            actual_device = getattr(self.agent, "device", "cpu")
            actual_type = getattr(actual_device, "type", str(actual_device))
            is_cuda = actual_type == "cuda"
            self.metadata["hardware"] = (
                torch.cuda.get_device_name(0) if is_cuda else platform.machine()
            )
            self.metadata["engine"] = "PyTorch CUDA" if is_cuda else "PyTorch CPU"
        else:
            self.agent = agent
            self.metadata["engine"] = "injected agent (test)"

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [move for move in moves if move.safe]
        if self.guarded and not safe:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        descriptions = {
            m.direction: (
                "Blocked. Collision." if not m.legal else
                "Unsafe. Traps the snake." if not m.safe else
                "Safe. Eat food now. Best." if m.eats else
                "Safe. Best route to food." if m.direction == preferred else
                "Safe. Slower route."
            ) for m in moves
        }
        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        if self.prompt == "detailed":
            state += f" Open cells: {space}. Snake length: {len(game.body)}."
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Choose the best safe move toward food.",
                "criteria": descriptions,
            },
            "risk": {"type": "noul", "instructions": "Is a safe route available?"},
            "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
        }
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        device = getattr(self.agent, "device", None)
        if getattr(device, "type", str(device)) == "cuda":
            import torch
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        values = [*(probabilities.get(d, float("nan")) for d in DIRECTIONS),
                  answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output.get("usage", {}).get("input_tokens", 0),
            output_tokens=output.get("usage", {}).get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
