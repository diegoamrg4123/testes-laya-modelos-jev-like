"""Re-run game invariants and validate actual local Laya recordings.

Adapted from tests/test_snake.py of mizorewww/laya-mlx (Apache-2.0).
"""
import json
import random
from pathlib import Path

import pytest

from snake_linux.game import DIRECTIONS, SnakeGame, hamiltonian_cycle


@pytest.mark.parametrize("width,height", [(4, 4), (4, 5), (5, 4), (24, 16)])
def test_cycle_visits_every_cell_and_closes(width, height):
    cycle = hamiltonian_cycle(width, height)
    assert len(cycle) == len(set(cycle)) == width * height
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
               for a, b in zip(cycle, cycle[1:] + cycle[:1]))


@pytest.mark.parametrize("seed", range(12))
def test_arbitrary_safe_choices_fill_board(seed):
    game = SnakeGame(6, 6, seed=seed)
    rng = random.Random(seed + 100)
    for _ in range(game.capacity * (game.capacity - game.initial_length)):
        allowed = [m.direction for m in game.moves() if m.safe]
        assert allowed
        game.step(rng.choice(allowed))
        assert game.alive and game.cycle_order_valid()
        if game.won:
            break
    assert game.won


@pytest.mark.parametrize("name", ["snake-20.jsonl", "snake-120.jsonl"])
def test_recorded_laya_frames_match_all_moves(name):
    path = Path(__file__).parents[1] / "resultados" / name
    rows = [json.loads(line) for line in path.open()]
    settings = rows[0]["settings"]
    game = SnakeGame(settings["width"], settings["height"], settings["seed"],
                     settings["initial_length"])
    frames = [r for r in rows if r["type"] == "frame"]
    interventions = 0
    for frame in frames:
        assert frame["game"] == game.snapshot()
        decision = frame["decision"]
        probs = decision["probabilities"]
        assert set(probs) == set(DIRECTIONS)
        assert sum(probs.values()) == pytest.approx(1, abs=0.00021)
        assert decision["proposed"] == max(DIRECTIONS, key=probs.__getitem__)
        assert decision["executed"] in [m.direction for m in game.moves() if m.safe]
        interventions += decision["intervened"]
        assert frame["stats"]["interventions"] == interventions
        game.step(decision["executed"])
        assert game.alive and game.cycle_order_valid()
    assert rows[-1]["game"] == game.snapshot()
    assert rows[-1]["summary"]["steps"] == len(frames)
