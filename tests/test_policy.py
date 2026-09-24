import pytest
import torch

from snake_linux.game import SnakeGame, DIRECTIONS
from snake_linux.policy import LayaPolicy


class FakeAgent:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {
            "answers": {
                "move": {"probabilities": self.probabilities},
                "risk": {"noul": 0.9},
                "food": {"noul": 0.7},
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


def test_one_real_api_shape_call_per_move_and_shield_rejects_unsafe_top_choice():
    game = SnakeGame(8, 6, seed=7)
    safe = {m.direction for m in game.moves() if m.safe}
    unsafe = next(d for d in DIRECTIONS if d not in safe)
    probabilities = {d: 0.05 for d in DIRECTIONS}
    probabilities[unsafe] = 0.85
    agent = FakeAgent(probabilities)
    policy = LayaPolicy(agent=agent, guarded=True)
    decision = policy.decide(game)
    assert len(agent.calls) == 1
    assert set(agent.calls[0][1]) == {"move", "risk", "food"}
    assert decision.proposed == unsafe
    assert decision.executed in safe
    assert decision.intervened
    assert decision.dead_end_risk == pytest.approx(0.1)
    assert decision.input_tokens == 42


def test_unassisted_passes_model_choice_without_shield():
    game = SnakeGame(8, 6, seed=7)
    unsafe = next(m.direction for m in game.moves() if not m.safe)
    agent = FakeAgent({d: (1.0 if d == unsafe else 0.0) for d in DIRECTIONS})
    decision = LayaPolicy(agent=agent, guarded=False).decide(game)
    assert decision.executed == unsafe
    assert not decision.intervened


def test_cuda_sync_uses_actual_torch_device_before_latency_measurement(monkeypatch):
    agent = FakeAgent({d: 0.25 for d in DIRECTIONS})
    agent.device = torch.device("cuda")
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(True))
    LayaPolicy(agent=agent).decide(SnakeGame(8, 6, seed=7))
    assert calls == [True]
