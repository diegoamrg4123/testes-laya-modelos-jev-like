import json
from urllib.error import HTTPError

import pytest

from snake_linux.game import DIRECTIONS, SnakeGame
from snake_linux.policy import LayaPolicy


def test_decider_adapter_uses_system_one_and_preserves_typed_answers():
    from snake_linux.alternatives import DeciderAgent

    calls = []

    class FakeDecider:
        dev = "cpu"
        name = "decider-test"

        def system_one(self, state, questions):
            calls.append((state, questions))
            return {
                "answers": {
                    "move": {"probabilities": {d: (0.6 if d == "LEFT" else 0.4 / 3) for d in DIRECTIONS}},
                    "risk": {"noul": 0.8},
                    "food": {"noul": 0.7},
                },
                "usage": {"input_tokens": 91, "output_tokens": 0},
            }

    agent = DeciderAgent(FakeDecider(), checkpoint="models/test-decider")
    decision = LayaPolicy(agent=agent).decide(SnakeGame(8, 6, seed=7))

    assert len(calls) == 1
    assert set(calls[0][1]) == {"move", "risk", "food"}
    assert decision.probabilities["LEFT"] == 0.6
    assert decision.dead_end_risk == pytest.approx(0.2)
    assert decision.input_tokens == 91


def test_cli_selects_decider_and_records_checkpoint_version(monkeypatch, tmp_path):
    from snake_linux import cli

    model = tmp_path / "decider"
    model.mkdir()
    (model / "decider_config.json").write_text(
        '{"version": "2b-v11", "release_date": "2026-09-24"}', encoding="utf-8"
    )
    agent = type("Agent", (), {"device": "cpu", "name": "decider-2b-v11"})()
    calls = []
    monkeypatch.setattr(
        cli.DeciderAgent, "from_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)) or agent,
    )

    policy = cli.load_backend(
        "decider", str(model), guarded=True, prompt="compact", semif_threads=4,
        decider_device="cpu",
    )

    assert calls == [((str(model),), {"device": "cpu"})]
    assert policy.metadata["name"] == "Mapika decider-2b-v11"
    assert policy.metadata["checkpoint"]["version"] == "2b-v11"


def test_kev_adapter_posts_snake_questions_and_keeps_raw_probabilities(monkeypatch):
    from snake_linux.alternatives import KevAgent

    calls = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self):
            return json.dumps({
                "answers": {
                    "move": {"probabilities": {d: (0.7 if d == "UP" else 0.1) for d in DIRECTIONS}},
                    "risk": {"noul": 0.8}, "food": {"noul": 0.6},
                },
                "usage": {"input_tokens": 50, "output_tokens": 12},
            }).encode()
    def send(request, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return Response()
    monkeypatch.setattr("snake_linux.alternatives.urlopen", send)
    agent = KevAgent()
    decision = LayaPolicy(agent=agent).decide(SnakeGame(8, 6, seed=7))
    assert calls[0][0] == "http://127.0.0.1:8009/v1/systemone"
    assert calls[0][1]["model"] == "kev-latest"
    assert set(calls[0][1]["questions"]) == {"move", "risk", "food"}
    assert decision.probabilities["UP"] == 0.7
    assert decision.dead_end_risk == pytest.approx(0.2)
    assert decision.input_tokens == 50


def test_kev_adapter_refuses_a_different_model_on_the_local_port(monkeypatch):
    from snake_linux.alternatives import KevAgent

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self):
            return json.dumps({"models": [{"name": "kev-latest", "run": "jaredpalmer/kev-4b"}]}).encode()
    monkeypatch.setattr("snake_linux.alternatives.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="kev-0.8b"):
        KevAgent().describe()


def test_semif_adapter_scores_three_rows_for_one_state_and_maps_option_order():
    from snake_linux.alternatives import SemIfAgent

    calls = []
    def score(model, tokenizer, rows, metadata):
        calls.append(rows)
        return ([
            {"id": "move", "option_ids": ["UP", "DOWN", "LEFT", "RIGHT"],
             "probabilities": [0.4, 0.1, 0.3, 0.2], "input_tokens": 80},
            {"id": "risk", "option_ids": ["yes", "no"],
             "probabilities": [0.6, 0.4], "input_tokens": 40},
            {"id": "food", "option_ids": ["yes", "no"],
             "probabilities": [0.7, 0.3], "input_tokens": 40},
        ], {"total_seconds": 0.1})
    agent = SemIfAgent(model=object(), tokenizer=object(), metadata={}, score_fn=score)
    decision = LayaPolicy(agent=agent).decide(SnakeGame(8, 6, seed=7))
    assert len(calls) == 1
    assert [row["id"] for row in calls[0]] == ["move", "risk", "food"]
    assert len({json.dumps(row["state"]) for row in calls[0]}) == 1
    assert [opt["id"] for opt in calls[0][0]["options"]] == list(DIRECTIONS)
    assert decision.probabilities == dict(zip(DIRECTIONS, [0.4, 0.1, 0.3, 0.2]))
    assert decision.dead_end_risk == pytest.approx(0.4)
    assert decision.food_reachable == pytest.approx(0.7)
    assert decision.output_tokens == 0


def test_cli_selects_kev_without_loading_laya(monkeypatch):
    from snake_linux import cli

    class Agent:
        def describe(self):
            return {"device": "cuda", "backend": "torch", "dtype": "bfloat16"}
    agent = Agent()
    monkeypatch.setattr(cli, "KevAgent", lambda: agent)
    policy = cli.load_backend("kev", None, guarded=True, prompt="compact", semif_threads=4)
    assert policy.agent is agent
    assert policy.metadata["name"] == "Kev-0.8B"
    assert policy.metadata["network"] == "localhost only"
    assert policy.metadata["hardware"] == "cuda"
    assert policy.metadata["checkpoint"]["dtype"] == "bfloat16"


def test_cli_selects_semif_with_explicit_gguf_and_cpu(monkeypatch):
    from snake_linux import cli

    calls = []
    agent = type("Agent", (), {"metadata": {"gguf": {"sha256": "sha-test"}}})()
    monkeypatch.setattr(cli.SemIfAgent, "from_checkpoint", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or agent)
    policy = cli.load_backend("semif", "models/custom.gguf", guarded=False,
                              prompt="detailed", semif_threads=3)
    assert policy.agent is agent
    assert calls[0][0] == ("models/custom.gguf", "models/semif-tokenizer")
    assert calls[0][1] == {"threads": 3}
    assert policy.metadata["engine"] == "llama.cpp CPU"
    assert policy.metadata["checkpoint"]["gguf"]["sha256"] == "sha-test"
    assert not policy.guarded
