import laya
import torch

from snake_linux.policy import LayaPolicy


def test_bundled_multilingual_checkpoint_loads_subfolder(tmp_path, monkeypatch):
    checkpoint = tmp_path / "laya" / "multilingual"
    checkpoint.mkdir(parents=True)
    (checkpoint / "rl_agent_config.json").write_text("{}")
    calls = []
    monkeypatch.setattr(laya, "load", lambda *args, **kwargs: calls.append((args, kwargs)) or object())
    LayaPolicy(model=str(tmp_path / "laya"))
    assert calls[0][1] == {"subfolder": "multilingual"}


def test_laya_reports_actual_cpu_fallback_even_when_gpu_is_present(tmp_path, monkeypatch):
    (tmp_path / "laya").mkdir()
    monkeypatch.setattr(laya, "load", lambda *_: type("Agent", (), {"device": torch.device("cpu")})())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "pretend CUDA")
    policy = LayaPolicy(model=str(tmp_path / "laya"))
    assert policy.metadata["engine"] == "PyTorch CPU"
    assert policy.metadata["hardware"] != "pretend CUDA"
