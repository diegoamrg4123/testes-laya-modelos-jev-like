from snake_linux.game import SnakeGame
from snake_linux.ui import compose


def test_terminal_identifies_linux_pytorch_without_mlx_claims():
    game = SnakeGame(8, 6)
    frame = compose(game.snapshot(), {}, {"engine": "PyTorch CUDA", "hardware": "RTX 2050"})
    text = frame.rich_text().plain
    assert "PyTorch CUDA" in text
    assert "Laya MLX" not in text
    assert "RTX 2050" in text


def test_terminal_names_actual_backend_and_network_instead_of_laya():
    game = SnakeGame(8, 6)
    for name, network in (("Kev-0.8B", "localhost only"), ("SemIf-4B Q4_K_M", "offline")):
        text = compose(game.snapshot(), {}, {"name": name, "network": network,
                       "engine": "llama.cpp CPU", "hardware": "CPU"}).rich_text().plain
        assert name in text
        assert network.upper() in text
        assert "ESTIMATES BY LAYA" not in text
