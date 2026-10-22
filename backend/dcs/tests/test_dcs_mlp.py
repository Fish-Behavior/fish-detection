"""PyTorch MLP (plan U13, T3.1; PRD §6.3): config, early stopping, class weights, seeds (EC-14), device (EC-15).

Runs only where torch is installed (the training machine); CI has no torch (Q8).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dcs.config import ConfigError, load_settings  # noqa: E402
from dcs.mlp import MLP, class_weights, network, resolve_device  # noqa: E402
from dcs.models import make_model  # noqa: E402

SMALL_MLP = {
    "hidden_sizes": (16, 8),
    "dropout": 0.1,
    "learning_rate": 0.01,
    "weight_decay": 0.01,
    "batch_size": 16,
    "max_epochs": 60,
    "patience": 10,
    "inner_val_fraction": 0.2,
    "seeds": 2,
}


def separable(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.array(["A", "B", "C"] * (n // 3))
    X = rng.normal(size=(n, 4))
    X[:, 0] += 4 * (y == "B") + 8 * (y == "C")
    return X, y, np.array(["2000-01-01"] * n)


def test_mlp_shares_the_model_interface_and_learns_a_planted_signal() -> None:
    X, y, dates = separable()
    test_X, test_y, test_dates = separable(30, seed=1)
    model = MLP(SMALL_MLP, seed=0).fit(X, y, dates)
    proba = model.predict_proba(test_X, test_dates)
    assert list(model.classes_) == ["A", "B", "C"]
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert (model.classes_[proba.argmax(axis=1)] == test_y).mean() > 0.9


def test_make_model_builds_the_mlp_from_the_settings() -> None:
    training: dict[str, Any] = load_settings().training
    model = make_model("mlp", seed=0, mlp=training["mlp"])
    assert isinstance(model, MLP) and model.config["hidden_sizes"] == (128, 64)
    with pytest.raises(ValueError, match="training.mlp"):
        make_model("mlp", seed=0)


def test_architecture_follows_the_config() -> None:
    net = network(10, 4, (128, 64), 0.3)
    linear = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [(10, 128), (128, 64), (64, 4)]
    assert [layer.p for layer in net if isinstance(layer, torch.nn.Dropout)] == [0.3, 0.3]
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in net) == 2


def test_one_network_per_seed_averaged() -> None:
    X, y, dates = separable()
    model = MLP({**SMALL_MLP, "seeds": 3}, seed=0).fit(X, y, dates)
    assert len(model.networks_) == len(model.epochs_) == 3


def test_early_stopping_ends_before_max_epochs_on_noise() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), np.array(["A", "B"] * 30)
    model = MLP({**SMALL_MLP, "max_epochs": 300, "patience": 3}, seed=0).fit(X, y, y)
    assert all(epochs < 300 for epochs in model.epochs_)


def test_loss_weights_balance_the_classes() -> None:
    weights = class_weights(np.array([0, 0, 0, 1]), 3)
    np.testing.assert_allclose(weights, [4 / 6, 2.0, 0.0])  # n / (classes seen x count); an absent class gets 0


def test_same_seed_gives_identical_cpu_probabilities() -> None:
    """EC-14."""
    X, y, dates = separable()
    first, second, other = (MLP(SMALL_MLP, seed=s).fit(X, y, dates).predict_proba(X, dates) for s in (5, 5, 6))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_classes_missing_from_the_inner_split_still_train() -> None:
    """A class with one fish cannot be stratified; the split falls back to a plain random split."""
    X, y, dates = separable()
    y = y.copy()
    y[0] = "D"
    model = MLP(SMALL_MLP, seed=0).fit(X, y, dates)
    assert list(model.classes_) == ["A", "B", "C", "D"]


def test_device_cuda_without_a_gpu_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """EC-15: auto falls back to the CPU; cuda without a GPU stops with a hint."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ConfigError, match="CUDA"):
        resolve_device("cuda")


def test_auto_picks_the_gpu_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
