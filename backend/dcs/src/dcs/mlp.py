"""PyTorch MLP (plan U13, T3.2; PRD §6.3; D-024, D-059; EC-14, EC-15).

The only module that imports torch; `models.make_model` imports it when the MLP is asked for, so everything
else runs without torch (EC-16). Same interface as the baselines: `fit(X, y, dates)`, `predict_proba(X, dates)`,
`classes_`. One network per seed in `training.mlp.seeds`, probabilities averaged (a seed ensemble). Each network
trains with AdamW and a class-weighted cross-entropy on the training fold minus an inner validation split, and
keeps the weights of its best validation epoch (early stopping).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from dcs.config import ConfigError


def resolve_device(requested: str) -> str:
    """`auto` -> cuda when PyTorch sees a GPU, else cpu; `cuda` without a GPU stops with a hint (EC-15)."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            "--device cuda, but PyTorch sees no CUDA GPU (torch.cuda.is_available() is False). Install the CUDA build "
            "of torch (runbook, GB10 section), or use --device auto or cpu."
        )
    return requested


def cuda_version() -> str | None:
    return torch.version.cuda


def network(n_in: int, n_out: int, hidden: tuple[int, ...], dropout: float) -> nn.Sequential:
    """Linear -> ReLU -> Dropout per hidden layer, then a linear output (logits)."""
    layers: list[nn.Module] = []
    for size in hidden:
        layers += [nn.Linear(n_in, size), nn.ReLU(), nn.Dropout(dropout)]
        n_in = size
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


def class_weights(targets: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced weights n / (classes seen x count), as scikit-learn's class_weight="balanced"; 0 for an absent class."""
    counts = np.bincount(targets, minlength=n_classes).astype(float)
    weights = np.zeros(n_classes)
    seen = counts > 0
    weights[seen] = len(targets) / (seen.sum() * counts[seen])
    return weights


class MLP:
    def __init__(self, config: Mapping[str, Any], seed: int, device: str = "cpu") -> None:
        self.config = dict(config)
        self.seed = seed
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray) -> MLP:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        targets = np.searchsorted(self.classes_, y)
        self.n_inputs_ = int(np.shape(X)[1])
        self.networks_: list[nn.Sequential] = []
        self.epochs_: list[int] = []
        for child in np.random.SeedSequence(self.seed).spawn(self.config["seeds"]):
            net, epochs = self._train_one(np.asarray(X, dtype=np.float32), targets, int(child.generate_state(1)[0]))
            self.networks_.append(net)
            self.epochs_.append(epochs)
        return self

    def predict_proba(self, X: np.ndarray, dates: np.ndarray) -> np.ndarray:
        inputs = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            proba = torch.stack([torch.softmax(net(inputs), dim=1) for net in self.networks_]).mean(dim=0)
        out = proba.cpu().numpy().astype(float)
        return out / out.sum(axis=1, keepdims=True)  # float32 rows do not sum to exactly 1

    def save(self, path: Any) -> None:
        """Settings, classes and each network's weights (PRD §7.4 `mlp.pt`); plain types and tensors only."""
        torch.save(
            {
                "config": {k: list(v) if isinstance(v, tuple) else v for k, v in self.config.items()},
                "classes": [str(c) for c in self.classes_],
                "n_inputs": self.n_inputs_,
                "networks": [{k: v.cpu() for k, v in net.state_dict().items()} for net in self.networks_],
            },
            path,
        )

    @classmethod
    def load(cls, path: Any, device: str = "cpu") -> MLP:
        """The saved MLP, ready to predict; `weights_only` loading cannot run code from the file."""
        state = torch.load(path, map_location=device, weights_only=True)
        model = cls(state["config"], seed=0, device=device)
        model.classes_ = np.array(state["classes"], dtype=object)
        model.n_inputs_ = state["n_inputs"]
        model.networks_ = []
        for weights in state["networks"]:
            net = network(model.n_inputs_, len(model.classes_), tuple(model.config["hidden_sizes"]), model.config["dropout"])
            net.load_state_dict(weights)
            model.networks_.append(net.to(device).eval())
        model.epochs_ = []
        return model

    def _train_one(self, X: np.ndarray, targets: np.ndarray, seed: int) -> tuple[nn.Sequential, int]:
        c = self.config
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        fit_rows, val_rows = _inner_split(targets, c["inner_val_fraction"], seed)
        net = network(X.shape[1], len(self.classes_), tuple(c["hidden_sizes"]), c["dropout"]).to(self.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
        weight = torch.as_tensor(class_weights(targets[fit_rows], len(self.classes_)), dtype=torch.float32, device=self.device)
        loss = nn.CrossEntropyLoss(weight=weight)
        inputs = torch.as_tensor(X, device=self.device)
        labels = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        val_in, val_labels = inputs[val_rows], labels[val_rows]
        best, best_state, waited, epoch = math.inf, copy.deepcopy(net.state_dict()), 0, 0
        for epoch in range(1, c["max_epochs"] + 1):
            net.train()
            order = rng.permutation(fit_rows)
            for start in range(0, len(order), c["batch_size"]):
                batch = torch.as_tensor(order[start : start + c["batch_size"]], device=self.device)
                optimizer.zero_grad()
                loss(net(inputs[batch]), labels[batch]).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():  # unweighted, so a class absent from the fitting rows cannot zero it out
                current = nn.functional.cross_entropy(net(val_in), val_labels).item()
            if current < best:
                best, best_state, waited = current, copy.deepcopy(net.state_dict()), 0
            else:
                waited += 1
                if waited >= c["patience"]:
                    break
        net.load_state_dict(best_state)
        return net.eval(), epoch


def _inner_split(targets: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows to fit on and rows to stop on; stratified when every class has 2+ fish, else plain random."""
    rows = np.arange(len(targets))
    try:
        fit_rows, val_rows = train_test_split(rows, test_size=fraction, random_state=seed, stratify=targets)
    except ValueError:  # a class too small to appear on both sides
        fit_rows, val_rows = train_test_split(rows, test_size=fraction, random_state=seed)
    return np.sort(fit_rows), np.sort(val_rows)
