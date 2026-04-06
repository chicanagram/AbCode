from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _build_hidden_sizes(n_layers: int, hidden_size: int, hidden_sizes: List[int] | None) -> List[int]:
    if hidden_sizes:
        return [int(v) for v in hidden_sizes]
    layers = max(int(n_layers), 1)
    width = max(int(hidden_size), 1)
    return [width] * layers


def _set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    dropout: float,
):
    import torch.nn as nn

    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(nn.ReLU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        prev = int(h)
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


@dataclass
class _TorchTrainConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 400
    patience: int = 15
    min_delta: float = 1e-5
    validation_split: float = 0.2
    random_state: int = 42
    device: str = "cpu"


class TorchMLPRegressor:
    def __init__(self, **params: Any):
        self.n_layers = int(params.get("n_layers", 2))
        self.hidden_size = int(params.get("hidden_size", 32))
        self.hidden_sizes = params.get("hidden_sizes")
        self.dropout = float(params.get("dropout", 0.0))
        self.config = _TorchTrainConfig(
            learning_rate=float(params.get("learning_rate", 1e-3)),
            batch_size=int(params.get("batch_size", 32)),
            max_epochs=int(params.get("max_epochs", 400)),
            patience=int(params.get("patience", 15)),
            min_delta=float(params.get("min_delta", 1e-5)),
            validation_split=float(params.get("validation_split", 0.2)),
            random_state=int(params.get("random_state", 42)),
            device=str(params.get("device", "cpu")),
        )
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        _set_torch_seed(self.config.random_state)

        X_np = _as_float32(X)
        y_np = _as_float32(np.asarray(y).reshape(-1, 1))
        n = len(X_np)

        rng = np.random.default_rng(self.config.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_val = max(1, int(round(n * self.config.validation_split))) if n > 5 else 1
        n_val = min(n_val, n - 1) if n > 1 else 0
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_np[val_idx], dtype=torch.float32)

        hidden_sizes = _build_hidden_sizes(self.n_layers, self.hidden_size, self.hidden_sizes)
        model = _build_mlp(X_np.shape[1], 1, hidden_sizes, self.dropout)

        device = torch.device(self.config.device)
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=max(int(self.config.batch_size), 1),
            shuffle=True,
        )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for _ in range(max(int(self.config.max_epochs), 1)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = float(loss_fn(val_pred, y_val.to(device)).detach().cpu().item())

            if val_loss + float(self.config.min_delta) < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        self.model_.eval()
        X_t = torch.tensor(_as_float32(X), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            pred = self.model_(X_t).detach().cpu().numpy().reshape(-1)
        return pred


class TorchMLPClassifier:
    def __init__(self, **params: Any):
        self.n_layers = int(params.get("n_layers", 2))
        self.hidden_size = int(params.get("hidden_size", 32))
        self.hidden_sizes = params.get("hidden_sizes")
        self.dropout = float(params.get("dropout", 0.0))
        self.config = _TorchTrainConfig(
            learning_rate=float(params.get("learning_rate", 1e-3)),
            batch_size=int(params.get("batch_size", 32)),
            max_epochs=int(params.get("max_epochs", 400)),
            patience=int(params.get("patience", 15)),
            min_delta=float(params.get("min_delta", 1e-5)),
            validation_split=float(params.get("validation_split", 0.2)),
            random_state=int(params.get("random_state", 42)),
            device=str(params.get("device", "cpu")),
        )
        self.model_ = None
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        _set_torch_seed(self.config.random_state)

        X_np = _as_float32(X)
        y_raw = np.asarray(y)
        classes = np.unique(y_raw)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)
        self.classes_ = classes
        self.class_to_idx_ = class_to_idx

        n = len(X_np)
        rng = np.random.default_rng(self.config.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_val = max(1, int(round(n * self.config.validation_split))) if n > 5 else 1
        n_val = min(n_val, n - 1) if n > 1 else 0
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_idx[train_idx], dtype=torch.long)
        X_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_idx[val_idx], dtype=torch.long)

        hidden_sizes = _build_hidden_sizes(self.n_layers, self.hidden_size, self.hidden_sizes)
        model = _build_mlp(X_np.shape[1], len(classes), hidden_sizes, self.dropout)

        device = torch.device(self.config.device)
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=max(int(self.config.batch_size), 1),
            shuffle=True,
        )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for _ in range(max(int(self.config.max_epochs), 1)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(device))
                val_loss = float(loss_fn(val_logits, y_val.to(device)).detach().cpu().item())

            if val_loss + float(self.config.min_delta) < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self._device = device
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        self.model_.eval()
        X_t = torch.tensor(_as_float32(X), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
