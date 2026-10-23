"""Attacker models (Sec. 6.2).

* `RandomForestAttacker`: random forest on per-link / per-session features, as in
  prior traffic-analysis work on federated and multi-agent systems.
* `TimeSliceCNN`: small 1D CNN on per-link packet counts and byte volumes in time
  slices (AgentPrint representation). The paper does not specify the architecture
  (audit M27); the one below is our choice. Requires PyTorch, which is an optional
  dependency (`pip install .[cnn]`). Select it with `attack.models: [rf, cnn]`; its
  training is optional locally and has been exercised on the Mac Studio; see the dated pilot reports.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Attacker(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class RandomForestAttacker:
    def __init__(self, n_estimators: int = 200, seed: int = 0, n_jobs: int = 4) -> None:
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> RandomForestAttacker:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X))

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.model.classes_)


class TimeSliceCNN:
    """Small 1D CNN on (4 x n_slices) time-slice matrices. Optional PyTorch dependency.

    Training: Adam, mini-batches, a random `val_fraction` of the training question groups held out
    for early stopping (patience in epochs); the weights with the lowest validation loss
    are kept. Runs on Apple MPS when available, else CPU.
    """

    def __init__(self, epochs: int = 100, lr: float = 1e-3, seed: int = 0, batch_size: int = 256,
                 val_fraction: float = 0.2, patience: int = 8, temporal_bins: int = 1, log_input: bool = False) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("TimeSliceCNN needs PyTorch: pip install '.[cnn]' (Mac Studio: torch with MPS)") from e
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if temporal_bins not in (1, 4):
            raise ValueError('temporal_bins must be one of the prespecified values 1 or 4')
        self.temporal_bins, self.log_input = temporal_bins, log_input
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.batch_size, self.val_fraction, self.patience = batch_size, val_fraction, patience
        self.net: Any = None
        self.classes_: np.ndarray = np.array([])
        self.device = "cpu"
        self.history_: list[dict[str, float]] = []
        self.validation_groups_: list[int] = []
        self.training_groups_: list[int] = []
        self.mean_: Any = 0.0
        self.scale_: Any = 1.0

    def _build(self, n_classes: int) -> Any:  # pragma: no cover - needs torch
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(self.temporal_bins),
            nn.Flatten(), nn.Linear(32 * self.temporal_bins, n_classes))

    def _input(self, X: np.ndarray) -> np.ndarray:
        if self.log_input:
            if np.any(X < 0):
                raise ValueError('log input requires nonnegative counts and volumes')
            return np.log1p(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> TimeSliceCNN:  # pragma: no cover
        import copy

        import torch
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available():
            self.device = "mps"
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        rng = np.random.default_rng(self.seed)
        group = np.arange(len(yi)) if groups is None else np.asarray(groups)
        unique = rng.permutation(np.unique(group))
        n_groups = min(len(unique) - 1, max(1, int(self.val_fraction * len(unique)))) if self.val_fraction and len(unique) > 1 else 0
        val_groups = unique[:n_groups]
        # Keep every class represented in the fitting set, moving entire groups.
        for c in self.classes_:
            if not np.any((y == c) & ~np.isin(group, val_groups)):
                move = next(g for g in val_groups if np.any((group == g) & (y == c)))
                val_groups = val_groups[val_groups != move]
        val = np.flatnonzero(np.isin(group, val_groups))
        tr = np.flatnonzero(~np.isin(group, val_groups))
        n_val = len(val)
        self.validation_groups_, self.training_groups_ = val_groups.tolist(), np.unique(group[tr]).tolist()
        X = self._input(X)
        self.mean_ = np.mean(X[tr], axis=(0, 2), keepdims=True)
        self.scale_ = np.maximum(np.std(X[tr], axis=(0, 2), keepdims=True), 1e-6)
        dev = torch.device(self.device)
        xt = torch.tensor((X - self.mean_) / self.scale_, dtype=torch.float32, device=dev)
        yt = torch.tensor(yi, dtype=torch.long, device=dev)
        self.net = self._build(len(self.classes_)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, wait = float("inf"), copy.deepcopy(self.net.state_dict()), 0
        weights = 1 / np.maximum(np.bincount(yi[tr], minlength=len(self.classes_)), 1)
        weight = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=dev)
        self.history_ = []
        for epoch in range(self.epochs):
            self.net.train()
            losses = []
            for b in np.array_split(rng.permutation(tr), max(1, int(np.ceil(len(tr) / self.batch_size)))):
                bi = torch.tensor(b, dtype=torch.long, device=dev)
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.net(xt[bi]), yt[bi], weight=weight)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            self.history_.append({"epoch": float(epoch), "train_loss": float(np.mean(losses))})
            if n_val == 0:
                continue
            self.net.eval()
            with torch.no_grad():
                vi = torch.tensor(val, dtype=torch.long, device=dev)
                vloss = float(torch.nn.functional.cross_entropy(self.net(xt[vi]), yt[vi]))
            self.history_[-1]["validation_loss"] = vloss
            if vloss < best - 1e-6:
                best, best_state, wait = vloss, copy.deepcopy(self.net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if n_val:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - needs torch
        import torch
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor((self._input(X) - self.mean_) / self.scale_, dtype=torch.float32, device=torch.device(self.device)))
            return torch.softmax(out, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - needs torch
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
