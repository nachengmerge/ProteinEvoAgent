"""Small residual MLP for GB1 four-site one-hot fitness regression.

Each virtual round builds a new instance from only the revealed assays.
An internal observed-only holdout selects the epoch count; all observed
labels are then used for final training from a fresh initialization.
"""

from __future__ import annotations

import copy
import os

import numpy as np


class ResMLPRegressor:
    def __init__(self, seed: int = 20261027, epochs: int = 120,
                 width: int = 128, batch_size: int = 96):
        self.seed, self.epochs = seed, epochs
        self.width, self.batch_size = width, batch_size
        self.network = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.selected_epochs = 0
        self.device = "cpu"

    @staticmethod
    def _requested_device():
        """Opt into GPU training without changing historical CPU baselines."""
        import torch

        requested = os.getenv("PROTEINEVO_TORCH_DEVICE", "cpu").strip().lower()
        if requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("PROTEINEVO_TORCH_DEVICE must be cpu, cuda, or auto")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for ResMLP, but PyTorch cannot see a CUDA GPU")
        return "cuda" if requested == "cuda" or (
            requested == "auto" and torch.cuda.is_available()
        ) else "cpu"

    def _network(self, input_dim: int):
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                    nn.Linear(width, width),
                )

            def forward(self, x):
                return x + self.net(x)

        return nn.Sequential(
            nn.Linear(input_dim, self.width), nn.GELU(),
            Block(self.width), Block(self.width),
            nn.LayerNorm(self.width), nn.Linear(self.width, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def _sample_weights(self, log_targets: np.ndarray) -> np.ndarray | None:
        """No weighting for legacy P0/P1 models; subclass may opt in."""
        return None

    def _extra_training_loss(self, estimates, targets):
        """Optional observed-only ranking objective; no effect on P0/P1 fits."""
        return None

    def _fit_epochs(self, network, x, y, indices, count: int, validate=None, weights=None):
        import torch
        import torch.nn.functional as F

        def criterion(indices):
            idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
            batch_x = x.index_select(0, idx)
            batch_y = y.index_select(0, idx)
            estimates = network(batch_x).squeeze(-1)
            if weights is None:
                loss = F.mse_loss(estimates, batch_y)
            else:
                batch_weights = weights.index_select(0, idx)
                residual = (estimates - batch_y).square()
                loss = (residual * batch_weights).sum() / batch_weights.sum()
            extra = self._extra_training_loss(estimates, batch_y)
            return loss if extra is None else loss + extra

        optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.002)
        rng = np.random.default_rng(self.seed + count)
        best_score = float("inf")
        best_epoch = count
        stale = 0
        for epoch in range(1, count + 1):
            network.train()
            for mini in np.array_split(rng.permutation(indices), max(1, int(np.ceil(len(indices)/self.batch_size)))):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(mini)
                loss.backward()
                optimizer.step()
            if validate is not None:
                network.eval()
                with torch.no_grad():
                    score = float(criterion(validate).item())
                if score < best_score - 1e-5:
                    best_score, best_epoch, stale = score, epoch, 0
                else:
                    stale += 1
                    if stale >= 15:
                        break
        return best_epoch

    def fit(self, features: np.ndarray, log_targets: np.ndarray):
        import torch

        if features.ndim != 2 or len(features) != len(log_targets) or len(features) < 20:
            raise ValueError("ResMLP expects >=20 feature/target rows")
        torch.manual_seed(self.seed)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.device = self._requested_device()
        self.target_mean = float(np.mean(log_targets))
        self.target_std = max(float(np.std(log_targets)), 0.01)
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        y = torch.from_numpy(
            ((np.asarray(log_targets) - self.target_mean) / self.target_std).astype(np.float32)
        ).to(self.device)
        raw_weights = self._sample_weights(np.asarray(log_targets, dtype=float))
        if raw_weights is not None:
            raw_weights = np.asarray(raw_weights, dtype=float)
            if (raw_weights.shape != (len(y),) or not np.isfinite(raw_weights).all()
                    or (raw_weights <= 0).any()):
                raise ValueError("Invalid sample weights for fitness regression")
        weights = (None if raw_weights is None else
                   torch.from_numpy(raw_weights.astype(np.float32)).to(self.device))
        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(len(x))
        val_size = max(10, int(round(len(x) * 0.10)))
        train_indices, val_indices = permutation[val_size:], permutation[:val_size]
        probe = self._network(x.shape[1]).to(self.device)
        selected = self._fit_epochs(probe, x, y, train_indices, self.epochs, val_indices, weights)
        self.selected_epochs = int(selected)
        torch.manual_seed(self.seed)
        self.network = self._network(x.shape[1]).to(self.device)
        self._fit_epochs(self.network, x, y, np.arange(len(x)), self.selected_epochs, weights=weights)
        self.network.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        if self.network is None:
            raise RuntimeError("Residual model not trained")
        with torch.no_grad():
            normalized = self.network(
                torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
            ).squeeze(-1).cpu().numpy()
        return normalized * self.target_std + self.target_mean


class PairwiseResMLPRegressor(ResMLPRegressor):
    """Low-rank pairwise residue embeddings followed by residual MLP blocks.

    This is deliberately smaller than a dense linear projection of all 2400
    pair indicators; six outer products are factorized into learned embeddings
    to reduce overfitting at 512 labels.
    """

    def __init__(self, seed: int = 20261027, epochs: int = 120,
                 width: int = 96, batch_size: int = 96, embedding_dim: int = 12):
        super().__init__(seed=seed, epochs=epochs, width=width, batch_size=batch_size)
        self.embedding_dim = embedding_dim

    def _network(self, input_dim: int):
        import torch
        from torch import nn

        if input_dim != 80:
            raise ValueError("PairwiseResMLP requires GB1 80-dimensional one-hot input")

        class PairResidual(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.transform = nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                    nn.Linear(width, width),
                )

            def forward(self, x):
                return x + self.transform(x)

        class Network(nn.Module):
            def __init__(self, width: int, embedding_dim: int):
                super().__init__()
                self.embedding = nn.Parameter(torch.empty(4, 20, embedding_dim))
                nn.init.normal_(self.embedding, std=0.12)
                self.network = nn.Sequential(
                    nn.Linear(80 + 6 * embedding_dim, width), nn.GELU(),
                    PairResidual(width), PairResidual(width),
                    nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(),
                    nn.Linear(64, 1),
                )

            def forward(self, onehot):
                sites = onehot.reshape(-1, 4, 20)
                residue_embeddings = torch.einsum("bsa,sad->bsd", sites, self.embedding)
                pair_interactions = [
                    residue_embeddings[:, i, :] * residue_embeddings[:, j, :]
                    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
                ]
                return self.network(torch.cat((onehot, *pair_interactions), dim=1))

        return Network(self.width, self.embedding_dim)


def tail_weights(log_targets: np.ndarray) -> np.ndarray:
    """Bounded continuous emphasis on high experimental fitness, labels only.

    Weight 1 at fitness=0; 3 at wild type (fitness=1); at most 5 above 3.
    Hyperparameters are fixed before consulting the P2 validation outcomes.
    """
    x = np.asarray(log_targets, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("log1p fitness labels must be finite and nonnegative")
    return 1.0 + 2.0 * np.clip(x / np.log(2.0), 0.0, 2.0)


class TailWeightedResMLPRegressor(ResMLPRegressor):
    def _sample_weights(self, log_targets: np.ndarray) -> np.ndarray:
        return tail_weights(log_targets)


class TailWeightedPairwiseResMLPRegressor(PairwiseResMLPRegressor):
    def _sample_weights(self, log_targets: np.ndarray) -> np.ndarray:
        return tail_weights(log_targets)


class RankAwarePairwiseResMLPRegressor(PairwiseResMLPRegressor):
    """Pairwise interaction model with a small auxiliary ordering objective.

    The rank loss uses ONLY pairs from the already observed training mini-batch;
    it is not an active-learning rule and needs no uncertainty estimator.
    The 0.15 loss coefficient and 0.5 normalized-label threshold are frozen.
    """

    def _extra_training_loss(self, estimates, targets):
        import torch
        import torch.nn.functional as F

        if len(targets) < 2:
            return estimates.sum() * 0
        truth_diff = targets[:, None] - targets[None, :]
        selected = torch.triu(truth_diff.abs() > 0.5, diagonal=1)
        if not torch.any(selected):
            return estimates.sum() * 0
        prediction_diff = estimates[:, None] - estimates[None, :]
        direction = torch.sign(truth_diff[selected])
        importance = truth_diff[selected].abs().clamp(max=3.0)
        rank_loss = F.softplus(-direction * prediction_diff[selected])
        return 0.15 * torch.sum(importance * rank_loss) / importance.sum()


class ResMLPPairwiseBlendRegressor:
    """Fixed equal-weight regression blend, no uncertainty or active learning.

    Both networks train on the *same observed labels* and use their own internal
    observed-only epoch selection. Mixing is performed on the log1p fitness
    scale, with a frozen 50/50 weight; the development split never trains a
    blending meta-learner.
    """

    def __init__(self, seed: int = 20261027, epochs: int = 120):
        self.base = ResMLPRegressor(seed=seed, epochs=epochs)
        self.pair = PairwiseResMLPRegressor(seed=seed + 103, epochs=epochs)
        self.selected_epochs: tuple[int, int] | None = None

    def fit(self, features: np.ndarray, log_targets: np.ndarray):
        self.base.fit(features, log_targets)
        self.pair.fit(features, log_targets)
        self.selected_epochs = (self.base.selected_epochs, self.pair.selected_epochs)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.selected_epochs is None:
            raise RuntimeError("Both blend components must be fitted")
        return 0.5 * self.base.predict(features) + 0.5 * self.pair.predict(features)
