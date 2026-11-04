"""Epistasis-sensitive finite-sequence kernel baseline for four GB1 sites.

The product Hamming kernel contains interactions of every order while needing
no pretrained PLM, oracle labels, uncertainty or pseudo-labels. Its settings
are fixed a priori rather than selected on the existing validation/test data.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge


class HammingKernelRidgeRegressor:
    """RBF over Hamming distance on four discrete residue positions."""

    def __init__(self, alpha: float = 0.25, gamma: float = 1.0, block_size: int = 1024):
        if alpha <= 0 or gamma <= 0 or block_size < 1:
            raise ValueError("Kernel regularization, bandwidth and block size must be positive")
        self.alpha, self.gamma, self.block_size = alpha, gamma, block_size
        self._train_sites: np.ndarray | None = None
        self.estimator: KernelRidge | None = None

    @staticmethod
    def _sites(encoded: np.ndarray) -> np.ndarray:
        onehot = np.asarray(encoded, dtype=float)
        if onehot.ndim != 2 or onehot.shape[1] != 80:
            raise ValueError("Expected GB1 four-site one-hot features of shape (n,80)")
        groups = onehot.reshape(-1, 4, 20)
        if not np.allclose(groups.sum(axis=2), 1.0):
            raise ValueError("Each position must contain precisely one residue")
        return groups.argmax(axis=2).astype(np.float64)

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # cdist('hamming') divides mismatches by four; recover integer distance.
        return np.exp(-4.0 * self.gamma * cdist(left, right, metric="hamming"))

    def fit(self, encoded: np.ndarray, log_targets: np.ndarray):
        sites = self._sites(encoded)
        target = np.asarray(log_targets, dtype=float)
        if len(sites) != len(target) or not np.isfinite(target).all():
            raise ValueError("Unexpected target shape or nonfinite label")
        self._train_sites = sites
        kernel = self._kernel(sites, sites)
        self.estimator = KernelRidge(alpha=self.alpha, kernel="precomputed")
        self.estimator.fit(kernel, target)
        return self

    def predict(self, encoded: np.ndarray) -> np.ndarray:
        if self.estimator is None or self._train_sites is None:
            raise RuntimeError("Hamming kernel regression not fitted")
        sites = self._sites(encoded)
        outputs = []
        for start in range(0, len(sites), self.block_size):
            kernel = self._kernel(sites[start:start + self.block_size], self._train_sites)
            outputs.append(self.estimator.predict(kernel))
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=float)
