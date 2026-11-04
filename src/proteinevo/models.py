"""Baseline predictors and bootstrap-based epistemic uncertainty."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from .features import encode, encode_pairwise


def make_estimator(kind: str, seed: int):
    if kind == "hamming_krr":
        from .kernel_model import HammingKernelRidgeRegressor
        return HammingKernelRidgeRegressor(alpha=0.25, gamma=1.0)
    if kind == "blend_resmlp_pairwise":
        from .residual_model import ResMLPPairwiseBlendRegressor
        return ResMLPPairwiseBlendRegressor(seed=seed)
    if kind == "resmlp_onehot":
        from .residual_model import ResMLPRegressor
        return ResMLPRegressor(seed=seed)
    if kind == "tail_resmlp_onehot":
        from .residual_model import TailWeightedResMLPRegressor
        return TailWeightedResMLPRegressor(seed=seed)
    if kind == "pairwise_resmlp":
        from .residual_model import PairwiseResMLPRegressor
        return PairwiseResMLPRegressor(seed=seed)
    if kind == "tail_pairwise_resmlp":
        from .residual_model import TailWeightedPairwiseResMLPRegressor
        return TailWeightedPairwiseResMLPRegressor(seed=seed)
    if kind == "rank_pairwise_resmlp":
        from .residual_model import RankAwarePairwiseResMLPRegressor
        return RankAwarePairwiseResMLPRegressor(seed=seed)
    if kind in ("pairwise_ridge", "tail_pairwise_ridge"):
        return Ridge(alpha=3.0, solver="lsqr", tol=1e-5)
    if kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Please install the project: pip install -e .") from exc
        return XGBRegressor(
            n_estimators=140, max_depth=4, learning_rate=.045,
            subsample=.85, colsample_bytree=.9,
            reg_lambda=3.0, min_child_weight=3,
            objective="reg:squarederror", tree_method="hist",
            n_jobs=2, random_state=seed,
        )
    if kind == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=.08, batch_size=64,
            max_iter=400, n_iter_no_change=20, early_stopping=True,
            validation_fraction=.15, random_state=seed,
        )
    if kind == "ridge":
        return Ridge(alpha=10.0)
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model: {kind}")


@dataclass
class FitnessModel:
    kind: str = "xgboost"
    seed: int = 42
    estimator: object | None = None

    def _features(self, variants: list[str]):
        return encode_pairwise(variants) if self.kind in ("pairwise_ridge", "tail_pairwise_ridge") else encode(variants)

    def fit(self, variants: list[str], fitness: list[float] | np.ndarray):
        self.estimator = make_estimator(self.kind, self.seed)
        log_targets = np.log1p(np.asarray(fitness, dtype=float))
        if self.kind == "tail_pairwise_ridge":
            from .residual_model import tail_weights
            self.estimator.fit(self._features(variants), log_targets,
                               sample_weight=tail_weights(log_targets))
        else:
            self.estimator.fit(self._features(variants), log_targets)
        return self

    def predict(self, variants: list[str]) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        return np.maximum(np.expm1(self.estimator.predict(self._features(variants))), 0.0)


class BootstrapEnsemble:
    """Ensemble sample std reflects model disagreement, not calibrated confidence."""

    def __init__(self, n_models: int = 5, seed: int = 42, kind: str = "xgboost"):
        self.n_models, self.seed, self.kind = n_models, seed, kind
        self.models: list[FitnessModel] = []

    def fit(self, variants: list[str], fitness: list[float] | np.ndarray):
        vals = np.asarray(fitness)
        rng = np.random.default_rng(self.seed)
        self.models = []
        for i in range(self.n_models):
            idx = rng.choice(len(variants), size=len(variants), replace=True)
            chosen = [variants[j] for j in idx]
            self.models.append(FitnessModel(self.kind, self.seed + i).fit(chosen, vals[idx]))
        return self

    def predict(self, variants: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Ensemble is not fitted")
        predictions = np.stack([m.predict(variants) for m in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0, ddof=1) if len(self.models) > 1 else np.zeros(len(variants))


def evaluate(y_true, y_pred, top_k: int = 20) -> dict:
    """Ranking and regression metrics on a fixed blind split."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(y) != len(p) or not len(y):
        raise ValueError("Mismatched or empty evaluation vectors")
    k = min(top_k, len(y))
    best_true = set(np.argsort(y)[-k:])
    best_pred = set(np.argsort(p)[-k:])
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "mse": float(np.mean((y - p)**2)),
        "rmse": float(np.sqrt(np.mean((y - p)**2))),
        "spearman": float(spearmanr(y, p).statistic),
        "pearson": float(pearsonr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else float("nan"),
        "top_k": k,
        "top_k_recall": len(best_true & best_pred) / k,
        "top_k_precision": len(best_true & best_pred) / k,
        "true_best_fitness": float(np.max(y)),
        "predicted_top_k_true_best": float(np.max(y[list(best_pred)])),
        "hit_rate_above_wt_in_predicted_top_k": float(np.mean(y[list(best_pred)] > 1.0)),
    }
