"""Retrospective experimental oracle with an explicit, auditable measurement gate."""

import pandas as pd

from .config import SPLITS


class RetrospectiveOracle:
    def __init__(self, splits_dir=SPLITS):
        # Critically, the agent never gets this object or the hidden labels.
        secret = pd.read_csv(splits_dir / "gb1_oracle.csv.gz")
        self._truth = dict(zip(secret.variant, secret.fitness))
        self._available = set(self._truth)
        self._revealed: set[str] = set()

    def available_variants(self) -> set[str]:
        """Return public candidate IDs only, never ground-truth labels."""
        return self._available.copy()

    def measure(self, variants: list[str]) -> pd.DataFrame:
        if len(variants) != len(set(variants)):
            raise ValueError("Cannot assay a duplicate sequence in the same batch")
        invalid = set(variants) - self._available
        if invalid:
            raise ValueError(f"Unmeasurable, held-out or previously assayed variants: {sorted(invalid)[:5]}")
        self._available.difference_update(variants)
        self._revealed.update(variants)
        return pd.DataFrame({"variant": variants, "fitness": [self._truth[v] for v in variants]})

    @property
    def acquired_count(self) -> int:
        return len(self._revealed)


class PanelRetrospectiveOracle(RetrospectiveOracle):
    """Full discovery complement for a P1 initial panel of arbitrary size.

    P0 oracle excludes its *own* 512-label panel; P1 samples a DIFFERENT panel.
    Rebuild the discovery complement from P0's two disjoint discovery files.
    The secret labels remain private, exposed only through ``measure``.
    """

    def __init__(self, initial_variants, splits_dir=SPLITS):
        initials = set(initial_variants)
        if not initials:
            raise ValueError("An empty initial panel is invalid")
        parts = [
            pd.read_csv(splits_dir / "gb1_initial_labeled.csv", usecols=["variant", "fitness"]),
            pd.read_csv(splits_dir / "gb1_oracle.csv.gz", usecols=["variant", "fitness"]),
        ]
        discovery = pd.concat(parts, ignore_index=True)
        if (len(discovery) != 112040 or discovery.variant.duplicated().any()
                or not initials.issubset(set(discovery.variant))):
            raise ValueError("Initial panel must be a subset of frozen discovery partition")
        hidden = discovery.loc[~discovery.variant.isin(initials)]
        self._truth = dict(zip(hidden.variant, hidden.fitness))
        self._available = set(self._truth)
        self._revealed = set()
        if len(self._available) != 112040 - len(initials):
            raise AssertionError("P1 oracle complement has inconsistent cardinality")

