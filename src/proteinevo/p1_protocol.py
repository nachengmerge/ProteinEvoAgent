"""P1 experimental panels selected without ever inspecting fitness to sample IDs.

The original validation/test and P0 results remain frozen. Panel budgets are
nested within each sampling seed (128 subset 256 subset 512 etc.), permitting
meaningful label-efficiency comparisons without treating computer seeds as new
biological repeats. All single mutants and WT are retained in every panel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED, SPLITS, WT


P1_PANEL_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
P1_BUDGETS = (128, 256, 512, 1024, 2048)


def _sha_variants(variants) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, variants))).encode("ascii")).hexdigest()


def make_panel_ids(partition: pd.DataFrame, seed: int, budgets: tuple[int, ...]) -> dict[int, set[str]]:
    """The sampling step consumes only variant, split and mutation_count."""
    required = {"variant", "mutation_count", "split"}
    if set(partition.columns) != required:
        raise ValueError(f"Sampling must receive only {sorted(required)} columns")
    discovery = partition[partition.split == "discovery"]
    fixed = set(discovery.loc[discovery.mutation_count <= 1, "variant"])
    if len(fixed) != 77 or WT not in fixed:
        raise ValueError("The P1 panel must contain WT and all 76 single mutants")
    rng = np.random.default_rng(seed)
    reservoir = {}
    for depth in (2, 3, 4):
        group = np.sort(discovery.loc[discovery.mutation_count == depth, "variant"].to_numpy(dtype=str))
        rng.shuffle(group)
        reservoir[depth] = group

    result = {}
    for budget in sorted(set(int(x) for x in budgets)):
        if budget < 77:
            raise ValueError("Panel must accommodate WT and all single mutants")
        remaining = budget - 77
        chosen = set(fixed)
        for depth in (2, 3, 4):
            quota = remaining // 3 + (depth - 2 < remaining % 3)
            if quota > len(reservoir[depth]):
                raise ValueError(f"Budget exceeds available depth-{depth} discovery variants")
            chosen.update(map(str, reservoir[depth][:quota]))
        if len(chosen) != budget:
            raise AssertionError("Panel size mismatch")
        if result and not result[max(result)].issubset(chosen):
            raise AssertionError("Nested panel property broken")
        result[budget] = chosen
    return result


def build_panels(output: Path, seeds: tuple[int, ...] = P1_PANEL_SEEDS,
                 budgets: tuple[int, ...] = P1_BUDGETS) -> pd.DataFrame:
    source = SPLITS / "gb1_partition.csv.gz"
    partition = pd.read_csv(source, usecols=["variant", "mutation_count", "split"])
    split_ids = {name: set(partition.loc[partition.split == name, "variant"])
                 for name in ("discovery", "validation", "test")}
    if len(split_ids["discovery"]) != 112040 or len(split_ids["validation"]) != 14928 or len(split_ids["test"]) != 22393:
        raise AssertionError("Frozen P0 partition unexpectedly changed")
    if any(split_ids[a] & split_ids[b] for a, b in (("discovery", "test"), ("discovery", "validation"), ("validation", "test"))):
        raise AssertionError("P0 partition overlap")

    # Labels are loaded only after every panel ID is already determined.
    panels = {seed: make_panel_ids(partition, seed, budgets) for seed in seeds}
    chosen = set().union(*(ids for by_budget in panels.values() for ids in by_budget.values()))
    measured = pd.read_csv(PROCESSED, usecols=["variant", "fitness", "mutation_count"])
    labels = measured[measured.variant.isin(chosen)].set_index("variant")
    if len(labels) != len(chosen) or not np.isfinite(labels.fitness.to_numpy()).all():
        raise AssertionError("Incomplete measured fitness for sampled experiment IDs")

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed, by_budget in panels.items():
        for budget, ids in by_budget.items():
            name = f"initial_panel_seed{seed}_n{budget}.csv"
            samples = labels.loc[sorted(ids), ["fitness", "mutation_count"]].reset_index()
            samples.to_csv(output / name, index=False)
            records.append({
                "panel_seed": seed,
                "labeled_budget": budget,
                "initial_file": name,
                "n_depth_0_1": int((samples.mutation_count <= 1).sum()),
                "n_depth_2": int((samples.mutation_count == 2).sum()),
                "n_depth_3": int((samples.mutation_count == 3).sum()),
                "n_depth_4": int((samples.mutation_count == 4).sum()),
                "initial_ids_sha256": _sha_variants(ids),
                "oracle_complement_n": len(split_ids["discovery"]) - budget,
            })
            if ids & (split_ids["validation"] | split_ids["test"]):
                raise AssertionError("Panel includes a reserved validation/test variant")
    index = pd.DataFrame(records).sort_values(["panel_seed", "labeled_budget"])
    index.to_csv(output / "panel_index.csv", index=False)
    metadata = {
        "protocol": "P1.1 independent initial experimental panels",
        "seeds": list(map(int, seeds)), "budgets": sorted(set(map(int, budgets))),
        "partition_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "validation_ids_sha256": _sha_variants(split_ids["validation"]),
        "test_ids_sha256": _sha_variants(split_ids["test"]),
        "sampler_input_columns": ["variant", "mutation_count", "split"],
        "fitness_blind_sampling": True,
        "validation_for_p1_model_selection_only": True,
        "p0_test_already_reported": True,
        "test_access_during_p1_development": False,
        "no_new_truly_external_holdout_claim": True,
        "future_oracle_rule": "oracle candidates = discovery IDs minus each corresponding panel's initially measured IDs",
        "warning": "Panel seeds are alternative retrospective sampling designs, not independent wet-lab experiments.",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index
