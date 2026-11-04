"""Cleaned measurements and strict train/validation/test/oracle boundaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AMINO_ACIDS, POSITIONS, PROCESSED, RAW, SEED, SPLITS, WT, WT_SEQUENCE, full_sequence, mutation_count, mutation_names


def prepare(source: Path = RAW, output: Path = PROCESSED) -> dict:
    """Process ONLY Supplementary file 1, which contains measured variants."""
    if not source.exists():
        raise FileNotFoundError(
            f"Missing official GB1 source: {source}. Run scripts/download_gb1.py first."
        )
    raw = pd.read_excel(source, sheet_name="Supplementary Dataset 1", engine="openpyxl")
    expected = {"Variants", "HD", "Count input", "Count selected", "Fitness"}
    if not expected.issubset(raw):
        raise ValueError(f"Unexpected source schema. Found: {list(raw.columns)}")
    variants = raw["Variants"].astype(str).str.strip().str.upper()
    valid = variants.str.fullmatch(f"[{AMINO_ACIDS}]{{4}}")
    if not valid.all():
        raise ValueError(f"Invalid variants in rows: {raw.index[~valid].tolist()[:10]}")
    if variants.duplicated().any():
        raise ValueError("Repeated measured variant in original dataset")
    fitness = pd.to_numeric(raw["Fitness"], errors="raise")
    if not np.isfinite(fitness.to_numpy()).all() or (fitness < 0).any():
        raise ValueError("Fitness must be finite, non-negative; zero is a real observation")
    hd = pd.to_numeric(raw["HD"], errors="raise").astype(int)
    observed_hd = variants.apply(mutation_count)
    if not hd.equals(observed_hd):
        raise ValueError("Reported HD and computed number of WT substitutions disagree")
    result = pd.DataFrame({
        "variant": variants,
        "fitness": fitness.astype(float),
        "log1p_fitness": np.log1p(fitness.to_numpy()),
        "mutation_count": hd,
        "mutations": variants.apply(lambda v: ";".join(mutation_names(v))),
        "full_sequence": variants.apply(full_sequence),
        "count_input": raw["Count input"].astype(int),
        "count_selected": raw["Count selected"].astype(int),
        "is_experimentally_measured": True,
    })
    if len(result) != 149_361:
        raise ValueError(f"Expected 149361 measured variants, found {len(result)}")
    wildtype = result.loc[result.variant == WT]
    if len(wildtype) != 1 or not np.isclose(wildtype.fitness.iloc[0], 1.0):
        raise ValueError("WT VDGV missing or not fitness 1.0")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, compression="gzip", float_format="%.12g")
    stats = {
        "source": "Wu et al. (2016), eLife 5:e16965, Supplementary file 1 (v4)",
        "source_url": "https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx",
        "publication_doi": "10.7554/eLife.16965",
        "n_measured": int(len(result)),
        "n_possible": 20**4,
        "n_missing_not_imputed": int(20**4 - len(result)),
        "wt_variant": WT,
        "wt_full_sequence": WT_SEQUENCE,
        "positions_one_indexed": list(POSITIONS),
        "n_zero_fitness": int((fitness == 0).sum()),
        "n_above_wt": int((fitness > 1).sum()),
        "fitness_min": float(fitness.min()),
        "fitness_median": float(fitness.median()),
        "fitness_max": float(fitness.max()),
        "hd_counts": {str(k): int(v) for k, v in hd.value_counts().sort_index().items()},
    }
    (output.parent / "gb1_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def make_splits(
    measured: Path = PROCESSED,
    target: Path = SPLITS,
    seed: int = SEED,
    initial_budget: int = 512,
) -> dict:
    """Never sample seed experiments by looking at their fitness labels."""
    frame = pd.read_csv(measured, usecols=["variant", "fitness", "mutation_count"])
    rng = np.random.default_rng(seed)
    groups = {int(h): group.variant.to_numpy() for h, group in frame.groupby("mutation_count")}
    partitions: dict[str, str] = {}
    for h, variants in groups.items():
        v = variants.copy()
        rng.shuffle(v)
        if h <= 1:
            partitions.update({str(x): "discovery" for x in v})
        else:
            n_test, n_val = int(round(len(v) * .15)), int(round(len(v) * .10))
            partitions.update({str(x): "test" for x in v[:n_test]})
            partitions.update({str(x): "validation" for x in v[n_test:n_test + n_val]})
            partitions.update({str(x): "discovery" for x in v[n_test + n_val:]})
    partition = frame[["variant", "mutation_count"]].copy()
    partition["split"] = partition.variant.map(partitions)
    assert partition.split.notna().all()
    discovery = partition.loc[partition.split == "discovery"]
    initial = set(discovery.loc[discovery.mutation_count <= 1, "variant"])
    if len(initial) != 77:
        raise ValueError(f"Expected WT + 76 single mutants in seed, got {len(initial)}")
    remaining = initial_budget - len(initial)
    if remaining < 0:
        raise ValueError("Initial budget needs at least 77 measurements")
    for h in (2, 3, 4):
        budget = remaining // 3 + (h - 2 < remaining % 3)
        options = discovery.loc[(discovery.mutation_count == h) & ~discovery.variant.isin(initial), "variant"].to_numpy()
        if budget > len(options):
            raise ValueError(f"Insufficient depth-{h} discovery variants")
        initial.update(map(str, rng.choice(options, size=budget, replace=False)))
    partition["initially_labeled"] = partition.variant.isin(initial)
    if len(initial) != initial_budget or (partition.initially_labeled & (partition.split != "discovery")).any():
        raise AssertionError("Seed budget or split invariant failed")
    target.mkdir(parents=True, exist_ok=True)
    partition.to_csv(target / "gb1_partition.csv.gz", index=False, compression="gzip")
    joined = frame.merge(partition[["variant", "split", "initially_labeled"]], on="variant", validate="one_to_one")
    for name, subset in (
        ("gb1_initial_labeled.csv", joined[joined.initially_labeled]),
        ("gb1_oracle.csv.gz", joined[(joined.split == "discovery") & ~joined.initially_labeled]),
        ("gb1_validation.csv.gz", joined[joined.split == "validation"]),
        ("gb1_test.csv.gz", joined[joined.split == "test"]),
    ):
        cols = ["variant", "fitness", "mutation_count"]
        file = target / name
        subset[cols].sort_values("variant").to_csv(file, index=False, compression="gzip" if file.suffix == ".gz" else None)
    stats = {
        "random_seed": seed,
        "initial_labeled": int(joined.initially_labeled.sum()),
        "oracle_unlabeled": int(((joined.split == "discovery") & ~joined.initially_labeled).sum()),
        "validation": int((joined.split == "validation").sum()),
        "test": int((joined.split == "test").sum()),
        "partition_method": "Stratified by Hamming depth 2-4; all WT and 76 singles assigned to discovery",
        "seed_sampling_method": "WT + all singles + equal random quota of depth 2, 3, 4 (fitness-blind)",
        "warning": "Only oracle.measure reveals oracle fitness to the simulation; never give the oracle CSV to the agent or predictor",
    }
    (target / "gb1_split_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def read_initial(target: Path = SPLITS) -> pd.DataFrame:
    return pd.read_csv(target / "gb1_initial_labeled.csv")


def read_validation(target: Path = SPLITS) -> pd.DataFrame:
    return pd.read_csv(target / "gb1_validation.csv.gz")


def read_test(target: Path = SPLITS) -> pd.DataFrame:
    return pd.read_csv(target / "gb1_test.csv.gz")

