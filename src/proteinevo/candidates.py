"""Candidate proposal policies operate solely on measured data and public IDs."""

import numpy as np
import pandas as pd

from .agents import Hypothesis
from .config import AMINO_ACIDS, POSITIONS


def sample_uniform(available: set[str], count: int, rng: np.random.Generator) -> list[str]:
    population = np.asarray(sorted(available), dtype="U4")
    return list(map(str, rng.choice(population, size=min(count, len(population)), replace=False)))


def generate_proposals(
    available: set[str],
    observed: pd.DataFrame,
    hypotheses: list[Hypothesis],
    budget: int,
    rng: np.random.Generator,
    *, provenance: dict[str, dict] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """One- and two-step neighborhood moves plus hypothesis-guided substitutions."""
    if not available:
        return [], {}
    elites = observed.nlargest(min(30, len(observed)), "fitness").variant.tolist()
    if not elites:
        values = sample_uniform(available, budget, rng)
        if provenance is not None:
            provenance.update({v: {"source": "uniform_fallback", "generator_parent": None,
                                   "edits": [], "hypothesis_applications": []} for v in values})
        return values, {v: "Uniform exploration without an elite parent" for v in values}
    priorities: dict[int, list[str]] = {}
    reasons: dict[tuple[int, str], str] = {}
    for h in hypotheses:
        if h.position not in POSITIONS or len(h.amino_acid) != 1 or h.amino_acid not in set(AMINO_ACIDS):
            continue
        p = POSITIONS.index(h.position)
        priorities.setdefault(p, []).append(h.amino_acid)
        reasons[(p, h.amino_acid)] = h.reason
    result: set[str] = set()
    explanations: dict[str, str] = {}
    # Cap attempts to ensure reproducible finite work even if many proposals are unavailable.
    for _ in range(max(1000, budget * 32)):
        if len(result) >= budget:
            break
        parent = str(rng.choice(elites))
        positions = rng.choice(4, size=int(rng.choice([1, 2], p=[.8, .2])), replace=False)
        candidate = list(parent)
        notes = []
        applications = []
        for p in positions:
            p = int(p)
            if p in priorities and rng.random() < .8:
                aa = str(rng.choice(priorities[p]))
                notes.append(reasons[(p, aa)])
                applications.append({"hypothesis_id": f"{POSITIONS[p]}{aa}",
                                     "position": POSITIONS[p], "amino_acid": aa,
                                     "changed_parent_residue": aa != parent[p]})
            else:
                aa = str(rng.choice(list(AMINO_ACIDS)))
            candidate[p] = aa
        variant = "".join(candidate)
        if variant in available and variant not in result:
            result.add(variant)
            explanations[variant] = f"Explore {parent} neighborhood; " + ("; ".join(notes[:2]) if notes else "diversify local substitutions")
            if provenance is not None:
                provenance[variant] = {
                    "source": "hypothesis_neighborhood" if any(a["changed_parent_residue"] for a in applications) else "local_diversification",
                    "generator_parent": parent,
                    "edits": [{"position": POSITIONS[p], "before": a, "after": b}
                              for p, (a, b) in enumerate(zip(parent, variant)) if a != b],
                    "hypothesis_applications": applications,
                }
    if len(result) < budget:
        remainder = sample_uniform(available - result, budget - len(result), rng)
        result.update(remainder)
        for v in remainder:
            explanations[v] = "Uniform exploration fallback; limited evidence from local hypotheses"
            if provenance is not None:
                provenance[v] = {"source": "uniform_fallback", "generator_parent": None,
                                 "edits": [], "hypothesis_applications": []}
    return sorted(result), explanations


def generate_recombination_proposals(
    available: set[str], observed: pd.DataFrame, budget: int,
    rng: np.random.Generator,
    *, provenance: dict[str, dict] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Combine residues from two already measured elite parents.

    All parent fitness values are previously observed; the candidate pool is
    public IDs only. A fraction of recombinants receive one new random site
    edit to avoid collapsing onto an elite-only set. Fallback retains exactly
    the same total screening budget as existing methods.
    """
    if not available or budget < 0:
        return [], {}
    if not {"variant", "fitness"}.issubset(observed.columns):
        raise ValueError("Observed sequence and measured fitness are required")
    elites = observed.nlargest(min(30, len(observed)), "fitness").variant.tolist()
    if len(elites) < 2:
        return generate_proposals(available, observed, [], budget, rng, provenance=provenance)
    result: set[str] = set()
    reasons: dict[str, str] = {}
    for _ in range(max(1500, budget * 48)):
        if len(result) >= budget:
            break
        i, j = rng.choice(len(elites), size=2, replace=False)
        left, right = str(elites[i]), str(elites[j])
        candidate = [left[p] if rng.random() < 0.5 else right[p] for p in range(4)]
        perturbed = rng.random() < 0.35
        if perturbed:
            position = int(rng.integers(0, 4))
            options = sorted(set(AMINO_ACIDS) - {candidate[position]})
            candidate[position] = str(rng.choice(options))
        sequence = "".join(candidate)
        if sequence in available and sequence not in result:
            result.add(sequence)
            reasons[sequence] = (
                f"Recombine two measured elite parents {left}/{right}"
                + (" and diversify one residue" if perturbed else "")
            )
            if provenance is not None:
                provenance[sequence] = {"source": "recombination", "generator_parent": None,
                                        "generator_parents": [left, right],
                                        "edits": [], "hypothesis_applications": []}
    if len(result) < budget:
        remainder, fallback_reasons = generate_proposals(
            available - result, observed, [], budget - len(result), rng,
            provenance=provenance,
        )
        result.update(remainder)
        reasons.update(fallback_reasons)
    return sorted(result), reasons
