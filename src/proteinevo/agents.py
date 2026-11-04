"""Scientific analysis, hypothesis generation, design, and critical review."""

import json
import os
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AMINO_ACIDS, POSITIONS, ROOT, WT, mutation_names, validate_variant
from .knowledge import AA_GROUP, describe, substitution_risk


@dataclass
class Hypothesis:
    position: int
    amino_acid: str
    reason: str
    source: str = "data_heuristic"


def parse_hypothesis(item: dict, source: str) -> Hypothesis:
    """Validate an external hypothesis without coercing floats or residue strings."""
    if not isinstance(item, dict) or type(item.get("position")) is not int:
        raise ValueError("position must be a JSON integer")
    position = item["position"]
    residue = item.get("amino_acid")
    reason = item.get("reason")
    if not isinstance(residue, str):
        raise ValueError("amino_acid must be a single standard residue")
    residue = residue.strip().upper()
    if position not in POSITIONS or len(residue) != 1 or residue not in set(AMINO_ACIDS):
        raise ValueError("Invalid mutable position or amino acid")
    if residue == WT[POSITIONS.index(position)]:
        raise ValueError("Unchanged WT residue is not a mutation hypothesis")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A non-empty reason is required")
    return Hypothesis(position, residue, reason.strip()[:350], source)


class DataAnalyst:
    """Summarize only the measurements that have already been revealed."""

    def analyze(self, measured: pd.DataFrame, top_n: int = 25) -> dict:
        top = measured.nlargest(min(top_n, len(measured)), "fitness")
        # Marginal associations are not causal single-site effects; epistasis confounds them.
        preferences = []
        for i, position in enumerate(POSITIONS):
            for aa in AMINO_ACIDS:
                if aa == WT[i]:
                    continue
                block = measured[measured.variant.str[i] == aa]
                if len(block) < 3:
                    continue
                preferences.append({
                    "position": position, "amino_acid": aa,
                    "count": int(len(block)),
                    "mean_log1p_fitness": round(float(np.log1p(block.fitness).mean()), 5),
                    "top_fraction": round(float(np.mean(top.variant.str[i] == aa)), 3),
                })
        preferences.sort(key=lambda x: (x["mean_log1p_fitness"], x["top_fraction"]), reverse=True)
        return {
            "n_observed": len(measured),
            "best_seen_fitness": float(top.fitness.max()),
            "top_variants": top[["variant", "fitness"]].head(12).to_dict("records"),
            "site_associations": preferences[:24],
            "caution": "Marginal associations from observed sequences do not establish causality or guarantee epistatic compatibility",
        }


class HypothesisGenerator:
    def propose(self, summary: dict, n: int = 12) -> list[Hypothesis]:
        return [
            Hypothesis(
                position=int(p["position"]), amino_acid=str(p["amino_acid"]),
                reason=f"Observed mean log(1+fitness)={p['mean_log1p_fitness']:.3f} (n={p['count']}); epistasis unverified",
            ) for p in summary["site_associations"][:n]
        ]


class OpenAICompatibleHypothesisGenerator:
    """Optional real LLM via a configured, OpenAI-compatible chat API."""

    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        self.api_key = os.environ.get("PROTEINEVO_LLM_API_KEY", "").strip()
        self.base_url = os.environ.get("PROTEINEVO_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("PROTEINEVO_LLM_MODEL", "").strip()
        if not self.api_key or not self.model:
            raise RuntimeError("Real LLM mode requires PROTEINEVO_LLM_API_KEY and PROTEINEVO_LLM_MODEL; see .env.example")

    def propose(self, summary: dict, n: int = 12) -> list[Hypothesis]:
        instructions = (
            "You are a protein engineering scientific hypothesis generator. "
            "Target: GB1 IgG-Fc relative binding fitness. WT VDGV; four mutable "
            "positions 39,40,41,54. Based ONLY on the supplied observed data, "
            "propose up to 12 single-site residue hypotheses. Reply with JSON "
            "object {\"hypotheses\": [{\"position\":39,\"amino_acid\":\"A\",\"reason\":\"...\"}]}. "
            "Use canonical amino acids, distinguish correlation from causal evidence, "
            "and explicitly acknowledge possible epistasis. Do not invent wet-lab results."
        )
        payload = {
            "model": self.model, "temperature": .4,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        # Accommodate models that enclose JSON in ```json code fences.
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
        parsed = json.loads(content)
        output = []
        seen = set()
        for h in parsed.get("hypotheses", []):
            try:
                hypothesis = parse_hypothesis(h, "llm")
                key = (hypothesis.position, hypothesis.amino_acid)
                if key in seen:
                    continue
                seen.add(key)
                output.append(hypothesis)
            except (TypeError, ValueError, KeyError):
                continue
        return output[:n]


class ScientificCritic:
    """Evidence-linked review; *no* access to unmeasured experimental labels.

    A critique is an annotation, not a biochemical verification or an oracle
    query. `observed` must be the exact training table available at this turn.
    """

    def review(self, variant: str, predicted: float, uncertainty: float | None,
               parent: str, reason: str, observed: pd.DataFrame | None = None) -> dict:
        validate_variant(variant)
        validate_variant(parent)
        edits = sum(a != b for a, b in zip(variant, parent))
        radical = sum(substitution_risk(a, b) >= 1 for a, b in zip(parent, variant))
        notes = []
        if edits > 2:
            notes.append("More than two mutations relative to stated parent; check lineage")
        if radical:
            notes.append(f"{radical} property-class-changing substitution(s) may alter folding/binding")

        observed_support = []
        nearest_distance = None
        nearest_measured = []
        if observed is not None:
            expected = {"variant", "fitness"}
            if not expected.issubset(observed.columns):
                raise ValueError("ScientificCritic requires only observed variant/fitness")
            if observed.variant.duplicated().any():
                raise ValueError("Duplicate observed variants in critic input")
            if variant in set(observed.variant):
                notes.append("Candidate already measured; reject duplicate experiment")
            distances = observed.variant.map(
                lambda seen: sum(a != b for a, b in zip(variant, str(seen)))
            )
            nearest_distance = int(distances.min())
            nearest = observed.loc[distances == nearest_distance]
            nearest_measured = nearest.nlargest(3, "fitness")[["variant", "fitness"]].to_dict("records")
            if nearest_distance > 1:
                notes.append("No directly observed single-edit neighbor; ranking extrapolates")
            for site, position in enumerate(POSITIONS):
                if variant[site] == WT[site]:
                    continue
                matching = observed.loc[observed.variant.str[site] == variant[site], "fitness"]
                count = len(matching)
                observed_support.append({
                    "position": position, "amino_acid": variant[site],
                    "observed_count": int(count),
                    "mean_observed_fitness": round(float(matching.mean()), 5) if count else None,
                })
                if count < 3:
                    notes.append(f"Insufficient observed support for {position}{variant[site]} (n={count})")

        if not notes:
            notes.append("No rule violation detected; experimental behavior is still unknown")
        result = {
            "candidate": variant,
            "mutations": mutation_names(variant),
            "predicted_fitness": round(float(predicted), 5) if np.isfinite(predicted) else None,
            "rationale": reason,
            "knowledge": describe(variant, parent),
            "parent": parent, "edits_from_parent": edits,
            "radical_substitution_count": radical,
            "nearest_observed_hamming_distance": nearest_distance,
            "nearest_observed_examples": nearest_measured,
            "marginal_mutation_support": observed_support,
            "warnings": notes,
            "status": "reject" if observed is not None and variant in set(observed.variant)
            else "caution" if edits > 2 or radical or nearest_distance is not None and nearest_distance > 1
            else "provisional",
            "limitations": (
                "Only observed labels were consulted. Marginal associations are not causal effects; "
                "fitness predictions and property rules do not establish binding, "
                "folding stability, or epistatic compatibility."
            ),
        }
        if uncertainty is not None and np.isfinite(uncertainty):
            result["bootstrap_std"] = round(float(uncertainty), 5)
            result["limitations"] += " Bootstrap spread, if present, is uncalibrated."
        return result

