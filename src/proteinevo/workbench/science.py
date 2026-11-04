"""Canonical contextual hypotheses, learned condition rules and explicit revision."""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from ..config import AMINO_ACIDS, POSITIONS
from ..contextual_science import observed_lookup, one_edit_evidence, evidence_examples, validate_action
from ..knowledge import AA_GROUP
from ..scientific_action_view import canonical_action_bank

EPS = 1e-9
KINDS = {"beneficial_both": ("increase", "increase"), "uncertain_transfer": ("uncertain", "uncertain")}


def sign(value):
    return "increase" if value > EPS else "decrease" if value < -EPS else "uncertain"


class Scoring:
    def __init__(self, pool, batch_size=16):
        if set(pool.columns) != {"variant", "predicted_fitness"} or pool.variant.duplicated().any():
            raise ValueError("Only public candidate IDs and predictions are allowed")
        self.pool = pool.copy()
        self.pool["predicted_fitness"] = self.pool.predicted_fitness.astype(float)
        if len(pool) < batch_size or not np.isfinite(self.pool.predicted_fitness).all():
            raise ValueError("Invalid scoring vector or budget")
        self.pool = self.pool.sort_values(["predicted_fitness", "variant"], ascending=[False, True], kind="stable")
        self.values = dict(zip(self.pool.variant, self.pool.predicted_fitness))
        self.ordered_ids = self.pool.variant.tolist()
        self.batch_size = batch_size
        self.greedy = self.pool.head(batch_size).variant.tolist()
        self.total = float(self.pool.head(batch_size).predicted_fitness.sum())

    def choose(self, actions):
        tests = [v for a in actions for v in a["children"]]
        if len(tests) != len(set(tests)) or len(tests) > self.batch_size or not set(tests) <= set(self.values):
            raise ValueError("Invalid/overlapping test budget")
        exploit = []
        used = set(tests)
        for variant in self.ordered_ids:
            if len(exploit) >= self.batch_size - len(tests): break
            if variant not in used: exploit.append(variant)
        return exploit + tests

    def cost(self, actions):
        return max((self.total - sum(self.values[v] for v in self.choose(actions))) / self.batch_size, 0.)


def nearest_effect(parent, site_index, rows):
    if not rows:
        return {"distance": None, "n": 0, "median_delta": None, "positive": 0, "negative": 0}
    distances = [sum(parent[i] != row["parent"][i] for i in range(4) if i != site_index) for row in rows]
    minimum = min(distances)
    nearest = [row for row, distance in zip(rows, distances) if distance == minimum]
    return {"distance": minimum, "n": len(nearest), "median_delta": round(float(np.median([r["delta"] for r in nearest])), 5),
            "positive": sum(r["delta"] > EPS for r in nearest), "negative": sum(r["delta"] < -EPS for r in nearest)}


def condition_rules(action, rows):
    """Observed conditional associations, never hard biochemical guarantees."""
    index = POSITIONS.index(action["site"])
    relevant = []
    for j, position in enumerate(POSITIONS):
        if j == index:
            continue
        for aa in sorted({p[j] for p in action["parents"]}):
            subset = [r for r in rows if r["parent"][j] == aa]
            if len(subset) >= 2:
                median = float(np.median([r["delta"] for r in subset]))
                relevant.append({"rule_id": f"K:{action['substitution']}|{position}{aa}",
                    "when": {str(position): aa}, "n": len(subset), "median_delta": round(median, 5),
                    "positive": sum(r["delta"] > EPS for r in subset), "negative": sum(r["delta"] < -EPS for r in subset),
                    "evidence_ids": [r["evidence_id"] for r in evidence_examples(subset, 2)],
                    "status": "observed_conditional_association_not_causal_rule"})
    return sorted(relevant, key=lambda r: (-r["n"], r["rule_id"]))[:2]


def build_bank(observed, scoring, *, limit=12, registry=None):
    known = observed_lookup(observed)
    evidence = one_edit_evidence(observed)
    groups = defaultdict(list)
    for child in sorted(scoring.values):
        if child in known:
            raise ValueError("Candidate was already observed")
        for i, site in enumerate(POSITIONS):
            for before in AMINO_ACIDS:
                if before == child[i]: continue
                parent = child[:i] + before + child[i + 1:]
                if parent in known: groups[(site, before, child[i])].append((parent, child))
    actions = []
    registry = registry or {}
    for (site, before, after), candidates in sorted(groups.items()):
        if len(candidates) < 2: continue
        def pair_key(pair):
            distance = sum(a != b for a, b in zip(pair[0][0], pair[1][0]))
            draft = {"children": [pair[0][1], pair[1][1]]}
            return (distance != 1, round(scoring.cost([draft]), 12), distance, pair)
        left, right = min(combinations(sorted(candidates), 2), key=pair_key)
        parents, children = [left[0], right[0]], [left[1], right[1]]
        key = f"{site}:{before}>{after}"
        rows = evidence.get(key, [])
        idx = POSITIONS.index(site)
        changed_background = [POSITIONS[j] for j in range(4) if parents[0][j] != parents[1][j]]
        action = {"action_id": key + "@" + "/".join(parents), "substitution": key,
            "site": site, "from_residue": before, "to_residue": after, "parents": parents, "children": children,
            "parent_fitness": [known[p] for p in parents], "evidence": evidence_examples(rows),
            "positive_context_count": sum(r["delta"] > EPS for r in rows), "negative_context_count": sum(r["delta"] < -EPS for r in rows),
            "observed_context_count": len(rows), "median_observed_delta": float(np.median([r["delta"] for r in rows])) if rows else None,
            "scope": {"backgrounds": parents, "differing_positions": changed_background,
                      "single_background_site": changed_background[0] if len(changed_background) == 1 else None},
            "nearest": [nearest_effect(p, idx, rows) for p in parents],
            "revision": copy.deepcopy(registry.get(key)), "knowledge_rules": condition_rules({"site": site, "parents": parents, "substitution": key}, rows)}
        predictions = [float(scoring.values[c]) for c in children]
        action["visible_stats"] = {"median_delta": round(action["median_observed_delta"], 5) if rows else None,
            "n_contexts": len(rows), "child_prediction": [round(v, 5) for v in predictions],
            "predicted_delta": [round(v - known[p], 5) for p, v in zip(parents, predictions)],
            "batch_cost_alone": round(scoring.cost([action]), 5),
            "unresolved": bool(action["positive_context_count"] and action["negative_context_count"] or len(rows) <= 2)}
        actions.append(action)
    selected = []
    for site in POSITIONS:
        subset = [a for a in actions if a["site"] == site]
        ranked = sorted(subset, key=lambda a: (a["visible_stats"]["batch_cost_alone"], -a["observed_context_count"], a["action_id"]))
        chosen = []
        for predicate in (lambda a: a["positive_context_count"] > 0 and not a["negative_context_count"],
                          lambda a: a["positive_context_count"] > 0 and a["negative_context_count"] > 0,
                          lambda a: a["observed_context_count"] <= 2):
            match = next((a for a in ranked if a not in chosen and predicate(a)), None)
            if match is not None: chosen.append(match)
        chosen += [a for a in ranked if a not in chosen]
        selected += chosen[:max(1, limit // 4)]
    normalized, _ = canonical_action_bank(selected[:limit])
    for action in normalized: validate_action(action, observed, set(scoring.values))
    return normalized


def rule_claim(action):
    directions = [sign(n["median_delta"]) if n["median_delta"] is not None else "uncertain" for n in action["nearest"]]
    if directions == ["increase", "increase"]: kind = "beneficial_both"
    elif set(directions) == {"increase", "decrease"}: kind = "background_dependent"
    else: kind, directions = "uncertain_transfer", ["uncertain", "uncertain"]
    return {"kind": kind, "expect_A": directions[0], "expect_B": directions[1],
        "response_to": action["revision"]["revision_id"] if action.get("revision") else "NEW",
        "reason": "Nearest observed background effects; revised after measured counterevidence when present"}


def ledger_update(actions, observed, measured, round_number, registry, selected_ids):
    known, truth = observed_lookup(observed), observed_lookup(measured)
    if set(truth) != set(selected_ids) or set(known) & set(truth): raise ValueError("Only new selected measurements may update beliefs")
    output = []
    for action in actions:
        claim = action["claim"]
        pairs = [{"parent": p, "child": c, "parent_fitness": known[p], "child_fitness": truth[c],
                  "delta": truth[c] - known[p], "expected": claim[f"expect_{label}"], "observed": sign(truth[c] - known[p])}
                 for label, p, c in zip(("A", "B"), action["parents"], action["children"]) if c in truth]
        if len(pairs) != 2:
            output.append({"round": round_number, "action_id": action["action_id"], "status": "untested" if not pairs else "incomplete", "pairs": pairs})
            continue
        if claim["kind"] == "uncertain_transfer": status = "exploration_observed"
        else: status = "supported" if all(p["expected"] == p["observed"] for p in pairs) else "refuted"
        observed_signs = [p["observed"] for p in pairs]
        revised_kind = "background_dependent" if set(observed_signs) == {"increase", "decrease"} else (
            "beneficial_both" if observed_signs == ["increase", "increase"] else "uncertain_transfer")
        prior = registry.get(action["substitution"], {})
        cumulative_pairs = prior.get("tested_pairs", []) + pairs
        if {"increase", "decrease"} <= {p["observed"] for p in cumulative_pairs}: revised_kind = "background_dependent"
        entry = {"revision_id": "R:" + action["substitution"], "substitution": action["substitution"],
            "last_status": status, "revised_kind": revised_kind, "tested_pairs": pairs,
            "has_refutation": prior.get("has_refutation", False) or status == "refuted",
            "revision_count": prior.get("revision_count", 0) + 1,
            "interpretation": "Specific measured backgrounds; no universal residue ban or significance claim"}
        entry["last_tested_pairs"] = pairs
        entry["tested_pairs"] = cumulative_pairs
        registry[action["substitution"]] = entry
        output.append({"round": round_number, "action_id": action["action_id"], "substitution": action["substitution"],
            "claim": claim, "scope": action["scope"], "status": status, "pairs": pairs, "revision": copy.deepcopy(entry)})
    return output


class Selector:
    def __init__(self, bank, scoring, allocation="adaptive"):
        self.bank, self.scoring, self.allocation = bank, scoring, allocation
        self.eligible = [a for a in bank if allocation == "fixed4" or
            (a["visible_stats"]["unresolved"] and max(a["visible_stats"]["predicted_delta"]) >= 0 and scoring.cost([a]) <= .1 + EPS)]
        self.pairs = [(a, b) for a, b in combinations(self.eligible, 2) if not set(a["children"]) & set(b["children"]) and
                      (allocation == "fixed4" or scoring.cost([a, b]) <= .1 + EPS)]
        self.target = 2 if self.pairs else 1 if self.eligible else 0

    def allowed(self, chosen):
        ids = {a["action_id"] for a in chosen}
        if len(chosen) >= self.target: return []
        permitted = ({a["action_id"] for pair in self.pairs if ids <= {b["action_id"] for b in pair} for a in pair}
                     if self.target == 2 else {a["action_id"] for a in self.eligible})
        return [a for a in self.bank if a["action_id"] in permitted - ids]

    def choose_rule(self):
        result = []
        for _ in range(self.target):
            chosen = copy.deepcopy(max(self.allowed(result), key=lambda a: (
                a["visible_stats"]["median_delta"] if a["visible_stats"]["median_delta"] is not None else -float("inf"),
                -a["visible_stats"]["batch_cost_alone"], a["action_id"])))
            chosen["claim"] = rule_claim(chosen)
            result.append(chosen)
        return result

    def metadata(self):
        return {"allocation": self.allocation, "target_actions": self.target, "eligible_ids": [a["action_id"] for a in self.eligible],
            "pair_ids": [[a["action_id"], b["action_id"]] for a, b in self.pairs], "cost_limit": .1,
            "calibrated_information_gain": False}
