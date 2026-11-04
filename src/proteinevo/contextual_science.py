"""Background-specific evidence and executable single-edit experiment choices.

Inputs are already observed fitness and public candidate IDs. No oracle object,
hidden label table, pretrained risk weights, or model checkpoint enters here.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .config import AMINO_ACIDS, POSITIONS, WT, validate_variant
from .knowledge import AA_GROUP

TOLERANCE = 1e-9  # Numerical comparison only, not an assay-noise threshold.


def observed_lookup(observed):
    if set(observed.columns) != {"variant", "fitness"} or observed.empty or observed.variant.duplicated().any():
        raise ValueError("Exactly distinct observed variant/fitness records are required")
    if not np.isfinite(observed.fitness).all() or (observed.fitness < 0).any():
        raise ValueError("Observed fitness must be finite and nonnegative")
    result = dict(zip(observed.variant, map(float, observed.fitness)))
    for variant in result:
        validate_variant(variant)
    return result


def substitution_key(site, before, after):
    return f"{site}:{before}>{after}"


def background(variant, index):
    return variant[:index] + "_" + variant[index + 1:]


def one_edit_evidence(observed):
    """All observed single-edit contrasts; edges sharing labels are dependent."""
    known = observed_lookup(observed)
    groups = defaultdict(list)
    for parent in sorted(known):
        for index, site in enumerate(POSITIONS):
            for after in AMINO_ACIDS:
                if after == parent[index]:
                    continue
                child = parent[:index] + after + parent[index + 1:]
                if child not in known:
                    continue
                key = substitution_key(site, parent[index], after)
                groups[key].append({
                    "evidence_id": f"C:{parent}>{child}", "substitution": key,
                    "parent": parent, "child": child, "background": background(parent, index),
                    "parent_fitness": known[parent], "child_fitness": known[child],
                    "delta": known[child] - known[parent],
                    "delta_log1p": float(np.log1p(known[child]) - np.log1p(known[parent])),
                    "kind": "observed_single_edit_contrast",
                    "other_sites_match_WT": all(parent[j] == WT[j] for j in range(4) if j != index),
                })
    return dict(groups)


def evidence_examples(rows, limit=3):
    """Keep explicit contexts, including opposing signs when available."""
    positive = sorted((r for r in rows if r["delta"] > TOLERANCE), key=lambda r: (-r["delta"], r["evidence_id"]))
    negative = sorted((r for r in rows if r["delta"] < -TOLERANCE), key=lambda r: (r["delta"], r["evidence_id"]))
    selected = []
    for subset in (positive, negative, sorted(rows, key=lambda r: r["evidence_id"])):
        for row in subset:
            if row["evidence_id"] not in {r["evidence_id"] for r in selected}:
                selected.append(row)
                break
    return selected[:limit]


def contextual_analysis(observed, evidence=None):
    known = observed_lookup(observed)
    evidence = one_edit_evidence(observed) if evidence is None else evidence
    singles = []
    if WT in known:
        for child in sorted(known):
            edits = [i for i in range(4) if child[i] != WT[i]]
            if len(edits) == 1:
                i = edits[0]
                singles.append({"site": POSITIONS[i], "from_residue": WT[i], "to_residue": child[i],
                                "parent": WT, "child": child, "delta": known[child] - known[WT]})
    return {"n_observed": len(known), "best_seen_fitness": max(known.values()),
            "WT_single_effects": singles,
            "top_variants": observed.nlargest(8, "fitness").to_dict("records"),
            "oriented_single_edit_contrasts": sum(map(len, evidence.values())),
            "interpretation": "Contrasts retain source residue and exact background; shared labels are not independent replicates"}


def validate_action(action, observed, pool):
    known = observed_lookup(observed)
    site = action["site"]
    if type(site) is not int or site not in POSITIONS:
        raise ValueError("Invalid action site")
    i = POSITIONS.index(site)
    before, after = action["from_residue"], action["to_residue"]
    if before not in AMINO_ACIDS or after not in AMINO_ACIDS or len(before) != 1 or len(after) != 1 or before == after:
        raise ValueError("Action must make one actual canonical substitution")
    parents, children = action["parents"], action["children"]
    if len(parents) != 2 or len(set(parents)) != 2 or len(children) != 2 or len(set(children)) != 2:
        raise ValueError("A test action needs two distinct backgrounds and children")
    for parent, child, value in zip(parents, children, action["parent_fitness"], strict=True):
        if parent not in known or parent[i] != before or value != known[parent]:
            raise ValueError("Incorrect observed parent or parent fitness")
        expected = parent[:i] + after + parent[i + 1:]
        if child != expected or child in known or child not in pool:
            raise ValueError("Test child must be exactly one edit and an unmeasured pool ID")
    if action["substitution"] != substitution_key(site, before, after):
        raise ValueError("Substitution key disagrees with action")
    for row in action["evidence"]:
        parent, child = row["parent"], row["child"]
        if parent not in known or child not in known or parent[i] != before or child != parent[:i] + after + parent[i + 1:]:
            raise ValueError("Evidence does not match the actual substitution")
        if row["evidence_id"] != f"C:{parent}>{child}" or row["delta"] != known[child] - known[parent]:
            raise ValueError("Invented evidence or effect")
    return True


def build_action_bank(observed, pool, seed, *, limit=24, evidence=None):
    """Create a policy-independent, site-balanced shortlist before any LLM call.

    Pair selection prefers backgrounds differing at one other site; it never
    consults child fitness. The shortlist is a search restriction, not a claim
    to cover the full action space or all sequence interactions.
    """
    known = observed_lookup(observed)
    candidates = sorted(set(pool))
    if len(candidates) != len(pool) or set(candidates) & set(known):
        raise ValueError("Candidate IDs must be distinct and unmeasured")
    evidence = one_edit_evidence(observed) if evidence is None else evidence
    groups = defaultdict(list)
    for child in candidates:
        validate_variant(child)
        for i, site in enumerate(POSITIONS):
            for before in AMINO_ACIDS:
                if before == child[i]:
                    continue
                parent = child[:i] + before + child[i + 1:]
                if parent in known:
                    groups[(site, before, child[i])].append((parent, child))
    actions = []
    for (site, before, after), pairs in sorted(groups.items()):
        if len(pairs) < 2:
            continue
        def pair_order(pair):
            distance = sum(a != b for a, b in zip(pair[0][0], pair[1][0]))
            return (distance != 1, distance, pair[0][0], pair[1][0])
        left, right = min(combinations(sorted(pairs), 2), key=pair_order)
        key = substitution_key(site, before, after)
        rows = evidence.get(key, [])
        positive = sum(r["delta"] > TOLERANCE for r in rows)
        negative = sum(r["delta"] < -TOLERANCE for r in rows)
        category = "mixed" if positive and negative else "positive" if positive else "negative" if negative else "unseen_or_flat"
        parents, children = [left[0], right[0]], [left[1], right[1]]
        identity = "|".join([key, *parents, *children])
        actions.append({
            "action_id": "H:" + hashlib.sha256(identity.encode()).hexdigest()[:12],
            "substitution": key, "site": site, "from_residue": before, "to_residue": after,
            "parents": parents, "children": children, "parent_fitness": [known[p] for p in parents],
            "parent_backgrounds": [background(p, POSITIONS.index(site)) for p in parents],
            "background_distance": sum(a != b for a, b in zip(*parents)),
            "restores_WT_residue": after == WT[POSITIONS.index(site)],
            "evidence": evidence_examples(rows), "observed_context_count": len(rows),
            "positive_context_count": positive, "negative_context_count": negative,
            "median_observed_delta": float(np.median([r["delta"] for r in rows])) if rows else None,
            "evidence_category": category,
        })
    def stable(a):
        return hashlib.sha256(f"{seed}|{a['action_id']}".encode()).hexdigest()
    shortlist = []
    cycle = ("positive", "mixed", "negative", "unseen_or_flat", "positive", "mixed")
    for site in POSITIONS:
        site_actions = sorted([a for a in actions if a["site"] == site], key=stable)
        selected = []
        reversions = [a for a in site_actions if a["restores_WT_residue"]]
        if reversions:
            selected.append(reversions[0])
        for category in cycle:
            if len(selected) >= max(1, limit // 4):
                break
            remaining = [a for a in site_actions if a not in selected and a["evidence_category"] == category]
            if remaining:
                selected.append(remaining[0])
        selected.extend(a for a in site_actions if a not in selected)
        shortlist.extend(selected[:max(1, limit // 4)])
    shortlist = sorted(shortlist[:limit], key=lambda a: a["action_id"])
    # Preserve a feasible disjoint pair if the balanced shortlist accidentally lost it.
    if not any(not set(a["children"]) & set(b["children"]) for a, b in combinations(shortlist, 2)):
        disjoint = next(((a, b) for a, b in combinations(sorted(actions, key=stable), 2)
                         if not set(a["children"]) & set(b["children"])), None)
        if disjoint and limit >= 2:
            shortlist = list(disjoint) + [a for a in shortlist if a not in disjoint][:limit - 2]
    for action in shortlist:
        validate_action(action, observed, set(candidates))
    return shortlist, {"all_feasible_substitutions": len(actions), "shortlisted_actions": len(shortlist),
                       "shortlist_limit": limit, "uses_unmeasured_fitness": False}


def compatible_actions(bank, selected):
    used = {v for a in selected for v in a["children"]}
    available = [a for a in bank if a not in selected and not set(a["children"]) & used]
    if not selected:
        pairable = [a for a in available if any(b is not a and not set(a["children"]) & set(b["children"]) for b in available)]
        if pairable:
            return pairable
    return available


def choose_actions(bank, policy, rng):
    if policy not in {"random", "rule"}:
        raise ValueError("Unknown non-LLM action policy")
    selected = []
    for _ in range(2):
        allowed = compatible_actions(bank, selected)
        if not allowed:
            break
        if policy == "random":
            action = allowed[int(rng.integers(len(allowed)))]
        else:
            action = max(allowed, key=lambda a: (
                a["median_observed_delta"] if a["median_observed_delta"] is not None else -float("inf"),
                a["positive_context_count"], -a["negative_context_count"], a["action_id"]))
        selected.append(action)
    return selected


def select_batch(pool, actions, observed, batch_size=16):
    if set(pool.columns) != {"variant", "predicted_fitness"} or pool.variant.duplicated().any():
        raise ValueError("Only unique candidate IDs and predictions are allowed")
    if not np.isfinite(pool.predicted_fitness).all():
        raise ValueError("Nonfinite prediction")
    tested = []
    for action in actions:
        validate_action(action, observed, set(pool.variant))
        tested.extend(action["children"])
    if len(tested) != len(set(tested)) or len(tested) > batch_size:
        raise ValueError("Duplicate tests or excess experiment budget")
    exploit = pool.loc[~pool.variant.isin(tested)].sort_values("predicted_fitness", ascending=False, kind="stable").head(batch_size - len(tested)).copy()
    exploit["role"] = "exploitation"
    tests = pool.set_index("variant").loc[tested].reset_index().copy()
    tests["role"] = "hypothesis_test"
    result = pd.concat([exploit, tests], ignore_index=True)
    if len(result) != batch_size or result.variant.duplicated().any():
        raise ValueError("Could not fill exact distinct experiment budget")
    result["action_id"] = result.variant.map({v: a["action_id"] for a in actions for v in a["children"]})
    result["test_parent"] = result.variant.map({v: p for a in actions for p, v in zip(a["parents"], a["children"])})
    return result


def update_ledger(actions, observed_before, measured, round_number, *, selected_variants):
    known = observed_lookup(observed_before)
    results = observed_lookup(measured)
    if len(selected_variants) != len(set(selected_variants)) or set(results) != set(selected_variants):
        raise ValueError("Feedback may contain only the actual selected assays")
    if set(known) & set(results):
        raise ValueError("New measurements overlap known observations")
    ledger = []
    for action in actions:
        pairs = []
        for parent, child in zip(action["parents"], action["children"]):
            if parent not in known:
                raise ValueError("Test parent was not known before experiment")
            if child in results:
                pairs.append({"parent": parent, "child": child, "background": background(parent, POSITIONS.index(action["site"])),
                              "parent_fitness": known[parent], "child_fitness": results[child],
                              "delta": results[child] - known[parent],
                              "delta_log1p": float(np.log1p(results[child]) - np.log1p(known[parent]))})
        deltas = [p["delta"] for p in pairs]
        if not pairs:
            status = "untested"
        elif len(pairs) != 2:
            status = "incomplete_test"
        elif all(d > TOLERANCE for d in deltas):
            status = "supported_in_tested_backgrounds"
        elif any(d > TOLERANCE for d in deltas) and any(d < -TOLERANCE for d in deltas):
            status = "mixed_sign_background_dependence"
        elif any(d > TOLERANCE for d in deltas):
            status = "partial_support"
        else:
            status = "not_supported_in_tested_backgrounds"
        ledger.append({"round": round_number, "action_id": action["action_id"], "substitution": action["substitution"],
                       "expected_direction": "increase", "status": status, "pairs": pairs,
                       "interpretation": "Only tested backgrounds; no noise/significance or universal-effect claim"})
    return ledger


def property_card(action):
    return {"substitution": action["substitution"], "from_group": AA_GROUP[action["from_residue"]],
            "to_group": AA_GROUP[action["to_residue"]],
            "same_coarse_group": AA_GROUP[action["from_residue"]] == AA_GROUP[action["to_residue"]],
            "caution": "Coarse group is a weak prior, not measured stability or a guarantee of improved binding"}
