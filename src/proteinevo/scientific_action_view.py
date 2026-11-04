"""Normalize scientific action sets before future LLM experiments.

This utility is not enabled in frozen P27/P28 studies. Input invariance is an
engineering property, not evidence of better fitness or scientific judgment.
"""
import copy


def canonical_action_bank(bank):
    """Return semantic IDs, deterministic ordering, and a reversible ID map.

    Parent/child correspondence and associated arrays are reordered together.
    Different fitness/statistics remain different; only incidental naming and
    presentation order are removed. Call normal experiment validators as usual.
    """
    if len({a["action_id"] for a in bank}) != len(bank):
        raise ValueError("Original action IDs must be unique")
    normalized, aliases = [], {}
    for original in bank:
        action = copy.deepcopy(original)
        if len(action["parents"]) != 2 or len(action["children"]) != 2:
            raise ValueError("Expected two complete parent/child tests")
        order = sorted(range(2), key=lambda i: (action["parents"][i], action["children"][i]))
        for field in ("parents", "children", "parent_fitness", "parent_backgrounds"):
            if field in action:
                if len(action[field]) != 2: raise ValueError("Background array length mismatch")
                action[field] = [action[field][i] for i in order]
        if "visible_stats" in action:
            for field in ("child_prediction", "predicted_delta"):
                action["visible_stats"][field] = [action["visible_stats"][field][i] for i in order]
        action["evidence"] = sorted(action["evidence"], key=lambda e: e["evidence_id"])
        semantic = f"{action['site']}:{action['from_residue']}>{action['to_residue']}@{'/'.join(action['parents'])}"
        if semantic in aliases:
            raise ValueError("Duplicate scientific action identity")
        aliases[semantic] = original["action_id"]
        action["action_id"] = semantic
        normalized.append(action)
    normalized.sort(key=lambda a: (a["site"], a["from_residue"], a["to_residue"], a["parents"], a["children"]))
    return normalized, aliases
