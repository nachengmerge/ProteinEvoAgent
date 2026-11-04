"""Explicit, inspectable amino-acid rules; heuristic, not biochemical ground truth."""

from .config import POSITIONS, WT, mutation_names, validate_variant

AA_GROUP = {
    "A": "nonpolar", "V": "nonpolar", "I": "nonpolar", "L": "nonpolar",
    "M": "nonpolar", "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar",
    "G": "special", "P": "special", "D": "negative", "E": "negative",
    "K": "positive", "R": "positive", "H": "positive",
}


def substitution_risk(before: str, after: str) -> float:
    """Coarse property-change penalty; does not model structure or actual stability."""
    if before == after:
        return 0.0
    if AA_GROUP[before] == AA_GROUP[after]:
        return 0.0
    if {AA_GROUP[before], AA_GROUP[after]} <= {"nonpolar", "aromatic"}:
        return 0.25
    if "special" in (AA_GROUP[before], AA_GROUP[after]):
        return 1.0
    if "positive" in (AA_GROUP[before], AA_GROUP[after]) or "negative" in (AA_GROUP[before], AA_GROUP[after]):
        return 1.0
    return 0.6


def validate_candidate(variant: str, measured: set[str], parent: str | None = None, max_edits: int = 2) -> tuple[bool, str]:
    try:
        validate_variant(variant)
    except ValueError as exc:
        return False, str(exc)
    if variant in measured:
        return False, "already experimentally measured"
    if parent is not None:
        validate_variant(parent)
        if sum(a != b for a, b in zip(variant, parent)) > max_edits:
            return False, "too many edits relative to proposed parent"
    return True, "valid canonical amino acids at the four mutable sites"


def knowledge_penalty(variant: str, parent: str) -> float:
    """Small soft penalty (not a hard filter) for unsupported radical edits."""
    return sum(substitution_risk(a, b) for a, b in zip(parent, variant))


def describe(variant: str, parent: str | None = None) -> str:
    validate_variant(variant)
    mutations = ", ".join(mutation_names(variant)) or "wild type"
    groups = [f"{pos}:{AA_GROUP[aa]}" for pos, aa in zip(POSITIONS, variant)]
    ancestor = f"; parent={parent}" if parent else ""
    return f"{mutations}; property groups [{', '.join(groups)}]{ancestor}"

