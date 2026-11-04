"""Fixed biological conventions used across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "elife-16965-supp1-v4.xlsx"
PROCESSED = ROOT / "data" / "processed" / "gb1_measured.csv.gz"
SPLITS = ROOT / "data" / "splits"
WT = "VDGV"
WT_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
POSITIONS = (39, 40, 41, 54)  # One-indexed on the 56-amino-acid domain.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20261027


def full_sequence(variant: str) -> str:
    """Insert a four-site variant into the published 56-aa GB1 WT sequence."""
    validate_variant(variant)
    residues = list(WT_SEQUENCE)
    for site, aa in zip(POSITIONS, variant):
        residues[site - 1] = aa
    return "".join(residues)


def validate_variant(variant: str) -> None:
    if not isinstance(variant, str) or len(variant) != 4 or any(
        aa not in AMINO_ACIDS for aa in variant
    ):
        raise ValueError(f"Invalid four-site GB1 variant: {variant!r}")


def mutation_count(variant: str) -> int:
    validate_variant(variant)
    return sum(a != b for a, b in zip(variant, WT))


def mutation_names(variant: str) -> list[str]:
    validate_variant(variant)
    return [f"{wt}{pos}{aa}" for pos, wt, aa in zip(POSITIONS, WT, variant) if wt != aa]

