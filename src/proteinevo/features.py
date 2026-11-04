"""GB1 sequence-only encoders; never incorporate experimental fitness labels."""

import numpy as np
from scipy.sparse import csr_matrix

from .config import AMINO_ACIDS, validate_variant

AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAIR_SITE_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def encode(variants: list[str] | np.ndarray) -> np.ndarray:
    """Four amino acids -> (n, 80) float32 one-hot design matrix."""
    result = np.zeros((len(variants), 4 * len(AMINO_ACIDS)), dtype=np.float32)
    for i, variant in enumerate(variants):
        validate_variant(str(variant))
        for site, aa in enumerate(str(variant)):
            result[i, site * 20 + AA_TO_IDX[aa]] = 1.0
    return result


def encode_pairwise(variants: list[str] | np.ndarray) -> csr_matrix:
    """Sparse main-effects + all six categorical pair interactions (n, 2480).

    One variant has exactly ten nonzero entries: four site identities and one
    residue-pair identity for each of the six possible position pairs. This
    allows strongly regularized additive/epistatic regression without creating
    dense 2480-dimensional matrices. No feature depends on measured fitness.
    """
    n = len(variants)
    row = np.repeat(np.arange(n, dtype=np.int32), 10)
    col = np.empty(n * 10, dtype=np.int32)
    for i, variant in enumerate(variants):
        sequence = str(variant)
        validate_variant(sequence)
        aa_ids = [AA_TO_IDX[aa] for aa in sequence]
        cols = [site * 20 + residue for site, residue in enumerate(aa_ids)]
        cols.extend(80 + pair * 400 + aa_ids[site_a] * 20 + aa_ids[site_b]
                    for pair, (site_a, site_b) in enumerate(PAIR_SITE_INDICES))
        col[i * 10:(i + 1) * 10] = cols
    values = np.ones(n * 10, dtype=np.float32)
    return csr_matrix((values, (row, col)), shape=(n, 2480), dtype=np.float32)

