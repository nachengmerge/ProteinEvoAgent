"""Observed-only, context-indexed evidence graph and matched flat retrieval.

The graph is an inspectable representation/retrieval index, not a learned GNN.
Shared measurements are retained as dependencies, not counted as replicates.
"""
import gzip
import hashlib
import json
from collections import Counter

from .config import POSITIONS, validate_variant
from .contextual_science import observed_lookup, one_edit_evidence
from .knowledge import AA_GROUP


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class EvidenceGraph:
    def __init__(self, observed):
        self.known = observed_lookup(observed)
        self.contrasts = one_edit_evidence(observed)
        self.nodes, self.edges = {}, []
        for variant, value in sorted(self.known.items()):
            vid = 'V:' + variant
            self.nodes[vid] = {'type': 'observed_variant', 'variant': variant, 'fitness': value}
            for site, aa in zip(POSITIONS, variant):
                rid = f'R:{site}:{aa}'
                self.nodes[rid] = {'type': 'site_residue', 'site': site, 'amino_acid': aa}
                pid = 'P:' + AA_GROUP[aa]
                self.nodes[pid] = {'type': 'coarse_property', 'property': AA_GROUP[aa]}
                self.edges.append((vid, 'contains', rid))
                self.edges.append((rid, 'has_property', pid))
        for key, rows in sorted(self.contrasts.items()):
            self.nodes['M:' + key] = {'type': 'substitution', 'substitution': key}
            site = int(key.split(':')[0])
            for row in rows:
                cid = row['evidence_id']
                bid = f'B:{site}:{row["background"]}'
                self.nodes[bid] = {'type': 'sequence_background', 'target_site': site, 'pattern': row['background']}
                self.nodes[cid] = dict(row, type='observed_contrast',
                    source_measurements=['V:' + row['parent'], 'V:' + row['child']],
                    interpretation='Observed association in this exact sequence background')
                self.edges.extend([(cid, 'has_parent', 'V:' + row['parent']),
                    (cid, 'has_child', 'V:' + row['child']),
                    (cid, 'applies_substitution', 'M:' + key),
                    (cid, 'has_background', bid)])
        self.edges = sorted(set(self.edges))
        self.state_sha256 = hashlib.sha256(canonical(sorted(self.known.items())).encode()).hexdigest()
        self.validate()

    def validate(self):
        for source, _, target in self.edges:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError('Dangling evidence graph edge')
        for rows in self.contrasts.values():
            for row in rows:
                if row['parent'] not in self.known or row['child'] not in self.known:
                    raise ValueError('Unknown outcomes cannot become observed evidence')
                if row['delta'] != self.known[row['child']] - self.known[row['parent']]:
                    raise ValueError('Evidence effect differs from source measurements')
        return True

    def record(self):
        return {'version': 'context-evidence-graph-v1', 'observed_state_sha256': self.state_sha256,
            'observed_n': len(self.known), 'nodes': self.nodes, 'edges': self.edges,
            'predicted_values_stored_as_observed': False,
            'independent_measurement_ids': ['V:' + v for v in sorted(self.known)]}

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fixed gzip timestamp makes graph exports reproducible across resumes.
        path.write_bytes(gzip.compress(canonical(self.record()).encode(), mtime=0))

    def retrieve(self, substitution, parents, *, mode, limit=3):
        if mode not in {'flat', 'graph'} or not 1 <= limit <= 4:
            raise ValueError('Fixed flat/graph retrieval with 1..4 evidence slots required')
        site = int(substitution.split(':')[0]); index = POSITIONS.index(site)
        for parent in parents:
            validate_variant(parent)
            if parent not in self.known:
                raise ValueError('A query parent must be an observed sequence')
        rows = list(self.contrasts.get(substitution, []))
        def distances(row):
            return [sum(row['parent'][j] != p[j] for j in range(4) if j != index) for p in parents]
        chosen = []
        def take(candidates):
            for row in candidates:
                if row['evidence_id'] not in {r['evidence_id'] for r in chosen}:
                    chosen.append(row); return
        if mode == 'flat':
            # Same available facts; query-independent positive/negative extremes.
            take(sorted((r for r in rows if r['delta'] > 1e-9), key=lambda r: (-r['delta'], r['evidence_id'])))
            take(sorted((r for r in rows if r['delta'] < -1e-9), key=lambda r: (r['delta'], r['evidence_id'])))
            remaining = sorted(rows, key=lambda r: r['evidence_id'])
        else:
            # Mutation -> contrast -> exact background; cover each queried context.
            for i in range(len(parents)):
                if len(chosen) >= limit: break
                take(sorted(rows, key=lambda r: (distances(r)[i], sum(distances(r)), r['evidence_id'])))
            local = [r for r in rows if min(distances(r)) <= 1]
            have_pos = any(r['delta'] > 1e-9 for r in chosen)
            have_neg = any(r['delta'] < -1e-9 for r in chosen)
            if len(chosen) < limit and have_pos != have_neg:
                opposing = [r for r in local if (r['delta'] < -1e-9 if have_pos else r['delta'] > 1e-9)]
                take(sorted(opposing, key=lambda r: (min(distances(r)), sum(distances(r)), r['evidence_id'])))
            remaining = sorted(rows, key=lambda r: (min(distances(r)), sum(distances(r)), r['evidence_id']))
        for row in remaining:
            if len(chosen) >= limit: break
            take([row])
        chosen = chosen[:limit]
        facts = [dict(id=r['evidence_id'], parent=r['parent'], child=r['child'],
            background=r['background'], observed_delta=r['delta'],
            distance_to_query=distances(r), source_measurements=['V:' + r['parent'], 'V:' + r['child']]) for r in chosen]
        return chosen, {'facts': facts, 'available_contrasts': len(rows),
            'distinct_measurements': len({v for r in chosen for v in (r['parent'], r['child'])}),
            'directions': dict(Counter('increase' if r['delta'] > .05 else 'decrease' if r['delta'] < -.05 else 'near_zero' for r in chosen)),
            'scope': 'Current observed contrasts only; related backgrounds are not causal guarantees; shared endpoints are dependent'}
