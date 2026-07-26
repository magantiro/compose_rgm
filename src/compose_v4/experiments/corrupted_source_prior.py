"""Corrupted-molecule source-prior training records (native Generator Matching).

Mirrors ``tracelet_conditional.build_tree_transport_path_records``, but each record's teacher trace
comes from the **corrupted-molecule source prior** (``rewrite.source_corruption.make_edit_pair``)
instead of a carbon-tree transport -- i.e. the SAME rewrite-CTMC GM training with the source prior
swapped from carbon trees to lightly-corrupted real molecules. Both directions are emitted per
molecule (trim: real source; grow: corrupted source), sized for a mix against carbon-tree records.
This is *not* an "edit flow" or a bridge -- it is a source-prior change, nothing else.

Layering: the coupling primitive lives in ``rewrite.source_corruption`` (no experiments deps); this
module adds the ``PathRecord`` wrapper the GM trainer consumes.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import (CNOF_VOCABULARY, MolecularGraphError,
                                             smiles_to_molecular_graph)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.source_corruption import make_edit_pair


def build_corrupted_prior_records(
    smiles,
    *,
    n_slots: int,
    depth_max: int = 5,
    system=None,
    catalog=None,
    vocabulary=CNOF_VOCABULARY,
    seed: int = 0,
    couplings_per_target: int = 1,
    checkpoint_interval: int | None = None,
    both_directions: bool = True,
):
    """Return ``(records, n_attempted)``. Each attempt corrupts a real molecule a short, varied number
    of legal mark-family edits (``depth ~ U[1, depth_max]``) and emits up to two ``PathRecord``s: the
    trim direction (real molecule as source) and the grow direction (corrupted source). Molecules
    outside the model vocabulary or larger than ``n_slots`` are skipped; the yield is worth logging."""
    system = system or de_novo_rewrite_system()
    rng = np.random.default_rng(seed)
    records = []
    attempted = 0
    for text in smiles:
        try:
            graph = smiles_to_molecular_graph(text)
        except MolecularGraphError:
            continue  # element outside the C/N/O/F/P/S/halogen vocabulary (e.g. Si); skip
        if graph is None:
            continue
        try:
            target = pad_molecular_graph(graph, n_slots)
        except ValueError:
            continue  # molecule needs more than n_slots slots; skip
        for _ in range(couplings_per_target):
            attempted += 1
            depth = int(rng.integers(1, depth_max + 1))  # varied, short "noise level"
            trim, grow = make_edit_pair(
                target, depth, system=system, rng=rng, catalog=catalog,
                vocabulary=vocabulary,
            )
            traces = (trim, grow) if both_directions else (grow,)
            for trace in traces:
                if trace is None:
                    continue
                records.append(
                    PathRecord(
                        canonical_state_key(trace.target),
                        TraceProgressCTMC(trace, checkpoint_interval=checkpoint_interval),
                    )
                )
    return tuple(records), attempted
