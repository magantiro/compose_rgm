"""Analogue-pair edit traces as GM training PathRecords (universal-edit-prior data).

Layer 1/3 of the master-plan training mixture (`docs/PAPER_MASTER_PLAN.md` §0b): real-molecule -> real-
molecule MMP/analogue A->B and B->A transitions, complementing the synthetic corruption traces (Layer 2)
so the edit prior learns real medicinal-chemistry transformations, not just corruption reconstruction
(the identity-bias guard). Reads a VERIFIED trace pool produced by ``scripts/build_analogue_trace_pool.py``
(each record already round-trip-checked through the real executor) and rebuilds each into the ``PathRecord``
the GM trainer consumes. The source is rebuilt from its exact SMILES (deterministic atom order) padded to
the recorded ``n_slots``; the target is the executed endpoint, so ``trace.target`` matches compile time.
"""
from __future__ import annotations

import json

from compose_v4.chem.molecular_graph import MolecularGraphError, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomDelete, AtomInsert, AtomRestate
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace, execute_trace

_SYSTEM = de_novo_rewrite_system()


def _dict_to_step(entry: dict) -> RewriteStep:
    if entry["rule"] == "atom_delete":
        return RewriteStep("atom_delete", AtomDelete(int(entry["v"])))
    if entry["rule"] == "atom_insert":
        return RewriteStep(
            "atom_insert",
            AtomInsert(
                slot=int(entry["slot"]),
                atom_type=int(entry["atom_type"]),
                formal_charge=int(entry["formal_charge"]),
                implicit_h_count=int(entry["implicit_h_count"]),
                neighbors=tuple((int(n), int(o)) for n, o in entry["neighbors"]),
            ),
        )
    if entry["rule"] == "atom_restate":
        return RewriteStep(
            "atom_restate",
            AtomRestate(
                v=int(entry["v"]),
                atom_type=int(entry["atom_type"]),
                formal_charge=int(entry["formal_charge"]),
                implicit_h_count=int(entry["implicit_h_count"]),
            ),
        )
    raise ValueError(f"unknown analogue-trace rule {entry['rule']!r}")


def rewrite_trace_from_record(record: dict) -> RewriteTrace:
    """Rebuild the ``RewriteTrace`` a verified pool record encodes."""
    source = pad_molecular_graph(
        smiles_to_molecular_graph(record["source_smiles"]), int(record["n_slots"])
    )
    steps = tuple(_dict_to_step(entry) for entry in record["steps"])
    target = execute_trace(source, steps, system=_SYSTEM)
    return RewriteTrace(source=source, target=target, steps=steps, metadata=dict(record.get("metadata", {})))


def build_analogue_prior_records(
    pool_path: str,
    *,
    count: int | None = None,
    checkpoint_interval: int | None = None,
) -> tuple[PathRecord, ...]:
    """Return ``PathRecord``s from a verified analogue trace pool (JSONL, one record per line). A record
    that fails to rebuild (out-of-vocab source, oversize, malformed) is skipped -- verified pools do not
    hit this, but stay defensive. ``count`` caps the number returned (for mixture sizing)."""
    records: list[PathRecord] = []
    with open(pool_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                trace = rewrite_trace_from_record(record)
            except (MolecularGraphError, ValueError, KeyError):
                continue
            records.append(
                PathRecord(
                    canonical_state_key(trace.target),
                    TraceProgressCTMC(trace, checkpoint_interval=checkpoint_interval),
                )
            )
            if count is not None and len(records) >= count:
                break
    return tuple(records)
