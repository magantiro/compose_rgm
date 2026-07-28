"""Global teacher-in-candidate invariant (owner mandate §5/§6): EVERY corruption teacher mark must belong to
the model's exact dynamic candidate set for the state it is scored against -- so the GM loss never raises
'outside exact dynamic candidates'. This covers the eval/validation path (sample_factorized_mark_batch),
which historically built its collator WITHOUT the editing-family enumeration flags, so ring_system_restate /
ring_system_delete / bond_reroute teachers landed outside the batch's candidates (on NEUTRAL and CHARGED
molecules alike -- charge was NOT the causal variable). The regression fails if that recurs for any family."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records  # noqa: E402
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.experiments.factorized_mark_conditional import (  # noqa: E402
    TeacherOutsideCandidatesError,
    assert_teachers_in_exact_candidates,
    factorized_mark_metrics,
    sample_factorized_mark_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
    OperatorCapabilities,
)

# Broad charge/element/topology strata (neutral, cation, anion, zwitterion, S/Cl, fused, hetero).
_LEADS = [
    "c1ccncc1", "CSc1ccc(N)cc1", "C1CCNCC1C(=O)O", "O=C(Nc1ccccc1)c1ccncc1",
    "CCN(CC)C(=O)c1ccc(N)cc1", "NC1CCCCN1C(=O)[O-]", "[O-]C(=O)c1ccccc1N",
    "O=[N+]([O-])c1ccccc1", "C[N+](C)(C)CCc1ccccc1", "Clc1ccc(CCN)cc1",
    "c1ccc2ccccc2c1CCN", "O=S(=O)(N)c1ccccc1",
]

_EDIT_FLAGS = dict(
    compute_ring_grow_support=False,
    compute_ring_restates=True,
    compute_cyclic_graft=True,
    compute_ring_opening=True,
)


def _catalog():
    from warmstart_dry_run import build_production_ring_catalog

    return build_production_ring_catalog(40)


def _model(catalog):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        catalog, hidden_dim=48, message_passing_steps=1, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_ring_restates=True, enable_cyclic_graft=True, enable_heteroatom_scan=True,
        enable_ring_opening=True, enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()


def _records(catalog):
    edit, _ = build_corrupted_prior_records(
        _LEADS, n_slots=40, depth_max=5, seed=17, catalog=catalog, vocabulary=ORGANIC_VOCABULARY
    )
    cyc, _ = build_cycle_op_records(_LEADS, n_slots=40, seed=18)
    return tuple(edit) + tuple(cyc)


def test_old_no_flags_reproduces_the_failure():
    """The historical bug: without the editing-family flags, some editing teachers fall outside candidates --
    now caught by the invariant at batch-construction time (TeacherOutsideCandidatesError) with rich context,
    before the scoring loss."""
    catalog = _catalog()
    records = _records(catalog)
    failures = 0
    for r in records:
        try:
            sample_factorized_mark_batch(
                (r,), batch_size=4, seed=0, late_time_fraction=0.5, operational_horizon=16.0,
                progress_stratification_fraction=0.5, use_aromatic_bond_view=True, workers=0,
                ring_catalog=catalog, ring_electronic_mode="factorized_local",
            )
        except TeacherOutsideCandidatesError as exc:
            assert "smiles=" in str(exc) and "capabilit" in str(exc)  # rich context
            failures += 1
    assert failures > 0, "expected the no-flags path to trip the teacher-in-candidate invariant"


def test_capability_object_drives_enumeration():
    """The immutable capability object is the single source of the editing-family enumeration; passing the
    model's operator_capabilities makes every teacher representable (no invariant trip)."""
    catalog = _catalog()
    model = _model(catalog)
    caps = model.operator_capabilities
    assert caps.compute_ring_restates and caps.compute_cyclic_graft and caps.compute_ring_opening
    assert not caps.compute_ring_grow_support  # RingCore: legacy grow disabled
    records = _records(catalog)
    for r in records:
        batch = sample_factorized_mark_batch(
            (r,), batch_size=4, seed=0, late_time_fraction=0.5, operational_horizon=16.0,
            progress_stratification_fraction=0.5, use_aromatic_bond_view=True, workers=0,
            ring_catalog=catalog, ring_electronic_mode="factorized_local", capabilities=caps,
        )
        assert_teachers_in_exact_candidates(batch)  # explicit invariant holds
    # de-novo capabilities are the editing-off defaults
    dn = OperatorCapabilities.de_novo()
    assert dn.compute_ring_grow_support and not dn.compute_ring_restates
    assert caps.fingerprint() != dn.fingerprint()


def test_teacher_in_candidates_with_editing_flags_all_strata():
    """The fix: with the editing-family flags, EVERY teacher (all families, all charge/element strata) is in
    the batch's dynamic candidates and scores to a finite GM loss -- zero mismatches."""
    catalog = _catalog()
    model = _model(catalog)
    records = _records(catalog)
    mismatches = []
    for r in records:
        fams = [s.rule_name for s in r.path.trace.steps]
        for seed in range(6):
            batch = sample_factorized_mark_batch(
                (r,), batch_size=4, seed=seed, late_time_fraction=0.5, operational_horizon=16.0,
                progress_stratification_fraction=0.5, use_aromatic_bond_view=True, workers=0,
                ring_catalog=catalog, ring_electronic_mode="factorized_local", **_EDIT_FLAGS,
            )
            try:
                metrics = factorized_mark_metrics(model, batch, use_bf16=False, microbatch_size=4)
            except RuntimeError as exc:
                mismatches.append((fams, str(exc)[:60]))
                break
            assert np.isfinite(metrics["factorized_gm_loss"])
    assert not mismatches, f"teacher-in-candidate mismatches: {mismatches}"
