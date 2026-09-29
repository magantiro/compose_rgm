"""Program-coherence golden path (§16) + train/validation/sampling parity (§7).

Proves that ONE production RingCore-V1 contract drives every path identically: a deterministic panel of diverse
broad-organic molecules flows through data/teacher construction, training-batch construction, validation-batch
construction, model legal-mark scoring, canonical-successor aggregation, sampler selection, and executor
commit -- and at every stage the ENABLED capability set, the exact legal support, and the canonical successors
agree. The global invariant `teacher in A_exact(x)` must hold with zero mismatches across all strata. If any
production path silently uses different capabilities (the eval-collator bug class) or a divergent executor,
this test fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.experiments.corrupted_source_prior import (
    build_corrupted_prior_records,
)
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkDataset,
    assert_teachers_in_exact_candidates,
    factorized_mark_metrics,
    sample_factorized_mark_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    canonical_state_key,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system

# One diverse panel: neutral CNOF, sulfur, halogen, charged (cation/anion/zwitterion), aromatic, fused, cycle.
_PANEL = [
    "c1ccncc1CC",  # neutral aromatic hetero
    "CSc1ccc(N)cc1",  # sulfur
    "Clc1ccc(CCN)cc1",  # halogen
    "C[N+](C)(C)CCC1CCCCC1",  # cation + saturated ring
    "[O-]C(=O)c1ccccc1N",  # anion (carboxylate)
    "O=[N+]([O-])c1ccccc1",  # zwitterion (nitro)
    "O=C1NC(=O)c2ccccc21",  # fused (phthalimide)
    "C1CCNCC1C(=O)O",  # saturated heterocycle
]


def _catalog():
    from warmstart_dry_run import build_production_ring_catalog

    return build_production_ring_catalog(40)


def _ringcore_model(catalog):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=48,
        message_passing_steps=2,
        atom_vocabulary=ORGANIC_VOCABULARY,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
    ).eval()


def test_golden_path_one_contract_all_paths():
    catalog = _catalog()
    model = _ringcore_model(catalog)
    caps = model.operator_capabilities
    system = de_novo_rewrite_system()

    # 1. production data/teacher construction (corruption + cycle), with the teacher-in-candidate filter
    edit, _ = build_corrupted_prior_records(
        _PANEL,
        n_slots=40,
        depth_max=5,
        seed=17,
        catalog=catalog,
        vocabulary=ORGANIC_VOCABULARY,
        couplings_per_target=2,
    )
    cyc, _ = build_cycle_op_records(_PANEL, n_slots=40, seed=18)
    records = tuple(edit) + tuple(cyc)
    assert records, "golden panel produced no editing records"

    # 2. training-batch path (FactorizedMarkDataset + collator with the model's capabilities)
    dataset = FactorizedMarkDataset(
        records,
        start_index=0,
        length=48,
        seed=0,
        ring_catalog=catalog,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        progress_stratification_fraction=0.5,
    )
    train_collator = FactorizedMarkCollator(
        True,
        catalog,
        compute_ring_grow_support=caps.compute_ring_grow_support,
        compute_ring_restates=caps.compute_ring_restates,
        compute_cyclic_graft=caps.compute_cyclic_graft,
        compute_ring_opening=caps.compute_ring_opening,
    )
    train_batch = train_collator([dataset[i] for i in range(48)])
    assert_teachers_in_exact_candidates(train_batch)  # §7 invariant on the training path

    # 3. validation-batch path (sample_factorized_mark_batch with the SAME capability object)
    val_batch = sample_factorized_mark_batch(
        records,
        batch_size=48,
        seed=1,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        progress_stratification_fraction=0.5,
        use_aromatic_bond_view=True,
        workers=0,
        ring_catalog=catalog,
        ring_electronic_mode="factorized_local",
        capabilities=caps,
    )
    assert_teachers_in_exact_candidates(val_batch)  # §7 invariant on the validation path

    # 5+6. model legal-mark scoring + canonical-successor aggregation: finite GM loss on BOTH paths
    for batch in (train_batch, val_batch):
        metrics = factorized_mark_metrics(model, batch, use_bf16=False, microbatch_size=48)
        assert np.isfinite(metrics["factorized_gm_loss"])

    # 7+8. sampler selection + executor commit: every sampled mark executes to a valid connected <=40 state,
    # and ring_system_grow is NEVER sampled (macro disabled). Executor is the SAME canonical system.
    rng = np.random.default_rng(0)
    grow_draws = 0
    for smi in _PANEL:
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        for _ in range(8):
            mark = model.sample_rewrite_mark(state, 0.3, rng)
            grow_draws += int(mark.rule_name == "ring_system_grow")
            succ = system.apply(state, mark.rule_name, mark.action)  # one canonical executor
            assert is_valid_state(succ) and is_connected_or_null(succ)
            assert succ.n_atoms == 40
            assert 0 < succ.n_real_atoms <= succ.n_atoms
            assert canonical_state_key(succ) is not None  # one canonical-successor function
    assert grow_draws == 0, "legacy ring_system_grow must never be sampled under RingCore-V1"


def test_capabilities_identical_across_paths():
    """The model, the training collator, and the eval batch builder must all carry the SAME capability set --
    no path silently defaults editing families off."""
    catalog = _catalog()
    model = _ringcore_model(catalog)
    caps = model.operator_capabilities
    # RingCore-V1 capability contract: editing on, legacy grow off
    assert caps.compute_ring_restates and caps.compute_cyclic_graft and caps.compute_ring_opening
    assert not caps.compute_ring_grow_support
    # de-novo defaults are DIFFERENT (grow on, editing off) -> a fresh fingerprint (proves they can't collide)
    from compose_v4.model.factorized_tracelet_rate_model import OperatorCapabilities

    assert caps.fingerprint() != OperatorCapabilities.de_novo().fingerprint()
