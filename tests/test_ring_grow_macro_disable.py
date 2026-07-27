"""RING_CORE_V1: the legacy whole-ring ``ring_system_grow`` macro must be disable-able so ring ADDITION is
purely compositional (cycle_close). ``enable_ring_grow_macro`` defaults True (byte-identical to B / current
B-edit: grow fires); when False the grow family is masked dead (never sampled, never taught) while its head
params are RETAINED (warm-start-safe)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target  # noqa: E402
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog  # noqa: E402

_CHAIN = "CCCCCCCC"  # a saturated carbon chain: legacy ring_system_grow fires here


def _catalog():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=40
        )
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)

    return build_typed_ring_catalog(tuple(trace(s) for s in ("c1ccccc1", "c1ccncc1")))


def _grow_draws(enable: bool, n: int = 400) -> Counter:
    catalog = _catalog()
    state = pad_molecular_graph(smiles_to_molecular_graph(_CHAIN), 40)
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_grow_macro=enable
    ).eval()
    fams: Counter = Counter()
    rng = np.random.default_rng(0)
    for _ in range(n):
        mark = model.sample_rewrite_mark(state, float(rng.uniform(0.1, 0.6)), rng)
        fams[mark.rule_name] += 1
    return fams


def test_default_retains_the_grow_macro():
    # default True == byte-identical to B: grow still fires on a saturated scaffold
    fams = _grow_draws(enable=True)
    assert fams.get("ring_system_grow", 0) > 0


def test_disable_kills_the_grow_family():
    fams = _grow_draws(enable=False)
    assert fams.get("ring_system_grow", 0) == 0
    # the mass is not lost -- other families still fire (the family is dead, the sampler is not)
    assert sum(fams.values()) == 400 and len(fams) >= 4


def test_grow_head_params_retained_when_disabled():
    # warm-start safety: disabling must NOT drop the grow head tensors, so a B checkpoint still loads them
    catalog = _catalog()
    enabled = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_grow_macro=True
    )
    disabled = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_grow_macro=False
    )
    ek = {k: tuple(v.shape) for k, v in enabled.state_dict().items()}
    dk = {k: tuple(v.shape) for k, v in disabled.state_dict().items()}
    assert ek == dk  # identical parameter set + shapes -> strict/compatible warm-start unaffected
    assert any("ring_system_template" in name for name in dk)
