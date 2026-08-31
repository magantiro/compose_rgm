"""The macro engine's job is to make three measured failures survivable.
Each test below pins one of them with numbers taken from the real audits."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from compose_v4.control.macro_engine import (
    MACRO_FAMILIES, proposal_support, macro_action_distribution)


def parp1_seed_like():
    """The measured parp1 seed shape: 57 cycle_close actions, all of whose
    global ranks fall past the top-300, plus a dominant atom_insert family."""
    fam = ["atom_insert"] * 216 + ["atom_restate_semantic"] * 123 + ["cycle_close"] * 57
    pr = np.concatenate([
        np.linspace(1.0, 0.10, 216),        # insert: the head
        np.linspace(0.09, 0.01, 123),       # restate
        np.full(57, 1e-6),                  # closures: all below the cap
    ])
    return fam, pr / pr.sum()


def test_global_cap_alone_loses_every_ring_closure():
    fam, pr = parp1_seed_like()
    top = np.argsort(-pr)[:300]
    assert not any(fam[i] == "cycle_close" for i in top), (
        "fixture wrong: closures must be below the cap, as measured")


def test_family_floor_recovers_the_ring_family():
    fam, pr = parp1_seed_like()
    s = proposal_support(fam, pr, cap=300, floor=20)
    n = sum(1 for i in s if fam[i] == "cycle_close")
    assert n == 20, f"floor must admit 20 closures, got {n}"


def test_support_growth_is_bounded():
    """Measured cost was ~+22 actions/state; the union must not blow up."""
    fam, pr = parp1_seed_like()
    s = proposal_support(fam, pr, cap=300, floor=20)
    assert 300 < len(s) <= 300 + 20 * len(set(fam))


def test_deep_within_family_tail_stays_reachable():
    """Two witness steps needed within-family ranks 137 and 109. Greedy
    top-1 gives them zero mass; the mixture must not."""
    fam = ["atom_insert"] * 293
    pr = np.exp(-np.arange(293) / 8.0)      # steep, like the real head
    pr = pr / pr.sum()
    support = proposal_support(fam, pr, cap=300, floor=20)
    clean = np.ones(293, dtype=bool)
    idx, q = macro_action_distribution(fam, pr, support, "grow", clean,
                                       temperature=2.0, epsilon=0.15)
    pos = int(np.flatnonzero(idx == 137)[0])
    assert q[pos] > 1e-4, f"rank-137 action effectively unreachable: q={q[pos]:.2e}"
    assert q[pos] > pr[137], "mixture must lift the tail above raw R_theta"


def test_masked_chemistry_gets_no_mass():
    fam = ["cycle_close"] * 10
    pr = np.full(10, 0.1)
    clean = np.ones(10, dtype=bool); clean[3] = False
    idx, q = macro_action_distribution(fam, pr, np.arange(10), "cyclize", clean)
    assert 3 not in idx
    assert abs(q.sum() - 1.0) < 1e-9, "must renormalise after masking"


def test_distribution_normalises_and_matches_macro_families():
    fam, pr = parp1_seed_like()
    s = proposal_support(fam, pr)
    clean = np.ones(len(fam), dtype=bool)
    for m in MACRO_FAMILIES:
        idx, q = macro_action_distribution(fam, pr, s, m, clean)
        if idx.size:
            assert abs(q.sum() - 1.0) < 1e-9
            assert all(fam[i] in MACRO_FAMILIES[m] for i in idx)


def test_empty_when_macro_has_no_legal_clean_action():
    fam = ["atom_insert"] * 5
    pr = np.full(5, 0.2)
    idx, q = macro_action_distribution(fam, pr, np.arange(5), "cyclize",
                                       np.ones(5, dtype=bool))
    assert idx.size == 0 and q.size == 0


def test_unknown_macro_raises():
    with pytest.raises(KeyError):
        macro_action_distribution(["atom_insert"], np.array([1.0]), np.array([0]),
                                  "not_a_macro", np.ones(1, dtype=bool))


def _fake_env(n=40, bad=frozenset()):
    """Deterministic stand-in for the model: action j maps to smiles 'S{j}'."""
    fams = ["atom_insert"] * (n // 2) + ["cycle_close"] * (n - n // 2)
    pr = np.exp(-np.arange(n) / 5.0); pr = pr / pr.sum()
    enumerate_fn = lambda smi: (fams, pr, list(range(n)))
    apply_fn = lambda smi, h: f"{smi}>{h}"
    gate_fn = lambda s: s.rsplit(">", 1)[-1] not in {str(b) for b in bad}
    return enumerate_fn, apply_fn, gate_fn


def test_rollout_executes_exactly_length_edits_without_scoring():
    from compose_v4.control.macro_engine import rollout
    e, a, g = _fake_env()
    r = rollout(e, a, g, "X", "grow", 5, np.random.default_rng(0))
    assert r.halted is None and r.length == 5
    assert r.endpoint.count(">") == 5, "each step must apply exactly one edit"


def test_rollout_halts_when_macro_has_no_legal_action():
    from compose_v4.control.macro_engine import rollout
    fams = ["atom_insert"] * 10
    pr = np.full(10, 0.1)
    e = lambda smi: (fams, pr, list(range(10)))
    r = rollout(e, lambda s, h: s + "!", lambda s: True, "X", "shrink", 3,
                np.random.default_rng(0))
    assert r.length == 0 and "no legal shrink action" in r.halted


def test_rollout_masks_gate_failures_and_continues():
    """A blocked action must not abort the program -- it must be redrawn."""
    from compose_v4.control.macro_engine import rollout
    e, a, _ = _fake_env()
    blocked = {"0", "1", "2", "3", "4"}
    g = lambda s: s.rsplit(">", 1)[-1] not in blocked
    r = rollout(e, a, g, "X", "grow", 3, np.random.default_rng(7))
    assert r.halted is None and r.length == 3
    assert not any(str(s.action_index) in blocked for s in r.steps)


def test_rollout_reaches_the_tail_not_just_the_head():
    """Over many rollouts the mixture must select beyond the top few actions."""
    from compose_v4.control.macro_engine import rollout
    e, a, g = _fake_env(n=40)
    seen = set()
    for s in range(60):
        r = rollout(e, a, g, "X", "grow", 1, np.random.default_rng(s))
        if r.length:
            seen.add(r.steps[0].action_index)
    assert max(seen) > 5, f"only head actions ever drawn: {sorted(seen)}"


def test_preference_is_not_a_gate_and_never_halts_a_macro():
    """If nothing satisfies the preference, the macro must still execute using
    the gate alone. A preference that can halt a program is a gate in disguise."""
    from compose_v4.control.macro_engine import rollout
    e, a, g = _fake_env(n=20)
    r = rollout(e, a, g, "X", "grow", 3, np.random.default_rng(3),
                prefer_fn=lambda s: False)          # nothing ever preferred
    assert r.halted is None and r.length == 3, (
        f"preference halted the macro: {r.halted}")


def test_preference_is_used_when_satisfiable():
    from compose_v4.control.macro_engine import rollout
    e, a, g = _fake_env(n=20)
    ok = {"2", "3", "4", "5", "6", "7"}
    r = rollout(e, a, g, "X", "grow", 3, np.random.default_rng(5),
                prefer_fn=lambda s: s.rsplit(">", 1)[-1] in ok)
    assert r.halted is None
    assert all(str(s.action_index) in ok for s in r.steps)


def test_synthesis_preference_matches_the_measurement():
    from compose_v4.control.macro_engine import synthesis_preference
    # the one clean molecule the grammar test produced: no bridgehead, no stereo
    assert synthesis_preference("CC(C)c1cnnc2c(Cl)c3c(cc12)CNC(=O)c1cccn1-3")
    # norbornane: the canonical bridged system SA punishes
    assert not synthesis_preference("C1CC2CCC1C2")
    assert not synthesis_preference("not_a_smiles")


def test_ring_systems_groups_fused_rings():
    from rdkit import Chem
    from compose_v4.control.macro_engine import ring_systems
    # naphthalene: two rings, ONE system
    assert len(ring_systems(Chem.MolFromSmiles("c1ccc2ccccc2c1"))) == 1
    # biphenyl: two rings, TWO systems
    assert len(ring_systems(Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1"))) == 2
    # the parp1 seed: one fused tricycle
    assert len(ring_systems(Chem.MolFromSmiles(
        "CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3"))) == 1


def test_disjoint_closure_rejects_annulation_accepts_pendant():
    from compose_v4.control.macro_engine import disjoint_closure_only
    ok = disjoint_closure_only("c1ccccc1")          # benzene: 1 system
    assert not ok("c1ccc2ccccc2c1")                  # naphthalene: still 1 -> reject
    assert ok("c1ccc(-c2ccccc2)cc1")                 # biphenyl: 2 -> accept
    assert not ok("not_a_smiles")


def test_ivg_winners_have_multiple_ring_systems_ours_do_not():
    """Pins the finding this macro exists to fix."""
    import json, pathlib
    from rdkit import Chem
    from compose_v4.control.macro_engine import ring_systems
    root = pathlib.Path(__file__).resolve().parents[1]
    W = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    for w in W["parp1_s0_d0.4"]["winners"]:
        n = len(ring_systems(Chem.MolFromSmiles(w["smiles"])))
        assert n >= 2, f"IVG winner unexpectedly single-system: {w['smiles']}"
    ours = "CC1=C(C(C)C)c2ccc3c(c2C1=O)CNC(=O)c1cccn1-3"
    assert len(ring_systems(Chem.MolFromSmiles(ours))) == 1


def test_backbone_only_rejects_halogen_inserts():
    from compose_v4.control.macro_engine import backbone_only
    ok = backbone_only("c1ccccc1")
    assert ok("Cc1ccccc1")            # added carbon
    assert ok("Nc1ccccc1")            # added nitrogen: still backbone-capable
    assert not ok("Fc1ccccc1")        # terminal halogen
    assert not ok("Clc1ccccc1")


def test_append_closure_requires_a_substantial_new_system():
    from compose_v4.control.macro_engine import append_system_closure
    ok = append_system_closure("c1ccccc1CCCCCC")     # benzene + a chain
    assert not ok("c1ccccc1C1CN1")                    # 3-atom aziridine: too small
    assert ok("c1ccccc1C1CCCCC1")                     # 6-atom ring: accepted
    assert not ok("c1ccc2ccccc2c1")                   # annulated, no new system


def test_append_minimum_is_documented_as_macro_scope_not_validity():
    """One parp1 IVG winner carries a 3-atom cyclopropane pendant. The minimum
    must NOT be a validity rule or it would reject real winning chemistry."""
    import json, pathlib
    from rdkit import Chem
    from compose_v4.gates.med_chem_gate import is_valid
    root = pathlib.Path(__file__).resolve().parents[1]
    W = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    # NOTE: bind the molecule. GetRingInfo() returns a reference OWNED by the
    # mol, so calling it on a temporary -- Chem.MolFromSmiles(x).GetRingInfo()
    # -- can read freed ring data and silently return no rings. That made this
    # test claim no winner had a 3-membered ring when one plainly does.
    small = []
    for w in W["parp1_s0_d0.4"]["winners"]:
        m = Chem.MolFromSmiles(w["smiles"])
        if 3 in [len(r) for r in m.GetRingInfo().AtomRings()]:
            small.append(w["smiles"])
    assert small, "expected a winner with a 3-membered ring"
    for smi in small:
        assert is_valid(smi), "gate must still accept a winner with a small ring"


# --- the four-layer hierarchy, tested explicitly -------------------------------

def _winners_and_seeds():
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    W = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    S = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
    return W, S


def test_layer1_global_validity_accepts_all_seeds_and_all_winners():
    from compose_v4.gates.med_chem_gate import is_valid, validity_reasons
    W, S = _winners_and_seeds()
    for s in S:
        assert is_valid(s["smiles"]), (s["target"], validity_reasons(s["smiles"]))
    for cell, rec in W.items():
        for w in rec["winners"]:
            assert is_valid(w["smiles"]), (cell, validity_reasons(w["smiles"]))


def test_layer1_rejects_only_genuine_pathology():
    from compose_v4.gates.med_chem_gate import is_valid
    for bad in ("C1#[SH2]CNC=N1", "O=C1NCCn2cc([IH4])c3cccc1c32",
                "O=C1NCCc2c(OC(O)=C[PH2](F)F)[nH]c3cccc1c23"):
        assert not is_valid(bad)
    # ...but NOT unconventional-yet-legal chemistry
    for fine in ("C1CC1", "C1CCCCCC1", "C1CC2CCC1C2", "c1ccsc1"):
        assert is_valid(fine), f"global gate must not ban {fine}"


def test_layer2_cyclopropane_fails_append_but_passes_small_ring():
    """The exact case that would break a global minimum-ring-size rule."""
    from compose_v4.control.macro_engine import contract_for
    before = "c1ccccc1CCCCCC"
    prod = "c1ccccc1C1CN1"
    assert not contract_for("append_system", before)(prod)
    assert contract_for("small_ring", before)(prod)


def test_layer2_annulate_and_append_are_mutually_exclusive():
    from compose_v4.control.macro_engine import contract_for
    before = "c1ccccc1CCCCCC"
    fused = "c1ccc2ccccc2c1"          # annulated onto the benzene
    pendant = "c1ccccc1C1CCCCC1"      # separate new system
    assert contract_for("annulate", before)(fused)
    assert not contract_for("append_system", before)(fused)
    assert contract_for("append_system", before)(pendant)
    assert not contract_for("annulate", before)(pendant)


def test_layer2_decorate_owns_halogens_scaffold_extend_refuses_them():
    from compose_v4.control.macro_engine import contract_for
    before = "c1ccccc1"
    assert contract_for("decorate", before)("Fc1ccccc1")
    assert not contract_for("scaffold_extend", before)("Fc1ccccc1")
    assert contract_for("scaffold_extend", before)("Cc1ccccc1")


def test_every_macro_with_a_contract_is_documented():
    from compose_v4.control.macro_engine import MACRO_CONTRACTS, contract_for
    for m in MACRO_CONTRACTS:
        assert contract_for(m, "c1ccccc1") is not None, f"{m} declares no contract"


def test_backbone_excludes_S_and_P_not_just_halogens():
    """Dropping halogens alone left S and P as ~52% of remaining support."""
    from compose_v4.control.macro_engine import backbone_only
    ok = backbone_only("c1ccccc1")
    assert ok("Cc1ccccc1") and ok("Nc1ccccc1") and ok("Oc1ccccc1")
    for bad in ("Sc1ccccc1", "Fc1ccccc1", "Clc1ccccc1"):
        assert not ok(bad), f"backbone must exclude {bad}"


def test_build_ring_system_is_a_program_over_existing_macros():
    from compose_v4.control.macro_engine import (BUILD_RING_SYSTEM, MACRO_FAMILIES,
                                                 MACRO_CONTRACTS)
    assert len(BUILD_RING_SYSTEM) >= 3
    for macro, length in BUILD_RING_SYSTEM:
        assert macro in MACRO_FAMILIES, f"{macro} is not an existing macro"
        assert length >= 1
    # it must not introduce a new primitive
    fams = {f for m, _ in BUILD_RING_SYSTEM for f in MACRO_FAMILIES[m]}
    # the executor's actual rule set, confirmed by enumerating the fiber on a
    # real state: bond_reroute, atom_insert, atom_restate, bond_insert,
    # bond_reorder, atom_delete, bond_delete, plus the cycle/ring rules
    known = {"atom_insert", "cycle_close", "cycle_open", "atom_restate_semantic",
             "bond_reorder", "bond_reroute", "ring_system_restate", "atom_delete",
             "bond_insert", "bond_delete", "atom_restate"}
    assert fams <= known, f"unknown primitive introduced: {fams - known}"


def test_build_ring_system_is_indivisible_and_endpoint_judged():
    """BUILD_RING_SYSTEM(state=aromatic) must execute grow->close->electronic
    completion as ONE action and report satisfaction only at the endpoint.

    The earlier design attached the aromatic condition to the CLOSURE, which no
    single action can satisfy, then let a generic restate sequence wander. This
    asserts the macro returns a single verdict rather than a menu of separately
    judged realisations."""
    from compose_v4.control.macro_engine import build_ring_system
    e, a, g = _fake_env(n=40)
    out = build_ring_system(e, a, g, "X", np.random.default_rng(1),
                            size=3, state="saturated")
    assert isinstance(out, dict), "one action, one verdict"
    assert "satisfied" in out and "trace" in out
    assert isinstance(out["trace"], list) and out["trace"]


def test_build_ring_system_uses_only_existing_primitives():
    from compose_v4.control.macro_engine import BUILD_RING_SYSTEM, MACRO_FAMILIES
    prims = {f for m, _ in BUILD_RING_SYSTEM for f in MACRO_FAMILIES[m]}
    assert prims <= {"atom_insert", "cycle_close", "bond_insert",
                     "atom_restate_semantic", "bond_reorder", "ring_system_restate"}


def test_decorate_is_not_part_of_ring_construction():
    """A halogen must never consume a backbone growth step."""
    from compose_v4.control.macro_engine import BUILD_RING_SYSTEM
    assert "decorate" not in {m for m, _ in BUILD_RING_SYSTEM}


# --- orthogonal ring taxonomy --------------------------------------------------

def test_all_four_topologies_are_distinguished():
    from compose_v4.control.macro_engine import _ring_relationship
    c = _ring_relationship("c1ccccc1")
    assert c("c1ccccc1-c1ccccc1") == "pendant"
    assert c("c1ccc2ccccc2c1") == "fused"
    assert c("C1CCC2(CC1)CCCCC2") == "spiro"
    assert c("c1ccccc1C1CC2CCC1C2") == "bridged"
    assert c("Cc1ccccc1") is None          # no new ring


def test_topology_contract_rejects_the_wrong_topology():
    from compose_v4.control.macro_engine import ring_topology_contract
    pendant = ring_topology_contract("c1ccccc1", "pendant")
    fused = ring_topology_contract("c1ccccc1", "fused")
    assert pendant("c1ccccc1-c1ccccc1") and not pendant("c1ccc2ccccc2c1")
    assert fused("c1ccc2ccccc2c1") and not fused("c1ccccc1-c1ccccc1")


def test_scale_is_orthogonal_to_topology():
    """Same topology, different scale -- the axes must not be entangled."""
    from compose_v4.control.macro_engine import ring_topology_contract
    small = ring_topology_contract("c1ccccc1", "pendant", scale="small")
    medium = ring_topology_contract("c1ccccc1", "pendant", scale="medium")
    assert small("c1ccccc1C1CC1") and not medium("c1ccccc1C1CC1")
    assert medium("c1ccccc1C1CCCCC1") and not small("c1ccccc1C1CCCCC1")


def test_no_topology_is_globally_banned():
    """The gate judges legitimacy; the contract judges intent. A bridged ring,
    a spirocycle and a cyclopropane are all legitimate chemistry."""
    from compose_v4.gates.med_chem_gate import is_valid
    for smi in ("c1ccccc1-c1ccccc1", "c1ccc2ccccc2c1",
                "C1CCC2(CC1)CCCCC2", "c1ccccc1C1CC2CCC1C2", "C1CC1"):
        assert is_valid(smi), f"global gate must not ban {smi}"


def test_unknown_topology_raises():
    from compose_v4.control.macro_engine import ring_topology_contract
    with pytest.raises(KeyError):
        ring_topology_contract("c1ccccc1", "helicene")


def test_scaffold_extend_composition_is_a_mode_not_a_definition():
    from compose_v4.control.macro_engine import backbone_only, COMPOSITION_MODES
    carbon = backbone_only("c1ccccc1", composition="carbon_rich")
    mixed = backbone_only("c1ccccc1", composition="mixed")
    hetero = backbone_only("c1ccccc1", composition="hetero_rich")
    assert carbon("Cc1ccccc1") and not carbon("Nc1ccccc1")
    assert mixed("Cc1ccccc1") and mixed("Nc1ccccc1")
    assert hetero("Nc1ccccc1") and not hetero("Cc1ccccc1")


def test_halogens_excluded_from_every_composition_mode():
    """They cannot extend a chain in any mode -- DECORATE owns them."""
    from compose_v4.control.macro_engine import backbone_only, COMPOSITION_MODES
    for mode in COMPOSITION_MODES:
        ok = backbone_only("c1ccccc1", composition=mode)
        for hal in ("Fc1ccccc1", "Clc1ccccc1", "Brc1ccccc1"):
            assert not ok(hal), f"{mode} admitted a terminal halogen"


def test_unknown_composition_raises():
    from compose_v4.control.macro_engine import backbone_only
    with pytest.raises(KeyError):
        backbone_only("c1ccccc1", composition="all_boron")


def test_ring_state_contract_distinguishes_aromatic_from_saturated():
    from compose_v4.control.macro_engine import ring_state_contract
    before = "c1ccccc1CCCCCC"                 # one aromatic ring, plus a chain
    aromatic = ring_state_contract(before, "aromatic")
    saturated = ring_state_contract(before, "saturated")
    assert aromatic("c1ccccc1-c1ccccc1")      # new ring IS aromatic
    assert not saturated("c1ccccc1-c1ccccc1")
    assert saturated("c1ccccc1C1CCCCC1")      # new ring is NOT aromatic
    assert not aromatic("c1ccccc1C1CCCCC1")


def test_no_new_ring_fails_every_state():
    from compose_v4.control.macro_engine import ring_state_contract, RING_STATES
    for st in RING_STATES:
        assert not ring_state_contract("c1ccccc1", st)("Cc1ccccc1")


def test_aromatisable_ring_screens_precursors():
    from rdkit import Chem
    from compose_v4.control.macro_engine import aromatisable_ring
    m = Chem.MolFromSmiles("C1CCCCC1")        # cyclohexane
    assert aromatisable_ring(m, m.GetRingInfo().AtomRings()[0])
    m2 = Chem.MolFromSmiles("C1CC1")          # cyclopropane: too small
    assert not aromatisable_ring(m2, m2.GetRingInfo().AtomRings()[0])
    m3 = Chem.MolFromSmiles("CC1(C)CCCCC1")   # quaternary carbon in the ring
    assert not aromatisable_ring(m3, m3.GetRingInfo().AtomRings()[0])


def test_saturated_remains_a_legitimate_branch():
    """Aromaticity is intent, not law -- purpose decides."""
    from compose_v4.control.macro_engine import RING_STATES
    assert "saturated" in RING_STATES and "aromatic" in RING_STATES


def test_local_extend_matches_the_measured_successes():
    """The three sim>=0.5 endpoints anchored at 2 adjacent aliphatic atoms;
    every failure spread across 5-8 including the aromatic core."""
    import json, pathlib
    from compose_v4.control.macro_engine import local_extend
    root = pathlib.Path(__file__).resolve().parents[1]
    seed = [s for s in json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
            if s["target"] == "parp1" and s["idx"] == 0][0]["smiles"]
    ok = local_extend(seed, seed, max_anchors=2, max_separation=2)
    good = "CCN1CN(Cc2ccc3c(c2)CNC(=O)c2cccn2-3)C(C)C(=O)C1S"   # sim 0.557
    assert ok(good), "the measured best endpoint must satisfy the contract"
    assert not ok(seed), "no growth at all is not an extension"


def test_ring_formation_is_not_confined_to_cycle_close():
    """Measured: a real six-carbon precursor offered ZERO cycle_close actions
    and 127 bond_insert, two of which close the chain into a six-ring."""
    from compose_v4.control.macro_engine import MACRO_FAMILIES
    for m in ("cyclize", "append_system", "annulate", "small_ring"):
        assert "bond_insert" in MACRO_FAMILIES[m], f"{m} cannot see bond_insert"


def test_provenance_is_scoped_to_topology_not_global():
    """A pendant must close among NEW atoms; an annulation must involve old ring
    atoms. A global new-atoms-only rule would break fused/spiro construction."""
    from compose_v4.control.macro_engine import closure_from_new_atoms
    before = "c1ccccc1"
    after = "c1ccccc1CCCCCC"          # six new carbons on a benzene
    pend = closure_from_new_atoms(before, after, topology="pendant", size=6)
    fuse = closure_from_new_atoms(before, after, topology="fused", size=6)
    ring_of_new = "c1ccccc1C1CCCCC1"  # closed among the new six
    assert pend(ring_of_new)
    assert not fuse(ring_of_new), "fused must require old ring atoms"


class _Ins:
    def __init__(self, slot, nbrs): self.slot=slot; self.neighbors=tuple((n,1) for n in nbrs)
class _Bond:
    def __init__(self, a, b): self.a=a; self.b=b


def test_ring_frontier_tracks_the_grown_chain():
    from compose_v4.control.macro_engine import RingFrontier
    f = RingFrontier(size=6, topology="pendant")
    f.observe_insert(_Ins(22, [6]))          # first new atom attaches to the seed
    for k, prev in zip(range(23, 28), range(22, 27)):
        f.observe_insert(_Ins(k, [prev]))
    assert f.path == [22, 23, 24, 25, 26, 27]
    assert f.anchors == [6]
    assert f.closure_pairs() == [(22, 27)], "exactly one endpoint pair, not 127"


def test_descriptor_matching_finds_only_the_requested_closure():
    from compose_v4.control.macro_engine import match_closure_descriptors
    fams = ["bond_insert"] * 5 + ["atom_insert"]
    acts = [_Bond(22, 24), _Bond(22, 27), _Bond(23, 25), _Bond(27, 22), _Bond(6, 9), None]
    idx = match_closure_descriptors(fams, acts, [(22, 27)])
    assert idx == [1, 3], "both orderings of the requested pair, nothing else"


def test_empty_intersection_is_unsat_not_fallback():
    from compose_v4.control.macro_engine import match_closure_descriptors
    fams = ["bond_insert", "bond_insert"]
    acts = [_Bond(1, 2), _Bond(3, 4)]
    assert match_closure_descriptors(fams, acts, [(22, 27)]) == []


def test_topology_changes_the_candidate_pairs():
    from compose_v4.control.macro_engine import RingFrontier
    f = RingFrontier(size=6, topology="pendant")
    f.observe_insert(_Ins(22, [6]))
    for k, prev in zip(range(23, 28), range(22, 27)):
        f.observe_insert(_Ins(k, [prev]))
    pend = f.closure_pairs()
    f.topology = "fused"
    assert f.closure_pairs() != pend, "topology must change the query"


def _t4_seed(target, idx):
    """Load a seed from the checked artifact. Never hand-copy a benchmark SMILES:
    the string typed here previously had 17 heavy atoms against the real seed's
    16, so the tests exercised a molecule that is not in the benchmark at all."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
    return next(r["smiles"] for r in rows
                if r["target"] == target and int(r["idx"]) == int(idx))


def _local_macro(seed, **kw):
    """Run BUILD_RING_SYSTEM against the real executor and real fiber with
    uniform probabilities. This exercises every mechanism except R_theta's
    ranking, and it runs on CPU in under a second."""
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
        molecular_graph_to_smiles, AtomVocabulary, ATOM_VALENCE_CLASSES)
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.rewrite.factorized_fiber import _factorized_candidates
    from compose_v4.rewrite.tracelets import _micro_runtime
    from compose_v4.control.macro_engine import build_ring_system_exact
    from compose_v4.gates.med_chem_gate import is_executable
    voc = AtomVocabulary(ATOM_VALENCE_CLASSES); sysm = _micro_runtime()
    cache = {}

    def enum_full(st):
        k = molecular_graph_to_smiles(st)
        if k not in cache:
            f, a = [], []
            for r, x in _factorized_candidates(st, allow_bond_reroute=True,
                                               vocabulary=voc):
                f.append(r); a.append(x)
            cache[k] = (f, a, [1.0] * len(f))
        return cache[k]

    def apply_fn(st, j):
        f, a, _ = enum_full(st)
        try:
            return sysm.apply(st, f[j], a[j])
        except Exception:
            return None

    def to_smiles(st):
        try:
            return molecular_graph_to_smiles(st)
        except Exception:
            return None

    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed), 48)
    return build_ring_system_exact(enum_full, apply_fn, to_smiles, is_executable,
                                   st0, **kw)


def test_build_ring_system_makes_an_aromatic_pendant():
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    seed = _t4_seed("5ht1b", 7)
    r = _local_macro(seed, size=6, topology="pendant", state="aromatic")
    assert r["status"] == "OK", r
    assert r["sys_gain"] == 1 and r["arom_gain"] == 1, r
    m = Chem.MolFromSmiles(r["smiles"])
    assert m is not None
    assert int(rdMD.CalcNumAromaticRings(m)) == \
        int(rdMD.CalcNumAromaticRings(Chem.MolFromSmiles(seed))) + 1


def test_closure_examines_two_candidates_not_the_whole_fiber():
    r = _local_macro(_t4_seed("5ht1b", 7), size=6, state="aromatic")
    assert r["status"] == "OK"
    assert r["n_candidates"] <= 4, r
    assert r["n_fiber"] > 100, "the fiber really is large; the point is we skip it"
    assert r["n_candidates"] < 0.05 * r["n_fiber"]


def test_trace_shows_grow_then_new_new_closure_then_electronic():
    r = _local_macro(_t4_seed("5ht1b", 7), size=6, state="aromatic")
    tr = r["trace"]
    assert sum(1 for t in tr if t.startswith("atom_insert")) == 6, tr
    closure = [t for t in tr if t.startswith("bond_insert")]
    assert len(closure) == 1, tr
    # both endpoints must be atoms this macro grew
    a, b = closure[0].split("(")[1].split(")")[0].split(",")
    grown = {t.split("@")[1].split("(")[0] for t in tr if t.startswith("atom_insert")}
    assert a in grown and b in grown, (closure, grown)
    assert any("reorder" in t or "restate" in t for t in tr), tr


def test_product_is_t4_feasible_at_delta_0_4():
    import os, sys as _s
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    _s.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    seed = _t4_seed("5ht1b", 7)
    r = _local_macro(seed, size=6, state="aromatic")
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    p = Chem.MolFromSmiles(r["smiles"])
    sim = DataStructs.TanimotoSimilarity(g.GetFingerprint(Chem.MolFromSmiles(seed)),
                                         g.GetFingerprint(p))
    # similarity is the binding constraint and the one every earlier macro
    # measurement omitted; a product that passes QED and SA alone is not a result
    assert QED.qed(p) >= 0.6 and sascorer.calculateScore(p) <= 4.0
    assert sim >= 0.4, f"sim={sim:.3f}"


def test_unsat_when_the_requested_closure_is_not_legal():
    from compose_v4.control.macro_engine import match_closure_descriptors
    assert match_closure_descriptors(["bond_insert"], [type("A", (), {"a": 1, "b": 2})()],
                                     [(40, 41)]) == []


class _Chg:
    def __init__(self, a, b, o=2): self.a=a; self.b=b; self.new_order=o
class _RSR:
    def __init__(self, changes): self.changes=tuple(changes)


def test_ring_system_restate_is_matched_by_its_changes_payload():
    from compose_v4.control.macro_engine import match_aromatisation_descriptors
    ring = [16, 17, 18, 19, 20, 21]
    inside = _RSR([_Chg(16, 17), _Chg(18, 19), _Chg(20, 21)])
    outside = _RSR([_Chg(2, 3), _Chg(4, 5), _Chg(6, 7)])
    straddle = _RSR([_Chg(16, 17), _Chg(2, 3)])
    fams = ["ring_system_restate"] * 3
    idx = match_aromatisation_descriptors(fams, [inside, outside, straddle],
                                          [0.04, 0.9, 0.5], ring)
    assert idx == [0], "only the event confined to the macro's own ring"


def test_five_ring_aromatic_carbocycle_is_unsat_not_silently_wrong():
    """A neutral all-carbon five-ring cannot be aromatic. The macro must say so
    rather than return a cyclopentane labelled OK -- the controller would then
    keep selecting an action that can never satisfy its own contract."""
    r = _local_macro(_t4_seed("5ht1b", 7), size=5, topology="pendant",
                     composition="carbon_rich", state="aromatic")
    assert r["status"] == "UNSAT", r
    assert r.get("stage") == "aromatise", r


def test_five_ring_saturated_is_satisfiable():
    """The same request without the aromatic contract is buildable."""
    r = _local_macro(_t4_seed("5ht1b", 7), size=5, topology="pendant",
                     composition="carbon_rich", state="saturated")
    assert r["status"] == "OK", r
    assert r["sys_gain"] == 1


# --- regression: the control-flow bug that cost a sentinel round -------------

class _FakeMargins:
    def __init__(self, feasible): self.feasible = feasible; self.sim = 0.0
    

def test_prefix_archive_keeps_feasible_state_when_a_later_action_ruins_it():
    """BUILD_RING_SYSTEM6 -> BUILD_RING_SYSTEM6 -> restate1 on the real 5HT1B
    seed: macro 2 yields a feasible terphenyl, restate then de-aromatises it.
    The feasible prefix must survive with prefix_len == 2."""
    from compose_v4.control.constrained_search import PrefixArchive
    good = "FC(F)(F)c1cc(-c2cccc(-c3ccccc3)c2)cc(N2CC[NH2+]CC2)c1"
    wrecked = "FC(F)(F)c1cc(C2CCCC(c3ccccc3)C2)cc(N2CC[NH2+]CC2)c1"
    feasible = {"FC(F)(F)c1cc(-c2ccccc2)cc(N2CC[NH2+]CC2)c1", good}
    a = PrefixArchive(lambda s: _FakeMargins(s in feasible))
    a.note("FC(F)(F)c1cc(-c2ccccc2)cc(N2CC[NH2+]CC2)c1", 0)   # after macro 1
    a.note(good, 1)                                            # after macro 2
    a.note(wrecked, 2)                                         # after restate
    smiles, prefix_len, _ = a.best()
    assert smiles == good, smiles
    assert prefix_len == 2, f"credit must stop at the producing prefix, got {prefix_len}"
    assert len(a) == 2, "both feasible states are candidates, not just the last"


def test_prefix_archive_credits_only_the_producing_prefix():
    from compose_v4.control.constrained_search import PrefixArchive
    a = PrefixArchive(lambda s: _FakeMargins(True))
    a.note("X", 0)
    a.note("X", 3)          # same molecule reached again later in the program
    assert a.entries()[0][1] == 1, "shortest producing prefix wins"


def test_prefix_archive_ignores_infeasible_and_empty():
    from compose_v4.control.constrained_search import PrefixArchive
    a = PrefixArchive(lambda s: _FakeMargins(False))
    assert not a.note("X", 0)
    assert len(a) == 0
    a2 = PrefixArchive(lambda s: _FakeMargins(True))
    assert not a2.note("", 0)
    assert not a2.note(None, 0)


def test_seed_can_never_enter_the_candidate_pool():
    """x0 is a realized state and may be noted, but must never be dockable.

    The seed trivially has sim = 1, and on a QED-rich seed it satisfies every
    other constraint too -- so it would dominate any archive it entered. This
    is the trivial-solution failure that made STOP0 the dominant action and put
    the seed itself in the archive, and STOP at step 0 is a direct route back
    to it."""
    from compose_v4.control.constrained_search import PrefixArchive
    seed = _t4_seed("5ht1b", 7)
    a = PrefixArchive(lambda s: _FakeMargins(True), seed_smiles=seed)
    assert not a.note(seed, 0)
    assert len(a) == 0 and a.best() is None
    # a non-canonical spelling of the same molecule is still the seed
    from rdkit import Chem
    assert not a.note(Chem.MolToSmiles(Chem.MolFromSmiles(seed), canonical=False), 0)
    assert len(a) == 0
    # anything else still gets through
    assert a.note("FC(F)(F)c1cc(-c2ccccc2)cc(N2CC[NH2+]CC2)c1", 1)
    assert len(a) == 1


def test_fused_restate_scope_is_the_whole_block_not_the_new_ring():
    """A fused aromatic system is one correlated electronic block, so the
    restate filter must admit actions touching the pre-existing ring.

    Scoping to the new ring alone rejected every legal block restate on the
    production law and sent all four PARP1 fused realisations into the Kekule
    fallback, which cannot fire for fused."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.macro_engine import ring_block_atoms, fusable_edges
    st = pad_molecular_graph(smiles_to_molecular_graph(_t4_seed("parp1", 0)), 48)
    edges = fusable_edges(st)
    assert edges, "parp1 seed must expose fusable ring edges"
    u, v = edges[0]
    block = ring_block_atoms(st, [u, v])
    assert u in block and v in block
    assert len(block) > 2, "the block must extend beyond the shared edge itself"


def test_partial_pooling_keeps_sites_near_the_macro_value():
    """Sites of one ring spec must inherit the macro's evidence, not compete
    as independent tokens. Measured: unpooled, the controller put 0.254 on @0
    versus 0.096 on @1 while replicated docking showed @1 marginally better."""
    import numpy as np
    from compose_v4.control.constrained_search import CEMController
    rng = np.random.default_rng(0)
    groups = [[0, 1, 2]]
    # every elite used site 0; sites 1 and 2 saw nothing
    pooled = CEMController(4, rng)
    pooled.update_by_rank([0, 0, 0, 3], [-11, -11, -11, -7], groups=groups,
                          site_shrink=0.25)
    solo = CEMController(4, rng)
    solo.update_by_rank([0, 0, 0, 3], [-11, -11, -11, -7])
    p, s = pooled.probs(), solo.probs()
    assert p[0] - p[1] < s[0] - s[1], "pooling must narrow the site gap"
    assert p[1] > s[1], "unobserved sites inherit the macro's evidence"
    assert p[0] > p[3], "the macro itself is still preferred over a non-ring action"


def test_sites_are_not_top_level_actions():
    """Attachment sites must NOT be independent CEM actions.

    With total ring mass matched at 0.24, six site-visible actions and one
    opaque action performed identically (best -11.3 vs -11.2, mean -8.46 vs
    -8.57, two 28-heavy molecules each) -- all gaps far under the 0.70 noise
    floor. Splitting ~30 docking observations across six unrelated global
    parameters only bought noise-chasing.

    The pooling machinery stays available (see the partial-pooling test) for
    when a contextual value model can score candidate endpoints directly."""
    from compose_v4.control.constrained_search import (ACTION_SPACE,
                                                       realization_groups)
    assert not any(n.startswith("build_ring_system@") for n, _k in ACTION_SPACE)
    assert realization_groups(ACTION_SPACE) == []


def test_ring_actions_are_semantic_not_positional():
    """The controller decides topology/size/composition/state -- choices that
    change the reachable molecular class -- not which atom index to use."""
    from compose_v4.control.constrained_search import ACTION_SPACE, parse_ring_spec
    specs = [parse_ring_spec(n) for n, _k in ACTION_SPACE if n.startswith("ring:")]
    specs = [s for s in specs if s]
    assert len(specs) >= 2
    assert {s["topology"] for s in specs} == {"linked", "fused"}
    assert all(s["size"] == 6 for s in specs)
    assert {s["state"] for s in specs} >= {"aromatic"}


def test_select_realization_defaults_to_the_goal_independent_prior():
    """With no trustworthy site-level objective evidence, selection must reduce
    to R_theta alone rather than inventing a preference."""
    import numpy as np
    from compose_v4.control.constrained_search import select_realization
    cands = [{"site": 3, "smiles": "A"}, {"site": 7, "smiles": "B"}]
    i, p = select_realization(cands, prior=[0.2, 0.8])
    assert i == 1 and p[1] > p[0]
    i2, p2 = select_realization(cands)          # no prior at all -> uniform
    assert abs(p2[0] - p2[1]) < 1e-9


def test_select_realization_accepts_a_contextual_value_on_ENDPOINTS():
    """The hook must score the candidate MOLECULE, so a future h_phi(y_xi,z,b)
    can distinguish realizations without arbitrary rank tokens."""
    from compose_v4.control.constrained_search import select_realization
    cands = [{"site": 3, "smiles": "bad"}, {"site": 7, "smiles": "good"}]
    seen = []

    def h(c):
        seen.append(c["smiles"])
        return 10.0 if c["smiles"] == "good" else 0.1

    i, p = select_realization(cands, prior=[0.9, 0.1], value_fn=h)
    assert seen == ["bad", "good"], "value_fn must receive candidate endpoints"
    assert i == 1, "objective value must be able to overturn the prior"


def test_empty_realization_set_is_handled():
    from compose_v4.control.constrained_search import select_realization
    i, p = select_realization([])
    assert i is None and len(p) == 0


def test_total_ring_mass_is_invariant_to_variant_count():
    """Total P(build a ring) must be a deliberate parent-level choice, not a
    side effect of how many semantic ring variants exist.

    Adding fused took the family from 1 variant to 3. A prior that missed the
    new names gave 3/22 = 0.136 instead of the requested 0.24."""
    from compose_v4.control.constrained_search import (ACTION_SPACE,
        ACTION_SPACE_OPAQUE, matched_prior, is_ring_action)
    for space in (ACTION_SPACE, ACTION_SPACE_OPAQUE):
        q = matched_prior(space, 0.24)
        ring = [i for i, (n, _k) in enumerate(space) if is_ring_action(n)]
        assert ring, f"no ring actions detected in {len(space)}-action space"
        assert abs(float(q[ring].sum()) - 0.24) < 1e-9, \
            f"{len(ring)} variants gave {q[ring].sum():.4f}, expected 0.2400"
    # and adding a hypothetical variant must not change the parent total
    grown = tuple(list(ACTION_SPACE) + [("ring:fused/5/C/saturated", 5)])
    qg = matched_prior(grown, 0.24)
    rg = [i for i, (n, _k) in enumerate(grown) if is_ring_action(n)]
    assert abs(float(qg[rg].sum()) - 0.24) < 1e-9


def test_ring_action_detection_covers_both_naming_schemes():
    from compose_v4.control.constrained_search import is_ring_action
    assert is_ring_action("ring:fused/6/C/aromatic")
    assert is_ring_action("build_ring_system")
    assert is_ring_action("build_ring_system@2")
    assert not is_ring_action("local_extend")
    assert not is_ring_action("restate")


def test_primary_feasibility_is_the_published_criterion_only():
    """GenMol's criterion is QED>=0.6, SA<=4, Tanimoto>=delta on a parseable
    molecule. Our stricter med-chem screen must NOT gate the primary number --
    the comparator has no such rule, so enforcing it would run the benchmark
    harder than GenMol and understate COMPOSE."""
    from compose_v4.control.constrained_search import margins, med_chem_ok
    seed = _t4_seed("5ht1b", 7)
    # a molecule our med-chem screen rejects (saturated polyaza) but which is
    # a perfectly parseable molecule
    odd = "FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1N1NNNNN1"
    m = margins(odd, seed, 0.4)
    assert m.valid, "primary feasibility must not apply the med-chem screen"
    assert not med_chem_ok(odd), "but the secondary screen must still flag it"


def test_med_chem_screen_still_available_for_secondary_reporting():
    from compose_v4.control.constrained_search import med_chem_ok
    assert med_chem_ok(_t4_seed("parp1", 0))
    assert not med_chem_ok("N1NNNNN1")


def test_frontier_module_never_imports_a_docking_oracle():
    """Phase 1 is constraint satisfaction and must cost ZERO oracle calls.

    Checks the parsed IMPORT statements, not the source text -- the first
    version grepped for 'dock' and tripped over the word 'docking' in the
    module's own docstring explaining that it performs none."""
    import ast, pathlib
    tree = ast.parse(pathlib.Path(
        'src/compose_v4/control/feasibility_frontier.py').read_text())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or '')
            mods += [f"{node.module}.{a.name}" for a in node.names]
    banned = ('dock', 'qvina', 'oracle', 'molleo_task3')
    hits = [m for m in mods if any(b in (m or '').lower() for b in banned)]
    assert not hits, f"feasibility phase must not import an oracle: {hits}"


def test_is_feasible_uses_published_criterion_only():
    """QED>=0.6, SA<=4, sim>=delta on a parseable molecule -- the med-chem
    screen must not gate it, since GenMol has no such rule."""
    from compose_v4.control.feasibility_frontier import is_feasible
    seed = _t4_seed("5ht1b", 7)
    good = "FC(F)(F)c1cc(-c2cccc(-c3ccccc3)c2)cc(N2CC[NH2+]CC2)c1"
    assert is_feasible(good, seed, 0.6)
    assert not is_feasible(seed, seed, 0.6) is False   # seed itself is feasible here
    # a molecule our med-chem screen rejects still counts if it meets the criterion
    from compose_v4.control.constrained_search import med_chem_ok
    odd = "FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1N1NNNNN1"
    if is_feasible(odd, seed, 0.4):
        assert not med_chem_ok(odd), "secondary screen still flags it"


def test_frontier_finds_feasible_states_from_an_infeasible_seed():
    """braf_s10 starts at QED 0.346 against a 0.6 gate. The controller got
    0 feasible over ten rounds; the cheap frontier must find some."""
    from compose_v4.control.feasibility_frontier import build_frontier
    seed = _t4_seed("braf", 10)
    feas, stats = build_frontier(seed, 0.6, target=5, max_depth=6, beam=60)
    assert stats['seed_shortfall'] > 0, "seed must actually be infeasible"
    assert len(feas) >= 1, f"found none in {stats['expanded']} expansions"
    for smi, pr in feas.items():
        assert pr['qed'] >= 0.6 and pr['sa'] <= 4.0 and pr['sim'] >= 0.6
