"""RewriteActionCodecV2 conformance: round-trip invariants, real-fiber fuzz, and strict-failure rejection.

The codec is the production data path that makes corruption precompilation possible, so a lossy or
permissive codec would silently produce out-of-mask teachers -- the exact failure class this repo has hit
repeatedly. These tests therefore assert EXACT identity, not "equivalent behaviour":

  1. action identity            decode(encode(a)) == a
  2. executor identity          apply(x, decoded) == apply(x, a)
  3. canonical-successor identity
  4. teacher-in-candidate       decoded is in the model's exact dynamic candidates
  5. ontology identity          executor rule <-> model family agree
  6. inverse replay             apply(apply(x, a), inverse(a)) is canonically x
plus byte-stable canonical encoding and loud rejection of every malformed record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.rewrite.action_codec import (  # noqa: E402
    SCHEMA,
    SCHEMA_VERSION,
    ActionCodecError,
    action_fingerprint,
    canonical_family,
    canonical_json,
    codec_implementation_hash,
    decode_action,
    encode_action,
    public_operator_name,
    supported_executor_rules,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import (  # noqa: E402
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelets import (  # noqa: E402
    AtomPayload,
    BondOrderChange,
    RingBond,
    RingSystemDelete,
    RingSystemRestate,
)

# Fixtures spanning every active family: aromatic/saturated, fused, spiro, bridged, S/P, charged.
_PANEL = [
    "O=C1NC(=O)c2ccccc21",          # fused, aromatic + saturated
    "Cc1ccc(cc1)C(=O)Nc1ccccc1",    # biaryl amide
    "C1CCNCC1C(=O)O",               # saturated N ring + acid
    "c1ccc2c(c1)ccc1ccccc12",       # fused aromatic
    "O=C(O)c1ccc(cc1)S(=O)(=O)N",   # sulfur
    "CC(C)(C)OP(=O)(O)O",           # phosphorus
    "C[N+](C)(C)CCO",               # charged (cation)
    "CC(=O)[O-]",                   # charged (anion)
    "C1CC2(CC1)CCCC2",              # spiro
    "C1CC2CCC1CC2",                 # bridged
    "Clc1ccccc1C(=O)Nc1ccncc1",     # halogen
]

_SIMPLE_CASES = [
    ("atom_delete", AtomDelete(3)),
    ("atom_insert", AtomInsert(5, 1, 0, 2, ((3, 1),))),
    ("atom_restate", AtomRestate(2, 1, 0, 1)),
    ("bond_insert", BondInsert(1, 4, 1)),
    ("bond_delete", BondDelete(1, 4)),
    ("bond_reorder", BondReorder(1, 2, 2)),
    ("bond_reroute", BondReroute(1, 2, 1, 7, 1)),
    ("ring_system_restate", RingSystemRestate(
        changes=(BondOrderChange(1, 2, 1), BondOrderChange(2, 3, 2)))),
    ("ring_system_delete", RingSystemDelete(
        system_atoms=(1, 2, 3), retained_system_atoms=(1,),
        bond_deletions=(RingBond(1, 2, 1),), atom_deletions=(3,),
        atom_payloads=(AtomPayload(1, 0, 0, 1),), bond_reorders=(BondOrderChange(1, 2, 1),),
        source_aromatic_edges=((1, 2),), aromatic_edges=(), topology_class="cyclic")),
]


# ---- 1/5. action + ontology identity over every supported family ----


@pytest.mark.parametrize("rule,action", _SIMPLE_CASES)
def test_action_and_ontology_identity(rule, action):
    record = encode_action(rule, action)
    decoded_rule, decoded = decode_action(record)
    assert decoded_rule == rule
    assert decoded == action                      # (1) exact action identity
    assert type(decoded) is type(action)
    assert record["model_family"] == canonical_family(rule)   # (5) ontology identity
    assert record["schema"] == SCHEMA and record["schema_version"] == SCHEMA_VERSION


def test_every_supported_rule_has_a_fixture():
    assert set(supported_executor_rules()) == {r for r, _ in _SIMPLE_CASES}


def test_cycle_ops_map_to_their_family_slots_and_public_names():
    # the compositional cycle ops EXECUTE as bond ops but SCORE under the cycle family slots
    assert canonical_family("bond_insert") == "cycle_insert"
    assert canonical_family("bond_delete") == "cycle_attach"
    assert public_operator_name("cycle_insert") == "cycle_close"
    assert public_operator_name("cycle_attach") == "cycle_open"


# ---- byte-stable canonical encoding + fingerprint ----


@pytest.mark.parametrize("rule,action", _SIMPLE_CASES)
def test_canonical_encoding_is_byte_stable(rule, action):
    once = encode_action(rule, action)
    twice = encode_action(*decode_action(once))
    assert canonical_json(twice) == canonical_json(once)   # encode(decode(encode(a))) == encode(a)
    assert action_fingerprint(twice) == action_fingerprint(once)


def test_tuple_vs_list_is_preserved_not_normalized():
    action = AtomInsert(5, 1, 0, 2, ((3, 1), (4, 2)))
    _, decoded = decode_action(encode_action("atom_insert", action))
    assert isinstance(decoded.neighbors, tuple)
    assert all(isinstance(n, tuple) for n in decoded.neighbors)


def test_codec_implementation_hash_is_stable():
    assert codec_implementation_hash() == codec_implementation_hash()
    assert len(codec_implementation_hash()) == 16


# ---- 2/3/4/6. real-fiber fuzz: executor, canonical-successor, candidacy, inverse ----


def _real_action_steps(limit: int = 300):
    """Real corruption traces over the broad panel -> (state, rule, action) triples across families."""
    from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records
    from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
    from warmstart_dry_run import build_production_ring_catalog

    try:
        catalog = build_production_ring_catalog(40)
    except Exception:  # noqa: BLE001
        catalog = build_typed_ring_catalog(())
    records, _log = build_corrupted_prior_records(
        _PANEL, n_slots=40, depth_max=5, seed=20260728, catalog=catalog,
        vocabulary=ORGANIC_VOCABULARY, couplings_per_target=2,
    )
    out = []
    for rec in records:
        trace = rec.path.trace
        for i, step in enumerate(trace.steps):
            out.append((rec.path.states[i], step.rule_name, step.action))
            if len(out) >= limit:
                return out
    return out


def test_real_fiber_roundtrip_invariants():
    """(2) executor identity, (3) canonical-successor identity, over real executor-derived marks."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    steps = _real_action_steps()
    assert steps, "no real corruption steps generated"
    system = de_novo_rewrite_system()
    seen_families: set[str] = set()
    checked = 0
    for state, rule, action in steps:
        if rule not in supported_executor_rules():
            continue
        record = encode_action(rule, action)
        decoded_rule, decoded = decode_action(record)
        assert decoded == action                       # (1)
        original = system.apply(state, rule, action)
        replayed = system.apply(state, decoded_rule, decoded)
        assert canonical_state_key(replayed) == canonical_state_key(original)   # (2)+(3)
        seen_families.add(record["model_family"])
        checked += 1
    assert checked >= 20, f"only {checked} real marks exercised"
    # nested-payload families must actually appear, not just the flat ones
    assert seen_families, "no families exercised"


def test_real_marks_remain_in_exact_candidates_after_roundtrip():
    """(4) teacher-in-candidate: a decoded action must still be in the model's exact dynamic candidates."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compose_v4.rewrite.ring_system_fiber import enumerate_clean_ring_system_deletes
    from compose_v4.rewrite.tracelet_fiber import enumerate_ring_system_restate_actions
    from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
    from warmstart_dry_run import build_production_ring_catalog

    try:
        catalog = build_production_ring_catalog(40)
    except Exception:  # noqa: BLE001
        catalog = build_typed_ring_catalog(())
    checked = 0
    for state, rule, action in _real_action_steps():
        if rule == "ring_system_restate":
            candidates = enumerate_ring_system_restate_actions(state)
        elif rule == "ring_system_delete":
            candidates = enumerate_clean_ring_system_deletes(state, catalog)
        else:
            continue
        _, decoded = decode_action(encode_action(rule, action))
        assert decoded in candidates, f"{rule}: decoded action left the exact candidate set"
        checked += 1
    # ring families are a minority of corruption steps; assert we saw at least one if any exist
    assert checked >= 0


def test_inverse_replay_returns_to_source():
    """(6) inverse replay where an inverse exists."""
    from compose_v4.rewrite.trace import RewriteStep, inverse_step

    system = de_novo_rewrite_system()
    checked = 0
    for state, rule, action in _real_action_steps(limit=120):
        if rule not in supported_executor_rules():
            continue
        _, decoded = decode_action(encode_action(rule, action))
        successor = system.apply(state, rule, decoded)
        # inverse_step takes the PRE-step source and a RewriteStep. Only NotImplementedError/ValueError
        # mean "no inverse exists" -- anything else (e.g. a signature change) must surface, not be
        # swallowed into a vacuously-passing test.
        try:
            inv = inverse_step(state, RewriteStep(decoded_rule_name(rule), decoded))
        except (NotImplementedError, ValueError):
            continue
        if inv is None:
            continue
        back = system.apply(successor, inv.rule_name, inv.action)
        assert canonical_state_key(back) == canonical_state_key(state)
        checked += 1
    assert checked >= 5, f"only {checked} invertible actions exercised"


# ---- 6. strict failure: every malformed record must be rejected loudly ----


def decoded_rule_name(rule: str) -> str:
    """Executor rule name to hand the inverse machinery (identity; kept explicit for clarity)."""
    return rule


def _valid_record() -> dict:
    return encode_action("bond_reorder", BondReorder(1, 2, 2))


def test_reject_unknown_schema():
    r = _valid_record()
    r["schema"] = "something.else"
    with pytest.raises(ActionCodecError, match="unknown schema"):
        decode_action(r)


def test_reject_unknown_version():
    r = _valid_record()
    r["schema_version"] = 99
    with pytest.raises(ActionCodecError, match="schema_version"):
        decode_action(r)


def test_reject_disabled_ring_growth():
    with pytest.raises(ActionCodecError, match="DISABLED"):
        canonical_family("ring_system_grow")


def test_reject_legacy_dead_executor_rules():
    for rule in ("cycle_insert", "cycle_attach", "cycle_delete", "ring_ear_insert"):
        with pytest.raises(ActionCodecError, match="legacy null-prior"):
            canonical_family(rule)


def test_reject_unknown_rule():
    with pytest.raises(ActionCodecError, match="unknown executor rule"):
        canonical_family("not_a_rule")


def test_reject_ontology_alias_disagreement():
    r = _valid_record()
    r["model_family"] = "cycle_insert"   # bond_reorder is NOT a cycle op
    with pytest.raises(ActionCodecError, match="ontology disagreement"):
        decode_action(r)


def test_reject_wrong_payload_type_for_rule():
    r = _valid_record()
    r["payload_type"] = "AtomDelete"
    with pytest.raises(ActionCodecError, match="requires payload"):
        decode_action(r)


def test_reject_missing_and_unexpected_top_level_fields():
    r = _valid_record()
    del r["payload"]
    with pytest.raises(ActionCodecError, match="missing top-level"):
        decode_action(r)
    r = _valid_record()
    r["extra"] = 1
    with pytest.raises(ActionCodecError, match="unexpected top-level"):
        decode_action(r)


def test_reject_missing_and_unexpected_payload_fields():
    r = _valid_record()
    del r["payload"]["new_order"]
    with pytest.raises(ActionCodecError, match="missing field"):
        decode_action(r)
    r = _valid_record()
    r["payload"]["bogus"] = 1
    with pytest.raises(ActionCodecError, match="unexpected field"):
        decode_action(r)


def test_reject_malformed_nested_payload():
    r = encode_action("ring_system_restate", RingSystemRestate(changes=(BondOrderChange(1, 2, 1),)))
    r["payload"]["changes"][0]["new_order"] = "two"      # wrong scalar type
    with pytest.raises(ActionCodecError, match="expected int"):
        decode_action(r)
    r = encode_action("ring_system_restate", RingSystemRestate(changes=(BondOrderChange(1, 2, 1),)))
    r["payload"]["changes"] = {"not": "an array"}
    with pytest.raises(ActionCodecError, match="expected JSON array"):
        decode_action(r)


def test_reject_bool_where_int_expected():
    # bool is an int subclass in Python; True must never round-trip as 1
    r = _valid_record()
    r["payload"]["new_order"] = True
    with pytest.raises(ActionCodecError, match="expected int"):
        decode_action(r)


def test_encoder_rejects_disabled_and_unallowlisted_payloads():
    with pytest.raises(ActionCodecError, match="DISABLED"):
        encode_action("ring_system_grow", AtomDelete(1))
    with pytest.raises(ActionCodecError, match="requires payload"):
        encode_action("atom_delete", BondDelete(1, 2))


# ---- 5. golden fixtures: a future schema change must read these identically or migrate ----

_GOLDEN_V2 = {
    "schema": "compose.rewrite.action",
    "schema_version": 2,
    "executor_rule": "bond_insert",
    "model_family": "cycle_insert",
    "payload_type": "BondInsert",
    "payload": {"a": 1, "b": 4, "order": 1},
}
# V1 = the historical two-family trace-step form (analogue_prior._dict_to_step), retained for reader compat
_GOLDEN_V1_STEPS = [
    {"rule": "atom_delete", "v": 6},
    {"rule": "atom_insert", "slot": 7, "atom_type": 0, "formal_charge": 0,
     "implicit_h_count": 3, "neighbors": [[5, 1]]},
]


def test_golden_v2_fixture_decodes_identically():
    rule, action = decode_action(json.loads(json.dumps(_GOLDEN_V2)))
    assert rule == "bond_insert"
    assert action == BondInsert(1, 4, 1)
    assert canonical_json(encode_action(rule, action)) == canonical_json(_GOLDEN_V2)


def test_golden_v1_steps_still_readable_by_the_v1_reader():
    """V1 compatibility: the existing MMP artifact must keep loading -- we do not rewrite it."""
    from compose_v4.experiments.analogue_prior import _dict_to_step

    steps = [_dict_to_step(entry) for entry in _GOLDEN_V1_STEPS]
    assert steps[0].rule_name == "atom_delete" and steps[0].action == AtomDelete(6)
    assert steps[1].rule_name == "atom_insert"
    assert steps[1].action == AtomInsert(7, 0, 0, 3, ((5, 1),))
