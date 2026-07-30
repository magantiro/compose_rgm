"""A1.1: the shared evaluator constructs, validates and reports provenance -- or refuses.

The evaluator's job is to be the only place the molecular process is derived, so these tests concentrate on
the ways a WRONG evaluation could look right:

  * an experiment handed a checkpoint whose capabilities differ from the declared production
    configuration (a de-novo or pre-RingCore artifact) must be refused, not silently scored;
  * the whole-ring macro must be off, which matters because the production loader DEFAULTS it to on when
    the metadata key is absent;
  * the output envelope must stay stable and additive, since Phase B/D consume it.

A conforming checkpoint is built from a FRESH model rather than by editing an old payload's metadata: the
strict loader correctly refuses a frankenstein artifact whose weights lack the cycle-op heads, and relying
on that trick would have produced a test that only passed by accident.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from compose_v4.experiments.checkpoint_evaluator import (  # noqa: E402
    CAPABILITY_FLAGS,
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    EvaluationError,
    build_context,
    build_support_signature,
    evaluation_envelope,
    file_sha256,
    read_checkpoint_provenance,
    resolve_capability_flags,
    validate_capabilities,
    write_envelope,
)
from compose_v4.experiments.registry import load_registry  # noqa: E402
from compose_v4.experiments.successor_kernel import assert_arms_comparable  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY = _ROOT / "configs" / "experiment_registry.yaml"


@pytest.fixture(scope="module")
def registry():
    return load_registry(_REGISTRY)


def _ring_catalog():
    """A minimal typed ring catalog -- the model requires one, but A1.1 exercises no ring sampling."""
    from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

    return build_typed_ring_catalog(())


def _build_model(*, editing: bool, cycle_ops: bool, macro: bool):
    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
    from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

    return FactorizedTraceletRateModel(
        _ring_catalog(),
        hidden_dim=8,
        message_passing_steps=1,
        atom_vocabulary=ORGANIC_VOCABULARY,
        enable_ring_restates=editing,
        enable_cyclic_graft=editing,
        enable_heteroatom_scan=editing,
        enable_ring_opening=editing,
        enable_cycle_ops=cycle_ops,
        enable_ring_grow_macro=macro,
    )


def _write_checkpoint(path: Path, *, editing: bool, cycle_ops: bool, macro: bool) -> Path:
    """Save a checkpoint the PRODUCTION loader can reconstruct, with production metadata keys."""
    model = _build_model(editing=editing, cycle_ops=cycle_ops, macro=macro)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ring_catalog": _ring_catalog(),
            "hidden_dim": 8,
            "message_passing_steps": 1,
            "training_backend": "factorized_marks",
            # The production loader requires these rollout-metadata keys, and currently accepts only
            # source_prior "carbon_tree" -- which is what the production recipe sets, so real checkpoints
            # satisfy it. (The EDITING source prior is a recipe-level mixture, not this field.)
            "source_prior": "carbon_tree",
            "tree_source_prior": None,
            "organic_vocabulary": True,
            "corrupted_prior_mix": editing,
            "enable_cycle_ops": cycle_ops,
            "enable_ring_grow_macro": macro,
            "max_atoms": 8,
            "bond_representation": "aromatic",
            "rate_factorization": "hierarchical",
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def conforming(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "ringcore.pt"
    return _write_checkpoint(path, editing=True, cycle_ops=True, macro=False)


# ---- provenance ---------------------------------------------------------------------------------------


def test_provenance_reads_the_keys_the_gate_actually_persists(conforming):
    """The gate writes enable_cycle_ops into checkpoint_metadata, NOT cycle_op_mix."""
    provenance = read_checkpoint_provenance(conforming)
    assert provenance.organic_vocabulary is True
    assert provenance.corrupted_prior_mix is True
    assert provenance.enable_cycle_ops is True
    assert provenance.sha256 == file_sha256(conforming)
    assert provenance.bytes > 0


def test_missing_and_empty_checkpoints_fail_loudly(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        read_checkpoint_provenance(tmp_path / "absent.pt")
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(EvaluationError, match="empty"):
        read_checkpoint_provenance(empty)


# ---- capability validation ----------------------------------------------------------------------------


def test_flags_report_what_was_actually_applied():
    model = _build_model(editing=True, cycle_ops=True, macro=False)
    flags = resolve_capability_flags(model)
    assert set(flags) == set(CAPABILITY_FLAGS)
    assert all(flags.values())


def test_the_model_itself_forbids_macro_together_with_cycle_ops():
    """A stronger guard already exists upstream: the two are mutually exclusive at construction."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_model(editing=True, cycle_ops=True, macro=True)


def test_the_whole_ring_macro_must_be_off():
    """Covers the case the model DOES permit: the legacy macro-only configuration.

    Non-tautological for a second reason: the production loader defaults enable_ring_grow_macro to True
    when the metadata key is absent, so a checkpoint predating RingCore-V1 arrives with it on.
    """
    model = _build_model(editing=True, cycle_ops=False, macro=True)
    with pytest.raises(EvaluationError, match="whole-ring macro"):
        validate_capabilities(model)


def test_capabilities_must_match_what_the_registry_requires():
    """The realistic mistake: an editing experiment handed a de-novo checkpoint."""
    required = {
        "enable_ring_restates": True,
        "enable_cyclic_graft": True,
        "enable_heteroatom_scan": True,
        "enable_ring_opening": True,
        "enable_cycle_ops": True,
        "enable_ring_grow_macro": False,
    }
    de_novo = _build_model(editing=False, cycle_ops=False, macro=False)
    with pytest.raises(EvaluationError, match="do not match the configuration"):
        validate_capabilities(de_novo, required=required)
    # the conforming configuration passes the same check
    validate_capabilities(_build_model(editing=True, cycle_ops=True, macro=False), required=required)


def test_cycle_ops_off_is_refused_when_the_registry_requires_them():
    required = {"enable_cycle_ops": True}
    model = _build_model(editing=True, cycle_ops=False, macro=False)
    with pytest.raises(EvaluationError, match="enable_cycle_ops"):
        validate_capabilities(model, required=required)


# ---- support signature --------------------------------------------------------------------------------


def test_support_signature_is_populated_from_the_model(conforming):
    model = _build_model(editing=True, cycle_ops=True, macro=False)
    flags = resolve_capability_flags(model)
    signature = build_support_signature(
        model, flags, provenance=read_checkpoint_provenance(conforming)
    )
    assert signature.capability_flags == tuple((name, True) for name in CAPABILITY_FLAGS)
    assert signature.atom_insert_arity_support == (0, 1)
    assert signature.embedded_jump_chain_policy == "fixed_step_embedded_jump_chain"
    assert signature.ringcore_configuration == "ringcore_v1_compositional_cycle_ops"
    assert signature.canonicalizer_version == "canonical_state_key"
    assert signature.max_atoms == 8
    assert signature.aromaticity_policy == "aromatic"
    assert signature.charge_vocabulary == (-2, -1, 0, 1, 2)
    # the organic vocabulary must actually be read, not left empty
    assert len(signature.element_vocabulary) > 4, signature.element_vocabulary


def test_two_contexts_from_the_same_checkpoint_are_comparable_arms(conforming, registry):
    left = build_context(checkpoint=conforming, registry=registry, experiment_id="E2", seed=0)
    right = build_context(checkpoint=conforming, registry=registry, experiment_id="E2", seed=1)

    class _Arm:
        def __init__(self, context, name):
            self._context, self._name = context, name

        def successors(self, state):  # pragma: no cover - identity is what is compared
            raise NotImplementedError

        def identity(self):
            return self._context.kernel_identity(self._name)

    assert_arms_comparable(_Arm(left, "learned"), _Arm(right, "uniform"))


# ---- context construction -----------------------------------------------------------------------------


def test_context_builds_for_an_editing_experiment(conforming, registry):
    context = build_context(
        checkpoint=conforming, registry=registry, experiment_id="E2", seed=0
    )
    assert context.experiment_id == "E2"
    assert context.seed == 0
    assert context.capability_flags["enable_cycle_ops"] is True
    assert context.registry_protocol_hash == (
        registry["protocol"]["protocol_freeze"]["content_hash"]
    )


def test_context_constructs_the_shared_production_successor_kernel(
    conforming, registry
):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    context = build_context(
        checkpoint=conforming,
        registry=registry,
        experiment_id="E2",
        seed=0,
    )
    kernel = context.successor_kernel(time=0.5)
    batch = kernel.successors(
        pad_molecular_graph(smiles_to_molecular_graph("CCO"), 8)
    )
    assert batch.identity.checkpoint_sha256 == context.checkpoint.sha256
    assert batch.identity.support_signature == context.support_signature
    assert batch.support_size > 0


def test_unknown_experiment_id_fails_loudly(conforming, registry):
    from compose_v4.experiments.registry import RegistryViolation

    with pytest.raises(RegistryViolation, match="unknown experiment id"):
        build_context(checkpoint=conforming, registry=registry, experiment_id="E99", seed=0)


def test_off_protocol_seed_is_refused(conforming, registry):
    with pytest.raises(EvaluationError, match="frozen protocol seeds"):
        build_context(checkpoint=conforming, registry=registry, experiment_id="E2", seed=99)


def test_an_experiment_declaring_no_checkpoint_refuses_one(conforming, registry):
    """E6 is checkpoint-free; handing it one would misreport what produced the number."""
    assert registry["experiments"]["E6"]["checkpoint"] == "none"
    with pytest.raises(EvaluationError, match="must not be given one"):
        build_context(checkpoint=conforming, registry=registry, experiment_id="E6", seed=0)


def test_a_pre_ringcore_checkpoint_is_refused(tmp_path, registry):
    stale = _write_checkpoint(tmp_path / "stale.pt", editing=True, cycle_ops=False, macro=True)
    with pytest.raises(EvaluationError, match="whole-ring macro"):
        build_context(checkpoint=stale, registry=registry, experiment_id="E2", seed=0)


def test_a_de_novo_checkpoint_is_refused_for_an_editing_experiment(tmp_path, registry):
    de_novo = _write_checkpoint(tmp_path / "denovo.pt", editing=False, cycle_ops=False, macro=False)
    with pytest.raises(EvaluationError, match="do not match the configuration"):
        build_context(checkpoint=de_novo, registry=registry, experiment_id="E2", seed=0)


# ---- output envelope ----------------------------------------------------------------------------------


def test_envelope_schema_is_stable(conforming, registry):
    context = build_context(checkpoint=conforming, registry=registry, experiment_id="E2", seed=0)
    envelope = evaluation_envelope(
        context, metrics={"example": 1.0}, registry_path=str(_REGISTRY)
    )
    assert envelope["schema"] == OUTPUT_SCHEMA
    assert envelope["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert set(envelope) == {
        "schema", "schema_version", "experiment_id", "seed", "checkpoint", "capability_flags",
        "support_signature", "registry", "environment", "metrics", "per_sample_records",
    }
    # provenance sufficient to say which artifact produced the number
    assert envelope["checkpoint"]["sha256"]
    assert envelope["registry"]["protocol_content_hash"]


def test_envelope_round_trips_to_disk(conforming, registry, tmp_path):
    import json

    context = build_context(checkpoint=conforming, registry=registry, experiment_id="E2", seed=0)
    path = write_envelope(evaluation_envelope(context), tmp_path / "out" / "E2.json")
    assert json.loads(path.read_text())["experiment_id"] == "E2"
