"""Process V2 must be constructible by every runtime and checkpoint path.

Round 1 shipped a Process-V2 mask that no batch builder, evaluator, validation
cache, trainer, distillation run, or checkpoint reader could actually construct:
each of them hand-copied a subset of the capability fields and silently dropped
the atom-delete mode, and the scientific identity hard-bound the V1 model.

These tests pin the two properties that fix is worth:

1. **Constructibility.** Every owned builder threads the complete capability
   object, so a Process-V2 model produces a Process-V2 batch, and direct and
   multiworker collation agree exactly.
2. **Fail-closed persistence.** A Process-V2 model round-trips through its
   persisted metadata to an identical configuration and capability fingerprint;
   a historical Process-V1 payload reconstructs byte-identically; and missing,
   mixed, or stale semantics raise instead of defaulting to legacy.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_v2_scientific_identity import (
    ARTIFACT_KEYS,
    PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA,
    PROCESS_V2_SCIENTIFIC_IDENTITY_STATUS,
    SCIENTIFIC_IDENTITY_SCHEMA,
    SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    SCIENTIFIC_IDENTITY_STATUS,
    PRODUCTIVE_TRAINING_OBJECT,
    EditingV2ScientificIdentityError,
    build_editing_v2_scientific_identity,
    canonical_sha256,
    process_v2_semantic_model_identity,
    semantic_model_identity,
    validate_editing_v2_scientific_identity,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    ATOM_DELETE_ACTION_SEMANTICS_FIELD,
    SemanticScratchRuntimeError,
    build_model_from_semantic_model_identity,
    model_identity_atom_delete_action_semantics,
    semantic_runtime_model_identity,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkDataset,
    _require_complete_capability_keywords,
    operator_capability_batch_kwargs,
    sample_factorized_mark_batch,
)
from compose_v4.experiments.production_successor_kernel import (
    _SEMANTIC_PROCESS_CONFIGURATION_LABELS,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    ATOM_DELETE_ACTION_SEMANTICS,
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_VERSIONS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
    OperatorCapabilities,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import enumerate_semantic_atom_restates
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 24
_HIDDEN_DIM = 12
_MESSAGE_PASSING_STEPS = 1
_MARK_DIM = 16
_SEED = 11


def _state(smiles: str) -> MolecularGraph:
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


@pytest.fixture(scope="module")
def ring_catalog():
    return build_process_v2_ring_catalog()


def _build_model(ring_catalog, *, process: str, delete_mode: str):
    torch.manual_seed(_SEED)
    return FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=_HIDDEN_DIM,
        message_passing_steps=_MESSAGE_PASSING_STEPS,
        mark_dim=_MARK_DIM,
        enable_ring_restates=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        editing_process_semantics=process,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        atom_delete_action_semantics=delete_mode,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        enable_ring_opening=True,
        enable_cyclic_graft=True,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


@pytest.fixture(scope="module")
def semantic_v1_model(ring_catalog):
    return _build_model(
        ring_catalog,
        process=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        delete_mode=LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    )


@pytest.fixture(scope="module")
def process_v2_model(ring_catalog):
    return _build_model(
        ring_catalog,
        process=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        delete_mode=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    )


def build_semantic_record() -> PathRecord:
    """One real semantic-editing trace, so the sampler has a productive record.

    Module level, not fixture-local, so the bounded multiworker subprocess in
    `test_direct_and_multiworker_collation_produce_identical_masks` can rebuild
    the identical record by importing this module.
    """

    runtime = editing_v2_semantic_rewrite_system()
    source = _state("C1CCCCC1")
    action = enumerate_semantic_atom_restates(source)[0]
    target = runtime.apply(source, "atom_restate_semantic", action)
    return PathRecord(
        target_key=canonical_state_key(target),
        path=TraceProgressCTMC(
            RewriteTrace(
                source=source,
                target=target,
                steps=(RewriteStep("atom_restate_semantic", action),),
                metadata={},
            ),
            system=runtime,
        ),
    )


def build_process_v2_ring_catalog():
    """The `ring_catalog` fixture body, importable for the same reason."""

    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(1), n_slots=_SLOTS
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    return build_typed_ring_catalog((trace,))


@pytest.fixture(scope="module")
def semantic_record():
    return build_semantic_record()


# ---- 1. The capability keywords are derived, never hand-copied ----


def test_capability_keywords_cover_every_capability_field() -> None:
    """The derived expansion is what stops a builder dropping a later field."""

    expected = {field.name for field in dataclass_fields(OperatorCapabilities)}
    keywords = operator_capability_batch_kwargs(OperatorCapabilities.de_novo())
    assert set(keywords) == expected
    assert ATOM_DELETE_ACTION_SEMANTICS_FIELD in expected


@pytest.mark.parametrize(
    "builder",
    [prepare_factorized_mark_batch, FactorizedMarkCollator],
)
def test_every_capability_keyword_is_accepted_by_the_batch_builders(builder) -> None:
    """A capability the builders cannot accept must fail here, not at a call site."""

    accepted = set(inspect.signature(builder).parameters)
    assert set(operator_capability_batch_kwargs(OperatorCapabilities.de_novo())) <= (
        accepted
    )


def test_an_incomplete_hand_listed_keyword_set_is_rejected() -> None:
    keywords = operator_capability_batch_kwargs(OperatorCapabilities.de_novo())
    keywords.pop(ATOM_DELETE_ACTION_SEMANTICS_FIELD)
    with pytest.raises(ValueError, match="capability keywords are incomplete"):
        _require_complete_capability_keywords(keywords)


def test_capability_keywords_require_a_capability_object() -> None:
    with pytest.raises(TypeError, match="OperatorCapabilities"):
        operator_capability_batch_kwargs({"atom_delete_action_semantics": "x"})


def test_collator_from_capabilities_carries_the_atom_delete_mode(
    process_v2_model,
    semantic_v1_model,
) -> None:
    """The collator factory the Gate-0/trainer/T1/P50 builders now use."""

    v2 = FactorizedMarkCollator.from_capabilities(
        process_v2_model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=process_v2_model.ring_catalog,
    )
    assert v2.atom_delete_action_semantics == PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    assert v2.editing_process_semantics == PROCESS_V2_EDITING_PROCESS_SEMANTICS

    v1 = FactorizedMarkCollator.from_capabilities(
        semantic_v1_model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=semantic_v1_model.ring_catalog,
    )
    assert v1.atom_delete_action_semantics == LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    assert v1.editing_process_semantics == SEMANTIC_EDITING_V2_PROCESS_SEMANTICS


def test_every_semantic_process_version_has_a_kernel_configuration_label() -> None:
    """A membership test plus a lookup, so a new version cannot fall through."""

    assert set(_SEMANTIC_PROCESS_CONFIGURATION_LABELS) == set(
        SEMANTIC_EDITING_V2_PROCESS_VERSIONS
    )
    assert len(set(_SEMANTIC_PROCESS_CONFIGURATION_LABELS.values())) == len(
        _SEMANTIC_PROCESS_CONFIGURATION_LABELS
    )


# ---- 2. Direct versus multiworker collation ----


# `workers >= 1` builds a real torch DataLoader, and on macOS the start method
# is `spawn`, so the worker is a fresh interpreter that re-imports the package
# and unpickles the dataset and collate function.  Nothing in that path has a
# deadline: if a worker never becomes ready, `iter(loader)` blocks forever and
# the whole suite stops rather than failing.  This repository has a recorded
# history of exactly that on macOS multiprocessing.  The bound below turns a
# stall into a reported failure, and it is generous enough that a merely slow
# machine cannot trip it.
_MULTIWORKER_DEADLINE_SECONDS = 300

_MULTIWORKER_CHILD = """
import hashlib, json, sys
sys.path.insert(0, {tests!r})
sys.path.insert(0, {src!r})
import test_process_v2_runtime_checkpoint as harness
from compose_v4.experiments.factorized_mark_conditional import (
    sample_factorized_mark_batch,
)

catalog = harness.build_process_v2_ring_catalog()
model = harness._build_model(
    catalog,
    process=harness.PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    delete_mode=harness.PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
)
batch = sample_factorized_mark_batch(
    (harness.build_semantic_record(),),
    batch_size=4,
    seed=5,
    late_time_fraction=0.0,
    operational_horizon=1.0,
    workers={workers},
    ring_catalog=catalog,
    capabilities=model.operator_capabilities,
)


def digest(tensor):
    array = tensor.detach().cpu().numpy()
    return f"{{array.dtype}}|{{array.shape}}|" + hashlib.sha256(
        array.tobytes()
    ).hexdigest()


print(
    json.dumps(
        {{
            "atom_delete_action_semantics": batch.atom_delete_action_semantics,
            "atom_delete_mask": digest(batch.atom_delete_mask),
            "atom_delete_admission_mask": digest(batch.atom_delete_admission_mask),
        }}
    )
)
"""


def _sampled_mask_digests(workers: int) -> dict:
    """Run one sampling in a child process under a hard deadline."""

    root = Path(__file__).resolve().parents[1]
    script = _MULTIWORKER_CHILD.format(
        tests=str(root / "tests"), src=str(root / "src"), workers=workers
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=_MULTIWORKER_DEADLINE_SECONDS,
            cwd=str(root),
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"multiworker collation at workers={workers} did not finish within "
            f"{_MULTIWORKER_DEADLINE_SECONDS}s; the DataLoader worker never "
            "delivered a batch. This is the invariant failing, not a flaky test."
        ) from error
    if completed.returncode != 0:
        raise AssertionError(
            f"multiworker collation at workers={workers} exited "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_multiworker_fixtures_are_picklable(
    process_v2_model,
    semantic_record,
) -> None:
    """Spawn transfers these by pickle, so an unpicklable one is a startup hang.

    Checked directly rather than inferred from the loader succeeding: a pickling
    regression here is one of the three named mechanisms (pickling, startup,
    teardown) by which the worker path fails, and it is the only one that can be
    localized without running a worker at all.
    """

    dataset = FactorizedMarkDataset(
        (semantic_record,),
        start_index=0,
        length=4,
        seed=5,
        late_time_fraction=0.0,
        operational_horizon=1.0,
        progress_stratification_fraction=0.0,
        ring_catalog=process_v2_model.ring_catalog,
    )
    collator = FactorizedMarkCollator.from_capabilities(
        process_v2_model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=process_v2_model.ring_catalog,
    )
    for label, obj in (
        ("ring_catalog", process_v2_model.ring_catalog),
        ("semantic_record", semantic_record),
        ("dataset", dataset),
        ("collator", collator),
    ):
        restored = pickle.loads(pickle.dumps(obj))
        assert restored is not None, label
    # The collator must survive the round trip with its Process-V2 modes intact,
    # or a worker would collate a legacy mask under a V2 identity.
    revived = pickle.loads(pickle.dumps(collator))
    assert revived.atom_delete_action_semantics == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert revived.editing_process_semantics == PROCESS_V2_EDITING_PROCESS_SEMANTICS


def test_direct_and_multiworker_collation_produce_identical_masks(
    process_v2_model,
    semantic_record,
) -> None:
    """Both eval paths must consume the same Process-V2 mask, not just the direct one.

    The worker-backed half runs in a child process under a deadline. That keeps
    the invariant intact while making a worker that never starts a bounded
    failure instead of an unbounded hang, and it isolates the DataLoader's
    process teardown from the pytest session.
    """

    direct = sample_factorized_mark_batch(
        (semantic_record,),
        batch_size=4,
        seed=5,
        late_time_fraction=0.0,
        operational_horizon=1.0,
        workers=0,
        ring_catalog=process_v2_model.ring_catalog,
        capabilities=process_v2_model.operator_capabilities,
    )
    assert direct.atom_delete_action_semantics == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert direct.atom_delete_admission_mask is not None

    def digest(tensor) -> str:
        array = tensor.detach().cpu().numpy()
        return f"{array.dtype}|{array.shape}|" + hashlib.sha256(
            array.tobytes()
        ).hexdigest()

    # The child recomputes the direct path too, so a mismatch between this
    # process and the child is attributable before the worker comparison runs.
    child_direct = _sampled_mask_digests(0)
    assert child_direct["atom_delete_mask"] == digest(direct.atom_delete_mask)
    assert child_direct["atom_delete_admission_mask"] == digest(
        direct.atom_delete_admission_mask
    )

    multiworker = _sampled_mask_digests(1)
    assert multiworker["atom_delete_action_semantics"] == (
        direct.atom_delete_action_semantics
    )
    assert multiworker["atom_delete_mask"] == digest(direct.atom_delete_mask)
    assert multiworker["atom_delete_admission_mask"] == digest(
        direct.atom_delete_admission_mask
    )


# ---- 3. Checkpoint metadata: write, read, round trip ----


def _round_trip_identity(identity: dict) -> dict:
    """Persist and reload through the deterministic serialization the repo uses."""

    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return json.loads(payload)


def test_process_v2_model_round_trips_through_checkpoint_metadata(
    ring_catalog,
    process_v2_model,
) -> None:
    identity = semantic_runtime_model_identity(process_v2_model)
    assert identity[ATOM_DELETE_ACTION_SEMANTICS_FIELD] == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert identity["editing_process_semantics"] == PROCESS_V2_EDITING_PROCESS_SEMANTICS

    restored = build_model_from_semantic_model_identity(
        _round_trip_identity(identity),
        catalog=ring_catalog,
        hidden_dim=_HIDDEN_DIM,
        message_passing_steps=_MESSAGE_PASSING_STEPS,
        mark_dim=_MARK_DIM,
        initialization_seed=_SEED,
    )
    assert restored.operator_capabilities == process_v2_model.operator_capabilities
    assert restored.operator_capabilities.fingerprint() == (
        process_v2_model.operator_capabilities.fingerprint()
    )
    assert semantic_runtime_model_identity(restored) == identity


def test_a_process_v1_payload_reconstructs_byte_identically(
    ring_catalog,
    semantic_v1_model,
) -> None:
    """A historical identity never carried the atom-delete key; it must stay absent."""

    identity = semantic_runtime_model_identity(semantic_v1_model)
    assert ATOM_DELETE_ACTION_SEMANTICS_FIELD not in identity

    restored = build_model_from_semantic_model_identity(
        _round_trip_identity(identity),
        catalog=ring_catalog,
        hidden_dim=_HIDDEN_DIM,
        message_passing_steps=_MESSAGE_PASSING_STEPS,
        mark_dim=_MARK_DIM,
        initialization_seed=_SEED,
    )
    assert restored.operator_capabilities == semantic_v1_model.operator_capabilities
    assert restored.operator_capabilities.fingerprint() == (
        semantic_v1_model.operator_capabilities.fingerprint()
    )
    assert restored.atom_delete_action_semantics == (
        LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert semantic_runtime_model_identity(restored) == identity
    # Byte-identical state, not merely an equal configuration.
    assert all(
        torch.equal(restored.state_dict()[name], tensor)
        for name, tensor in semantic_v1_model.state_dict().items()
    )


def test_a_missing_atom_delete_key_means_legacy() -> None:
    assert model_identity_atom_delete_action_semantics({}) == (
        LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert model_identity_atom_delete_action_semantics(
        {ATOM_DELETE_ACTION_SEMANTICS_FIELD: PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS}
    ) == PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS


def test_the_declared_atom_delete_modes_are_exactly_two() -> None:
    """No alias may be retained for a superseded mode name.

    A retired mode string kept as a third declared value would let a stale
    artifact keep validating under a name whose semantics have moved, which is
    exactly what the Process-V2 rename exists to prevent.
    """

    assert set(ATOM_DELETE_ACTION_SEMANTICS) == {
        LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    }
    assert len(ATOM_DELETE_ACTION_SEMANTICS) == 2


# Stale values are DERIVED from the live constants rather than retyped, so this
# test keeps testing "a superseded mode string" across a mode rename instead of
# accidentally naming the current mode.
_STALE_ATOM_DELETE_MODES = (
    f"{PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS}_superseded",
    f"{LEGACY_ATOM_DELETE_ACTION_SEMANTICS}_superseded",
    "",
    None,
)


@pytest.mark.parametrize("stale", _STALE_ATOM_DELETE_MODES)
def test_a_stale_atom_delete_mode_raises_instead_of_defaulting(stale) -> None:
    assert stale not in ATOM_DELETE_ACTION_SEMANTICS
    with pytest.raises(SemanticScratchRuntimeError, match="unknown atom-delete"):
        model_identity_atom_delete_action_semantics(
            {ATOM_DELETE_ACTION_SEMANTICS_FIELD: stale}
        )


def test_a_mixed_process_and_delete_mode_identity_raises(ring_catalog) -> None:
    """V2 process with the legacy delete mode, and the reverse, are both illegal."""

    v2_identity = dict(
        semantic_runtime_model_identity(
            _build_model(
                ring_catalog,
                process=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
                delete_mode=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
            )
        )
    )
    mixed = dict(v2_identity)
    mixed[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    with pytest.raises(ValueError, match="Process-V2 editing process requires"):
        build_model_from_semantic_model_identity(
            mixed,
            catalog=ring_catalog,
            hidden_dim=_HIDDEN_DIM,
            message_passing_steps=_MESSAGE_PASSING_STEPS,
            mark_dim=_MARK_DIM,
            initialization_seed=_SEED,
        )

    reversed_mix = dict(v2_identity)
    reversed_mix["editing_process_semantics"] = SEMANTIC_EDITING_V2_PROCESS_SEMANTICS
    with pytest.raises(ValueError, match="atom-delete semantics require"):
        build_model_from_semantic_model_identity(
            reversed_mix,
            catalog=ring_catalog,
            hidden_dim=_HIDDEN_DIM,
            message_passing_steps=_MESSAGE_PASSING_STEPS,
            mark_dim=_MARK_DIM,
            initialization_seed=_SEED,
        )


def test_a_process_v2_identity_never_reconstructs_a_legacy_delete_model(
    ring_catalog,
    process_v2_model,
) -> None:
    """Dropping the persisted key must not silently rebuild the V1 delete fiber."""

    identity = dict(semantic_runtime_model_identity(process_v2_model))
    identity.pop(ATOM_DELETE_ACTION_SEMANTICS_FIELD)
    with pytest.raises(ValueError, match="Process-V2 editing process requires"):
        build_model_from_semantic_model_identity(
            identity,
            catalog=ring_catalog,
            hidden_dim=_HIDDEN_DIM,
            message_passing_steps=_MESSAGE_PASSING_STEPS,
            mark_dim=_MARK_DIM,
            initialization_seed=_SEED,
        )


# ---- 4. Scientific identity: two contracts, no masquerade ----

_CAPABILITY_FINGERPRINT_V1 = "d246bc88d8440d31"


def _artifact(label: str) -> dict[str, str]:
    return {
        "physical_sha256": canonical_sha256({"fixture": f"{label}:physical"}),
        "semantic_sha256": canonical_sha256({"fixture": f"{label}:semantic"}),
    }


def _identity(model: dict) -> dict:
    artifacts = {key: _artifact(key) for key in ARTIFACT_KEYS}
    return build_editing_v2_scientific_identity(
        model_identity=model,
        semantic_corpus_global_registry=artifacts["semantic_corpus_global_registry"],
        active8_admission=artifacts["active8_admission"],
        successor_cache=artifacts["successor_cache"],
        sampling_sidecar=artifacts["sampling_sidecar"],
        gate_zero=artifacts["gate_zero"],
        t1=artifacts["t1"],
        p50_recipe=artifacts["p50_recipe"],
    )


def _process_v2_capability_fingerprint() -> str:
    return OperatorCapabilities(
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
        compute_ring_system_delete=False,
        editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    ).fingerprint()


def test_the_scientific_identity_accepts_a_process_v2_model() -> None:
    model = process_v2_semantic_model_identity(
        mark_dim=32,
        operator_capability_fingerprint=_process_v2_capability_fingerprint(),
    )
    assert model[ATOM_DELETE_ACTION_SEMANTICS_FIELD] == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    identity = _identity(model)
    assert identity["schema"] == PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA
    assert identity["status"] == PROCESS_V2_SCIENTIFIC_IDENTITY_STATUS
    assert identity["training_object"] == PRODUCTIVE_TRAINING_OBJECT
    assert identity["process_identity"]["process_semantics"] == (
        PROCESS_V2_EDITING_PROCESS_SEMANTICS
    )
    assert validate_editing_v2_scientific_identity(identity) == identity


def test_the_scientific_identity_still_accepts_the_historical_v1_payload() -> None:
    """The V1 contract, its field set, and its hash derivation are unchanged."""

    model = semantic_model_identity(
        mark_dim=32,
        operator_capability_fingerprint=_CAPABILITY_FINGERPRINT_V1,
    )
    assert ATOM_DELETE_ACTION_SEMANTICS_FIELD not in model
    identity = _identity(model)
    assert identity["schema"] == SCIENTIFIC_IDENTITY_SCHEMA
    assert identity["schema_version"] == SCIENTIFIC_IDENTITY_SCHEMA_VERSION
    assert identity["status"] == SCIENTIFIC_IDENTITY_STATUS
    assert identity["model_identity"] == model
    assert validate_editing_v2_scientific_identity(identity) == identity
    # The body shape, and therefore the identity hash, is the historical one.
    body = {
        "schema": identity["schema"],
        "schema_version": identity["schema_version"],
        "status": identity["status"],
        "training_object": identity["training_object"],
        "process_identity": identity["process_identity"],
        "model_identity": identity["model_identity"],
        "artifact_identities": identity["artifact_identities"],
    }
    assert identity["scientific_identity_sha256"] == canonical_sha256(body)


def test_the_two_scientific_identity_contracts_are_distinct() -> None:
    v1 = _identity(
        semantic_model_identity(
            mark_dim=32,
            operator_capability_fingerprint=_CAPABILITY_FINGERPRINT_V1,
        )
    )
    v2 = _identity(
        process_v2_semantic_model_identity(
            mark_dim=32,
            operator_capability_fingerprint=_process_v2_capability_fingerprint(),
        )
    )
    assert v1["schema"] != v2["schema"]
    assert v1["status"] != v2["status"]
    assert v1["process_identity"] != v2["process_identity"]
    assert v1["scientific_identity_sha256"] != v2["scientific_identity_sha256"]


def test_a_v1_scientific_identity_cannot_carry_the_atom_delete_key() -> None:
    model = dict(
        semantic_model_identity(
            mark_dim=32,
            operator_capability_fingerprint=_CAPABILITY_FINGERPRINT_V1,
        )
    )
    model[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    with pytest.raises(EditingV2ScientificIdentityError, match="unexpected"):
        _identity(model)


def test_a_process_v2_identity_relabelled_with_the_v1_schema_is_rejected() -> None:
    """Re-hashing alone must not let a V2 payload validate under the V1 contract."""

    identity = _identity(
        process_v2_semantic_model_identity(
            mark_dim=32,
            operator_capability_fingerprint=_process_v2_capability_fingerprint(),
        )
    )
    relabelled = dict(identity)
    relabelled["schema"] = SCIENTIFIC_IDENTITY_SCHEMA
    relabelled["status"] = SCIENTIFIC_IDENTITY_STATUS
    body = {
        key: value
        for key, value in relabelled.items()
        if key != "scientific_identity_sha256"
    }
    relabelled["scientific_identity_sha256"] = canonical_sha256(body)
    with pytest.raises(EditingV2ScientificIdentityError):
        validate_editing_v2_scientific_identity(relabelled)


def test_a_mixed_process_and_delete_mode_scientific_identity_is_rejected() -> None:
    model = dict(
        process_v2_semantic_model_identity(
            mark_dim=32,
            operator_capability_fingerprint=_process_v2_capability_fingerprint(),
        )
    )
    model[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    with pytest.raises(EditingV2ScientificIdentityError, match="exact semantic"):
        _identity(model)


def test_a_stale_atom_delete_mode_in_a_scientific_identity_is_rejected() -> None:
    model = dict(
        process_v2_semantic_model_identity(
            mark_dim=32,
            operator_capability_fingerprint=_process_v2_capability_fingerprint(),
        )
    )
    model[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = _STALE_ATOM_DELETE_MODES[0]
    with pytest.raises(EditingV2ScientificIdentityError, match="exact semantic"):
        _identity(model)
