"""Guards on corpus-scale editing training over published prep slices.

The interesting failures here are not arithmetic, so these run the REAL path:
a scratch model, genuine Active8 transitions compiled through the production
teacher-fiber compiler, published as real slices, and trained on.  A mocked
batch could not catch the two defects this suite exists for -- a published
batch whose successor-family coordinate was never attached, and a capability
regression that fails to abort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.experiments import editing_v2_process_v2_training as training
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    FrozenGateZeroSemanticContract,
    build_gate_zero_process_v2_contract,
)
from compose_v4.experiments.editing_v2_process_v2_chunk_compile import (
    RECEIPT_FILENAME,
    publish_process_v2_chunk_shard,
)
from compose_v4.experiments.editing_v2_process_v2_p50_runtime import (
    _COMPILE_SUPPORT_TIME,
    compile_prepared_entries,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    build_semantic_scratch_runtime,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
)
from compose_v4.rewrite.trace_shard import decode_state

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import process_v2_genuine_transitions as genuine_fixture  # noqa: E402


# ---- Fixture ----


def _catalog_fingerprint() -> str:
    """The catalog fingerprint this machine actually reconstructs.

    ``build_production_ringcore_catalog`` drifts on some machines -- an
    environment fact, recorded in ``learnings.md``.  The pin itself is the
    subject of ``test_editing_gate_zero_runtime.py``; neutralizing it here is
    what lets a builder path be exercised at all, and it is the ONE constant
    these tests neutralize.
    """

    import compose_v4.experiments.editing_gate_zero_runtime as gate_zero

    try:
        gate_zero.build_production_ringcore_catalog(max_atoms=40)
    except gate_zero.EditingGateZeroRuntimeError as error:
        if "catalog fingerprint drifted" not in str(error):
            raise
        observed = str(error).rsplit(": ", 1)[-1].split(" != ")[0].strip()
        gate_zero.PRODUCTION_RINGCORE_CATALOG_FINGERPRINT = observed
        return observed
    return gate_zero.PRODUCTION_RINGCORE_CATALOG_FINGERPRINT


def _model():
    fingerprint = _catalog_fingerprint()
    contract = FrozenGateZeroSemanticContract(
        source=Path("configs/editing_v2_semantic_process_v2.json"),
        payload=build_gate_zero_process_v2_contract(),
        file_sha256="0" * 64,
    )
    config = SemanticScratchModelConfig(
        initialization_seed=104729,
        max_atoms=40,
        hidden_dim=32,
        message_passing_steps=2,
        mark_dim=32,
        dtype="torch.float32",
        atom_vocabulary_class_count=15,
        catalog_fingerprint=fingerprint,
    )
    return build_semantic_scratch_runtime(config, contract).model


def _publish(root: Path, model, *, role: str, task_id: str) -> dict:
    """Publish one real slice from the genuine-transition fixture.

    Entries are compiled ONE AT A TIME and the few the scratch model cannot
    represent are skipped, so the fixture is every transition this model can
    genuinely teach rather than a subset chosen to make a number come out.
    """

    transitions = list(genuine_fixture.discover_genuine_transitions().values())
    collator = FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        chemistry_feature_cache_limit=4096,
    )
    compiled: list[dict] = []
    for index, transition in enumerate(transitions):
        body = {
            "task_identity_sha256": task_id,
            "entry_index": index,
            "partition_role": role,
            "model_family": transition.model_family,
            "capability_cell_id": transition.capability_cell_id,
            "source_state_sha256": transition.source_state_sha256,
            "target_state_sha256": transition.successor_state_sha256,
            "canonical_successor_key": transition.canonical_successor_key,
            "action_sha256": transition.action_sha256,
        }
        entry = {**body, "p50_entry_sha256": canonical_sha256(body)}
        try:
            compiled.extend(
                compile_prepared_entries(
                    model,
                    [transition.source_state],
                    [transition.successor_state],
                    [entry],
                    task_identity_sha256=task_id,
                    chemistry_feature_cache=collator._chemistry_feature_cache,
                )
            )
        except Exception:  # noqa: BLE001 - unrepresentable teachers are skipped
            continue
    assert compiled, "the genuine-transition fixture compiled nothing"

    batch = collator(
        [
            FactorizedMarkExample(
                state=decode_state(entry["exact_state"]),
                time=_COMPILE_SUPPORT_TIME,
                teacher_action=None,
                teacher_rule_name=None,
                teacher_rate=1.0,
                importance_weight=1.0,
            )
            for entry in compiled
        ]
    )
    families: dict[str, int] = {}
    cells: dict[str, int] = {}
    for entry in compiled:
        families[str(entry["model_family"])] = families.get(str(entry["model_family"]), 0) + 1
        cells[str(entry["capability_cell_id"])] = (
            cells.get(str(entry["capability_cell_id"]), 0) + 1
        )
    return publish_process_v2_chunk_shard(
        {
            "task_identity_sha256": task_id,
            "partition_role": role,
            "entry_offset": 0,
            "entry_count": len(compiled),
            "entries": compiled,
            "batch": batch,
            "family_counts": dict(sorted(families.items())),
            "capability_cell_counts": dict(sorted(cells.items())),
            "chemistry_states_cached": len(collator._chemistry_feature_cache),
        },
        output_root=root,
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("prep")
    model = _model()
    _publish(root / "train", model, role="train", task_id="a" * 64)
    _publish(root / "validation", model, role="validation", task_id="b" * 64)
    return root


# ---- The loader ----


def test_a_published_slice_loads_with_its_successor_family_coordinate_attached(prepared):
    """The chunk compile stores RAW collator output.

    Every P50 batch has the family coordinate attached before it is scored; the
    published one does not, so the loader must attach it.  Without this the
    trainer would score a batch with no selecting coordinate at all.
    """

    directory = training.iter_slice_directories(prepared / "train")[0]
    loaded = training.load_published_slice(directory, expected_role="train")

    assert loaded.partition_role == "train"
    assert loaded.entry_count == loaded.batch.batch_size == len(loaded.fibers)
    assert all(name is not None for name in loaded.batch.teacher_rule_names)
    assert tuple(loaded.batch.teacher_rule_names) == tuple(
        str(entry["model_family"]) for entry in loaded.entries
    )


def test_a_slice_whose_bytes_changed_is_refused_before_it_is_unpickled(prepared, tmp_path):
    """The receipt binds the tensors by content hash, and it is checked FIRST.

    Order matters: ``torch.load`` executes pickle, so a byte check that ran
    after it would authenticate content already acted upon.
    """

    source = training.iter_slice_directories(prepared / "train")[0]
    directory = tmp_path / "chunks" / ("c" * 64) / "000000000-000000019"
    directory.mkdir(parents=True)
    for name in ("ENTRIES.json", "BATCH.pt", RECEIPT_FILENAME):
        (directory / name).write_bytes((source / name).read_bytes())
    (directory / "BATCH.pt").write_bytes(
        (source / "BATCH.pt").read_bytes() + b"trailing"
    )

    with pytest.raises(training.ProcessV2TrainingError, match="bytes disagree"):
        training.load_published_slice(directory)


def test_a_tampered_receipt_is_refused_by_its_own_self_hash(prepared, tmp_path):
    source = training.iter_slice_directories(prepared / "train")[0]
    directory = tmp_path / "chunks" / ("d" * 64) / "000000000-000000019"
    directory.mkdir(parents=True)
    for name in ("ENTRIES.json", "BATCH.pt", RECEIPT_FILENAME):
        (directory / name).write_bytes((source / name).read_bytes())
    receipt = json.loads((directory / RECEIPT_FILENAME).read_text())
    receipt["entry_count"] = int(receipt["entry_count"]) + 1
    (directory / RECEIPT_FILENAME).write_text(json.dumps(receipt))

    with pytest.raises(training.ProcessV2TrainingError, match="self-hash|receipt"):
        training.load_published_slice(directory)


def test_the_role_is_a_load_parameter_so_a_sealed_role_cannot_be_opened(prepared):
    """Asking for the wrong role fails rather than reading and filtering."""

    directory = training.iter_slice_directories(prepared / "validation")[0]
    with pytest.raises(training.ProcessV2TrainingError, match="not 'train'"):
        training.load_published_slice(directory, expected_role="train")


def test_a_prep_root_with_no_published_slice_refuses(tmp_path):
    with pytest.raises(training.ProcessV2TrainingError, match="no chunks/"):
        training.iter_slice_directories(tmp_path)
    (tmp_path / "chunks").mkdir()
    with pytest.raises(training.ProcessV2TrainingError, match="no published slice"):
        training.iter_slice_directories(tmp_path)


# ---- The validation panel ----


def test_the_panel_is_balanced_per_family_not_drawn_in_corpus_proportion(prepared):
    """A proportional panel would hide the very signal the panel exists for."""

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)

    assert panel.example_count == sum(panel.family_counts.values())
    assert set(panel.families) == set(panel.family_counts)
    assert max(panel.family_counts.values()) <= 2
    # The fixture reaches every Active8 family, so a balanced panel measures all
    # of them -- including the rare ones an aggregate metric cannot see.
    assert len(panel.families) >= 6


def test_the_panel_measures_every_family_it_reports(prepared):
    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    observed = training.evaluate_validation_panel(_model(), panel, batch_size=8)

    assert set(observed["by_family"]) == set(panel.families)
    assert observed["example_count"] == panel.example_count
    assert all(value == value for value in observed["by_family"].values())  # not NaN


# ---- Configuration guards ----


def test_a_cadence_without_ceilings_is_refused_as_protection_that_never_refuses(prepared):
    with pytest.raises(training.ProcessV2TrainingError, match="set together"):
        training.run_process_v2_training(
            _model(), prepared / "train", optimizer_steps=1, capability_check_every_steps=5
        )


def test_ceilings_without_a_cadence_are_refused(prepared):
    with pytest.raises(training.ProcessV2TrainingError, match="set together"):
        training.run_process_v2_training(
            _model(),
            prepared / "train",
            optimizer_steps=1,
            maximum_family_nll_regression={"atom_delete": 1.0},
        )


def test_a_ceilinged_family_the_panel_cannot_measure_is_refused(prepared):
    """"Not measured" and "did not regress" must never look alike."""

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    with pytest.raises(training.ProcessV2TrainingError, match="cannot measure"):
        training.run_process_v2_training(
            _model(),
            prepared / "train",
            optimizer_steps=1,
            panel=panel,
            capability_check_every_steps=1,
            maximum_family_nll_regression={"a_family_that_does_not_exist": 1.0},
        )


@pytest.mark.parametrize("steps", [0, -1, 1.0, "8"])
def test_a_nonsense_step_count_is_refused(prepared, steps):
    with pytest.raises(training.ProcessV2TrainingError, match="positive int"):
        training.run_process_v2_training(_model(), prepared / "train", optimizer_steps=steps)


# ---- The run ----


def test_a_bounded_run_trains_every_family_and_leaves_the_hazard_frozen(prepared, tmp_path):
    model = _model()
    # Measured, not asserted from the returned literal: a flag the function sets
    # unconditionally would agree with itself no matter what the run did.
    hazard_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(training.TOTAL_HAZARD_PREFIX)
    }
    assert hazard_before, "the model has no total-hazard parameters to freeze"
    other_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith(training.TOTAL_HAZARD_PREFIX)
    }

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=4)
    result = training.run_process_v2_training(
        model,
        prepared / "train",
        optimizer_steps=12,
        batch_size=8,
        learning_rate=1e-3,
        seed=17,
        panel=panel,
        capability_check_every_steps=6,
        # Deliberately loose: this test is about the run completing and the
        # bookkeeping being true, not about the abort (covered below).
        maximum_family_nll_regression={family: 50.0 for family in panel.families},
        checkpoint_every_steps=6,
        checkpoint_root=tmp_path,
    )

    assert result["optimizer_steps_completed"] == 12
    assert result["capability_abort"] is None
    assert len(result["capability_checks"]) == 2

    import torch

    after = model.state_dict()
    assert all(
        torch.equal(tensor, after[name].detach().cpu()) for name, tensor in hazard_before.items()
    ), "the total hazard moved under the hazard-free objective"
    # ...and the run has to have actually trained, or freezing proves nothing.
    assert any(
        not torch.equal(tensor, after[name].detach().cpu()) for name, tensor in other_before.items()
    ), "no productive parameter moved, so the frozen hazard is not evidence"
    assert result["hazard_frozen"] is True
    # Nothing here authorizes anything.
    assert result["artifact_published"] is False
    assert result["checkpoint_selection_authorized"] is False

    # Every family that appeared got a finite, nonzero route gradient on at
    # least one step -- the weaker per-step claim the P50 guard makes.
    seen = {
        family: row
        for family, row in result["family_exposure"].items()
        if row["observed_example_count"]
    }
    assert seen, "no family was exposed"
    for family, row in seen.items():
        assert row["exposure_steps"] > 0, family
        assert row["finite_nonzero_action_route_gradient_exposure_steps"] > 0, family
        assert (
            row["finite_nonzero_action_route_gradient_exposure_steps"]
            <= row["exposure_steps"]
        ), family


def test_the_objective_itself_never_reaches_the_total_hazard(prepared):
    """The real reason the hazard cannot move -- measured, not assumed.

    Unfreezing every parameter and taking one backward pass leaves the
    total-hazard head with NO gradient, so the hazard is frozen by the
    OBJECTIVE.  The ``requires_grad_`` freeze and the per-step identity
    assertion are defence in depth against a future change to the loss; this is
    what would actually break first, so this is what is pinned.
    """

    import torch

    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        _index_factorized_batch,
    )
    from compose_v4.experiments.factorized_successor_training import (
        factorized_successor_identity_loss,
        forward_teacher_successor_batch,
    )

    model = _model()
    loaded = training.load_published_slice(
        training.iter_slice_directories(prepared / "train")[0], expected_role="train"
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    indices = tuple(range(min(8, loaded.entry_count)))
    batch = _index_factorized_batch(loaded.batch, indices).to(model.device)
    prediction = forward_teacher_successor_batch(
        model, batch, tuple(loaded.fibers[index] for index in indices)
    )
    factorized_successor_identity_loss(prediction, batch).backward()

    hazard = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if name.startswith(training.TOTAL_HAZARD_PREFIX)
    ]
    assert hazard, "the model has no total-hazard parameters"
    assert all(gradient is None for _name, gradient in hazard)
    # ...and the same backward pass DID reach the productive parameters, or the
    # absence above would just mean nothing was differentiated.
    productive = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith(training.TOTAL_HAZARD_PREFIX) and parameter.grad is not None
    ]
    assert productive
    assert any(float(torch.linalg.vector_norm(gradient)) > 0.0 for gradient in productive)


def test_checkpoints_go_to_disk_rather_than_accumulating_in_memory(prepared, tmp_path):
    """A long run holding every state dict in memory is itself a failure mode."""

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    result = training.run_process_v2_training(
        _model(),
        prepared / "train",
        optimizer_steps=4,
        batch_size=8,
        seed=17,
        panel=panel,
        capability_check_every_steps=4,
        maximum_family_nll_regression={family: 50.0 for family in panel.families},
        checkpoint_every_steps=2,
        checkpoint_root=tmp_path,
    )

    assert [row["optimizer_step"] for row in result["checkpoints"]] == [2, 4]
    for row in result["checkpoints"]:
        assert Path(row["path"]).is_file()
        assert len(row["model_state_sha256"]) == 64
        assert row["checkpoint_selection_authorized"] is False
    assert not any("model_state" in row for row in result["checkpoints"])


def test_a_run_resumes_from_a_checkpoint_instead_of_repeating_its_steps(prepared, tmp_path):
    """A retry budget without a resume path just burns the same steps again.

    The checkpoint therefore has to carry the optimizer moments and the sampler
    state, not only the weights: restoring weights alone looks like a resume and
    behaves like a restart.
    """

    import torch

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    common = {
        "batch_size": 8,
        "seed": 17,
        "panel": panel,
        "capability_check_every_steps": 100,
        "maximum_family_nll_regression": {family: 50.0 for family in panel.families},
    }

    interrupted = training.run_process_v2_training(
        _model(),
        prepared / "train",
        optimizer_steps=4,
        checkpoint_every_steps=4,
        checkpoint_root=tmp_path,
        **common,
    )
    checkpoint = Path(interrupted["checkpoints"][-1]["path"])

    resumed_model = _model()
    resumed = training.run_process_v2_training(
        resumed_model,
        prepared / "train",
        optimizer_steps=8,
        resume_from=checkpoint,
        **common,
    )

    assert resumed["resumed_from_optimizer_step"] == 4
    assert resumed["optimizer_steps_completed"] == 8
    # It ran the REMAINING four, not another eight.
    assert len(resumed["losses"]) == 4

    # Resuming then continuing must land where an uninterrupted run of the same
    # length lands -- otherwise the "resume" is a differently-seeded restart.
    straight = training.run_process_v2_training(
        _model(), prepared / "train", optimizer_steps=8, **common
    )
    assert torch.allclose(
        torch.tensor(straight["losses"][4:]), torch.tensor(resumed["losses"]), atol=1e-5
    )
    assert resumed["final_model_state_sha256"] == straight["final_model_state_sha256"]


def test_a_checkpoint_that_cannot_resume_is_refused_rather_than_half_restored(tmp_path):
    """Weights without optimizer moments is a restart wearing a resume's name."""

    path = tmp_path / "partial.pt"
    import torch

    torch.save({"optimizer_step": 4, "model_state": {}}, path)
    model = _model()
    optimizer = torch.optim.AdamW([next(iter(model.parameters()))])
    with pytest.raises(training.ProcessV2TrainingError, match="lacks"):
        training.load_training_checkpoint(path, model, optimizer)


def test_resuming_past_the_requested_step_count_is_refused(prepared, tmp_path):
    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    common = {
        "batch_size": 8,
        "seed": 17,
        "panel": panel,
        "capability_check_every_steps": 100,
        "maximum_family_nll_regression": {family: 50.0 for family in panel.families},
    }
    done = training.run_process_v2_training(
        _model(),
        prepared / "train",
        optimizer_steps=4,
        checkpoint_every_steps=4,
        checkpoint_root=tmp_path,
        **common,
    )
    with pytest.raises(training.ProcessV2TrainingError, match="already at step"):
        training.run_process_v2_training(
            _model(),
            prepared / "train",
            optimizer_steps=4,
            resume_from=Path(done["checkpoints"][-1]["path"]),
            **common,
        )


def test_a_checkpoint_cadence_without_a_root_is_refused(prepared):
    with pytest.raises(training.ProcessV2TrainingError, match="checkpoint root"):
        training.run_process_v2_training(
            _model(), prepared / "train", optimizer_steps=1, checkpoint_every_steps=1
        )


def test_a_regressing_family_aborts_the_run_before_it_finishes(prepared):
    """The guard the barred 16,000-step run lacked.

    A zero ceiling makes any per-family worsening a regression, so this proves
    the abort is wired to the measurement rather than merely recorded beside it:
    the run must stop EARLY and name the family.
    """

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=4)
    result = training.run_process_v2_training(
        _model(),
        prepared / "train",
        optimizer_steps=12,
        batch_size=8,
        seed=17,
        panel=panel,
        capability_check_every_steps=3,
        maximum_family_nll_regression={family: 0.0 for family in panel.families},
    )

    abort = result["capability_abort"]
    assert abort is not None
    assert result["optimizer_steps_completed"] < 12
    assert abort["optimizer_step"] == result["optimizer_steps_completed"]
    assert abort["family"] in panel.families
    assert abort["regression_nats"] > abort["ceiling_nats"] == 0.0


def test_training_times_are_sampled_per_example_while_the_panel_stays_fixed(prepared):
    """Training draws U(0,1) times; validation stays at the collated time.

    Slices are collated at one support time, so a trainer that forgot to
    override ``times`` would train the whole corpus at a single time and still
    look healthy.  Validation must NOT be overridden, or successive capability
    checks would differ by time as well as by the model.
    """

    entries = training.load_published_slice(
        training.iter_slice_directories(prepared / "train")[0], expected_role="train"
    ).entries
    times = training._sampled_times(entries[:8], stream_index=0, seed=17)

    assert len(set(times)) == len(times), "times are not per-example"
    assert all(0.0 < value < 1.0 for value in times)
    assert times != training._sampled_times(entries[:8], stream_index=8, seed=17)
    # Deterministic in the seed, so a resumed run replays the same draws.
    assert times == training._sampled_times(entries[:8], stream_index=0, seed=17)

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    assert all(
        float(value) == _COMPILE_SUPPORT_TIME
        for shard in panel.shards
        for value in shard.batch.times
    )


def test_the_run_itself_scores_at_sampled_times_not_the_collated_one(prepared, monkeypatch):
    """Assert the PATH, not the helper.

    Testing ``_sampled_times`` alone passes even if the loop never calls it, so
    this records what the forward pass actually received.
    """

    observed: list[tuple[bool, tuple[float, ...]]] = []
    real = training.forward_teacher_successor_batch

    def _record(model, batch, fibers, **kwargs):
        # Keyed on train/eval mode, because the BASELINE panel evaluation runs
        # before the first optimizer step -- slicing by position would compare
        # panel forwards to the training claim.
        observed.append((model.training, tuple(float(value) for value in batch.times)))
        return real(model, batch, fibers, **kwargs)

    monkeypatch.setattr(training, "forward_teacher_successor_batch", _record)
    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    training.run_process_v2_training(
        _model(),
        prepared / "train",
        optimizer_steps=3,
        batch_size=8,
        seed=17,
        panel=panel,
        capability_check_every_steps=3,
        maximum_family_nll_regression={family: 50.0 for family in panel.families},
    )

    training_times = [times for is_training, times in observed if is_training]
    panel_times = [times for is_training, times in observed if not is_training]

    assert len(training_times) == 3
    for times in training_times:
        assert len(set(times)) > 1, "a training step scored every row at one time"
        assert all(value != _COMPILE_SUPPORT_TIME for value in times)
    assert len(set(training_times)) == 3, "steps reused one draw"

    # The panel, by contrast, must stay at the one time it was collated at.
    assert panel_times
    assert all(value == _COMPILE_SUPPORT_TIME for times in panel_times for value in times)


def test_a_run_reads_slices_rather_than_reselecting_one_cached_batch(prepared):
    """The barred run was data starved; a streamed run reads the corpus.

    ``slices_read`` counts loads, so it grows with epochs.  Reading once and
    replaying a cached batch would leave it pinned at the slice count.
    """

    panel = training.build_validation_panel(prepared / "validation", examples_per_family=2)
    result = training.run_process_v2_training(
        _model(),
        prepared / "train",
        optimizer_steps=6,
        batch_size=8,
        seed=17,
        panel=panel,
        capability_check_every_steps=6,
        maximum_family_nll_regression={family: 50.0 for family in panel.families},
    )

    assert result["epochs_started"] > 1
    assert result["slices_read"] >= result["epochs_started"]
    assert result["unique_training_examples_seen"] > 0
