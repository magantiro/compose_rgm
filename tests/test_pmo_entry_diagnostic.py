"""Guards for the PMO teacher-atlas ENTRY diagnostic.

The properties these pin are the ones that decide whether a number the harness
produces means anything: a seed that is reproducible across processes, a
threshold that cannot move after a measurement, a parent state that still
carries the ``atom_insert`` family, and an ordinal that counts refused draws.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from compose_v4.experiments import pmo_entry_diagnostic as entry

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sealed() -> dict:
    path = REPO_ROOT / "diagnostics/pmo_entry_diagnostic_v1/predicate_v1.json"
    if not path.is_file():
        pytest.skip("the entry predicate has not been sealed in this checkout")
    return json.loads(path.read_text())


# ---- Seeding ----


def test_stable_seed_is_a_pinned_digest_not_a_salted_hash():
    # A literal, so a switch to hash() -- which is PYTHONHASHSEED-salted and has
    # already made one table in this repository irreproducible -- fails here
    # rather than silently in a later run.
    assert entry.stable_seed("pmo_entry_diagnostic_v1", "arm1", "CCO") == 2569041677456740427
    assert entry.stable_seed("a", "b") != entry.stable_seed("b", "a")


def test_stable_seed_survives_a_fresh_interpreter():
    import subprocess
    import sys

    code = (
        "from compose_v4.experiments.pmo_entry_diagnostic import stable_seed;"
        "print(stable_seed('x','y'))"
    )
    env = {"PYTHONHASHSEED": "17", "PATH": "/usr/bin:/bin", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert int(out.stdout.strip()) == entry.stable_seed("x", "y")


# ---- Predicate sealing ----


def test_sealed_predicate_matches_the_module_constants():
    document = _sealed()
    payload = entry.assert_predicate_sealed(REPO_ROOT)
    assert payload["delta"] == entry.ENTRY_DELTA
    assert payload["fingerprint"] == entry.FINGERPRINT
    assert entry.payload_sha256(payload) == document["payload_sha256"]


def test_a_moved_threshold_is_refused(tmp_path):
    document = _sealed()
    payload = dict(document["payload"])
    payload["delta"] = entry.ENTRY_DELTA + 0.05
    target = tmp_path / "diagnostics/pmo_entry_diagnostic_v1/predicate_v1.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps({"payload": payload, "payload_sha256": entry.payload_sha256(payload)})
    )
    with pytest.raises(ValueError, match="may not move after measurement"):
        entry.assert_predicate_sealed(tmp_path)


def test_a_tampered_predicate_payload_is_refused(tmp_path):
    document = _sealed()
    payload = dict(document["payload"])
    payload["statement"] = "tampered"
    target = tmp_path / "diagnostics/pmo_entry_diagnostic_v1/predicate_v1.json"
    target.parent.mkdir(parents=True)
    # The recorded hash is deliberately left addressing the ORIGINAL payload, so
    # only the hash check can refuse this.
    target.write_text(
        json.dumps({"payload": payload, "payload_sha256": document["payload_sha256"]})
    )
    with pytest.raises(ValueError, match="hash has moved"):
        entry.assert_predicate_sealed(tmp_path)


def test_the_sealed_calibration_justifies_its_own_threshold():
    payload = _sealed()["payload"]
    calibration = payload["calibration"]
    # The threshold must sit strictly above every objective-blind chance pair.
    assert calibration["chance_reference"]["max"] < payload["delta"]
    assert calibration["chance_reference"]["fraction_at_or_above_delta"] == 0.0
    assert calibration["cross_answer_reference"]["fraction_at_or_above_delta"] == 0.0
    # ... and no higher than the measured still-productive band at the median.
    assert payload["delta"] <= calibration["spine_ladder_summary"]["near_anchor_median"]


# ---- Regions ----


def test_primary_region_is_the_measured_spine_anchor_for_every_task():
    # Rebuilt from the atlas here rather than read back from the module, so a
    # region rule that quietly widened to sibling endpoints fails.
    payload = entry.load_atlas_payload(REPO_ROOT)
    expected: dict[str, set[str]] = {}
    endpoints: dict[str, set[str]] = {}
    for route in payload["routes"]:
        anchors = {c["smiles"] for c in route["checkpoints"] if c["label"] == "anchor"}
        endpoints.setdefault(route["task"], set()).update(anchors)
        if route.get("is_spine"):
            expected.setdefault(route["task"], set()).update(anchors)
    regions = entry.load_productive_regions(REPO_ROOT)
    assert len(regions) == 11
    for task, region in regions.items():
        declared = payload["per_task"][task]["destination_smiles"]
        assert set(region.primary) == expected[task], task
        assert declared in region.primary, task
        assert len(region.primary) <= 2, task
        assert set(region.primary) < set(region.wide) or len(endpoints[task]) == 1


def test_fingerprint_agrees_with_the_atlas_fingerprint():
    # Derived from the other module rather than transcribed, so a drift in
    # either definition fails here.
    from rdkit import DataStructs

    from compose_v4.experiments.pmo_atlas_discovery import _fingerprint as atlas_print

    for smiles in ("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"):
        assert (
            DataStructs.TanimotoSimilarity(entry.fingerprint(smiles), atlas_print(smiles))
            == 1.0
        )


# ---- Parents ----


def _blind_rows(task: str):
    folder = Path.home() / "compose_pmo_atlas_runs" / "test_c_blind" / f"{task}__blind_250"
    if not folder.is_dir():
        pytest.skip("the blind run artifacts are not present in this checkout")
    from compose_v4.experiments.pmo_atlas_discovery import read_blind_trajectory

    return read_blind_trajectory(folder)


def test_parents_are_48_slot_states_not_smiles_reparses():
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph

    rows = _blind_rows("celecoxib_rediscovery")
    parents = entry.select_parents(rows, "celecoxib_rediscovery", per_stratum=1)
    assert parents
    for parent in parents:
        graph = parent.graph()
        assert graph.n_atoms == 48, "the PMO proposal path requires a 48-slot source"
    tight = smiles_to_molecular_graph(parents[0].endpoint)
    assert tight.n_atoms < 48, (
        "the trap this guard exists for: a SMILES round trip yields a tight graph, "
        "which silently deletes the atom_insert family from the legal support"
    )


def test_parent_selection_is_stratified_and_deterministic():
    rows = _blind_rows("celecoxib_rediscovery")
    first = entry.select_parents(rows, "celecoxib_rediscovery", per_stratum=2)
    again = entry.select_parents(rows, "celecoxib_rediscovery", per_stratum=2)
    assert [p.endpoint for p in first] == [p.endpoint for p in again]
    strata = {p.score_stratum for p in first}
    assert strata == {"q1_worst", "q2", "q3", "q4_best"}
    # The best-scoring blind molecule is always present: it is the parent a
    # score-greedy controller is most likely to expand.
    best = max(rows, key=lambda row: row.score)
    assert best.endpoint in {p.endpoint for p in first}
    # ... and it is not merely the first N of the artifact's own ordering.
    assert [p.index for p in first] != sorted(row.index for row in rows)[: len(first)]


# ---- Curves ----


def _rows(spec):
    out = []
    for at, (status, similarity) in enumerate(spec, start=1):
        out.append(
            {
                "draw": at,
                "status": status,
                "endpoint": None if status == "refused" else f"m{at}",
                "similarity": similarity,
                "primitive_count": None if status == "refused" else 1,
                "heavy_atoms": None if status == "refused" else 10,
            }
        )
    return out


def test_first_entry_counts_refused_draws():
    rows = _rows([("refused", None), ("refused", None), ("executed", 0.9)])
    assert entry.first_entry_draw(rows, 0.3) == 3, (
        "a refusal is a proposal the mechanism made; dropping it would flatter "
        "a mechanism that mostly fails to execute"
    )
    assert entry.first_entry_draw(_rows([("executed", 0.1)]), 0.3) is None


def test_entry_curve_is_monotone_and_counts_parents():
    per_parent = [
        _rows([("executed", 0.1)] * 3 + [("executed", 0.5)]),
        _rows([("executed", 0.1)] * 4),
    ]
    curve = entry.entry_curve(per_parent, 0.3, 4)
    values = [point["probability"] for point in curve]
    assert values == sorted(values)
    assert curve[-1]["parents_entered"] == 1
    assert curve[-1]["parents"] == 2
    assert curve[0]["proposals"] == 1 and curve[0]["parents_entered"] == 0


def test_rate_bound_at_zero_is_the_rule_of_three():
    assert entry.rate_bound(0, 1000) == pytest.approx(3.0 / 1000, rel=0.02)
    assert entry.rate_bound(0, 100) > entry.rate_bound(0, 1000)
    with pytest.raises(ValueError):
        entry.rate_bound(0, 0)


# ---- Arm registry ----


def test_every_arm_has_the_registration_signature():
    for name, record in entry.ARM_REGISTRY.items():
        parameters = list(inspect.signature(record["function"]).parameters)
        assert parameters == ["source", "rng", "draws"], name
        assert record["describe"]["role"] in {"arm", "negative_control"}


def test_registering_a_duplicate_arm_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        entry.register_arm(
            "arm1_baseline_b", entry.baseline_b_proposals, describe={"role": "arm"}
        )
    with pytest.raises(ValueError, match="registered:"):
        entry.arm("no_such_arm")


def test_the_negative_control_spans_the_pmo_primitive_vocabulary():
    rows = _blind_rows("celecoxib_rediscovery")
    parent = entry.select_parents(rows, "celecoxib_rediscovery", per_stratum=1)[0]
    families = {family for family, _ in entry._legal_primitive_actions(parent.graph())}
    assert {"atom_insert", "atom_delete"} <= families, (
        "a control that cannot change heavy-atom count is not matched to a "
        "mechanism that can"
    )
    assert "bond_reorder" not in families


def test_both_registered_arms_execute_on_a_real_parent():
    rows = _blind_rows("celecoxib_rediscovery")
    parent = entry.select_parents(rows, "celecoxib_rediscovery", per_stratum=1)[0]
    for name in ("arm1_baseline_b", "nc1_uniform_legal_edits"):
        rng = np.random.default_rng(entry.stable_seed("test", name))
        proposals = list(entry.ARM_REGISTRY[name]["function"](parent.graph(), rng, 4))
        assert [p.draw for p in proposals] == [1, 2, 3, 4]
        executed = [p for p in proposals if p.status == "executed"]
        assert executed, name
        for proposal in executed:
            assert entry.canonical(proposal.endpoint) is not None


# ---- Rung ladder and the predeclared gate ----


def test_rung_ladder_is_monotone_and_starts_above_chance():
    thresholds = [threshold for _, threshold in entry.RUNGS]
    assert thresholds == sorted(thresholds)
    assert len(set(thresholds)) == len(thresholds)
    payload = _sealed()["payload"]
    # The first rung must sit above every objective-blind chance pair, or
    # "above_chance" would be a name rather than a measurement.
    assert thresholds[0] > payload["calibration"]["chance_reference"]["max"]
    assert entry.ENTRY_DELTA in thresholds


def test_best_rung_and_rung_index_order_correctly():
    assert entry.best_rung(None) is None
    assert entry.best_rung(0.2) is None
    assert entry.best_rung(0.25) == "above_chance"
    assert entry.best_rung(0.30) == "entry"
    assert entry.best_rung(0.99) == "anchor"
    assert entry.rung_index(None) == -1
    assert entry.rung_index("entry") > entry.rung_index("above_chance")
    assert entry.rung_index("anchor") == len(entry.RUNGS) - 1
    with pytest.raises(ValueError):
        entry.rung_index("not_a_rung")


def test_the_pass_criterion_is_sealed_with_the_predicate():
    payload = _sealed()["payload"]
    criterion = payload["pass_criterion"]
    assert json.loads(json.dumps(criterion)) == json.loads(
        json.dumps(entry.PASS_CRITERION)
    )
    assert set(criterion["primary_tasks"]) == {
        "celecoxib_rediscovery",
        "albuterol_similarity",
        "jnk3",
    }
    assert criterion["control_task"] == "qed"
    assert set(payload["gate_tasks"]) == set(entry.GATE_TASKS)
    assert payload["matched_proposals"] == list(entry.MATCHED_PROPOSALS)


def test_a_baseline_only_report_cannot_declare_the_gate_passed():
    import pmo_entry_diagnostic as driver

    verdict = driver._pass_criterion({"arm1_baseline_b": {"per_task": {}}}, None)
    assert verdict["verdict"] == "UNEVALUATED_NO_CHALLENGER_ARM"
    assert "not a verdict on C" in verdict["statement"]


# ---- Manifold column ----


def test_drug_likeness_reads_qed_and_heavy_atoms():
    aspirin = entry.drug_likeness("CC(=O)Oc1ccccc1C(=O)O")
    assert 0.0 <= aspirin["qed"] <= 1.0
    assert aspirin["heavy_atoms"] == 13
    assert entry.drug_likeness("not a molecule") == {"qed": None, "heavy_atoms": None}


def test_manifold_quartiles_report_drift_in_the_right_direction():
    import pmo_entry_diagnostic as driver

    rows = [
        {
            "proposal_index": at,
            "status": "executed",
            "qed": 0.8 - 0.006 * at,
            "heavy_atoms": 20,
            "rung": None,
        }
        for at in range(1, 41)
    ]
    block = driver._manifold(rows, [])
    assert len(block["qed_by_proposal_quartile"]) == 4
    assert block["qed_drift_q1_to_q4"] < 0, "a falling QED series must report negative drift"
    assert block["entrant_descriptors"] == []
    assert block["entrant_median_qed"] is None
    assert block["init_median_qed"] == entry.INIT_MEDIAN_QED


def test_manifold_block_is_empty_rather_than_zero_without_data():
    import pmo_entry_diagnostic as driver

    block = driver._manifold([], [])
    assert block["median_qed"] is None
    assert block["qed_drift_q1_to_q4"] is None


# ---- End to end ----


def test_measure_interleaves_parents_and_seals_its_shard(tmp_path):
    import pmo_entry_diagnostic as driver

    if not (Path.home() / "compose_pmo_atlas_runs" / "test_c_blind").is_dir():
        pytest.skip("the blind run artifacts are not present in this checkout")
    out = tmp_path / "shard.json"
    args = driver.argparse.Namespace(
        repo_root=str(REPO_ROOT),
        arm="arm1_baseline_b",
        task="celecoxib_rediscovery",
        draws=12,
        parents_per_stratum=2,
        descendant_entrants=0,
        descendant_draws=1,
        parent_source="blind_visited_stratified",
        output=str(out),
    )
    assert driver.phase_measure(args) == 0
    document = json.loads(out.read_text())
    payload = document["payload"]
    assert entry.payload_sha256(payload) == document["payload_sha256"]
    assert payload["charged_oracle_calls"] == 0
    assert payload["proposals_drawn"] == 12
    assert [row["proposal_index"] for row in payload["proposals"]] == list(range(1, 13))
    # Round robin: the first proposals must come from DIFFERENT parents, not one
    # parent's whole run. Slice order in this repository is lane- and
    # family-sorted and taking the first N has produced bad measurements before.
    first = [row["parent_endpoint"] for row in payload["proposals"][: payload["parents"]]]
    assert len(set(first)) == payload["parents"]
    for row in payload["proposals"]:
        assert row["provenance_class"] in {"local_search", "broad_exploration"}
        if row["status"] == "executed":
            assert row["qed"] is not None or row["endpoint"] is not None


# ---- the round-zero population axis ----


def test_every_parent_source_has_the_registration_signature():
    assert entry.PARENT_SOURCES, "no parent source is registered"
    for name, record in entry.PARENT_SOURCES.items():
        assert record["name"] == name
        assert callable(record["function"])
        describe = record["describe"]
        assert describe["role"] and describe["label"]
        assert isinstance(describe["initialization_count"], int)
        assert isinstance(describe["carries_scores"], bool)


def test_registering_a_duplicate_parent_source_is_refused():
    name = next(iter(entry.PARENT_SOURCES))
    with pytest.raises(ValueError):
        entry.register_parent_source(
            name, lambda *a, **k: (), describe={"role": "x", "label": "x"}
        )


def test_an_unknown_parent_source_is_refused_by_name():
    with pytest.raises(KeyError):
        entry.parent_source("no_such_population")


def test_the_sixteen_molecule_source_reproduces_the_sealed_production_file():
    """Arm 2 is a COUNT change, so its control must be production's own draw.

    If this ever diverges, the 100-molecule arm is no longer the same bank, the
    same selector and the same seed with one value changed, and the comparison
    stops isolating the initialization.
    """

    sealed = json.loads(
        (
            REPO_ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
        ).read_text()
    )
    parents = entry.parent_source("init_bank_16")["function"](
        REPO_ROOT, "celecoxib_rediscovery"
    )
    assert len(parents) == sealed["count"] == 16
    assert {parent.endpoint for parent in parents} == {
        candidate["endpoint"] for candidate in sealed["candidates"]
    }


def test_the_hundred_molecule_source_is_a_superset_at_48_slots():
    small = entry.parent_source("init_bank_16")["function"](
        REPO_ROOT, "celecoxib_rediscovery"
    )
    large = entry.parent_source("init_bank_100")["function"](
        REPO_ROOT, "celecoxib_rediscovery"
    )
    assert len(large) == 100
    assert {parent.endpoint for parent in small} <= {parent.endpoint for parent in large}
    for parent in large:
        # PMO states are 48 slots and the path hard-refuses 40. A SMILES round
        # trip yields a TIGHT graph and silently deletes the whole atom_insert
        # family, so this is the load-bearing check on the construction path.
        assert len(parent.state["atom_types"]) == 48
        assert parent.graph().n_real_atoms <= 40
        assert parent.score is None
        assert parent.score_stratum == "objective_blind"


def test_the_hundred_molecule_arm_declares_its_budget_limit():
    """100 initialization molecules are 40% of a 250-call budget.

    The arm must carry that where a reader of the shard sees it, not only in a
    handback, or the number gets quoted against a budget it is inadmissible at.
    """

    describe = entry.parent_source("init_bank_100")["describe"]
    assert "250" in describe["budget_caveat"]
    assert "NOT admissible" in describe["budget_caveat"]


def test_the_default_population_keeps_the_original_seed_stream():
    """Shards measured before the population axis existed must stay re-derivable.

    The two negative controls cost roughly ten seconds a draw; silently moving
    their seeds would have invalidated them with nothing to show it.
    """

    import pmo_entry_diagnostic as driver

    if not (Path.home() / "compose_pmo_atlas_runs" / "test_c_blind").is_dir():
        pytest.skip("the blind run artifacts are not present in this checkout")

    # Drive the PRODUCTION path. Recomputing the seed from a transcription of
    # the rule here would be a comparison whose expectation moves with the code
    # under test -- it cannot fail, and a mutation battery proved exactly that
    # against the first version of this test.
    drawn = {}

    def capture(source, rng, draws):
        drawn.setdefault(len(drawn), rng.bit_generator.state)
        return iter(())

    entry.register_arm(
        "seedprobe_arm",
        capture,
        describe={"role": "probe", "label": "seed capture", "provenance_class": {}},
    )
    try:
        states = {}
        for population in ("blind_visited_stratified", "init_bank_16"):
            drawn.clear()
            args = driver.argparse.Namespace(
                repo_root=str(REPO_ROOT),
                arm="seedprobe_arm",
                task="celecoxib_rediscovery",
                draws=0,
                parents_per_stratum=2,
                descendant_entrants=0,
                descendant_draws=1,
                parent_source=population,
                output=str(Path(tempfile.mkdtemp()) / "shard.json"),
            )
            driver.phase_measure(args)
            states[population] = drawn[0]

        default_parent = entry.parent_source("blind_visited_stratified")["function"](
            REPO_ROOT, "celecoxib_rediscovery", per_stratum=2
        )[0]
        expected = np.random.default_rng(
            entry.stable_seed(
                entry.DIAGNOSTIC_SCHEMA,
                "seedprobe_arm",
                "celecoxib_rediscovery",
                default_parent.endpoint,
                "r0",
            )
        ).bit_generator.state
        # The pre-axis rule, written out in full, must still be what the default
        # population produces.
        assert states["blind_visited_stratified"]["state"] == expected["state"]
        # A different population must not collide with it.
        assert states["init_bank_16"]["state"] != expected["state"]
    finally:
        entry.ARM_REGISTRY.pop("seedprobe_arm", None)


def test_a_ladder_row_is_a_mechanism_and_a_population():
    import pmo_entry_diagnostic as driver

    assert driver._arm_key({"arm": "a", "parent_source": "init_bank_100"}) == (
        "a@init_bank_100"
    )
    # Absent or default must keep the original key so committed shards stay
    # addressed by the name they were filed under.
    assert driver._arm_key({"arm": "a"}) == "a"
    assert driver._arm_key({"arm": "a", "parent_source": "blind_visited_stratified"}) == "a"


def test_the_sealed_criterion_refuses_a_subset_verdict():
    """The criterion spans three tasks plus a control; a one-task run cannot fire it."""

    import pmo_entry_diagnostic as driver

    arms = {
        "base": {"per_task": {"celecoxib_rediscovery": {"prefixes": {}}}},
        "chal": {"per_task": {"celecoxib_rediscovery": {"prefixes": {}}}},
    }
    verdict = driver._pass_criterion(arms, "chal", "base")
    assert verdict["verdict"] == "UNEVALUATED_CRITERION_SPANS_TASKS_NOT_MEASURED"
    assert set(verdict["tasks_missing_from_this_run"]) == {
        "albuterol_similarity",
        "jnk3",
        "qed",
    }


def test_a_histogram_of_non_numeric_values_does_not_raise():
    """The production arm once filed a LIST under a count column and crashed the report."""

    import pmo_entry_diagnostic as driver

    assert driver._histogram([[0, 1], [0], [0, 1]]) == {"[0, 1]": 2, "[0]": 1}
    assert driver._histogram([3, 1, 2]) == {"1": 1, "2": 1, "3": 1}
