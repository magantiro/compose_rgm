"""Guards for the fragment-constrained benchmark protocol.

Each test here exists because a specific way of inflating a reported number is
cheap and invisible in the artifact: counting a failed attempt as no attempt,
re-drawing until a sample lands, tuning the sampler per instance, deriving a
"reproducible" seed from a salted hash, or auditing fragment preservation with
a check that cannot fail.  Every test below is paired with a negative control:
it asserts the guard FIRES on the bad input, not merely that it passes on the
good one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    assert_emission_invariants,
    official_unique_valid,
)

MANIFEST = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"


def _prompts():
    return load_genmol_prompts(MANIFEST)


def _run_suite():
    import run_fragment_constrained_suite as suite

    return suite


# ---- Denominator: a failed attempt is an attempt ----


def test_failure_placeholder_is_counted_invalid_not_dropped():
    """An attempt that produced nothing must lower validity, not vanish.

    ``FAILED_SAMPLE_PLACEHOLDER`` is the empty string, and an empty SMILES
    PARSES to a zero-atom Mol rather than returning None.  The official metric
    survives that only because it tests the truthiness of the canonical string.
    If that ever changed, every failed attempt would silently score as valid,
    so pin the behaviour rather than trusting it.
    """
    mol = Chem.MolFromSmiles(FAILED_SAMPLE_PLACEHOLDER)
    assert mol is not None, "an empty SMILES is expected to parse to a 0-atom Mol"
    assert not Chem.MolToSmiles(mol, canonical=True)

    real = "c1ccccc1"
    samples = [real] * 70 + [FAILED_SAMPLE_PLACEHOLDER] * 30
    assert len(official_unique_valid(samples)) == 1
    # The valid count the official validity ratio is built from.
    valid = [s for s in samples if s and Chem.MolToSmiles(Chem.MolFromSmiles(s))]
    assert len(valid) == 70
    assert len(valid) / len(samples) * 100 == pytest.approx(70.0)


def test_official_metrics_refuse_a_short_sample_list():
    """Scoring 40 survivors as if 40 were requested is the cherry-pick to block."""
    from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics

    with pytest.raises(ValueError, match="expected exactly 100"):
        official_prompt_metrics(["c1ccccc1"] * 40, expected_samples=100)


def test_emission_invariants_reject_noncanonical_and_disconnected():
    """Both failures inflate a reported number, so both must raise."""
    assert_emission_invariants(["c1ccccc1", FAILED_SAMPLE_PLACEHOLDER])
    with pytest.raises(ValueError, match="not canonical"):
        assert_emission_invariants(["C1=CC=CC=C1"])
    with pytest.raises(ValueError, match="disconnected"):
        assert_emission_invariants(["c1ccccc1.CO"])


# ---- Reproducibility ----


def test_prompt_rng_seed_is_stable_across_processes():
    """The published seed must name one stream, not one process's stream."""
    suite = _run_suite()
    assert suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0) == (
        suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0)
    )
    # Pinned so a future refactor of the derivation is visible as a diff.
    assert suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0) == 722072583
    assert suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 1) != (
        suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0)
    )


def test_frozen_sampler_identity_moves_when_any_knob_moves():
    """A per-instance tune must be visible in the artifact's hash."""
    from compose_v4.benchmark.fragment_conditioned_sampler import SamplerConfig

    suite = _run_suite()
    base = suite.frozen_sampler_identity(SamplerConfig())
    assert base["config_sha256"] == suite.frozen_sampler_identity(SamplerConfig())["config_sha256"]
    for field, value in (
        ("max_events", 64),
        ("mark_attempts_per_event", 48),
        ("operational_horizon", 32.0),
        ("n_slots", 56),
    ):
        tuned = suite.frozen_sampler_identity(SamplerConfig(**{field: value}))
        assert tuned["config_sha256"] != base["config_sha256"], field


# ---- Fragment preservation, with a negative control ----


def test_preservation_audit_accepts_a_superstructure_and_rejects_a_near_miss():
    """The audit must FAIL on a molecule that lost the core.

    A containment check that returns True for everything would report a
    preservation rate of 100% no matter what the sampler did.
    """
    suite = _run_suite()
    prompt = next(
        p
        for p in _prompts()
        if p.drug_name == "BARICITINIB" and p.task is FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    queries = suite.audit_queries(prompt)
    core = prompt.fragments[0]

    # A genuine superstructure: the core plus a methyl.
    grown = "Cc1nc(-c2cnn(C3CNC3)c2)c2cc[nH]c2n1"
    assert suite.contains_all_fragments(grown, queries)

    # Negative controls.
    assert not suite.contains_all_fragments("c1ccccc1", queries)
    assert not suite.contains_all_fragments("", queries)
    # Disconnected: the core is present but the molecule is not one piece.
    assert not suite.contains_all_fragments(f"{core}.CO", queries)
    # A ring deletion inside the core must not pass.
    broken = "Cc1nc(-c2cnn(CCCN)c2)c2cc[nH]c2n1"
    assert not suite.contains_all_fragments(broken, queries)


def test_linker_audit_requires_two_disjoint_embeddings():
    """Two fragments matching the SAME atoms is not two fragments."""
    suite = _run_suite()
    prompt = next(
        p
        for p in _prompts()
        if p.drug_name == "BARICITINIB" and p.task is FragmentTask.LINKER_DESIGN
    )
    queries = suite.audit_queries(prompt)
    assert len(queries) == 2
    assert suite.contains_all_fragments(prompt.original_smiles, queries)
    # One of the two cores alone must not satisfy the pair.
    alone = Chem.MolToSmiles(queries[1])
    assert not suite.contains_all_fragments(alone, queries)


def test_distance_reference_is_the_dummy_stripped_prompt():
    """Distance depends on the reference, so pin which molecule it is."""
    suite = _run_suite()
    prompt = next(
        p
        for p in _prompts()
        if p.drug_name == "BARICITINIB" and p.task is FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    reference = suite.prompt_reference_smiles(prompt)
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(prompt.fragments[0]), canonical=True)
    assert reference == expected
    assert "*" not in reference


def test_distance_reference_parses_for_every_released_prompt():
    """An aromatic-N attachment point makes the dummy-stripped SMILES unparseable.

    Upstream's ``calculate_average_tanimoto`` raises ``Invalid prompt SMILES
    string`` on such a reference, which killed three shards mid-sweep.  Capping
    the dummy with hydrogen fixes it; this pins that EVERY released prompt
    yields a parseable reference, and that the fix is confined to the
    pathological cases -- for any fragment whose stripped form already parsed,
    the reference must be unchanged, or previously reported distances would
    have silently moved.
    """
    suite = _run_suite()
    changed = []
    for prompt in _prompts():
        stripped = ".".join(
            Chem.MolToSmiles(q, canonical=True) for q in suite.audit_queries(prompt)
        )
        reference = suite.prompt_reference_smiles(prompt)
        assert Chem.MolFromSmiles(reference) is not None, (
            prompt.drug_name,
            prompt.task.value,
            reference,
        )
        if stripped != reference:
            changed.append((prompt.drug_name, prompt.task.value))
            # It may only differ where the old form could not be used at all.
            assert Chem.MolFromSmiles(stripped) is None, (
                prompt.drug_name,
                prompt.task.value,
            )
    # LESINURAD (motif/linker/morphing) and MARIBAVIR (linker/morphing) attach at
    # an aromatic nitrogen. Nothing else may move.
    assert sorted(changed) == sorted(
        [
            ("LESINURAD", "motif_extension"),
            ("LESINURAD", "linker_design"),
            ("LESINURAD", "scaffold_morphing"),
            ("MARIBAVIR", "linker_design"),
            ("MARIBAVIR", "scaffold_morphing"),
        ]
    ), changed


def test_task_success_is_never_reported_as_chemical_validity():
    """The two quantities must stay distinct, and ordered.

    The suite's original "validity" column was task success: the sampler
    withholds any endpoint failing the task constraint and emits an unparseable
    placeholder, which the official function counts as invalid. Placed beside a
    baseline whose evaluator never inspects the prompt fragment, that understated
    COMPOSE by 42 points on motif extension and 90 on scaffold decoration.

    Two invariants hold by construction and are asserted over every shard:
    a molecule cannot satisfy the task without being chemically valid, and it
    cannot contain the fragment without having been produced at all.
    """
    import json

    shard_dir = ROOT / "diagnostics/fragment_official_suite_v2/shards"
    shards = sorted(shard_dir.glob("*.json"))
    assert shards, "no shards to audit"

    checked = 0
    for path in shards:
        shard = json.loads(path.read_text())
        for task, result in shard["results"].items():
            for drug, entries in result["per_drug"].items():
                for row in entries:
                    where = f"{task}/{drug}/seed{row['seed']}"
                    produced = row["committed_endpoints"]
                    valid = row["committed_chemically_valid"]
                    contained = row["committed_fragment_preserving"]
                    task_ok = row["emitted_nonempty"]

                    # Every committed endpoint is a valid connected molecule.
                    assert valid == produced, where
                    # Containment and task success are subsets, in that order.
                    assert contained <= produced, where
                    assert task_ok <= produced, where
                    # The old "validity" column can only ever be <= the
                    # comparator's definition. If these were ever equal-by-
                    # construction the distinction would be vacuous; assert the
                    # ordering, not equality.
                    assert task_ok <= valid, where
                    checked += 1
    assert checked == 90, checked


def test_corrected_table_reports_task_success_separately():
    """The corrected artifact must carry both columns, not one relabelled."""
    import json

    path = ROOT / "diagnostics/fragment_official_suite_v2/corrected_table.json"
    payload = json.loads(path.read_text())
    for task, block in payload["tasks"].items():
        keys = set(block["summary"])
        assert {"chemical_validity", "task_success", "fragment_containment"} <= keys, task
        # Secondary metrics must be flagged wherever the task filter removed a
        # material share of the committed endpoints.
        if block["censoring_pct_of_committed"] > 5.0:
            assert not block["secondary_metrics_valid"], task


# ---- Global sampler knobs ----
#
# A knob that is parsed but never reaches SamplerConfig is INERT, and an inert
# knob is invisible to every check that reads the recorded configuration,
# because the requested value is written into the artifact either way. These
# drive the suite's real parser rather than reconstructing one.


def _suite_module():
    import sys
    from pathlib import Path

    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import run_fragment_constrained_suite as suite

    return suite


def _parsed(*extra: str):
    suite = _suite_module()
    argv = ["--checkpoint", "ckpt.pt", "--output", "out.json", *extra]
    return suite, suite.build_parser().parse_args(argv)


def test_mark_attempts_flag_reaches_the_sampler_configuration():
    suite, args = _parsed("--mark-attempts-per-event", "96")
    assert suite.sampler_config_from_args(args).mark_attempts_per_event == 96


def test_mark_attempts_default_is_the_frozen_value():
    """The default must stay 24 or every landed row silently changes meaning."""
    suite, args = _parsed()
    assert suite.sampler_config_from_args(args).mark_attempts_per_event == 24


def test_the_knob_moves_the_recorded_sampler_identity():
    """Two settings must not be combinable by accident.

    The aggregator refuses shards whose sampler hashes disagree, which only
    protects the panel if the knob is actually part of the hashed payload.
    """
    suite, low = _parsed("--mark-attempts-per-event", "24")
    _, high = _parsed("--mark-attempts-per-event", "96")
    left = suite.frozen_sampler_identity(suite.sampler_config_from_args(low))
    right = suite.frozen_sampler_identity(suite.sampler_config_from_args(high))
    assert left["config_sha256"] != right["config_sha256"]
    assert left["config"]["mark_attempts_per_event"] == 24
    assert right["config"]["mark_attempts_per_event"] == 96


def test_no_per_task_or_per_drug_sampler_knob_is_expressible():
    """The line between a general capability and benchmark engineering.

    A global knob is legitimate; a per-instance one is not. The parser must
    offer no option that scopes a sampler knob to a drug or a task.
    """
    suite = _suite_module()
    options = {
        option
        for action in suite.build_parser()._actions
        for option in action.option_strings
    }
    for forbidden in (
        "--mark-attempts-for-drug",
        "--mark-attempts-for-task",
        "--per-drug-mark-attempts",
        "--per-task-mark-attempts",
        "--max-events-for-task",
    ):
        assert forbidden not in options
    # --drug and --task SELECT which instances to run; they must not carry a value.
    for action in suite.build_parser()._actions:
        if "--drug" in action.option_strings or "--task" in action.option_strings:
            assert action.type is not int, (
                f"{action.option_strings} looks like it carries a numeric knob"
            )


# ---- The two-core knobs: parsed is not consumed ----


def test_the_path_program_flag_reaches_the_controller_configuration():
    suite, args = _parsed("--attachment-control", "--path-program")
    control = suite.AttachmentControlConfig(
        enabled=bool(args.attachment_control), path_program=bool(args.path_program)
    )
    assert control.path_program is True
    _, off = _parsed("--attachment-control")
    assert bool(off.path_program) is False


def test_the_path_program_flag_moves_the_recorded_controller_identity():
    """Two arms that differ in the program must not be combinable by accident.

    The aggregator refuses shards whose controller hashes disagree, which only
    protects the panel if the program is part of the hashed payload.
    """
    suite = _suite_module()
    on = suite.frozen_attachment_identity(
        suite.AttachmentControlConfig(enabled=True, path_program=True)
    )
    off = suite.frozen_attachment_identity(
        suite.AttachmentControlConfig(enabled=True, path_program=False)
    )
    assert on["config_sha256"] != off["config_sha256"]
    assert on["config"]["path_program"] is True
    assert off["config"]["path_program"] is False


def test_the_bridge_knob_reaches_the_prompt_context_call_site():
    """A knob that is parsed and never passed on is INERT.

    This drives the production ``run_task`` and observes the argument arriving
    at the real call site, rather than checking that the parser holds a value.
    A region-law repair on the T4 path was merged, tested and hash-pinned while
    no production caller passed its keyword, so it ran as if it did not exist;
    the tell was available at the CALL site and nowhere else.
    """
    suite = _suite_module()
    seen: dict = {}
    real = suite.build_prompt_context

    def recorder(prompt, **kwargs):
        seen.update(kwargs)
        raise suite.FragmentConditioningError("probe")

    suite.build_prompt_context = recorder
    try:
        suite.run_task(
            None,
            None,
            _prompts(),
            FragmentTask.LINKER_DESIGN,
            seeds=1,
            samples=1,
            config=suite.SamplerConfig(),
            control=suite.AttachmentControlConfig(enabled=True, path_program=True),
            verbose=False,
            seed_list=[0],
            drugs=["BARICITINIB"],
            linker_bridge_atoms=3,
        )
    finally:
        suite.build_prompt_context = real
    assert seen.get("linker_bridge_atoms") == 3, (
        "run_task must pass the seeded bridge through to build_prompt_context"
    )


def test_the_bridge_knob_defaults_to_the_direct_join_at_the_call_site():
    """Default 0 keeps every landed row reproducible.

    Checked where it is CONSUMED, not where it is parsed: a default that stops
    at the parser changes nothing and would leave the landed rows intact by
    accident rather than by contract.
    """
    suite = _suite_module()
    seen: dict = {}
    real = suite.build_prompt_context

    def recorder(prompt, **kwargs):
        seen.update(kwargs)
        raise suite.FragmentConditioningError("probe")

    suite.build_prompt_context = recorder
    try:
        suite.run_task(
            None,
            None,
            _prompts(),
            FragmentTask.LINKER_DESIGN,
            seeds=1,
            samples=1,
            config=suite.SamplerConfig(),
            control=suite.AttachmentControlConfig(),
            verbose=False,
            seed_list=[0],
            drugs=["BARICITINIB"],
        )
    finally:
        suite.build_prompt_context = real
    assert seen.get("linker_bridge_atoms") == 0
    _, args = _parsed()
    assert args.linker_bridge_atoms == 0


def test_the_seeded_bridge_is_recorded_in_the_protocol_block():
    """A realized-length number is unreadable without the length it started from.

    The seeded bridge is a property of the START STATE and belongs to no config
    dataclass, so it appears in no hash; if the artifact does not carry it, a
    reader cannot tell a designed linker from an inherited seed.
    """
    suite, args = _parsed("--linker-bridge-atoms", "2")
    assert suite.protocol_block(args)["linker_bridge_atoms"] == 2
    _, default = _parsed()
    assert suite.protocol_block(default)["linker_bridge_atoms"] == 0


def test_no_per_task_or_per_drug_two_core_knob_is_expressible():
    suite = _suite_module()
    options = {
        option
        for action in suite.build_parser()._actions
        for option in action.option_strings
    }
    for forbidden in (
        "--path-program-for-drug",
        "--path-program-for-task",
        "--linker-bridge-atoms-for-drug",
        "--per-drug-linker-bridge-atoms",
        "--path-length-for-drug",
    ):
        assert forbidden not in options
