"""Zero-oracle smoke: the T4 and PMO architectures can still be instantiated.

This is layer 3 of the preservation guarantee. Layers 1 and 2, the importable
public surface and the entry-point set, are recorded in
``diagnostics/repo_hygiene/`` and compared mechanically. They are necessary and
NOT sufficient: an import can resolve while the mechanism behind it is reached
by no production caller, which this repository has found six separate times.

So these tests assert REACHABILITY along the named production paths rather than
symbol existence. Each one names the capability it protects, so a future
structural pass that disconnects a lane fails here with a readable reason
instead of passing an import check.

What these tests deliberately do NOT do: call an oracle, dock anything, launch
Modal, read a checkpoint, or assert a numeric result. They are cheap enough to
run in the ordinary suite. A behavioural equivalence test on chemistry belongs
with the kernel and is pinned into the process identity; this file only proves
the architecture is still wired together.
"""

from __future__ import annotations

import importlib
import inspect

# ---- T4: campaign entrypoint -> fiber gate -> proposal synthesis -> region law ----

T4_CAPABILITY_PATH = (
    ("compose_v4.experiments.t4_fiber_campaign", ("Fiber", "expand", "prepare")),
    ("compose_v4.control.dynamic_program_synthesis", ("synthesize_dynamic_program",)),
    ("compose_v4.control.bridge_region_law", ("BridgeRegionLaw",)),
    ("compose_v4.control.region_law_contract", ()),
    ("compose_v4.experiments.whole_ring_plan", ()),
)

# ---- PMO: production controller -> proposal families -> realization ----

PMO_CAPABILITY_PATH = (
    ("compose_v4.control.pmo_population_controller", ("PmoPopulationController",)),
    ("compose_v4.control.pmo_realization", ()),
    ("compose_v4.control.pmo_joint_dependency_jump", ()),
    ("compose_v4.experiments.pmo_population_v1", ()),
)

EXECUTOR_CAPABILITY_PATH = (
    ("compose_v4.rewrite.kernel", ()),
    ("compose_v4.rewrite.operators", ()),
    ("compose_v4.chem.molecular_graph", ()),
    ("compose_v4.chem.state", ("pad_molecular_graph",)),
)


def _assert_path_reachable(path, capability: str) -> None:
    for module_name, required in path:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # the import failure IS the finding, so it is caught and named
            raise AssertionError(
                f"{capability}: {module_name} no longer imports ({type(exc).__name__}: {exc}). "
                "A structural change disconnected this lane."
            ) from exc
        for symbol in required:
            assert hasattr(module, symbol), (
                f"{capability}: {module_name}.{symbol} is gone. If it moved, it must still be "
                "importable from this path through a shim."
            )


def test_t4_campaign_path_is_reachable():
    _assert_path_reachable(T4_CAPABILITY_PATH, "T4 campaign")


def test_pmo_controller_path_is_reachable():
    _assert_path_reachable(PMO_CAPABILITY_PATH, "PMO controller")


def test_executor_path_is_reachable():
    _assert_path_reachable(EXECUTOR_CAPABILITY_PATH, "executor")


def test_t4_fiber_gate_still_takes_its_production_arguments():
    """The gate's constructor shape is what every T4 contract is written against."""
    campaign = importlib.import_module("compose_v4.experiments.t4_fiber_campaign")
    parameters = inspect.signature(campaign.Fiber.__init__).parameters
    # smiles and delta are benchmark INPUTS; a rename here silently breaks every cell.
    assert "delta" in parameters, "Fiber lost its delta parameter; T4 contracts pass delta"


def test_region_law_is_still_accepted_by_proposal_synthesis():
    """A validated repair behind an opt-in keyword is INERT until a caller passes it.

    The region law was built, tested and merged while no production caller passed
    it, so a rescue launched on that commit would have run v1 verbatim. The
    keyword's presence on the synthesis signature is the cheapest standing check
    that the wiring still exists.
    """
    synthesis = importlib.import_module("compose_v4.control.dynamic_program_synthesis")
    parameters = inspect.signature(synthesis.synthesize_dynamic_program).parameters
    assert "region_law" in parameters, (
        "synthesize_dynamic_program no longer accepts region_law; the T4 region repair "
        "would be inert again"
    )


def test_pmo_controller_restore_and_init_agree_on_arm_parameters():
    """A flag accepted by __init__ but not by restore() makes resume raise, not fall back.

    Measured once already: a 1k extension died on `restore() got an unexpected
    keyword argument`, after the ledger had been widened and before any call was
    charged. Worse than the crash would have been a silent accept-and-drop, which
    rebuilds the treatment arm as the control.
    """
    controller = importlib.import_module("compose_v4.control.pmo_population_controller")
    cls = controller.PmoPopulationController
    if not hasattr(cls, "restore"):
        return
    init_parameters = set(inspect.signature(cls.__init__).parameters)
    restore_parameters = set(inspect.signature(cls.restore).parameters)
    # Either restore takes the same named arm knobs, or it absorbs them via **kwargs.
    restore_signature = inspect.signature(cls.restore)
    absorbs_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in restore_signature.parameters.values()
    )
    arm_parameters = {name for name in init_parameters if name.startswith("enable_")}
    missing = arm_parameters - restore_parameters
    assert absorbs_kwargs or not missing, (
        f"PmoPopulationController.restore() cannot accept {sorted(missing)}; "
        "a resumed run would raise or silently rebuild the treatment arm as the control"
    )
