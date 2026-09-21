"""The memory must survive a resume exactly, or arm B silently reverts to arm A."""
import json

from compose_v4.control.pmo_online_memory import OnlineProposalMemory, RegionContext


def _warm() -> OnlineProposalMemory:
    m = OnlineProposalMemory()
    m.frontier.observe("CCO", 0.4); m.frontier.observe("CCN", 0.7)
    m.ordinal = 11
    ctx = RegionContext(size_bucket=2, anchor_element=6, bond_order=1,
                        anchor_in_ring=False, fragment_has_ring=True)
    key = tuple(ctx) if isinstance(ctx, tuple) else (ctx.size_bucket, ctx.anchor_element,
                 ctx.bond_order, ctx.anchor_in_ring, ctx.fragment_has_ring)
    m.edits.totals[key] += 0.35; m.edits.counts[key] += 3
    m.edits.rows.append({"ctx": list(key), "gain": 0.35})
    m.donors.totals[key] += 0.2;  m.donors.counts[key] += 2
    m.donors.rows.append({"endpoint": "CCN", "score": 0.7, "claim": "association_only"})
    m.size.totals[-2] += 0.1; m.size.counts[-2] += 4
    m.cost["syntheses"] = 9; m.cost["region_weight_evaluations"] = 120
    return m

def test_roundtrip_is_exact():
    a = _warm()
    payload = json.loads(json.dumps(a.payload()))   # must survive real JSON
    b = OnlineProposalMemory(); b.restore_payload(payload)
    assert b.payload() == a.payload(), "round trip is not exact"
    assert b.ordinal == 11 and b.cost["syntheses"] == 9
    assert dict(b.edits.counts) == dict(a.edits.counts)
    assert dict(b.size.totals) == dict(a.size.totals)

def test_restore_refuses_a_foreign_schema():
    b = OnlineProposalMemory()
    try:
        b.restore_payload({"schema_version": "something_else"})
    except ValueError:
        return
    raise AssertionError("restore accepted a payload it did not write")

def test_a_dropped_field_is_caught():
    """If a future edit forgets a component, the round trip must stop matching."""
    a = _warm(); payload = a.payload()
    payload["edits"]["counts"] = []          # simulate dropping one component
    b = OnlineProposalMemory(); b.restore_payload(json.loads(json.dumps(payload)))
    assert b.payload() != a.payload(), "dropping a component went undetected"


def test_restore_accepts_the_arm_flag_like_the_constructor():
    """`run_program_campaign` passes optimizer_kwargs to BOTH __init__ and restore().

    A flag accepted by only one of them makes the 250-call path work and the RESUME
    path raise -- which is exactly how this surfaced, on the first extension.
    """
    import inspect

    from compose_v4.control.pmo_population_controller import PmoPopulationController

    init = inspect.signature(PmoPopulationController.__init__).parameters
    restore = inspect.signature(PmoPopulationController.restore).parameters
    for name in ("jump_checkpoint", "enable_online_memory"):
        assert name in init, f"{name} missing from __init__"
        assert name in restore, f"{name} missing from restore -- resume will raise"
