"""Production preflight gate (wiring validation on a B-compatible fixture).

The gate must run every check across the chemistry buckets and, on a structurally sound model, return
GO_FOR_FULL_A100: the widened organic heads present; per-bucket rollouts all-intermediate valid + editable +
finite; and the charged bucket editable under the charge-preserving policy (charge-changing marks rejected,
not advanced). This exercises the gate's plumbing before the real checkpoint exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from broad_preflight_gate import _fixture_sampler, check_heads, run_preflight  # noqa: E402


def test_fixture_preflight_returns_go() -> None:
    sampler = _fixture_sampler()
    report = run_preflight(sampler, scope_ok=True, scope_detail="fixture")
    assert report["verdict"] == "GO_FOR_FULL_A100", report["verdict"]
    assert report["heads"]["ok"]
    for name, bucket in report["buckets"].items():
        assert bucket["validity_ok"], (name, bucket)
        assert bucket["editable"], (name, bucket)
        assert bucket["finite"], (name, bucket)


def test_broad_heads_are_15_class() -> None:
    ok, detail = check_heads(_fixture_sampler())
    assert ok, detail
    assert "15-class" in detail


def test_charged_bucket_editable_under_charge_preserving_policy() -> None:
    report = run_preflight(_fixture_sampler(), scope_ok=True, scope_detail="fixture")
    charged = report["buckets"]["charged"]
    # every counted charged intermediate is valid + charge-preserving (charge-changers are rejected, not
    # advanced), and the charged leads still edit (advanced > 0).
    assert charged["valid"] == charged["intermediates"] and charged["intermediates"] > 0
    assert charged["advanced"] > 0
    assert charged["charge_ok"]


def test_scope_mismatch_forces_no_go() -> None:
    # a wrong scope with otherwise-sound buckets must not return GO.
    report = run_preflight(_fixture_sampler(), scope_ok=False, scope_detail="wrong-scope")
    assert report["verdict"] != "GO_FOR_FULL_A100"
