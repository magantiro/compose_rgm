"""Mutation audit: break each resumability invariant and require a NAMED test to go red.

A test suite that has never been seen to fail is a decoration.  Each entry below
disables exactly one production guard, runs the single test that is supposed to
catch it, and requires that test to FAIL.  The audit refuses to report a verdict
when a mutation did not actually apply -- a mutation string that silently fails to
match makes an unmutated file look like a weak guard, which has happened here
before -- and it carries a COSMETIC positive control that changes bytes and nothing
else, because "everything refused" and "the harness is broken" are otherwise
indistinguishable.

Run:  PYTHONPATH=src python tests/resumability_mutation_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parent
_MODULE = _REPO / "src" / "compose_v4" / "control" / "durable_resume.py"
_WORKER = _TESTS / "resumability_kill_worker.py"
_SUITE = _TESTS / "test_pmo_resumability.py"

# (name, file, old, new, test that must go red, expectation)
_MUTATIONS = [
    (
        "drop_rng_from_the_snapshot",
        _MODULE,
        '        "rng",\n',
        "",
        "test_dropping_the_rng_from_the_snapshot_is_refused",
        "red",
    ),
    (
        "skip_the_memory_payload",
        _MODULE,
        '        "pmo_population.online_memory",\n',
        "",
        "test_skipping_the_memory_payload_is_refused",
        "red",
    ),
    (
        # Sequence addressing is the naive design this module rejects: it appends a
        # new record per reservation, so a replayed round charges twice.
        "let_a_reserved_call_be_re_spent_on_resume",
        _MODULE,
        '        slot = self._slot(round_index, ordinal)\n',
        ('        slot = self._slot(round_index, ordinal)'
        ' + "_%d" % len(list((self.root / "calls").glob("round_*")))\n'),
        "test_a_reserved_call_is_never_re_spent_on_resume",
        "red",
    ),
    (
        "stop_detecting_a_diverged_proposal",
        _MODULE,
        "        if reserved_path.exists():\n            existing = read_json(reserved_path)",
        "        if False:\n            existing = read_json(reserved_path)",
        "test_a_resumed_run_that_proposes_a_different_molecule_diverges_loudly",
        "red",
    ),
    (
        "let_the_lease_never_expire",
        _MODULE,
        '            if age <= float(prior["lease_seconds"]):',
        "            if True:",
        "test_an_expired_lease_allows_another_worker_to_take_the_run_over",
        "red",
    ),
    (
        "let_a_lease_expire_while_the_worker_is_alive_unfenced",
        _MODULE,
        '        if not self.path.exists():\n            raise LeaseHeld("lease record vanished; this worker no longer owns the run")',
        "        if True:\n            return",
        "test_a_worker_whose_lease_lapsed_while_it_was_alive_is_fenced_out_of_commits",
        "red",
    ),
    (
        "fill_in_a_missing_component_at_load_instead_of_refusing",
        _MODULE,
        ('        missing = sorted(\n'
         '            name\n'
         '            for name in self.required_components\n'
         '            if resolve_component(stored, name) is _MISSING\n'
         '        )'),
        "        missing = []",
        "test_a_stored_snapshot_missing_a_component_is_refused_at_load_not_filled_in",
        "red",
    ),
    (
        "stop_verifying_the_snapshot_content_hash",
        _MODULE,
        '        if identity(body) != record.get("snapshot_sha256"):',
        "        if False:",
        "test_a_tampered_snapshot_fails_its_content_hash",
        "red",
    ),
    (
        "stop_charging_a_reservation_whose_worker_died_before_observing",
        _MODULE,
        "        for reservation in self.reconcile().pending:",
        "        for reservation in []:",
        "test_a_reservation_made_before_the_oracle_ran_stays_charged_and_is_repaired_once",
        "red",
    ),
    # The acceptance test itself must be able to fail: a resumed worker that does
    # NOT restore its random stream proposes different molecules, which the ledger
    # must catch as divergence rather than quietly producing a different run.
    (
        "resume_without_restoring_the_rng_stream",
        _WORKER,
        '        state = json.loads(json.dumps(latest["components"]))\n',
        ('        state = json.loads(json.dumps(latest["components"]))\n'
         '        state["rng"] = _initial_state(args.seed)["rng"]\n'),
        "test_kill_and_resume_reproduces_the_uninterrupted_run",
        "red",
    ),
    # POSITIVE CONTROL: changes bytes, changes no behaviour.  Must stay green, or
    # every "refused" above is the harness failing rather than the guard working.
    (
        "cosmetic_control_comment_only",
        _MODULE,
        "# ---- Errors ---",
        "# a cosmetic comment that changes nothing\n# ---- Errors ---",
        "test_dropping_the_rng_from_the_snapshot_is_refused",
        "green",
    ),
]


def _run(test_name: str) -> bool:
    """True when the named test PASSES."""
    environment = {
        **os.environ,
        "PYTHONPATH": str(_REPO / "src"),
        "COMPOSE_ALLOW_REAPABLE_PATH": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", f"{_SUITE}::{test_name}", "-q", "-p", "no:randomly"],
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
        cwd=str(_REPO),
        check=False,
    )
    return completed.returncode == 0


def main() -> int:
    baseline_failures = []
    results = []
    for name, path, old, new, test_name, expectation in _MUTATIONS:
        original = path.read_text()
        if old not in original:
            results.append(
                {"mutation": name, "verdict": "NOT_APPLIED", "detail": "anchor text not found"}
            )
            continue
        mutated = original.replace(old, new, 1)
        if mutated == original:
            results.append({"mutation": name, "verdict": "NOT_APPLIED", "detail": "no byte change"})
            continue
        try:
            path.write_text(mutated)
            assert path.read_text() != original, "mutation did not reach disk"
            passed = _run(test_name)
        finally:
            path.write_text(original)
            assert path.read_text() == original, "FAILED TO RESTORE SOURCE"
        if expectation == "red":
            verdict = "CAUGHT" if not passed else "SURVIVED"
        else:
            verdict = "CONTROL_OK" if passed else "CONTROL_BROKEN"
        results.append(
            {"mutation": name, "test": test_name, "expected": expectation, "verdict": verdict}
        )

    report = {
        "schema_version": "resumability_mutation_audit_v1",
        "mutations": len(_MUTATIONS),
        "caught": sum(r["verdict"] == "CAUGHT" for r in results),
        "survived": sum(r["verdict"] == "SURVIVED" for r in results),
        "not_applied": sum(r["verdict"] == "NOT_APPLIED" for r in results),
        "control_ok": sum(r["verdict"] == "CONTROL_OK" for r in results),
        "results": results,
        "baseline_failures": baseline_failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    clean = (
        report["survived"] == 0
        and report["not_applied"] == 0
        and report["control_ok"] == sum(m[5] == "green" for m in _MUTATIONS)
    )
    print("\nMUTATION AUDIT:", "PASS" if clean else "FAIL")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
