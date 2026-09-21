"""Seal the QED / GrIDDD-Jin controller-comparison contract, capsule and receipt.

NEW FILE, modelled on ``tools/prepare_pmo_250_pilot.py`` and
``tools/launch_pmo_population_v1_corrected.py``: the binding chain is
contract -> payload hash -> source capsule -> authorization receipt, and every
link is recomputed here so no stale pointer can survive.

FAIL CLOSED, ALWAYS
-------------------
The sealed contract carries ``scored_launch_authorized: false``,
``modal_launch_authorized: false`` and ``oracle_calls_authorized: 0``.  This tool
cannot write any other value: the three flags are constants in the payload and
``verify`` re-reads the written file and refuses it unless all three still hold.
Arm B needs no authorization because it charges no paid oracle -- QED and
Tanimoto are local RDKit calls -- and that is recorded as the reason rather than
left implicit.

THE CAPSULE IS DERIVED BY EXECUTION, NOT BY READING IMPORTS
-----------------------------------------------------------
A hand-written file list is wrong in both directions: it names modules that are
not on the path and misses modules reached only through a lazy import inside a
function.  This tool runs one real source at the smallest legal budget and then
hashes every ``compose_v4`` module present in ``sys.modules`` afterwards, which
is the closure the run actually touched.

CONTRACT-AUTHORITY
------------------
Each declared field carries its runtime consumption site, so a reviewer can check
it is operative rather than prose.  Fields that are NOT operative on the paper-era
arm are recorded under ``decorative_on_arm_a`` instead of being quietly dropped:
that arm reads its thresholds from module constants, so a contract cannot move
them without editing frozen bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / "configs/qed_griddd_controller_comparison_v1.json"
CAPSULE_PATH = REPO / "diagnostics/qed_griddd_controller_comparison_v1/source_capsule_manifest.json"
RECEIPT_PATH = REPO / "diagnostics/qed_griddd_controller_comparison_v1/authorization_receipt.json"

PANEL = REPO / "data/jin/qed_test.txt"
PANEL_SHA256 = "704103777e8050eb59f4d15d9997ca6070ba05a6b878b8e18738bfb1e706a090"

TOOLS = (
    "tools/run_qed_griddd_controller_comparison_v1.py",
    "tools/score_qed_griddd_arm_a_v1.py",
    "tools/prepare_qed_griddd_controller_comparison_v1.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def execution_closure() -> list[str]:
    """Every ``compose_v4`` module a real arm-B source touches, by execution."""

    sys.path.insert(0, str(REPO / "src"))
    import tempfile

    from tools.run_qed_griddd_controller_comparison_v1 import run_one

    with tempfile.TemporaryDirectory() as tmp:
        source = next(ln.strip() for ln in PANEL.read_text().split("\n") if ln.strip())
        record = run_one(
            {
                "index": 0,
                "source": source,
                "out_root": tmp,
                "budget": 2,
                "rounds": 1,
                "queries_per_round": 1,
                "attempts_per_batch": 8,
                "wall_seconds": 5.0,
            }
        )
        if record.get("status") != "complete":
            raise SystemExit(f"closure probe did not complete: {record.get('error')}")
    modules = set()
    for name, module in list(sys.modules.items()):
        if not name.startswith("compose_v4"):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        try:
            modules.add(str(path.relative_to(REPO)))
        except ValueError:
            continue
    return sorted(modules)


def build_payload(files: dict[str, str], run: dict) -> dict:
    return {
        "schema_version": "qed_griddd_controller_comparison_v1",
        "question": (
            "On the 800 Jin ICLR-2019 QED sources, at the same number of returned "
            "candidates, does the current structural-program + FiberControl controller "
            "match the paper-era region-h_phi + twisted-SMC controller?"
        ),
        # ---- Fail closed. These three are constants, never arguments. ----
        "scored_launch_authorized": False,
        "modal_launch_authorized": False,
        "oracle_calls_authorized": 0,
        "no_paid_oracle_reason": (
            "QED and Morgan-Tanimoto are local RDKit calls. Arm B charges a query "
            "ledger for accounting only; no external or paid oracle is contacted. "
            "Arm A is not re-run: its per-source records are read-only."
        ),
        "panel": {
            "name": "jin_iclr19_qed_test_800",
            "path": "data/jin/qed_test.txt",
            "sha256": PANEL_SHA256,
            "count": 800,
            "provenance": (
                "wengong-jin/iclr19-graph2graph @ e02c14ae, data/qed/test.txt; the exact "
                "public artifact, matching griddd_qed_lead_manifest_v1.jin_iclr19_exact"
            ),
            "max_heavy_atoms": 33,
        },
        "task": {
            "success_event": "QED(y) >= 0.90 AND Tanimoto(y, x0) >= 0.40",
            "conjunction_is_not_blended": True,
            "qed": "rdkit.Chem.QED.qed",
            "similarity": {
                "metric": "Tanimoto",
                "fingerprint": "Morgan",
                "radius": 2,
                "bits": 2048,
                "use_chirality": False,
                "reference": "always the ORIGINAL source, never a branch point",
            },
            "returned_candidates_per_source": run["k"],
            "no_sa_constraint": (
                "the GrIDDD/Jin QED task has no synthetic-accessibility floor; the T4 "
                "endpoint gate's sa_max=4.0 must not be imported into it"
            ),
        },
        "arm_a_paper_era": {
            "controller": "region h_phi + twisted SMC over the exact legal fiber",
            "status": "banked, read-only, NOT re-run",
            "records": "modal volume compose-v4-artifacts: editing_v2/r_theta_run/hphi_official800_k8",
            "scored_sources": 798,
            "unscored_sources": [135, 408],
            "rederivation_tool": "tools/score_qed_griddd_arm_a_v1.py",
            "rederivation_artifact": "diagnostics/qed_griddd_arm_a_rederivation_v1.json",
        },
        "arm_b_current": {
            "controller": "structural program synthesis + FiberControl allocation",
            "entrypoint": "compose_v4.control.program_campaign.run_program_campaign",
            "objective_injection": "ProgramQueryLedger(evaluate=...) -- a free local scorer",
            "score": "QED(y) * 1[Tanimoto(y, x0) >= 0.40]",
            "task_kind": "pmo",
            "task_kind_reason": (
                "kind='t4' would apply ProgramTask.endpoint_evaluator's hardcoded "
                "qed_min=0.6 and sa_max=4.0, which are not part of this benchmark"
            ),
            "canonical_slots": 48,
            "canonical_slots_reason": (
                "whole_ring_plan.execute_program requires an exact supported 48-slot "
                "source; a TIGHT graph from smiles_to_molecular_graph removes the whole "
                "atom_insert family from the legal support"
            ),
            **run,
        },
        "contract_authority": {
            "operative": {
                "similarity_threshold_0.40": "arm B: the injected evaluate() closure",
                "qed_threshold_0.90": "arm B: run_one success derivation and the scorer",
                "returned_candidates_k": "arm A: task dict k_start/k_end; arm B: post-hoc over the ledger",
                "charged_budget": (
                    "ProgramQueryLedger.__init__ -- required keyword, type-checked, "
                    "no default (program_campaign.py:29)"
                ),
                "seed": (
                    "ProgramSearchConfig.seed -> np.random.default_rng "
                    "(adaptive_program_optimizer.py:222)"
                ),
                "rounds / queries_per_round": "run_program_campaign validation (program_campaign.py:153)",
                "wall_seconds": "adaptive_program_optimizer.py:860",
                "attempts_per_batch": "adaptive_program_optimizer.py:857",
                "candidates_per_batch": "adaptive_program_optimizer.py:859",
                "parent_allocation": "adaptive_program_optimizer.py:329-331",
                "score_direction": (
                    "adaptive_program_optimizer.py:326 AND parent_edit_search.py:23, which "
                    "FORCES 'minimize' for kind='t4' and 'maximize' for kind='pmo'"
                ),
            },
            "decorative_on_arm_a": {
                "qed_threshold": "modal_apps/hphi_h40head_ab_app.py:78 REGION = (0.90, 0.40) -- module constant",
                "similarity_threshold": "same module constant",
                "particles": "N_PARTICLES = 32 -- module constant, not task-configurable",
                "canonical_slots": "CANONICAL_SLOTS = 48 -- module constant",
                "time_point": "TIME_POINT = 0.5 -- module constant",
                "note": (
                    "a contract field naming any of these would be prose: the arm reads "
                    "them from module scope and no task dict can move them"
                ),
            },
            "silent_default_hazards": {
                "budget_max": "task.get('budget_max', 24) -- a caller that omits it inherits 24",
                "horizon": "task.get('horizon', HORIZON=24) -- official tasks pass 40",
                "qed_task_success_threshold": (
                    "griddd_jin_benchmark.qed_task_success(threshold=0.4) -- a default "
                    "argument, and the 0.9 QED band is a hardcoded literal in its body"
                ),
            },
        },
        "implementation_sha256": files,
    }


def seal() -> dict:
    files = {rel: digest(REPO / rel) for rel in TOOLS}
    files[str(PANEL.relative_to(REPO))] = digest(PANEL)
    for rel in execution_closure():
        files[rel] = digest(REPO / rel)
    run = {
        "k": 8,
        "budget": 48,
        "rounds": 5,
        "queries_per_round": 8,
        "attempts_per_batch": 64,
        "wall_seconds": 20.0,
        "seed_rule": "uint64(sha256('qed-griddd-controller-comparison-v1|<source>|<index>')[:8])",
    }
    payload = build_payload(files, run)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(envelope, indent=1, sort_keys=True))

    capsule = {
        "schema_version": "qed_griddd_controller_comparison_capsule_v1",
        "contract": str(CONTRACT_PATH.relative_to(REPO)),
        "contract_payload_sha256": envelope["payload_sha256"],
        "closure_derivation": "executed one real source and hashed every compose_v4 module reached",
        "file_count": len(files),
        "files": files,
    }
    CAPSULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPSULE_PATH.write_text(json.dumps(capsule, indent=1, sort_keys=True))

    receipt_body = {
        "schema_version": "qed_griddd_controller_comparison_receipt_v1",
        "contract_payload_sha256": envelope["payload_sha256"],
        "capsule_sha256": identity(capsule),
        "scored_launch_authorized": False,
        "modal_launch_authorized": False,
        "oracle_calls_authorized": 0,
        "statement": (
            "Preparation and local free-scorer execution only. This receipt authorizes "
            "no scored launch, no Modal operation and no paid oracle call."
        ),
    }
    receipt = {**receipt_body, "receipt_sha256": identity(receipt_body)}
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return envelope


def verify() -> None:
    """Re-read what was written; refuse anything that is not fail-closed."""

    envelope = json.loads(CONTRACT_PATH.read_text())
    payload = envelope["payload"]
    if envelope["payload_sha256"] != identity(payload):
        raise SystemExit("contract envelope hash does not match its payload")
    for flag, expected in (
        ("scored_launch_authorized", False),
        ("modal_launch_authorized", False),
        ("oracle_calls_authorized", 0),
    ):
        if payload.get(flag) != expected:
            raise SystemExit(f"contract is not fail-closed: {flag} = {payload.get(flag)!r}")
    if payload["panel"]["sha256"] != digest(PANEL):
        raise SystemExit("panel file does not match its pinned digest")
    capsule = json.loads(CAPSULE_PATH.read_text())
    if capsule["contract_payload_sha256"] != envelope["payload_sha256"]:
        raise SystemExit("capsule does not pin the sealed contract")
    drifted = [rel for rel, want in capsule["files"].items() if digest(REPO / rel) != want]
    if drifted:
        raise SystemExit(f"capsule drift in {len(drifted)} files: {drifted[:5]}")
    receipt = json.loads(RECEIPT_PATH.read_text())
    if receipt["capsule_sha256"] != identity(capsule):
        raise SystemExit("receipt does not pin the capsule")
    if receipt["oracle_calls_authorized"] != 0:
        raise SystemExit("receipt authorizes oracle calls")
    print(
        f"VERIFIED fail-closed  contract={envelope['payload_sha256'][:16]}"
        f"  capsule_files={capsule['file_count']}"
        f"  receipt={receipt['receipt_sha256'][:16]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        envelope = seal()
        print(f"sealed contract payload {envelope['payload_sha256']}")
    verify()


if __name__ == "__main__":
    main()
