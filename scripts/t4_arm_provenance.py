"""Reconstruct, per published T4 cell, the EXACT historical method that produced it.

Motivation
----------
``diagnostics/T4_FROZEN_RESULT_v1.json`` reports 30 (target, seed, delta) cells with a
two-valued ``source`` label (``panel`` / ``support_expansion``).  That label is far coarser
than the executable configuration: the 30 rows were produced by **15 registered arms** under
**three different proposal operators**, spanning **more than one contract identity per arm**
and **six distinct code revisions**.  Cloning the method for an independent replicate from
today's defaults would silently test a different method.

This module walks the chain

    frozen table row  ->  registered arm  ->  contract payload identity  ->  code revision
                      ->  app module + wrapper  ->  controller family  ->  support operator

and emits ``diagnostics/t4_replication_manifest/provenance_v1.json``.

Everything the script can verify, it verifies live: contract payload hashes are recomputed
canonically and compared with the declared field, every ``runtime_inputs_sha256`` pin is
compared against the bytes in the tree, launch receipts are scanned from disk, and contract
identity chains are read out of ``git log``.  Facts that could only be established by
archaeology -- which arm owns which published row -- are carried as data with their evidence
recorded beside them, never inferred from a filename.

Invariants this module exists to protect
----------------------------------------
* ``delta`` is read from the CONTRACT, never from an arm or volume name.  The repo has
  shipped a contract named ``..._jak2_d06_250.json`` whose executable ``delta`` is 0.4.
* The protonation operator and the region-repair operator are DIFFERENT and must never be
  merged under one label.  ``support_operator`` distinguishes them by executable field.
* A realized charged-call count is a RESULT.  A replicate clones the CEILING and the
  STOPPING RULE, never the realized count.

Read-only.  No network, no Modal, no oracle calls.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = "diagnostics/T4_FROZEN_RESULT_v1.json"
OUT = "diagnostics/t4_replication_manifest/provenance_v1.json"
# Written by a SEPARATE agent from the Modal volumes; read-only cross-reference here.  Its
# receipts come from the volumes, whereas ``_receipts`` below scans only the in-repo
# ``diagnostics/**/launches/`` directories, so the two sources are genuinely independent and
# a disagreement between them is a real finding rather than a duplicate reading.
CENSUS = "diagnostics/t4_replication_manifest/run_census_v1.json"

# ---- Registered arms -------------------------------------------------------------------
# Mirrors tools/t4_reconcile_ledger.py ARMS (volume, target, contract).  ``wrapper`` is the
# thin Modal launcher that selects the contract/volume/output namespace; note it is NOT a
# member of any contract's runtime_inputs_sha256, which is recorded as a pin gap below.

ARMS: dict[str, dict] = {
    "parp1_d04": {"volume": "compose-t4-held-target-distilled-parp1-d04-250",
                  "contract": "configs/t4_held_target_distilled_parp1_d04_250.json",
                  "wrapper": "modal_apps/t4_integrated_route_fiber_held_parp1_d04_app.py"},
    "parp1_d06": {"volume": "compose-t4-held-target-distilled-parp1-d06-250",
                  "contract": "configs/t4_held_target_distilled_parp1_d06_250.json",
                  "wrapper": "modal_apps/t4_integrated_route_fiber_held_parp1_app.py"},
    "braf_d04": {"volume": "compose-t4-held-target-distilled-braf-d04-250",
                 "contract": "configs/t4_held_target_distilled_braf_d04_250.json",
                 "wrapper": "modal_apps/t4_integrated_route_fiber_held_braf_d04_app.py"},
    "braf_d06": {"volume": "compose-t4-held-target-distilled-braf-d06-250",
                 "contract": "configs/t4_held_target_distilled_braf_d06_250.json",
                 "wrapper": "modal_apps/t4_integrated_route_fiber_held_braf_app.py"},
    "fa7_d04": {"volume": "compose-t4-held-target-distilled-fa7-d04-250",
                "contract": "configs/t4_held_target_distilled_fa7_d04_250.json",
                "wrapper": "modal_apps/t4_integrated_route_fiber_held_fa7_d04_app.py"},
    "fa7_d06": {"volume": "compose-t4-held-target-distilled-fa7-d06-250",
                "contract": "configs/t4_held_target_distilled_fa7_d06_250.json",
                "wrapper": "modal_apps/t4_integrated_route_fiber_held_fa7_app.py"},
    "5ht1b_d04": {"volume": "compose-t4-held-target-distilled-5ht1b-d04-250",
                  "contract": "configs/t4_held_target_distilled_5ht1b_d04_250.json",
                  "wrapper": "modal_apps/t4_integrated_route_fiber_held_5ht1b_d04_app.py"},
    "5ht1b_d06": {"volume": "compose-t4-held-target-distilled-5ht1b-d06-250",
                  "contract": "configs/t4_held_target_distilled_5ht1b_d06_250.json",
                  "wrapper": "modal_apps/t4_integrated_route_fiber_held_5ht1b_app.py"},
    "jak2_named_d06": {"volume": "compose-t4-held-target-distilled-jak2-d06-250",
                       "contract": "configs/t4_held_target_distilled_jak2_d06_250.json",
                       "wrapper": "modal_apps/t4_integrated_route_fiber_held_jak2_app.py"},
    "jak2_true_d06": {"volume": "compose-t4-held-target-jak2-true-d06-250",
                      "contract": "configs/t4_held_target_distilled_jak2_true_d06_250.json",
                      "wrapper": "modal_apps/t4_integrated_route_fiber_held_jak2_d06_app.py"},
    "braf_d06_region_repair": {"volume": "compose-t4-region-repair-rescue-braf-d06",
                               "contract": "configs/t4_region_repair_rescue_braf_d06_v1.json",
                               "wrapper": "modal_apps/t4_region_repair_rescue_braf_d06_app.py"},
    "fa7_d06_region_repair": {"volume": "compose-t4-region-repair-rescue-fa7-d06",
                              "contract": "configs/t4_region_repair_rescue_fa7_d06_v1.json",
                              "wrapper": "modal_apps/t4_region_repair_rescue_fa7_d06_app.py"},
    "fa7_d04_region_repair": {"volume": "compose-t4-region-repair-rescue-fa7-d04",
                              "contract": "configs/t4_region_repair_rescue_fa7_d04_v1.json",
                              "wrapper": "modal_apps/t4_region_repair_rescue_fa7_d04_app.py"},
    "5ht1b_d04_protonation": {"volume": "compose-t4-5ht1b2-protonation-rescue-d04",
                              "contract": "configs/t4_5ht1b2_protonation_rescue_d04_v1.json",
                              "wrapper": "modal_apps/t4_5ht1b2_protonation_rescue_d04_app.py"},
    "5ht1b_d06_protonation": {"volume": "compose-t4-5ht1b2-protonation-rescue-d06",
                              "contract": "configs/t4_5ht1b2_protonation_rescue_d06_v1.json",
                              "wrapper": "modal_apps/t4_5ht1b2_protonation_rescue_d06_app.py"},
}

# ---- Row ownership ---------------------------------------------------------------------
# (target, seed, delta) -> arm.  Seeds are 1-based in the frozen table and map to 0-based
# cell names (seed 1 -> <protein>_0).  Every assignment is justified by the arm contract's
# own ``cells`` list plus its executable ``delta``; ``_check_ownership`` re-derives it and
# raises rather than letting a stale mapping through.

OWNER: dict[tuple[str, int, str], str] = {
    **{("PARP1", s, "0.4"): "parp1_d04" for s in (1, 2, 3)},
    **{("PARP1", s, "0.6"): "parp1_d06" for s in (1, 2, 3)},
    **{("BRAF", s, "0.4"): "braf_d04" for s in (1, 2, 3)},
    ("BRAF", 1, "0.6"): "braf_d06_region_repair",
    ("BRAF", 2, "0.6"): "braf_d06_region_repair",
    ("BRAF", 3, "0.6"): "braf_d06",
    ("FA7", 1, "0.4"): "fa7_d04",
    ("FA7", 2, "0.4"): "fa7_d04",
    ("FA7", 3, "0.4"): "fa7_d04_region_repair",
    # FA7 seed 1 at delta 0.6 produced NO value (scoped negative, compose=None). The frozen
    # table labels it source='panel', but no arm produced a number, so "which arm produced the
    # reported value" has no answer. Both arms that ATTEMPTED it are recorded in BLANK_CELL_
    # ATTEMPTS; the owner here is the last arm to hold the cell.
    ("FA7", 1, "0.6"): "fa7_d06_region_repair",
    ("FA7", 2, "0.6"): "fa7_d06",
    ("FA7", 3, "0.6"): "fa7_d06_region_repair",
    ("5HT1B", 1, "0.4"): "5ht1b_d04",
    ("5HT1B", 2, "0.4"): "5ht1b_d04",
    ("5HT1B", 3, "0.4"): "5ht1b_d04_protonation",
    ("5HT1B", 1, "0.6"): "5ht1b_d06",
    ("5HT1B", 2, "0.6"): "5ht1b_d06",
    ("5HT1B", 3, "0.6"): "5ht1b_d06_protonation",
    # NOTE: there is NO jak2 delta=0.4 contract.  The d0.4 JAK2 rows come from the arm whose
    # files and volume are named "..._jak2_d06_250" but whose executable delta is 0.4.  The
    # d0.6 rows come from the separately sealed "jak2_true" arm.  Verified by hash, not name.
    **{("JAK2", s, "0.4"): "jak2_named_d06" for s in (1, 2, 3)},
    **{("JAK2", s, "0.6"): "jak2_true_d06" for s in (1, 2, 3)},
}

BLANK_CELL_ATTEMPTS = {
    ("FA7", 1, "0.6"): {
        "produced_a_value": False,
        "frozen_source_label": "panel",
        "arms_that_attempted_this_cell": ["fa7_d06", "fa7_d06_region_repair"],
        "fa7_d06_panel": ("charged 2 calls across prior attempts and terminated at "
                          "candidate_exhaustion; recorded as prior_charged_calls fa7_0: 2 in the "
                          "region-repair contract's ceiling_derivation"),
        "fa7_d06_region_repair": ("held the cell with a 248-call ceiling and produced no "
                                  "dockable eligible endpoint; docking calls were declined"),
        "caution": ("The table's source='panel' on this row is NOT an attribution of a produced "
                    "value -- there is none. A replicate must decide deliberately which arm it "
                    "is replicating for this cell."),
    },
}

PROTEIN_CELL = {"PARP1": "parp1", "BRAF": "braf", "FA7": "fa7", "5HT1B": "5ht1b",
                "JAK2": "jak2"}

# Seed derivation, read from the app source that ran -- not retyped from memory.  The same
# two lines appear in all three app families (panel, region-repair, protonation).
SEED_MODEL = {
    "controller_stream": "rng = numpy.random.default_rng(cell['controller_seed'])",
    "proposal_seed_formula": ("proposal_seed = cell['controller_seed'] + 1_000_003 * round_index"
                              " + 10_007 * parent_index + 101 * expert_index"),
    "docking_seed": "contract['docking_seed'], passed to qvina02 by t4_docking_adapter.dock_t4",
    "verified_identical_in": [
        "modal_apps/t4_integrated_route_fiber_parp1_app.py @8ab529cc (panel, lines 386/471-475)",
        "modal_apps/t4_integrated_route_fiber_parp1_app.py @c639437c (region repair, 422/509-511)",
        "modal_apps/t4_5ht1b2_protonation_rescue_d06_app.py @e86a2181 (protonation, 731/807-811)",
    ],
    "places_a_seed_enters": [
        "cell.controller_seed -> the per-cell controller RNG (parent selection, exploration)",
        "cell.controller_seed -> every proposal_seed via the formula above",
        "contract.docking_seed -> qvina02",
        ("RESUMED RUNS restore rng_state from the checkpoint, so a resumed controller stream "
         "does NOT restart from controller_seed -- a replicate that only sets the seed will "
         "not reproduce a resumed trajectory"),
    ],
    "conformer_generation_is_UNSEEDED": (
        "obabel --gen3D takes no seed; the frozen result's docking_reproducibility_caveat "
        "measures a 1.3 kcal/mol spread on one molecule across three runs. Cloning every "
        "seed here does NOT make a per-row docking value reproducible."
    ),
}

STOPPING_RULE = {
    "terminal_statuses": {
        "candidate_exhaustion": ("published when the round's selected batch is empty "
                                 "(`if not selected:`) -- the search stops with budget left"),
        "complete_budget": "published when the round loop exits having consumed the budget",
        "root_oracle_failure": "published when the single root docking returns score None",
    },
    "budget_mechanics": ("SearchState.budget starts at contract['charged_calls_per_cell']; the "
                         "root docking charges 1 before the loop; each round charges len(observed)"),
    "source": "modal_apps/t4_integrated_route_fiber_parp1_app.py (panel/region-repair) and "
              "modal_apps/t4_5ht1b2_protonation_rescue_d0{4,6}_app.py (protonation)",
    "replicate_rule": ("clone the CEILING and these terminal conditions. Never clone a realized "
                       "charged-call count -- it is an outcome of the stopping rule, not an input."),
}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonical(obj) -> str:
    return _sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def _git(*args: str) -> str | None:
    done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=False)
    return done.stdout if done.returncode == 0 else None


def _flat(obj, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


EXEC_PREFIXES = (
    "/proposal", "/delta", "/batch", "/parents", "/exploration", "/expert_floor_rounds",
    "/parent_explore", "/value_penalty", "/support", "/charged_calls_per_cell",
    "/total_charged_call_ceiling", "/total_new_charged_call_ceiling", "/docking_seed",
    "/docking_box", "/cells", "/route_scale_floor_rounds", "/cold_start_floor_rounds",
    "/evaluator_sha256", "/runtime_inputs_sha256", "/allocation_policy", "/endpoint_gate",
)


def _load_contract(rel: str, rev: str | None = None) -> dict | None:
    """Return the envelope, verifying the declared payload hash against a recomputation."""
    if rev is None:
        raw = (REPO / rel).read_bytes()
        body = json.loads(raw)
        file_sha = _sha256_bytes(raw)
    else:
        text = _git("show", f"{rev}:{rel}")
        if text is None:
            return None
        body = json.loads(text)
        file_sha = _sha256_bytes(text.encode())
    payload = body["payload"]
    declared = body.get("payload_sha256")
    recomputed = _canonical(payload)
    return {
        "payload": payload,
        "declared_payload_sha256": declared,
        "recomputed_payload_sha256": recomputed,
        "payload_hash_self_consistent": declared == recomputed,
        "file_sha256": file_sha,
    }


def _identity_chain(rel: str) -> list[dict]:
    """Every distinct payload identity this contract has held, oldest first."""
    log = _git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--", rel) or ""
    seen: list[dict] = []
    for line in reversed([x for x in log.strip().split("\n") if x]):
        sha, date, subject = line.split("|", 2)
        env = _load_contract(rel, sha)
        if env is None:
            continue
        pay_sha = env["declared_payload_sha256"]
        if seen and seen[-1]["payload_sha256"] == pay_sha:
            continue
        seen.append({
            "payload_sha256": pay_sha,
            "file_sha256": env["file_sha256"],
            "introduced_by_commit": sha,
            "date": date,
            "subject": subject,
            "delta": env["payload"].get("delta"),
            "charged_calls_per_cell": env["payload"].get("charged_calls_per_cell"),
        })
    for older, newer in pairwise(seen):
        a = dict(_flat(_load_contract(rel, older["introduced_by_commit"])["payload"]))
        b = dict(_flat(_load_contract(rel, newer["introduced_by_commit"])["payload"]))
        changes = []
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key, "<ABSENT>"), b.get(key, "<ABSENT>")
            if x != y and key.startswith(EXEC_PREFIXES):
                changes.append({"field": key, "from": x, "to": y})
        newer["executable_changes_from_previous"] = changes
    return seen


def _verify_pins(payload: dict) -> dict:
    drift, missing, ok = [], [], 0
    for rel, pinned in sorted(payload.get("runtime_inputs_sha256", {}).items()):
        path = REPO / rel
        if not path.exists():
            missing.append(rel)
            continue
        current = _sha256_bytes(path.read_bytes())
        if current == pinned:
            ok += 1
        else:
            recoverable = _find_blob_revision(rel, pinned)
            drift.append({"path": rel, "pinned_sha256": pinned, "tree_sha256": current,
                          "pinned_bytes_recoverable_at_commit": recoverable})
    return {"matching": ok, "drifted": drift, "missing": missing,
            "all_pins_reproducible_from_tree": not drift and not missing}


def _find_blob_revision(rel: str, target_sha256: str) -> str | None:
    """Locate a commit whose version of ``rel`` hashes to ``target_sha256``."""
    log = _git("log", "--all", "--format=%H", "--", rel) or ""
    for sha in [x for x in log.strip().split("\n") if x]:
        text = _git("show", f"{sha}:{rel}")
        if text is not None and _sha256_bytes(text.encode()) == target_sha256:
            return sha
    return None


def _commit_facts(sha: str) -> dict:
    full = _git("rev-parse", f"{sha}^{{commit}}")
    if full is None:
        return {"revision": sha, "exists_in_repo": False}
    full = full.strip()
    head = (_git("rev-parse", "HEAD") or "").strip()
    anc = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", full, head],
                         capture_output=True, check=False)
    branches = [b.strip().lstrip("*+ ") for b in
                (_git("branch", "-a", "--contains", full) or "").strip().split("\n") if b.strip()]
    local = sorted({b for b in branches if not b.startswith("remotes/")})
    return {
        "revision": full,
        "exists_in_repo": True,
        "is_ancestor_of_checkout_head": anc.returncode == 0,
        "branch_count": len(local),
        "branches_sample": local[:6],
        "subject": (_git("log", "-1", "--format=%s", full) or "").strip(),
    }


def _receipts() -> dict[str, list[dict]]:
    """Scan launch receipts from every readable checkout, keyed by volume."""
    by_volume: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    roots = [REPO, Path("/Users/rmaganti/compose_fa7_run")]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*/launches/*.json")):
            try:
                body = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            payload = body.get("payload", body)
            volume = payload.get("volume")
            if not volume:
                continue
            task = payload.get("task", {})
            key = (volume, payload.get("run_id"), payload.get("mode"))
            if key in seen:
                continue
            seen.add(key)
            by_volume.setdefault(volume, []).append({
                "receipt": str(path.relative_to(root)),
                "found_in_checkout": root.name,
                "mode": payload.get("mode") or "launch(implicit)",
                "run_id": payload.get("run_id"),
                "code_revision": task.get("code_revision"),
                "contract_payload_sha256": task.get("contract_payload_sha256"),
                "contract_file_sha256": task.get("contract_file_sha256"),
                "charged_call_ceiling": task.get("charged_call_ceiling"),
            })
    return by_volume


def _census_by_arm() -> dict[str, list[dict]]:
    """Cross-reference the volume-side run census, if the other agent has written it."""
    path = REPO / CENSUS
    if not path.exists():
        return {}
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, list[dict]] = {}
    for run in body.get("runs", []):
        arm = run.get("arm")
        if not arm:
            continue
        out.setdefault(arm, []).append({
            "run_id": run.get("run_id"),
            "code_revision": run.get("code_revision"),
            "contract_payload_sha256": run.get("contract_payload_sha256"),
            "contract_payload_sha256_in_cell": run.get("contract_payload_sha256_in_cell"),
            "contract_payload_sha256_in_launch": run.get("contract_payload_sha256_in_launch"),
            "charged_call_ceiling": run.get("charged_call_ceiling"),
            "status": run.get("status"),
        })
    return out


def _support_operator(payload: dict) -> dict:
    """Classify the arm's proposal operator by EXECUTABLE field, never by label."""
    proposal = payload.get("proposal", {})
    shallow = proposal.get("shallow", {})
    has_region_law = "region_law" in shallow
    has_protonation = "protonation_aware_retained_subgraph" in proposal
    if has_region_law and has_protonation:
        kind = "CONFLICT_BOTH_DECLARED"
    elif has_region_law:
        kind = "region_repair"
    elif has_protonation:
        kind = "protonation"
    else:
        kind = "none"
    evidence = {
        "proposal.shallow.region_law": shallow.get("region_law", "<ABSENT>"),
        "proposal.protonation_aware_retained_subgraph": (
            "PRESENT" if has_protonation else "<ABSENT>"),
        "expert_floor_rounds": payload.get("expert_floor_rounds", "<ABSENT>"),
        "cold_start_floor_rounds": payload.get("cold_start_floor_rounds", "<ABSENT>"),
        "route_scale_floor_rounds": payload.get("route_scale_floor_rounds", "<ABSENT>"),
        "route_complete_region.pool_size": proposal.get("route_complete_region", {}).get(
            "pool_size", "<ABSENT>"),
        "route_complete_region.realization_limit": proposal.get("route_complete_region", {}).get(
            "realization_limit", "<ABSENT>"),
        "route_complete_region.training_split": proposal.get("route_complete_region", {}).get(
            "training_split", "<ABSENT>"),
    }
    if has_protonation:
        evidence["protonation_lane_settings"] = proposal["protonation_aware_retained_subgraph"]
    return {"operator": kind, "evidence": evidence}


def _controller_family(payload: dict, arm: str) -> dict:
    return {
        "proposal": payload.get("proposal"),
        "batch": payload.get("batch"),
        "parents": payload.get("parents"),
        "exploration": payload.get("exploration"),
        "expert_floor_rounds": payload.get("expert_floor_rounds", "ABSENT"),
        "cold_start_floor_rounds": payload.get("cold_start_floor_rounds", "ABSENT"),
        "route_scale_floor_rounds": payload.get("route_scale_floor_rounds", "ABSENT"),
        "parent_explore": payload.get("parent_explore"),
        "value_penalty": payload.get("value_penalty"),
        "support": payload.get("support"),
        "docking_seed": payload.get("docking_seed"),
        "route_expert_checkpoint": _route_checkpoint(payload),
        "allocation_policy_present": "allocation_policy" in payload,
        "arm": arm,
    }


def _route_checkpoint(payload: dict) -> dict:
    for rel, sha in payload.get("runtime_inputs_sha256", {}).items():
        if "checkpoint.json" in rel:
            path = REPO / rel
            present = path.exists()
            return {"path": rel, "pinned_sha256": sha, "present_in_tree": present,
                    "tree_sha256": _sha256_bytes(path.read_bytes()) if present else None,
                    "bytes_match_pin": present and _sha256_bytes(path.read_bytes()) == sha}
    return {"path": None}


def _oracle_ceiling(payload: dict) -> dict:
    return {
        "charged_calls_per_cell": payload.get("charged_calls_per_cell"),
        "total_charged_call_ceiling": payload.get("total_charged_call_ceiling"),
        "total_new_charged_call_ceiling": payload.get("total_new_charged_call_ceiling", "ABSENT"),
        "prior_operational_waste": payload.get("prior_operational_waste", "ABSENT"),
        "ceiling_derivation": payload.get("ceiling_derivation", "ABSENT"),
        "replicate_note": ("Clone charged_calls_per_cell and the stopping rule. A rescue arm's "
                           "sub-250 ceiling is a DEBIT of measured prior calls on that cell; an "
                           "independent replicate starting fresh has no such prior and should use "
                           "the full 250 unless it inherits the same prior spend."),
    }


def _check_ownership(contracts: dict[str, dict]) -> list[str]:
    """Re-derive each row's owning arm from contract cells+delta; report disagreements."""
    problems = []
    for (target, seed, delta), arm in OWNER.items():
        payload = contracts[arm]["payload"]
        cell = f"{PROTEIN_CELL[target]}_{seed - 1}"
        if str(payload.get("delta")) != delta:
            problems.append(f"{target} seed {seed} d{delta}: arm {arm} has executable delta "
                            f"{payload.get('delta')}")
        if cell not in {c["cell"] for c in payload.get("cells", [])}:
            problems.append(f"{target} seed {seed} d{delta}: arm {arm} does not list cell {cell}")
    return problems


def build() -> dict:
    frozen = json.loads((REPO / FROZEN).read_text())
    table = frozen["payload"]["deltas"]

    contracts = {arm: _load_contract(meta["contract"]) for arm, meta in ARMS.items()}
    missing = [a for a, c in contracts.items() if c is None]
    if missing:
        raise SystemExit(f"contract(s) unreadable: {missing}")

    ownership_problems = _check_ownership(contracts)
    receipts = _receipts()
    census = _census_by_arm()

    arm_records: dict[str, dict] = {}
    for arm, meta in ARMS.items():
        env = contracts[arm]
        payload = env["payload"]
        chain = _identity_chain(meta["contract"])
        arm_receipts = receipts.get(meta["volume"], [])
        arm_census = census.get(arm, [])
        revs = {r["code_revision"] for r in arm_receipts if r.get("code_revision")}
        revs |= {r["code_revision"] for r in arm_census if r.get("code_revision")}
        revisions = [_commit_facts(sha) for sha in sorted(revs)]
        # The identity the run was BOUND to, preferring volume-side evidence (the cell/launch
        # artifacts the run itself wrote) over the in-repo receipt, and over the tree file --
        # which for an arm whose commit is not in this checkout is a SUPERSEDED payload.
        bound = [r["contract_payload_sha256"] for r in arm_census
                 if r.get("contract_payload_sha256")]
        tree_matches_run = (not bound) or env["declared_payload_sha256"] in bound
        wrapper = REPO / meta["wrapper"]
        arm_records[arm] = {
            "volume": meta["volume"],
            "contract_path": meta["contract"],
            "contract_payload_sha256": env["declared_payload_sha256"],
            "contract_file_sha256": env["file_sha256"],
            "payload_hash_self_consistent": env["payload_hash_self_consistent"],
            "delta_EXECUTABLE": payload.get("delta"),
            "cells": [c["cell"] for c in payload.get("cells", [])],
            "controller_seeds": {c["cell"]: c["controller_seed"] for c in payload.get("cells", [])},
            "contract_identity_chain": chain,
            "identity_count": len(chain),
            "launch_receipts_in_repo": arm_receipts,
            "run_census_cross_reference": arm_census,
            "contract_payload_the_runs_were_BOUND_to": sorted(set(bound)) or None,
            "tree_contract_is_the_one_that_ran": tree_matches_run,
            "tree_contract_warning": None if tree_matches_run else (
                "The contract file in THIS CHECKOUT is a SUPERSEDED payload "
                f"({env['declared_payload_sha256'][:16]}...). The run was bound to "
                f"{min(bound)[:16]}..., which exists only at a commit that is not an "
                "ancestor of this checkout's HEAD. Replicating from the tree file would clone a "
                "contract that never ran. Recover it from the identity chain below."),
            "code_revisions": revisions,
            "app_module": {
                "shared_app": "modal_apps/t4_integrated_route_fiber_parp1_app.py"
                if "protonation" not in arm else meta["wrapper"],
                "wrapper": meta["wrapper"],
                "wrapper_present_in_tree": wrapper.exists(),
                "wrapper_tree_sha256": _sha256_bytes(wrapper.read_bytes())
                if wrapper.exists() else None,
                "wrapper_is_pinned_in_runtime_inputs": meta["wrapper"] in payload.get(
                    "runtime_inputs_sha256", {}),
                "pinned_app_modules": {k: v for k, v in payload.get(
                    "runtime_inputs_sha256", {}).items() if k.startswith("modal_apps/")},
            },
            "runtime_input_pin_check": _verify_pins(payload),
            "controller_family": _controller_family(payload, arm),
            "support_operator": _support_operator(payload),
            "oracle_ceiling": _oracle_ceiling(payload),
        }

    rows = []
    for delta, block in table.items():
        for row in block["rows"]:
            key = (row["target"], row["seed"], delta)
            arm = OWNER[key]
            rec = arm_records[arm]
            cell = f"{PROTEIN_CELL[row['target']]}_{row['seed'] - 1}"
            finished = None
            if rec["contract_payload_the_runs_were_BOUND_to"]:
                finished = rec["contract_payload_the_runs_were_BOUND_to"][0]
            else:
                for receipt in rec["launch_receipts_in_repo"]:
                    if receipt["mode"].startswith("resume"):
                        finished = receipt["contract_payload_sha256"]
                if finished is None and rec["launch_receipts_in_repo"]:
                    finished = rec["launch_receipts_in_repo"][0]["contract_payload_sha256"]
            rows.append({
                "target": row["target"],
                "seed": row["seed"],
                "cell": cell,
                "delta": delta,
                "frozen_table": {"compose": row["compose"], "ivg": row["ivg"], "gap": row["gap"],
                                 "realized_charged_calls": row["charged_calls"],
                                 "source_label": row["source"]},
                "arm": arm,
                "arm_evidence": {
                    "contract_lists_this_cell": cell in rec["cells"],
                    "contract_delta_matches_row": str(rec["delta_EXECUTABLE"]) == delta,
                    "volume": rec["volume"],
                    "receipt_count_in_repo": len(rec["launch_receipts_in_repo"]),
                    "run_census_records": len(rec["run_census_cross_reference"]),
                },
                "contract_path": rec["contract_path"],
                "contract_payload_sha256": rec["contract_payload_sha256"],
                "contract_file_sha256": rec["contract_file_sha256"],
                "contract_identity_count": rec["identity_count"],
                "contract_payload_the_run_FINISHED_under": finished,
                "code_revisions": [c["revision"] for c in rec["code_revisions"]],
                "app_module": rec["app_module"]["wrapper"],
                "support_operator": rec["support_operator"]["operator"],
                "oracle_ceiling": rec["oracle_ceiling"]["charged_calls_per_cell"],
                "controller_seed": rec["controller_seeds"].get(cell),
                "produced_a_value": row["compose"] is not None,
                "blank_cell_note": BLANK_CELL_ATTEMPTS.get(key, {}).get("caution"),
                "configuration_family": None,  # filled below
            })

    # ---- Configuration families ---------------------------------------------------------
    families: dict[str, dict] = {}
    for row in rows:
        rec = arm_records[row["arm"]]
        fam = rec["controller_family"]
        fingerprint = _canonical({
            "proposal": fam["proposal"], "batch": fam["batch"], "parents": fam["parents"],
            "exploration": fam["exploration"], "expert_floor_rounds": fam["expert_floor_rounds"],
            "cold_start_floor_rounds": fam["cold_start_floor_rounds"],
            "route_scale_floor_rounds": fam["route_scale_floor_rounds"],
            "parent_explore": fam["parent_explore"], "value_penalty": fam["value_penalty"],
            "support": fam["support"], "docking_seed": fam["docking_seed"],
            "route_expert_checkpoint": fam["route_expert_checkpoint"].get("pinned_sha256"),
            "operator": rec["support_operator"]["operator"],
            "ceiling": rec["oracle_ceiling"]["charged_calls_per_cell"],
        })[:16]
        row["configuration_family"] = fingerprint
        entry = families.setdefault(fingerprint, {
            "fingerprint": fingerprint,
            "support_operator": rec["support_operator"]["operator"],
            "arms": set(), "cells": [], "docking_seed": fam["docking_seed"],
            "charged_calls_per_cell": rec["oracle_ceiling"]["charged_calls_per_cell"],
            "route_complete_region": fam["proposal"].get("route_complete_region"),
            "shallow": fam["proposal"].get("shallow"),
            "anchored_replacement": fam["proposal"].get("anchored_replacement"),
            "expert_floor_rounds": fam["expert_floor_rounds"],
            "cold_start_floor_rounds": fam["cold_start_floor_rounds"],
            "route_scale_floor_rounds": fam["route_scale_floor_rounds"],
            "route_expert_checkpoint": fam["route_expert_checkpoint"].get("path"),
        })
        entry["arms"].add(row["arm"])
        entry["cells"].append(f"{row['target']}/s{row['seed']}/d{row['delta']}")
    for entry in families.values():
        entry["arms"] = sorted(entry["arms"])
        entry["cell_count"] = len(entry["cells"])

    label_audit = _label_audit(rows, arm_records)
    broken = _broken_links(arm_records, ownership_problems)

    return {
        "schema_version": "t4_replication_provenance_v1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "purpose": ("Per-cell historical method for the 30 published T4 cells, so two further "
                    "independent replicates clone the method that ran rather than today's "
                    "defaults."),
        "read_only": "No network, no Modal, no oracle calls were used to build this.",
        "frozen_table": {"path": FROZEN, "payload_sha256": json.loads(
            (REPO / FROZEN).read_text()).get("payload_sha256")},
        "checkout": {
            "path": str(REPO),
            "head": (_git("rev-parse", "HEAD") or "").strip(),
            "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
        },
        "seed_model": SEED_MODEL,
        "stopping_rule": STOPPING_RULE,
        "rows": rows,
        "arms": arm_records,
        "configuration_families": {
            "count": len(families),
            "count_note": (
                "delta is a benchmark INPUT, not a controller setting, so the primary grouping "
                "deliberately EXCLUDES it: two rows at different thresholds running the same "
                "controller are one method. Splitting additionally by delta gives "
                f"{len({(r['configuration_family'], r['delta']) for r in rows})} "
                "(configuration, threshold) combinations."),
            "count_split_by_delta": len(
                {(r["configuration_family"], r["delta"]) for r in rows}),
            "note": ("Two rows share a family only if their whole executable controller "
                     "configuration agrees, INCLUDING docking seed, route-expert checkpoint, "
                     "floor schedule, operator and ceiling."),
            "families": sorted(families.values(), key=lambda e: -e["cell_count"]),
        },
        "blank_cells": {f"{k[0]}/s{k[1]}/d{k[2]}": v for k, v in BLANK_CELL_ATTEMPTS.items()},
        "label_audit": label_audit,
        "broken_links": broken,
    }


def _label_audit(rows: list[dict], arm_records: dict[str, dict]) -> dict:
    by_label: dict[str, set] = {}
    for row in rows:
        by_label.setdefault(row["frozen_table"]["source_label"], set()).add(
            row["support_operator"])
    conflated = {k: sorted(v) for k, v in by_label.items() if len(v) > 1}
    return {
        "frozen_source_label_to_executable_operator": {k: sorted(v) for k, v in by_label.items()},
        "labels_covering_more_than_one_operator": conflated,
        "verdict": ("CONFLATED" if conflated else "one label per operator"),
        "detail": (
            "The frozen table's source='support_expansion' covers BOTH the region-repair "
            "operator (proposal.shallow.region_law='free_gate_margin_v1', an extra key on the "
            "EXISTING shallow lane) and the protonation operator (an entirely NEW fourth "
            "proposal lane 'protonation_aware_retained_subgraph' with 17 settings, plus "
            "expert_floor_rounds REPLACED by cold_start_floor_rounds=1, a different route "
            "checkpoint, and route_complete_region pool_size 64->192 / realization_limit "
            "64->96 with training_split REMOVED). The protonation contract itself asserts "
            "'THIS ARM USES THE PROTONATION-AWARE PROPOSAL, NOT THE REGION-REPAIR LAW ... the "
            "two mechanisms must never be mixed in one arm or the mechanism becomes "
            "unattributable'. The shared label defeats that separation at reporting time."),
        "third_meaning_of_the_same_string": (
            "'support_expansion' is ALSO the name of an unrelated module, "
            "compose_v4.experiments.t4_support_expansion, used in the fa7_0 escalation run. "
            "That module contributed NO row to this table -- FA7 seed 1 at delta 0.6 is blank. "
            "So the string denotes three different things across the record."),
    }


def _broken_links(arm_records: dict[str, dict], ownership_problems: list[str]) -> list[dict]:
    issues: list[dict] = []
    for arm, rec in arm_records.items():
        pins = rec["runtime_input_pin_check"]
        if pins["drifted"] or pins["missing"]:
            issues.append({
                "arm": arm, "kind": "runtime_input_pin_drift",
                "severity": "HIGH" if any(
                    d["pinned_bytes_recoverable_at_commit"] is None for d in pins["drifted"]
                ) else "MEDIUM-recoverable-from-git",
                "detail": pins,
            })
        if not rec["launch_receipts_in_repo"]:
            resolved = bool(rec["run_census_cross_reference"])
            issues.append({
                "arm": arm, "kind": "no_launch_receipt_in_repo",
                "severity": "LOW-resolved-by-run-census" if resolved else "HIGH",
                "detail": ("No launch receipt exists under diagnostics/**/launches/ in any "
                           f"readable checkout for volume {rec['volume']}."
                           + (" RESOLVED: the volume-side run census supplies run_id, "
                              "code_revision, bound contract payload and ceiling."
                              if resolved else
                              " UNRESOLVED: code_revision, bound contract identity and ceiling "
                              "cannot be confirmed.")),
            })
        if not rec["tree_contract_is_the_one_that_ran"]:
            issues.append({
                "arm": arm, "kind": "tree_contract_is_superseded", "severity": "HIGH",
                "detail": rec["tree_contract_warning"],
            })
        if not rec["app_module"]["wrapper_is_pinned_in_runtime_inputs"]:
            issues.append({
                "arm": arm, "kind": "launch_wrapper_not_pinned", "severity": "MEDIUM",
                "detail": (f"{rec['app_module']['wrapper']} selects the contract, route "
                           "checkpoint, volume, output namespace and receptor, but is absent "
                           "from runtime_inputs_sha256, so it is free to drift unnoticed."),
            })
        if not rec["payload_hash_self_consistent"]:
            issues.append({"arm": arm, "kind": "payload_hash_self_inconsistent",
                           "severity": "HIGH", "detail": "declared != recomputed canonical hash"})
    for problem in ownership_problems:
        issues.append({"arm": None, "kind": "row_ownership_disagreement", "severity": "HIGH",
                       "detail": problem})
    return issues


def main() -> None:
    manifest = build()
    out = REPO / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT}")
    print(f"  rows: {len(manifest['rows'])}")
    print(f"  configuration families: {manifest['configuration_families']['count']}")
    print(f"  label verdict: {manifest['label_audit']['verdict']}")
    print(f"  broken links: {len(manifest['broken_links'])}")


if __name__ == "__main__":
    main()
