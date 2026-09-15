"""Prospective zero-oracle lock for compositional-generator T4 endpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.control.docking_value import identity
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import InvalidRewrite, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state

PRELOCK_SCHEMA = "t4_compositional_generator_utility_prelock_contract_v1"
BINDING_SCHEMA = "t4_compositional_generator_utility_input_binding_v1"
SOURCE_LOCK_SCHEMA = "t4_compositional_structural_subgoal_candidate_lock_v1"
LEDGER_SCHEMA = "t4_compositional_generator_utility_eligibility_ledger_v1"
CANDIDATE_SCHEMA = "t4_compositional_generator_utility_candidate_lock_v1"
REQUEST_SCHEMA = "t4_compositional_generator_utility_request_lock_v1"
ABSTENTION_SCHEMA = "t4_compositional_generator_utility_abstention_ledger_v1"
RESULT_SCHEMA = "t4_compositional_generator_utility_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_git_path(root: Path, revision: str, path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root)
    return hashlib.sha256(content).hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _read_json(path: Path) -> dict:
    encoded = path.read_bytes()
    raw = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sealed_payload(path: Path, *, expected_payload_sha256: str | None = None) -> dict:
    envelope = _read_json(path)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    if expected_payload_sha256 is not None and claimed != expected_payload_sha256:
        raise ValueError(f"artifact payload identity changed: {path}")
    return payload


def _walk_hashed_inputs(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        else:
            for child in value.values():
                yield from _walk_hashed_inputs(child)


def load_prelock(root: Path, path: Path) -> dict:
    payload = _sealed_payload(path)
    if payload.get("schema_version") != PRELOCK_SCHEMA:
        raise ValueError(f"unexpected compositional utility prelock schema: {path}")
    authority = payload.get("authority", {})
    if (
        authority.get("oracle_calls_authorized") != 0
        or authority.get("docking_calls_authorized") != 0
        or authority.get("modal_launches_authorized") != 0
        or authority.get("live_run_access_authorized") is not False
        or authority.get("scored_launch_authorized") is not False
    ):
        raise ValueError("compositional utility prelock must remain zero-oracle")
    if len(payload.get("cells", ())) != 5 or len(payload.get("arms", ())) != 4:
        raise ValueError(
            "compositional utility prelock must contain five cells and four arms"
        )
    for entry in _walk_hashed_inputs(payload["immutable_inputs"]):
        input_path = root / entry["path"]
        if sha256_file(input_path) != entry["sha256"]:
            raise ValueError(f"immutable prelock input changed: {input_path}")
    return payload


def load_binding(root: Path, path: Path, prelock_path: Path, prelock: dict) -> dict:
    """Validate every physical identity before a candidate payload may be opened."""
    payload = _sealed_payload(path)
    if payload.get("schema_version") != BINDING_SCHEMA:
        raise ValueError(f"unexpected compositional utility binding schema: {path}")
    bound_prelock = payload.get("prelock", {})
    prelock_envelope = _read_json(prelock_path)
    if (
        bound_prelock.get("path") != str(prelock_path.relative_to(root))
        or bound_prelock.get("sha256") != sha256_file(prelock_path)
        or bound_prelock.get("contract_sha256")
        != prelock_envelope.get("contract_sha256")
        or payload.get("selection_semantics_sha256")
        != identity(prelock["selection_semantics"])
    ):
        raise ValueError("input binding does not preserve the frozen prelock")
    candidate_bindings = payload.get("candidate_locks", {})
    if set(candidate_bindings) != {"baseline", "expanded"}:
        raise ValueError(
            "input binding must contain exactly baseline and expanded locks"
        )
    for generator_revision, entry in sorted(candidate_bindings.items()):
        expected = prelock["candidate_bindings"][generator_revision]
        if entry.get("path") != expected["logical_path"]:
            raise ValueError(f"candidate-lock path changed: {generator_revision}")
        candidate_path = root / entry["path"]
        physical = sha256_file(candidate_path)
        if physical != entry.get("sha256"):
            raise ValueError(
                f"candidate-lock physical identity changed: {candidate_path}"
            )
        expected_physical = expected["expected_physical_sha256"]
        if (
            expected_physical != "PENDING_ARTIFACT_ONLY_COMMIT"
            and physical != expected_physical
        ):
            raise ValueError(
                f"candidate lock differs from predeclared identity: {candidate_path}"
            )
        if (
            not isinstance(entry.get("payload_sha256"), str)
            or len(entry["payload_sha256"]) != 64
        ):
            raise ValueError(
                f"candidate payload identity is not bound: {generator_revision}"
            )
        if (
            not isinstance(entry.get("source_revision"), str)
            or len(entry["source_revision"]) != 40
        ):
            raise ValueError(
                f"candidate source revision is not bound: {generator_revision}"
            )
        if sha256_git_path(root, entry["source_revision"], entry["path"]) != physical:
            raise ValueError(
                f"candidate lock is absent from its bound source revision: {generator_revision}"
            )
    implementation = payload.get("selection_implementation", {})
    implementation_path = root / implementation.get("path", "")
    implementation_revision = implementation.get("source_revision")
    if (
        implementation.get("path")
        != "src/compose_v4/experiments/t4_compositional_utility_lock.py"
        or sha256_file(implementation_path) != implementation.get("sha256")
        or not isinstance(implementation_revision, str)
        or len(implementation_revision) != 40
        or sha256_git_path(root, implementation_revision, implementation["path"])
        != implementation["sha256"]
    ):
        raise ValueError("selection implementation identity changed")
    return payload


def verify_environment(prelock: dict) -> None:
    expected = prelock["environment"]
    if rdBase.rdkitVersion != expected["rdkit"]:
        raise ValueError(
            f"utility lock requires RDKit {expected['rdkit']}; got {rdBase.rdkitVersion}"
        )
    scorer_path = Path(sascorer.__file__)
    if sha256_file(scorer_path) != expected["sa_scorer_sha256"]:
        raise ValueError("SA scorer implementation identity changed")
    if (
        sha256_file(scorer_path.with_name("fpscores.pkl.gz"))
        != expected["sa_fragment_scores_sha256"]
    ):
        raise ValueError("SA fragment-score identity changed")


def endpoint_exclusion_reasons(
    *,
    replay_exact: bool,
    valid: bool,
    connected: bool,
    is_null: bool,
    canonical_available: bool,
    active_heavy_match: bool | None,
    non_self: bool | None,
    similarity: float | None,
    qed: float | None,
    sa: float | None,
    active_atoms: int,
) -> list[str]:
    reasons = []
    if not replay_exact:
        reasons.append("candidate_actions_do_not_exactly_replay_endpoint_state")
    if not valid:
        reasons.append("invalid_molecular_graph")
    if not connected:
        reasons.append("disconnected_molecular_graph")
    if is_null:
        reasons.append("null_endpoint")
    if not canonical_available:
        reasons.append("canonical_molecule_unavailable")
    if active_heavy_match is False:
        reasons.append("active_heavy_atom_count_mismatch")
    if non_self is False:
        reasons.append("self_endpoint")
    if similarity is None or not similarity > 0.4:
        reasons.append("similarity_not_strictly_greater_than_0.4")
    if qed is None or not qed > 0.6:
        reasons.append("qed_not_strictly_greater_than_0.6")
    if sa is None or not sa < 4.0:
        reasons.append("sa_not_strictly_less_than_4.0")
    if active_atoms > 40:
        reasons.append("active_atoms_greater_than_40")
    return reasons


def _canonical_descriptor(state, generator) -> dict:
    active_atoms = int(state.n_real_atoms)
    smiles = molecular_graph_to_smiles(state) if active_atoms else None
    molecule = Chem.MolFromSmiles(smiles) if smiles is not None else None
    canonical = (
        Chem.MolToSmiles(molecule, isomericSmiles=False)
        if molecule is not None
        else None
    )
    return {
        "active_atoms": active_atoms,
        "smiles": smiles,
        "molecule": molecule,
        "canonical_smiles": canonical,
        "heavy_atoms": (
            int(molecule.GetNumHeavyAtoms()) if molecule is not None else None
        ),
        "fingerprint": (
            generator.GetFingerprint(molecule) if molecule is not None else None
        ),
    }


def _source_descriptor(source_state: dict, generator) -> dict:
    state = decode_state(source_state)
    if (
        not is_valid_state(state)
        or not is_connected_or_null(state)
        or state.n_real_atoms == 0
    ):
        raise ValueError("frozen source state is not a valid connected molecule")
    descriptor = _canonical_descriptor(state, generator)
    if descriptor["canonical_smiles"] is None:
        raise ValueError("frozen source state has no canonical molecule")
    if descriptor["active_atoms"] != descriptor["heavy_atoms"]:
        raise ValueError("frozen source active/heavy atom counts disagree")
    return {"state": state, **descriptor}


def _replay_candidate(
    source_state: dict, actions: list[dict]
) -> tuple[dict | None, str | None]:
    system = editing_v2_semantic_rewrite_system()
    current = decode_state(source_state)
    try:
        for action_record in actions:
            rule, action = decode_action(action_record)
            current = system.apply(current, rule, action)
    except (InvalidRewrite, KeyError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}:{exc}"
    return encode_state(current), None


def describe_candidate(
    candidate: dict,
    *,
    generator_revision: str,
    arm_id: str,
    policy_id: str,
    fold: int,
    cell: str,
    target: str,
    source_case_id: str,
    source_state: dict,
    source: dict,
    rank: int,
    fingerprint_generator,
    candidate_lock_sha256: str,
    candidate_lock_payload_sha256: str,
) -> dict:
    required = {
        "endpoint_state",
        "actions",
        "policy_id",
        "score",
        "goal",
        "bindings",
        "construction_dependencies",
        "patch_ids",
        "realization",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(
            f"candidate is missing required fields {missing}: {cell}/{arm_id}/{rank}"
        )
    if candidate["policy_id"] != policy_id:
        raise ValueError(f"candidate policy mismatch: {cell}/{arm_id}/{rank}")
    proposal_score = candidate["score"]
    if not isinstance(proposal_score, (int, float)) or not math.isfinite(
        float(proposal_score)
    ):
        raise ValueError(
            f"candidate proposal score is nonfinite: {cell}/{arm_id}/{rank}"
        )
    endpoint_state = candidate["endpoint_state"]
    endpoint = decode_state(endpoint_state)
    replayed_state, replay_error = _replay_candidate(source_state, candidate["actions"])
    replay_exact = replayed_state == endpoint_state
    descriptor = _canonical_descriptor(endpoint, fingerprint_generator)
    valid = bool(is_valid_state(endpoint))
    connected = bool(is_connected_or_null(endpoint))
    molecule = descriptor["molecule"]
    fingerprint = descriptor["fingerprint"]
    similarity = (
        float(DataStructs.TanimotoSimilarity(source["fingerprint"], fingerprint))
        if fingerprint is not None
        else None
    )
    qed = float(QED.qed(molecule)) if molecule is not None else None
    sa = float(sascorer.calculateScore(molecule)) if molecule is not None else None
    canonical = descriptor["canonical_smiles"]
    non_self = (
        canonical != source["canonical_smiles"] if canonical is not None else None
    )
    active_heavy_match = (
        descriptor["active_atoms"] == descriptor["heavy_atoms"]
        if descriptor["heavy_atoms"] is not None
        else None
    )
    reasons = endpoint_exclusion_reasons(
        replay_exact=replay_exact,
        valid=valid,
        connected=connected,
        is_null=descriptor["active_atoms"] == 0,
        canonical_available=canonical is not None,
        active_heavy_match=active_heavy_match,
        non_self=non_self,
        similarity=similarity,
        qed=qed,
        sa=sa,
        active_atoms=descriptor["active_atoms"],
    )
    candidate_id = identity(candidate)
    return {
        "candidate_uid": identity(
            {
                "generator_revision": generator_revision,
                "arm_id": arm_id,
                "cell": cell,
                "rank": rank,
                "candidate_id": candidate_id,
            }
        ),
        "candidate_id": candidate_id,
        "generator_revision": generator_revision,
        "arm_id": arm_id,
        "policy_id": policy_id,
        "fold": fold,
        "cell": cell,
        "target": target,
        "source_case_id": source_case_id,
        "rank": rank,
        "candidate_lock_sha256": candidate_lock_sha256,
        "candidate_lock_payload_sha256": candidate_lock_payload_sha256,
        "source_state": source_state,
        "source_state_sha256": identity(source_state),
        "source_canonical_smiles": source["canonical_smiles"],
        "endpoint_state": endpoint_state,
        "endpoint_state_sha256": identity(endpoint_state),
        "actions": candidate["actions"],
        "goal": candidate["goal"],
        "bindings": candidate["bindings"],
        "construction_dependencies": candidate["construction_dependencies"],
        "patch_ids": candidate["patch_ids"],
        "realization": candidate["realization"],
        "proposal_score": float(proposal_score),
        "proposal_score_role": "generator-internal ranking evidence; not a task or docking score",
        "exact_action_replay": replay_exact,
        "replay_error": replay_error,
        "valid": valid,
        "connected": connected,
        "canonical_smiles": canonical,
        "active_atoms": descriptor["active_atoms"],
        "heavy_atoms": descriptor["heavy_atoms"],
        "non_self": non_self,
        "morgan_radius2_2048_similarity_to_source": similarity,
        "qed": qed,
        "sa": sa,
        "eligible": not reasons,
        "eligibility_exclusion_reasons": reasons,
        "selection_exclusion_reasons": [],
        "selected": False,
    }


def select_cell_arm(rows: list[dict]) -> dict | None:
    """Apply the frozen eligibility-first, canonical-dedup, rank-first rule."""
    by_canonical: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            by_canonical[row["canonical_smiles"]].append(row)
    representatives = []
    for canonical, duplicates in sorted(by_canonical.items()):
        ordered = sorted(
            duplicates,
            key=lambda row: (row["rank"], canonical, row["candidate_id"]),
        )
        representatives.append(ordered[0])
        for duplicate in ordered[1:]:
            duplicate["selection_exclusion_reasons"].append(
                "eligible_canonical_duplicate_within_cell_arm"
            )
    representatives.sort(
        key=lambda row: (row["rank"], row["canonical_smiles"], row["candidate_id"])
    )
    if not representatives:
        return None
    selected = representatives[0]
    selected["selected"] = True
    for row in representatives[1:]:
        row["selection_exclusion_reasons"].append(
            "eligible_not_lowest_rank_representative"
        )
    return selected


def _evaluator(cell: dict, unit: dict, scoring_sha256: str) -> dict:
    domain = unit["oracle_domain"]
    if (
        unit["cell"] != cell["cell"]
        or unit["target"] != cell["target"]
        or unit["original_seed"] != cell["original_seed"]
        or unit["source_idx"] != cell["source_idx"]
    ):
        raise ValueError(f"replicate-0 unit disagrees with frozen cell: {cell['cell']}")
    return {
        "schema_version": "t4_quickvina_endpoint_evaluator_v1",
        "target": cell["target"],
        "box_definition": domain["box_definition"],
        "docking": domain["docking"],
        "qvina02_sha256": domain["qvina02_sha256"],
        "receptor_sha256": domain["receptor_sha256"],
        "scoring_implementation_sha256": scoring_sha256,
        "score_direction": "minimize",
        "score_unit": "kcal/mol as emitted by the bound QuickVina wrapper",
    }


def build_request_records(memberships: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in memberships:
        key = (
            row["target"],
            row["canonical_smiles"],
            row["docking_seed"],
            row["evaluator"],
        )
        grouped[key].append(row)
    requests = []
    for key, members in sorted(grouped.items()):
        target, canonical_smiles, docking_seed, evaluator = key
        request_identity = {
            "schema_version": "t4_docking_request_identity_v1",
            "target": target,
            "canonical_smiles": canonical_smiles,
            "docking_seed": docking_seed,
            "evaluator": evaluator,
        }
        requests.append(
            {
                "request_id": identity(request_identity),
                **request_identity,
                "evaluator_payload": members[0]["evaluator_payload"],
                "membership_ids": sorted(row["membership_id"] for row in members),
                "memberships": sorted(
                    [
                        {
                            field: row[field]
                            for field in (
                                "membership_id",
                                "generator_revision",
                                "arm_id",
                                "policy_id",
                                "fold",
                                "cell",
                                "source_case_id",
                                "rank",
                                "candidate_id",
                                "candidate_uid",
                            )
                        }
                        for row in members
                    ],
                    key=lambda row: (row["cell"], row["arm_id"], row["rank"]),
                ),
            }
        )
    return requests


def _candidate_case_index(payload: dict, t4_cells: dict) -> dict[str, dict]:
    if (
        payload.get("schema_version") != SOURCE_LOCK_SCHEMA
        or payload.get("teacher_fields_present") is not False
        or payload.get("task_identity_present") is not False
        or payload.get("new_oracle_calls") != 0
    ):
        raise ValueError("source candidate lock violates the zero-oracle schema")
    by_source = {identity(cell["source_state"]): cell for cell in t4_cells.values()}
    cases = {}
    for fold_record in payload.get("folds", []):
        fold = int(fold_record["fold"])
        for case in fold_record["cases"]:
            cell = by_source.get(identity(case["source_state"]))
            if cell is None:
                raise ValueError("candidate source is outside the frozen T4 contract")
            cell_name = cell["cell"]
            if cell_name in cases:
                raise ValueError(f"candidate lock repeats T4 cell: {cell_name}")
            policies = {row["policy_id"]: row for row in case["policies"]}
            cases[cell_name] = {"fold": fold, "case": case, "policies": policies}
    if set(cases) != set(t4_cells):
        raise ValueError("candidate lock does not cover the exact frozen 15-cell panel")
    return cases


def _code_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _source_dirty(root: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )


def build_payloads(
    root: Path,
    prelock_path: Path,
    binding_path: Path,
) -> dict[str, dict]:
    prelock = load_prelock(root, prelock_path)
    binding = load_binding(root, binding_path, prelock_path, prelock)
    verify_environment(prelock)
    fingerprint_spec = prelock["eligibility"]["fingerprint"]
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint_spec["radius"]),
        fpSize=int(fingerprint_spec["bits"]),
        includeChirality=bool(fingerprint_spec["include_chirality"]),
    )

    t4_input = prelock["immutable_inputs"]["frozen_t4_source_cell_contract"]
    t4 = _sealed_payload(
        root / t4_input["path"],
        expected_payload_sha256=t4_input["payload_sha256"],
    )
    cells = t4["cells"]
    expected_cells = {
        f"{target}_{source}"
        for target in ("5ht1b", "braf", "fa7", "jak2", "parp1")
        for source in range(3)
    }
    if set(cells) != expected_cells or set(prelock["cells"]) - set(cells):
        raise ValueError("frozen T4 source/cell contract changed")
    replicate_zero = {
        row["cell"]: row for row in t4["units"] if row.get("replicate") == 0
    }
    if set(replicate_zero) != set(cells):
        raise ValueError("frozen T4 contract lacks one replicate-0 unit per cell")

    source_locks = {}
    case_indexes = {}
    for generator_revision, entry in sorted(binding["candidate_locks"].items()):
        path = root / entry["path"]
        source_locks[generator_revision] = _sealed_payload(
            path, expected_payload_sha256=entry["payload_sha256"]
        )
        case_indexes[generator_revision] = _candidate_case_index(
            source_locks[generator_revision], cells
        )

    rows = []
    selections = []
    arm_statuses = []
    arm_specs = {row["arm_id"]: row for row in prelock["arms"]}
    scoring_sha256 = prelock["immutable_inputs"]["docking_scoring_implementation"][
        "sha256"
    ]
    for cell_name in prelock["cells"]:
        cell = cells[cell_name]
        for arm_id, arm in arm_specs.items():
            revision = arm["generator_revision"]
            policy_id = arm["policy_id"]
            indexed = case_indexes[revision][cell_name]
            case = indexed["case"]
            policy = indexed["policies"].get(policy_id)
            if policy is None:
                raise ValueError(
                    f"candidate lock lacks declared arm: {cell_name}/{arm_id}"
                )
            candidates = policy["candidates"]
            if len(candidates) > 128:
                raise ValueError(
                    f"candidate pool exceeds frozen cap: {cell_name}/{arm_id}"
                )
            source = _source_descriptor(case["source_state"], fingerprint_generator)
            binding_entry = binding["candidate_locks"][revision]
            arm_rows = [
                describe_candidate(
                    candidate,
                    generator_revision=revision,
                    arm_id=arm_id,
                    policy_id=policy_id,
                    fold=indexed["fold"],
                    cell=cell_name,
                    target=cell["target"],
                    source_case_id=case["source_case_id"],
                    source_state=case["source_state"],
                    source=source,
                    rank=rank,
                    fingerprint_generator=fingerprint_generator,
                    candidate_lock_sha256=binding_entry["sha256"],
                    candidate_lock_payload_sha256=binding_entry["payload_sha256"],
                )
                for rank, candidate in enumerate(candidates, 1)
            ]
            selected = select_cell_arm(arm_rows)
            rows.extend(arm_rows)
            status = {
                "cell": cell_name,
                "target": cell["target"],
                "fold": indexed["fold"],
                "arm_id": arm_id,
                "generator_revision": revision,
                "policy_id": policy_id,
                "source_case_id": case["source_case_id"],
                "pool_candidates": len(arm_rows),
                "pool_shortfall_from_128": 128 - len(arm_rows),
                "eligible_candidates": sum(row["eligible"] for row in arm_rows),
                "eligible_unique_canonical_endpoints": len(
                    {row["canonical_smiles"] for row in arm_rows if row["eligible"]}
                ),
                "selected_memberships": int(selected is not None),
                "status": (
                    "candidate_locked"
                    if selected is not None
                    else "abstain_no_eligible_canonical_endpoint"
                ),
            }
            arm_statuses.append(status)
            if selected is None:
                continue
            unit = replicate_zero[cell_name]
            evaluator_payload = _evaluator(cell, unit, scoring_sha256)
            evaluator = identity(evaluator_payload)
            membership_body = {
                field: selected[field]
                for field in (
                    "candidate_uid",
                    "candidate_id",
                    "generator_revision",
                    "arm_id",
                    "policy_id",
                    "fold",
                    "cell",
                    "target",
                    "source_case_id",
                    "rank",
                    "candidate_lock_sha256",
                    "candidate_lock_payload_sha256",
                    "source_state",
                    "source_state_sha256",
                    "source_canonical_smiles",
                    "endpoint_state",
                    "endpoint_state_sha256",
                    "actions",
                    "goal",
                    "bindings",
                    "construction_dependencies",
                    "patch_ids",
                    "realization",
                    "proposal_score",
                    "proposal_score_role",
                    "exact_action_replay",
                    "canonical_smiles",
                    "active_atoms",
                    "heavy_atoms",
                    "non_self",
                    "morgan_radius2_2048_similarity_to_source",
                    "qed",
                    "sa",
                )
            }
            membership_body.update(
                {
                    "selection_rule": "lowest_rank_eligible_canonical_representative",
                    "docking_seed": int(unit["docking_seed"]),
                    "evaluator": evaluator,
                    "evaluator_payload": evaluator_payload,
                }
            )
            selections.append(
                {"membership_id": identity(membership_body), **membership_body}
            )

    rows.sort(key=lambda row: (row["cell"], row["arm_id"], row["rank"]))
    for row in rows:
        row["final_exclusion_reasons"] = [
            *row["eligibility_exclusion_reasons"],
            *row["selection_exclusion_reasons"],
        ]
        if row["selected"] != (not row["final_exclusion_reasons"]):
            raise RuntimeError(
                f"candidate selection ledger is inconsistent: {row['candidate_uid']}"
            )
    selections.sort(key=lambda row: (row["cell"], row["arm_id"], row["rank"]))
    requests = build_request_records(selections)
    if len(selections) > 20 or len(requests) > 20:
        raise RuntimeError("prospective utility lock exceeded the 20-request ceiling")
    request_by_membership = {
        membership_id: request["request_id"]
        for request in requests
        for membership_id in request["membership_ids"]
    }
    selections = [
        {**row, "request_id": request_by_membership[row["membership_id"]]}
        for row in selections
    ]

    common = {
        "prelock": {
            "path": str(prelock_path.relative_to(root)),
            "sha256": sha256_file(prelock_path),
            "contract_sha256": _read_json(prelock_path)["contract_sha256"],
        },
        "input_binding": {
            "path": str(binding_path.relative_to(root)),
            "sha256": sha256_file(binding_path),
            "payload_sha256": _read_json(binding_path)["payload_sha256"],
        },
        "code_revision": _code_revision(root),
        "candidate_locks": binding["candidate_locks"],
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
    }
    ledger_payload = {
        "schema_version": LEDGER_SCHEMA,
        **common,
        "selection_inputs": {
            "teacher_endpoint_accessed": False,
            "teacher_transformation_accessed": False,
            "teacher_metric_accessed": False,
            "task_score_accessed": False,
            "docking_score_accessed": False,
            "evaluation_manifest_accessed": False,
            "candidate_regeneration": False,
            "candidate_reranking": False,
        },
        "eligibility": prelock["eligibility"],
        "selection_semantics": prelock["selection_semantics"],
        "candidates": rows,
    }
    candidate_payload = {
        "schema_version": CANDIDATE_SCHEMA,
        **common,
        "selection_semantics": prelock["selection_semantics"],
        "memberships": selections,
    }
    candidate_payload_sha256 = identity(candidate_payload)
    request_payload = {
        "schema_version": REQUEST_SCHEMA,
        **common,
        "candidate_lock_payload_sha256": candidate_payload_sha256,
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
        "call_ceiling_if_later_authorized": len(requests),
        "maximum_permitted_call_ceiling": 20,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "interpretation_limit": prelock["interpretation_limit"],
        "requests": requests,
    }
    abstention_payload = {
        "schema_version": ABSTENTION_SCHEMA,
        **common,
        "candidate_lock_payload_sha256": candidate_payload_sha256,
        "cell_arm_units": sorted(
            arm_statuses, key=lambda row: (row["cell"], row["arm_id"])
        ),
    }
    exclusion_counts = Counter(
        reason for row in rows for reason in row["final_exclusion_reasons"]
    )
    summary = {
        "declared_cells": 5,
        "declared_arms": 4,
        "declared_cell_arm_units": 20,
        "candidate_rows": len(rows),
        "exact_replay_candidates": sum(row["exact_action_replay"] for row in rows),
        "valid_candidates": sum(row["valid"] for row in rows),
        "connected_candidates": sum(row["connected"] for row in rows),
        "eligible_candidates": sum(row["eligible"] for row in rows),
        "eligible_unique_cell_arm_endpoints": len(
            {
                (row["cell"], row["arm_id"], row["canonical_smiles"])
                for row in rows
                if row["eligible"]
            }
        ),
        "selected_memberships": len(selections),
        "unique_future_requests": len(requests),
        "cell_arm_abstentions": sum(
            row["status"].startswith("abstain") for row in arm_statuses
        ),
        "merged_memberships": len(selections) - len(requests),
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
    }
    result_payload = {
        "schema_version": RESULT_SCHEMA,
        **common,
        "evidence": "computed zero-oracle score-blind endpoint eligibility and prospective request selection",
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "precision": prelock["environment"]["numeric_precision"],
            "randomness": "none",
        },
        "summary": summary,
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "cell_arm_units": sorted(
            arm_statuses, key=lambda row: (row["cell"], row["arm_id"])
        ),
        "interpretation": {
            "computed": "The two frozen generator candidate locks were screened using the preregistered task-score-blind rule.",
            "not_measured": "No docking utility, teacher recovery, route recovery, optimization improvement or IVG comparison was evaluated.",
            "later_launch_authority": prelock["authority"][
                "later_authorization_required"
            ],
        },
    }
    return {
        "eligibility_ledger.json.gz": ledger_payload,
        "candidate_lock.json": candidate_payload,
        "request_lock.json": request_payload,
        "abstention_ledger.json": abstention_payload,
        "result.json": result_payload,
    }


def _sealed_bytes(payload: dict, *, compressed: bool) -> bytes:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    raw = canonical_json(envelope) + b"\n"
    return gzip.compress(raw, mtime=0) if compressed else raw


def _readme(result: dict, physical: dict[str, str], request_payload_sha256: str) -> str:
    summary = result["summary"]
    return f"""# T4 compositional-generator prospective utility lock

## Outcome

The zero-oracle score-blind lock audited {summary['candidate_rows']:,} candidates
from 20 predeclared cell-arm units. It selected
{summary['selected_memberships']} memberships and merged them into
{summary['unique_future_requests']} unique prospective requests. There were
{summary['cell_arm_abstentions']} cell-arm abstentions and
{summary['merged_memberships']} cross-arm merged memberships.

No oracle, docking function, Modal job or live run was accessed. The request lock
is not launch authority. No teacher endpoint, teacher transformation, teacher
metric, task score, docking score or evaluation manifest entered selection.

## Artifacts

- `eligibility_ledger.json.gz`: every in-scope candidate, exact states/actions,
  recomputed descriptors, eligibility and selection exclusions
- `candidate_lock.json`: at most one selected membership per cell/arm
- `request_lock.json`: canonical requests with all arm memberships preserved
- `abstention_ledger.json`: all 20 predeclared cell-arm units
- `result.json`: aggregate and cell-arm findings

Request-lock payload SHA-256: `{request_payload_sha256}`

Physical SHA-256 values at publication:

{os.linesep.join(f'- `{name}`: `{digest}`' for name, digest in sorted(physical.items()))}

## Interpretation and launch boundary

This lock establishes only prospective, score-blind endpoint eligibility and
coverage. It does not establish endpoint utility, teacher or route recovery,
autonomous optimization, or an InVirtuoGen comparison. Any later score run needs
explicit authorization naming the exact physical and payload SHA-256 of
`request_lock.json`, its exact call ceiling, and zero retry, replacement or
backfill.
"""


def publish_bundle(output: Path, payloads: dict[str, dict]) -> dict[str, str]:
    if output.exists():
        raise ValueError(f"refusing to overwrite utility-lock output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"stale utility-lock temporary directory exists: {temporary}")
    temporary.mkdir()
    primary = [
        "eligibility_ledger.json.gz",
        "candidate_lock.json",
        "request_lock.json",
        "abstention_ledger.json",
    ]
    for name in primary:
        (temporary / name).write_bytes(
            _sealed_bytes(payloads[name], compressed=name.endswith(".gz"))
        )
    physical = {name: sha256_file(temporary / name) for name in primary}
    result = {
        **payloads["result.json"],
        "output_artifacts": {
            name: {
                "sha256": physical[name],
                "payload_sha256": identity(payloads[name]),
            }
            for name in primary
        },
    }
    (temporary / "result.json").write_bytes(_sealed_bytes(result, compressed=False))
    physical["result.json"] = sha256_file(temporary / "result.json")
    (temporary / "README.md").write_text(
        _readme(result, physical, identity(payloads["request_lock.json"]))
    )
    physical["README.md"] = sha256_file(temporary / "README.md")
    temporary.replace(output)
    return dict(sorted(physical.items()))


def run(
    root: Path,
    prelock_path: Path,
    binding_path: Path,
    output: Path,
) -> dict[str, str]:
    if _source_dirty(root):
        raise ValueError("utility lock requires a clean committed source worktree")
    payloads = build_payloads(root, prelock_path, binding_path)
    return publish_bundle(output, payloads)
