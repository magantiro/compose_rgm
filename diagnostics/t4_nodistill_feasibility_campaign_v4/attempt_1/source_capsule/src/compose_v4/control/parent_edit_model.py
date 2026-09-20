"""Small completed-program utility ensemble and auditable query allocation.

An endpoint predictor conditioned on parent and edit, NOT a future-value head.
Bootstrap disagreement is explicitly uncalibrated and never used as optimism.
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import solve

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram, environment
from compose_v4.control.edit_program_graph import compile_program_graph, program_size_profile
from compose_v4.control.program_task import predicted_archive_gains
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

RECIPE = {
    "schema_version": "parent_edit_utility_v1",
    "ridge": 1.0,
    "members": 5,
    "seed": 20260912,
    "fingerprint_bits": 512,
    "attachment_hash_bits": 128,
    "target": "measured_completed_oriented_utility",
    "weighting": "equal_endpoint_mass_then_equal_program_representation",
    "uncertainty": "uncalibrated_bootstrap_disagreement_not_used_for_optimism",
    "qualification": "retrospective_development_only",
}


MUTATION_RECIPE = {
    **RECIPE,
    "schema_version": "parent_edit_utility_v2_mutation_context",
    "features": "constructor_plus_selected_parent_delta_and_mutation_events",
    "seed": 20260913,
}


def _unit(vector):
    array = np.asarray(vector, dtype=float)
    return array / max(float(np.linalg.norm(array)), 1)


@dataclass
class ParentEditFeatures:
    fingerprints: dict = field(default_factory=dict)
    graphs: dict = field(default_factory=dict)

    def _graph(self, state):
        key = identity(state)
        if key not in self.graphs:
            self.graphs[key] = decode_state(state)
        return self.graphs[key]

    def _fp(self, smiles):
        if smiles not in self.fingerprints:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"invalid parent/edit fingerprint molecule: {smiles}")
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=RECIPE["fingerprint_bits"]
            )
            self.fingerprints[smiles] = generator.GetFingerprintAsNumPy(molecule).astype(float)
        return self.fingerprints[smiles]

    def __call__(self, candidate, *, parent_state, parent_scores=()):
        parent = self._graph(parent_state)
        source = self._graph(candidate["source_state"])
        program = EditProgram.from_payload(candidate["program"])
        size = program_size_profile(compile_program_graph(program), source.n_real_atoms)
        p = self._fp(canonical_state_key(parent))
        child = self._fp(candidate["endpoint"])
        s = self._fp(canonical_state_key(source))
        bags = np.zeros(RECIPE["attachment_hash_bits"])
        for slot in candidate["assignment"]:
            # Binding is against the constructor source, which may differ from
            # the measured parent. Both graphs are represented explicitly.
            bags[
                int(identity({"environment": environment(source, slot)})[:16], 16) % len(bags)
            ] += 1
        families = Counter(json.loads(mark)["executor_rule"] for mark in program.marks)
        for family, count in families.items():
            bags[int(identity({"operation": family})[:16], 16) % len(bags)] += count
        scores = np.asarray(parent_scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite measured parent observations")
        changes = candidate["trace"]["actual_changes"]
        properties = candidate.get("provenance", {}).get("properties", {})
        numeric = np.array(
            [
                parent.n_real_atoms / 40,
                source.n_real_atoms / 40,
                size["final_heavy_atoms"] / 40,
                size["peak_heavy_atoms"] / 40,
                size["delta_heavy_atoms"] / 40,
                len(program.marks) / 32,
                len(program.blocks) / 8,
                changes["changed_site_count"] / 4,
                changes["deleted_original_atoms"] / 40,
                changes["surviving_new_atoms"] / 40,
                float(len(scores) > 0),
                float(np.mean(scores)) / 20 if len(scores) else 0,
                float(np.std(scores)) / 5 if len(scores) > 1 else 0,
                min(len(scores), 10) / 10,
                float("sim" in properties),
                float(properties.get("sim", 0)),
            ]
        )
        return np.concatenate(
            [_unit(p), _unit(child), _unit(child - p), _unit(s), _unit(bags), numeric]
        ).tolist()

    def with_mutation_context(
        self, candidate, *, parent_state, parent_scores=(), parent_record=None
    ):
        """Parent/candidate graph comparison, not an invented execution trace."""
        base = self(candidate, parent_state=parent_state, parent_scores=parent_scores)
        parent = self._graph(parent_state)
        product = self._graph(candidate["trace"]["states"][-1])
        p, child = self._fp(canonical_state_key(parent)), self._fp(candidate["endpoint"])
        detail = candidate.get("provenance", {}).get("metadata", {})
        events = Counter(r["kind"] for r in detail.get("mutations", []))
        names = ("attachment", "extend_segment", "contract_segment", "created_atom_type")
        comparable = parent_record is not None and len(parent_record["assignment"]) == len(
            candidate["assignment"]
        )
        attachment_changes = (
            sum(
                a != b
                for a, b in zip(parent_record["assignment"], candidate["assignment"], strict=True)
            )
            if comparable
            else 0
        )
        direct = identity(parent_state) == identity(candidate["source_state"])
        return [
            *base,
            (product.n_real_atoms - parent.n_real_atoms) / 40,
            float(np.count_nonzero((child > 0) & (p == 0))) / RECIPE["fingerprint_bits"],
            float(np.count_nonzero((p > 0) & (child == 0))) / RECIPE["fingerprint_bits"],
            *[events[name] / 2 for name in names],
            float(comparable),
            attachment_changes / 4,
            len(detail.get("removed_branches", [])) / 4,
            len(detail.get("donors", [])) / 4,
            float("current_state_edit" in detail),
            float(direct),
            len(candidate["trace"]["actions"]) / 32 if direct else 0,
        ]


class ParentEditModel:
    def __init__(self, payload):
        if payload.get("recipe") not in (RECIPE, MUTATION_RECIPE) or identity(
            {k: v for k, v in payload.items() if k != "model_sha256"}
        ) != payload.get("model_sha256"):
            raise ValueError("parent-edit model recipe or content identity mismatch")
        self.payload = payload

    @classmethod
    def fit(cls, rows, *, oracle_protocol, input_sha256, recipe=RECIPE):
        if recipe not in (RECIPE, MUTATION_RECIPE):
            raise ValueError("unknown frozen parent-edit feature/fit recipe")
        if not rows or len(input_sha256) != 64 or not oracle_protocol:
            raise ValueError("measured parent-edit rows, protocol and input identity required")
        for row in rows:
            if (
                row["oracle_protocol"] != oracle_protocol
                or not row["receipt_id"]
                or not math.isfinite(row["utility"])
            ):
                raise ValueError(
                    "only identified measured outcomes from one oracle domain may be fitted"
                )
        x = np.asarray([r["features"] for r in rows], dtype=float)
        y = np.asarray([r["utility"] for r in rows], dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("nonfinite or misaligned parent-edit features")
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["endpoint"]].append(i)
        # An endpoint's repeats are pooled before representation balancing.
        labels = {}
        for endpoint, indices in groups.items():
            observed = {}
            for i in indices:
                receipt = rows[i]["receipt_id"]
                if receipt in observed and observed[receipt] != y[i]:
                    raise ValueError("one receipt has conflicting endpoint utilities")
                observed[receipt] = y[i]
            labels[endpoint] = float(np.mean(list(observed.values())))
            y[indices] = labels[endpoint]
        rng = np.random.default_rng(recipe["seed"])
        endpoints = sorted(groups)
        kernel = x @ x.T / 6
        members = []
        for _ in range(recipe["members"]):
            counts = Counter(rng.choice(endpoints, size=len(endpoints), replace=True))
            weights = np.zeros(len(rows))
            for endpoint, indices in groups.items():
                weights[indices] = counts[endpoint] / len(indices)
            keep = np.flatnonzero(weights)
            mean = float(np.average(y[keep], weights=weights[keep]))
            alpha = solve(
                kernel[np.ix_(keep, keep)] + np.diag(recipe["ridge"] / weights[keep]),
                y[keep] - mean,
                assume_a="pos",
            )
            members.append({"indices": keep.tolist(), "mean": mean, "alpha": alpha.tolist()})
        payload = {
            "recipe": recipe,
            "oracle_protocol": oracle_protocol,
            "input_sha256": input_sha256,
            "training_receipts": sorted({r["receipt_id"] for r in rows}),
            "training_endpoints": endpoints,
            "training_features": x.tolist(),
            "members": members,
            "training_endpoint_count": len(endpoints),
            "training_row_count": len(rows),
        }
        return cls({**payload, "model_sha256": identity(payload)})

    def predict(self, features, *, oracle_protocol):
        if oracle_protocol != self.payload["oracle_protocol"]:
            raise ValueError("parent-edit model cannot mix task/oracle domains")
        x = np.asarray(features, dtype=float)
        training = np.asarray(self.payload["training_features"])
        if x.ndim != 2 or x.shape[1] != training.shape[1] or not np.isfinite(x).all():
            raise ValueError("prediction feature shape or finiteness mismatch")
        kernel = x @ training.T / 6
        predictions = np.array(
            [m["mean"] + kernel[:, m["indices"]] @ m["alpha"] for m in self.payload["members"]]
        )
        return predictions.mean(axis=0), predictions.std(axis=0)


def select_parent_edits(
    candidate_ids,
    predictions,
    archive_utilities,
    *,
    k,
    count,
    seed,
    mode="score_blind",
    exploration=0.25,
    diagnostic=False,
):
    """Same-pool selection; learned mode is fail-closed outside development."""
    if mode not in ("score_blind", "learned") or (mode == "learned" and not diagnostic):
        raise ValueError("learned selector is unqualified; explicit diagnostic mode required")
    if len(set(candidate_ids)) != len(candidate_ids) or not 1 <= count <= len(candidate_ids):
        raise ValueError("selector needs unique candidate identities and a feasible batch size")
    if not 0 < exploration <= 1:
        raise ValueError("nonzero score-blind exploration required")
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(candidate_ids),):
        raise ValueError("candidate/prediction alignment mismatch")
    gains = predicted_archive_gains(predictions, archive_utilities, k=k)
    rng = np.random.default_rng(seed)
    audit_count = count if mode == "score_blind" else max(1, math.ceil(count * exploration))
    audit = [int(i) for i in rng.choice(len(candidate_ids), size=audit_count, replace=False)]
    # Recompute top-k marginal gain after each selected predicted endpoint.
    selected, virtual = list(audit), [*archive_utilities, *(predictions[i] for i in audit)]
    while len(selected) < count:
        marginal = predicted_archive_gains(predictions, virtual, k=k)
        remaining = [i for i in range(len(candidate_ids)) if i not in selected]
        best = max(remaining, key=lambda i: (marginal[i], predictions[i], candidate_ids[i]))
        selected.append(best)
        virtual.append(predictions[best])
    return {
        "selected_ids": [candidate_ids[i] for i in selected],
        "indices": selected,
        "audit_indices": audit,
        "initial_plugin_gains": gains.tolist(),
        "mode": mode,
        "uncertainty_used": False,
        "seed": seed,
    }
