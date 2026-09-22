"""Realized SHAPE of what a role-based rebinding produces, on the pairs that bind.

A binding RATE that rises while the realized scale, direction and retained fraction do
not has not repaired the located defect.  This recovers the MOLECULES -- endpoint SMILES,
ring and heavy-atom deltas against the parent, retained fraction, realized primitive
count -- so the question "would repairing this lane produce macros of the required
character?" can be answered from the artifact rather than re-derived.

Run at a NON-BINDING seconds cap so every number depends on the node budget and the
deterministic depth-first order alone, never on machine load.

ZERO oracle calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import rdkit

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control import pmo_realization as PR
from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_jump_binding_autopsy import (
    role_based_specification,
    structural_delta,
)
from compose_v4.rewrite.trace_shard import decode_state

JUMP = "joint_dependency_region_jump"
CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
NON_BINDING_CAP = 600.0
# The corrected engineering target: coherent options of 12-18 primitives, capacity ~20.
TARGET_BAND = (12, 18)


def _rings(smiles: str | None) -> int | None:
    if not smiles:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetRingInfo().NumRings()


def _band(count: int) -> str:
    if count < TARGET_BAND[0]:
        return "below_12"
    if count <= TARGET_BAND[1]:
        return "in_12_to_18_target"
    if count <= 20:
        return "19_to_20"
    return "above_20"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--probe", required=True, help="role_rebinding_rows_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    snapshot = json.loads(Path(args.result).read_text())["campaign"]["snapshot"]
    payload = json.loads((repo / CHECKPOINTS).read_text())["payload"]["checkpoints"][
        "shared_all_routes"
    ]
    if identity(payload) != snapshot["pmo_population"]["jump_checkpoint_id"]:
        raise SystemExit("jump checkpoint is not the run's checkpoint")
    plans = {plan["plan_id"]: plan for plan in payload["plan_latents"]}
    entries = snapshot["entries"]
    probe = json.loads(Path(args.probe).read_text())["rows"]
    # Any pair either arm bound, or whose search was time-limited and so could not be
    # proven either way at the production cap.
    # Only pairs that actually BOUND carry shape; an exhausted search produces no
    # realization and therefore no region to measure.  The count of exhausted searches is
    # reported by the probe instead, as the reason its binding rate is a LOWER BOUND.
    interesting = [
        row for row in probe if any(arm["bindings"] for arm in row["arms"].values())
    ]
    role_spec = role_based_specification()
    out_rows = []
    for row in interesting:
        entry = entries[row["entry_id"]]
        source = decode_state(entry["trace"]["states"][-1])
        plan = plans[row["plan_id"]]
        parent_smiles = molecular_graph_to_smiles(source)
        parent_rings = _rings(parent_smiles)
        for arm_name, spec in (
            ("v1_exact_production", PR.SPEC_V1_EXACT),
            ("role_based", role_spec),
        ):
            result = PR.realize_plan_binding(
                source,
                plan,
                specification=spec,
                node_budget=PR.PRODUCTION_NODE_BUDGET,
                seconds_cap=NON_BINDING_CAP,
                max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
            )
            for binding in result["bindings"]:
                endpoint = decode_state(binding["endpoint_state"])
                smiles = molecular_graph_to_smiles(endpoint)
                rings = _rings(smiles)
                delta = structural_delta(parent_smiles, smiles, timeout=10) or {}
                out_rows.append(
                    {
                        "arm": arm_name,
                        "plan_id": row["plan_id"],
                        "entry_id": row["entry_id"],
                        "parent_smiles": parent_smiles,
                        "parent_heavy_atoms": int(source.n_real_atoms),
                        "parent_rings": parent_rings,
                        "endpoint_smiles": smiles,
                        "endpoint_key": binding["endpoint_key"],
                        "endpoint_heavy_atoms": int(endpoint.n_real_atoms),
                        "endpoint_rings": rings,
                        "heavy_atom_delta": int(endpoint.n_real_atoms - source.n_real_atoms),
                        "ring_delta": None
                        if rings is None or parent_rings is None
                        else rings - parent_rings,
                        "retained_fraction": float(PR.retained_fraction(source, endpoint)),
                        "plan_primitive_count": int(plan["primitive_count"]),
                        "realized_primitive_count": int(binding["primitive_count"]),
                        "realized_band": _band(int(binding["primitive_count"])),
                        "component_count": int(binding["component_count"]),
                        "created_dependency_edges": int(binding["created_dependency_edges"]),
                        "exact_replay": bool(binding["exact_replay"]),
                        # PRIMARY metrics: the axes the structural census made primary.
                        "largest_changed_region": delta.get("largest_changed_region"),
                        "n_changed_regions": delta.get("n_changed_regions"),
                        "total_changed": delta.get("total_changed"),
                        "retained_fraction_mcs": delta.get("retained_fraction_mcs"),
                        "retained_core": delta.get("retained_core"),
                    }
                )
        print(f"{len(out_rows)} realizations so far", flush=True)

    def reduce(arm_name):
        sub = [row for row in out_rows if row["arm"] == arm_name]
        if not sub:
            return {"realizations": 0}
        return {
            "realizations": len(sub),
            "distinct_pairs": len({(r["plan_id"], r["entry_id"]) for r in sub}),
            "distinct_endpoints": len({r["endpoint_smiles"] for r in sub}),
            "realized_band": dict(Counter(r["realized_band"] for r in sub).most_common()),
            "in_target_band_fraction": sum(
                1 for r in sub if r["realized_band"] == "in_12_to_18_target"
            )
            / len(sub),
            "realized_primitive_count_median": statistics.median(
                r["realized_primitive_count"] for r in sub
            ),
            "heavy_atom_delta_median": statistics.median(r["heavy_atom_delta"] for r in sub),
            "heavy_atom_delta_min": min(r["heavy_atom_delta"] for r in sub),
            "heavy_atom_delta_max": max(r["heavy_atom_delta"] for r in sub),
            "ring_delta_distribution": dict(
                Counter(r["ring_delta"] for r in sub).most_common()
            ),
            "ring_delta_nonzero_fraction": sum(1 for r in sub if r["ring_delta"]) / len(sub),
            "retained_fraction_median": statistics.median(r["retained_fraction"] for r in sub),
            "retained_fraction_min": min(r["retained_fraction"] for r in sub),
            "purely_additive_fraction": sum(
                1 for r in sub if r["retained_fraction"] >= 0.9999
            )
            / len(sub),
            "both_excises_and_installs_fraction": sum(
                1 for r in sub if r["retained_fraction"] < 0.9999 and r["heavy_atom_delta"] > 0
            )
            / len(sub),
            "exact_replay_all": all(r["exact_replay"] for r in sub),
            # PRIMARY: the generic-sampler baseline to beat is largest_changed_region 7
            # at retained_fraction 0.64; the answer-known required macro sits at 19 and
            # 0.30.  Those two reference values are DIAGNOSTIC EVIDENCE about the missing
            # structural scale and are never encoded in any mechanism.
            "PRIMARY_largest_changed_region_median": statistics.median(
                r["largest_changed_region"] for r in sub if r["largest_changed_region"] is not None
            ),
            "PRIMARY_largest_changed_region_max": max(
                (r["largest_changed_region"] for r in sub if r["largest_changed_region"] is not None),
                default=None,
            ),
            "PRIMARY_retained_fraction_mcs_median": statistics.median(
                r["retained_fraction_mcs"] for r in sub if r["retained_fraction_mcs"] is not None
            ),
            "n_changed_regions_median": statistics.median(
                r["n_changed_regions"] for r in sub if r["n_changed_regions"] is not None
            ),
            "d_heavy_median": statistics.median(r["heavy_atom_delta"] for r in sub),
            "reaches_region_15_or_more": sum(
                1 for r in sub if (r["largest_changed_region"] or 0) >= 15
            ),
        }

    summary = {
        "schema_version": "pmo_jump_rebinding_shape_v1",
        "answer_known_offline_diagnostic": True,
        "new_oracle_calls": 0,
        "rdkit_version": rdkit.__version__,
        "seconds_cap": NON_BINDING_CAP,
        "node_budget": PR.PRODUCTION_NODE_BUDGET,
        "target_band": list(TARGET_BAND),
        "reference_values_are_diagnostic_not_encoded": {
            "generic_sampler_baseline": {"largest_changed_region": 7, "retained_fraction": 0.64},
            "answer_known_required_macro": {"largest_changed_region": 19, "retained_fraction": 0.30},
            "note": "answer-known offline diagnostic evidence about the missing structural "
                    "scale; never a value a production mechanism may encode",
        },
        "pairs_examined": len(interesting),
        "arms": {
            name: reduce(name) for name in ("v1_exact_production", "role_based")
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rebinding_shape_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (out / "rebinding_shape_molecules_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "pmo_jump_rebinding_shape_v1_molecules",
                "answer_known_offline_diagnostic": True,
                "rows": out_rows,
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
