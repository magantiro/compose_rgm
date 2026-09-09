"""Offline, target-informed reachability audit; no oracle or generator fitting."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import DockingValue, molecular_features
from compose_v4.control.macro_engine import MACRO_FAMILIES
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import admissible_indices
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
from compose_v4.experiments.t4_warm_continuation import exact_context
from compose_v4.experiments.winner_paths import PathConfig, find_path, replay
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CENSUS_SHA = "2c06fdfe65fc31327d865381ccb8f4d2c40553c3925d9dc0a03386c22fda05bb"
SNAPSHOT_SHA = "b3454d5105ab6b800f1665c01eb31b206ebe8dad9a1d18eda6638ace08e15de4"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def publish(path, value, *, compressed=False):
    data = canonical(value) + b"\n"
    if compressed:
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def implementation_closure(entry):
    """Conservative static local import closure, not a repository-wide cache key."""
    pending, seen = [entry], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        if path.is_relative_to(ROOT / "src"):
            package = path.relative_to(ROOT / "src").with_suffix("").parts[:-1]
        else:
            package = ()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = ".".join(package[: len(package) - node.level + 1]) if node.level else ""
                module = ".".join(x for x in (prefix, node.module or "") if x)
                modules = [module] + [f"{module}.{alias.name}" for alias in node.names]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if parts[0] != "compose_v4":
                    continue
                for end in range(1, len(parts) + 1):
                    base = ROOT / "src" / Path(*parts[:end])
                    for dependency in (base.with_suffix(".py"), base / "__init__.py"):
                        if dependency.is_file() and dependency not in seen:
                            pending.append(dependency)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(seen)}


def pairs_from_census(census, seeds):
    groups = {}
    for cell in census["cells"]:
        registered = [row for row in seeds if row["target"] == cell["target"]]
        if registered[cell["source_idx"]]["smiles"] != cell["source_smiles"]:
            raise ValueError(f"seed registry disagrees with {cell['cell']}")
        for run in cell["runs"]:
            for winner in run["winners"]:
                identity = {"source": cell["source_smiles"], "target": winner["canonical_smiles"]}
                key = digest(identity)
                pair = groups.setdefault(key, {"pair_id": key, **identity, "references": []})
                pair["references"].append(
                    {
                        "cell": cell["cell"],
                        "target": cell["target"],
                        "source_idx": cell["source_idx"],
                        "delta": cell["delta"],
                        "run_seed": run["run_seed"],
                        "reported_docking_score": run["reported_docking_score"],
                        "source_rows": winner["source_rows"],
                    }
                )
    return [groups[key] for key in sorted(groups)]


def load_snapshot(path):
    if sha(path) != SNAPSHOT_SHA:
        raise ValueError(f"{path}: frozen value snapshot input hash mismatch")
    envelope = json.loads(path.read_text())
    if digest(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"{path}: corrupt lock envelope")
    model = DockingValue.from_payload(envelope["payload"]["lock"]["value_snapshot"])
    match = all(
        canonical(molecular_features(row["smiles"])[1]) == canonical(feature)
        for row, feature in zip(
            model.payload["training_rows"], model.payload["features"], strict=True
        )
    )
    return model if match else None, {
        "snapshot_sha256": model.payload["snapshot_sha256"],
        "feature_identity_verified": match,
        "training_rows": len(model.payload["training_rows"]),
        "applicability": "docking_parp1_idx0_thr4 only; uncalibrated frozen predictor, not docking",
    }


def annotate(pair, path, model):
    if path["status"] != "witness_found":
        return None
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = gm.GetFingerprint(Chem.MolFromSmiles(pair["source"]))
    applicable = any(r["cell"] == "docking_parp1_idx0_thr4" for r in pair["references"])
    steps = []
    for index, state in enumerate(path["states"]):
        graph = decode_state(state)
        smiles = canonical_state_key(graph)
        mol = Chem.MolFromSmiles(smiles)
        qed = float(QED.qed(mol))
        sa = float(sascorer.calculateScore(mol))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gm.GetFingerprint(mol)))
        taxonomy = ring_taxonomy_report((smiles,))
        row = {
            "step": index,
            "smiles": smiles,
            "qed": qed,
            "sa": sa,
            "sim": sim,
            "heavy_atoms": graph.n_real_atoms,
            "cycle_rank": int(taxonomy["means"]["cycle_rank"]),
            "ring_systems": int(taxonomy["means"]["ring_systems"]),
            "ring_size_counts": taxonomy["ring_size_counts"],
            "state_sha256": digest(state),
            "failed_constraints_by_delta": {
                str(delta): [
                    name
                    for name, failed in (
                        ("qed", qed < 0.6),
                        ("sa", sa > 4.0),
                        ("similarity", sim < delta),
                    )
                    if failed
                ]
                for delta in sorted({r["delta"] for r in pair["references"]})
            },
        }
        if applicable and model is not None:
            row["predicted_docking"] = float(model.predict([smiles])[0])
            row["feasibility_gated_value"] = model.desirability(
                smiles, qed >= 0.6 and sa <= 4 and sim >= 0.4
            )
        else:
            row["value_abstention"] = "no applicable feature-verified frozen snapshot"
        if index < len(path["actions"]):
            family, action = decode_action(path["actions"][index])
            scopes = []
            regions = [r for r in enumerate_regions(smiles) if 1 <= len(r.atoms) <= 24]
            for region in regions:
                ctx = exact_context(graph, smiles, region)
                if admissible_indices((family,), (action,), ctx)[0]:
                    scopes.append(len(region.atoms) / graph.n_real_atoms)
            row["next_edit"] = {
                "executor_rule": family,
                "family_compatible_macros_not_application_proof": sorted(
                    name for name, families in MACRO_FAMILIES.items() if family in families
                ),
                "touch_filter_regions": len(scopes),
                "regions_enumerated": len(regions),
                "touch_filter_scope_range": [min(scopes), max(scopes)] if scopes else None,
                "learned_mark_support_and_probability": "not_evaluated",
            }
        steps.append(row)
    return steps


def run(args):
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise ValueError("run from a clean committed source tree; put results outside that tree")
    if sha(args.census) != CENSUS_SHA:
        raise ValueError(f"{args.census}: expected the complete pinned IVG census")
    census = json.loads(args.census.read_text())
    expected_seed = census["implementation_and_input_sha256"]["docs/GENMOL_T4_SEEDS.json"]
    if sha(args.seeds) != expected_seed:
        raise ValueError(f"{args.seeds}: seed registry hash mismatch")
    pairs = pairs_from_census(census, json.loads(args.seeds.read_text()))
    model, snapshot = load_snapshot(args.snapshot)
    config = PathConfig()
    implementation = implementation_closure(Path(__file__).resolve())
    software = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
    }
    # SA's external score table and implementation are part of annotation identity.
    sa_path = Path(sascorer.__file__)
    scorer_identity = {str(p): sha(p) for p in (sa_path, sa_path.parent / "fpscores.pkl.gz")}
    common = {
        "schema_version": "ivg_winner_path_audit_v1",
        "config": asdict(config),
        "census_sha256": CENSUS_SHA,
        "seed_sha256": expected_seed,
        "value_input_sha256": SNAPSHOT_SHA,
        "implementation_sha256": implementation,
        "software": software,
        "sa_score_inputs": scorer_identity,
        "snapshot": snapshot,
    }
    scientific_id = digest(common)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipts, rows = [], []
    started = time.monotonic()
    for number, pair in enumerate(pairs, 1):
        cache_id = digest({"scientific_identity": scientific_id, "pair": pair})
        receipt_path = args.output / "pairs" / f"{cache_id}.json.gz"
        cached = receipt_path.exists()
        if cached:
            receipt = json.loads(gzip.decompress(receipt_path.read_bytes()))
            if receipt["scientific_identity"] != scientific_id or receipt["pair"] != pair:
                raise ValueError(f"{receipt_path}: incompatible cache entry")
            if digest(receipt["payload"]) != receipt["payload_sha256"]:
                raise ValueError(f"{receipt_path}: corrupt cache entry")
            payload = receipt["payload"]
            if payload["path"]["status"] == "witness_found":
                p = payload["path"]
                if replay(p["source_state"], p["actions"], p["target_2d"]) != p["states"]:
                    raise ValueError(f"{receipt_path}: cached exact replay mismatch")
        else:
            clock = time.monotonic()
            path = find_path(pair["source"], pair["target"], config)
            search_seconds = time.monotonic() - clock
            annotations = annotate(pair, path, model)
            payload = {
                "path": path,
                "annotations": annotations,
                "search_seconds": search_seconds,
                "total_seconds": time.monotonic() - clock,
            }
            receipt = {
                "scientific_identity": scientific_id,
                "pair": pair,
                "code_revision": revision,
                "payload": payload,
                "payload_sha256": digest(payload),
            }
            publish(receipt_path, receipt, compressed=True)
        p = payload["path"]
        rows.append(
            {
                **pair,
                **{
                    k: p[k]
                    for k in (
                        "status",
                        "primitive_lower_bound",
                        "witness_steps",
                        "within_16_edit_witness",
                        "source_stats",
                        "target_stats",
                    )
                    if k in p
                },
                "search_seconds": payload["search_seconds"],
                "total_seconds": payload["total_seconds"],
                "receipt": str(receipt_path.relative_to(args.output)),
                "receipt_sha256": sha(receipt_path),
            }
        )
        receipts.append({"pair_id": pair["pair_id"], "cache_hit": cached})
        print(
            f"{number}/{len(pairs)} {pair['references'][0]['cell']} {p['status']} "
            f"steps={p.get('witness_steps')} search={payload['search_seconds']:.2f}s "
            f"total={payload['total_seconds']:.2f}s cached={cached}",
            flush=True,
        )
    report = {
        **common,
        "scientific_identity": scientific_id,
        "producer_revision": revision,
        "input_paths": {
            k: str(getattr(args, k).resolve()) for k in ("census", "seeds", "snapshot")
        },
        "access_basis_and_upstream": {
            k: census[k] for k in ("access_basis", "upstream_revision", "upstream_sources")
        },
        "configuration": {
            "random_seed": None,
            "seed_reason": "deterministic ordered graph search; wall-time-limited MCS may vary",
            "workers": 1,
            "dtype": "integer graph states; float64 diagnostic scores",
            "new_oracle_calls": 0,
            "r_theta_enumerations": 0,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "accelerator": "none",
        },
        "split_identity": "all inspected development cells, no training or held-out inference",
        "evidence": "target-informed locally replayed upper-bound witnesses, not blind recovery or shortest paths",
        "limitations": [
            "RDKit differs from pinned Modal runtime; remote replay not performed",
            "only two bounded MCS mappings; failed search is unresolved",
            "touch-filter and family compatibility do not prove Q(M,o) or learned mark support",
            "fresh source initialization uses 2D projection; no stereo or formal-charge-change claim",
            "task-value predictions are uncalibrated, not new docking observations",
        ],
        "pairs": rows,
        "counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        "n_pairs": len(rows),
        "n_cell_run_winners": sum(len(r["references"]) for r in rows),
        "execution": {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds_this_invocation": time.monotonic() - started,
            "cache": receipts,
        },
    }
    publish(args.output / "audit.json", report)
    print(json.dumps(report["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--census", type=Path, default=ROOT / "diagnostics/ivg_t4_census/census.json"
    )
    parser.add_argument("--seeds", type=Path, default=ROOT / "docs/GENMOL_T4_SEEDS.json")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "diagnostics/t4_lazy_reference_probe/cache/candidate_lock.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/ivg_winner_paths")
    RDLogger.DisableLog("rdApp.warning")
    run(parser.parse_args())
