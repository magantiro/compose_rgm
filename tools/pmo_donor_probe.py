"""One fixed, executor-replayed donor batch using an existing prescreen bank."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase

from compose_v4.chem.molecular_graph import MolecularGraphError, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.donor_program import PendantCut, compile_transplant, pendant_cuts
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_macro_probe import make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp
from compose_v4.experiments.winner_paths import PathConfig
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT = Path(__file__).resolve().parents[1]


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f"restart changed {path}")
    else:
        publish_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serialization-repair-from", type=str)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw output must be outside the source worktree")
    if rdBase.rdkitVersion != "2024.03.5" or importlib.metadata.version("PyTDC") != "0.3.6":
        raise ValueError("probe requires pinned RDKit 2024.03.5 and PyTDC 0.3.6")
    sources = [
        "tools/pmo_donor_probe.py",
        "docs/PMO_DONOR_PROGRAM.md",
        "src/compose_v4/control/donor_program.py",
        "src/compose_v4/experiments/winner_paths.py",
        "src/compose_v4/experiments/pmo_macro_probe.py",
    ]
    sources += [
        str(p.relative_to(ROOT))
        for folder in ("chem", "rewrite", "data")
        for p in sorted((ROOT / f"src/compose_v4/{folder}").glob("*.py"))
    ]
    cfg = {
        "schema_version": "donor_probe_v1",
        "task": "perindopril_mpo",
        "seed": 20260925,
        "attempts": 128,
        "new_endpoint_limit": 128,
        "parity_repeat_limit": 100,
        "time_limit_seconds": 300,
        "compiler": asdict(PathConfig()),
        "bank": {"path": str(args.bank.resolve()), "sha256": sha256_file(args.bank)},
        "implementation_sha256": {p: sha256_file(ROOT / p) for p in sorted(sources)},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "software": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "PyTDC")},
        "rdkit": rdBase.rdkitVersion,
        "python": platform.python_version(),
        "hardware": {
            "platform": platform.platform(),
            "device": "cpu",
            "workers": 1,
            "precision": "integer molecular arrays; float64 oracle and reductions",
        },
        "role": "exposed prescreen development; optional executor-supported channel only",
        "winner_donors": False,
        "reference_probability_certified": False,
        "split": "existing development bank, no held-out claim",
    }
    tdc_root = Path(importlib.util.find_spec("tdc").origin).parent
    cfg["oracle_implementation_sha256"] = {
        str(p.relative_to(tdc_root)): sha256_file(p) for p in sorted(tdc_root.rglob("*.py"))
    }
    configuration_path = args.output / "configuration.json"
    if args.serialization_repair_from:
        if sha256_file(configuration_path) != args.serialization_repair_from:
            raise ValueError("serialization repair origin hash mismatch")
        old = json.loads(configuration_path.read_text())
        variable = {"implementation_sha256", "worktree_status"}
        if {k: v for k, v in old.items() if k not in variable} != {
            k: v for k, v in cfg.items() if k not in variable
        }:
            raise ValueError("serialization repair changed scientific configuration")
        changed = [
            p
            for p, h in old["implementation_sha256"].items()
            if cfg["implementation_sha256"][p] != h
        ]
        if set(changed) != {"src/compose_v4/control/donor_program.py", "tools/pmo_donor_probe.py"}:
            raise ValueError("serialization repair modified an unexpected scientific dependency")
        frozen(
            args.output / "serialization_repair.json",
            {
                "original_configuration_sha256": args.serialization_repair_from,
                "new_configuration": cfg,
                "change": "Convert NumPy failure residual to Python int; no search, score, draw or successful-path change",
                "reused_files": {
                    str(p.relative_to(args.output)): sha256_file(p)
                    for folder in ("attempts", "oracle")
                    for p in sorted((args.output / folder).rglob("*.json"))
                },
            },
        )
    else:
        frozen(configuration_path, cfg)
    if (args.output / "result.json").exists():
        print((args.output / "result.json").read_text())
        return
    started = perf_counter()
    bank = json.loads(args.bank.read_text())
    parents, excluded, seen = [], [], set()
    for i, row in enumerate(bank["bank"]):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"invalid historical molecule {i}")
        Chem.RemoveStereochemistry(mol)
        smiles = Chem.MolToSmiles(mol)
        if not 1 <= mol.GetNumHeavyAtoms() <= 40:
            excluded.append({"index": i, "reason": "unsupported_size", "smiles": smiles})
            continue
        try:
            graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
        except MolecularGraphError as exc:
            excluded.append(
                {"index": i, "reason": "unsupported_representation", "detail": str(exc)}
            )
            continue
        if canonical_state_key(graph) != smiles:
            raise ValueError(f"unexpected 2D state conversion at row {i}")
        if smiles in seen:
            excluded.append({"index": i, "reason": "duplicate_2d", "smiles": smiles})
            continue
        seen.add(smiles)
        parents.append(
            {
                "source_index": i,
                "smiles": smiles,
                "historical_score": row["u"],
                "state": encode_state(graph),
                "cuts": [asdict(c) for c in pendant_cuts(graph)],
            }
        )
    if any(not p["cuts"] for p in parents) or len(parents) < 2:
        raise ValueError(
            "initial panel includes a molecule without a pendant cut or fewer than two parents"
        )
    rng = np.random.default_rng(cfg["seed"])
    jobs = []
    for index in range(cfg["attempts"]):
        i, j = (int(k) for k in rng.choice(len(parents), size=2, replace=False))
        jobs.append(
            {
                "index": index,
                "parent": i,
                "donor": j,
                "parent_cut": parents[i]["cuts"][int(rng.integers(len(parents[i]["cuts"])))],
                "donor_cut": parents[j]["cuts"][int(rng.integers(len(parents[j]["cuts"])))],
            }
        )
    frozen(args.output / "batch.json", {"parents": parents, "exclusions": excluded, "jobs": jobs})
    oracle = make_oracle(cfg["task"], ROOT, {})

    def evaluate(smiles, role, number):
        path = args.output / "oracle" / role / f"{number:04}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["status"] != "complete" or result["smiles"] != smiles:
                raise ValueError(f"ambiguous or changed oracle attempt: {path}")
            return result["score"], result["seconds"]
        publish_json(path, {"status": "started", "smiles": smiles, "role": role, "at": _stamp()})
        t = perf_counter()
        score = oracle(smiles)
        if not np.isfinite(score):
            raise ValueError(f"nonfinite oracle result for {smiles}")
        seconds = perf_counter() - t
        publish_json(
            path,
            {
                "status": "complete",
                "smiles": smiles,
                "score": score,
                "seconds": seconds,
                "role": role,
                "at": _stamp(),
            },
        )
        return score, seconds

    observed, parity = {}, []
    for i, parent in enumerate(parents):
        score, seconds = evaluate(parent["smiles"], "historical_parity", i)
        if abs(score - parent["historical_score"]) > 1e-12:
            raise ValueError(
                f"historical score mismatch at source {parent['source_index']}: {score} vs {parent['historical_score']}"
            )
        observed[parent["smiles"]] = score
        parity.append({"parent": i, "score": score, "seconds": seconds})
    initial = dict(observed)
    attempts, new_rows = [], []
    graphs = [decode_state(p["state"]) for p in parents]
    for job in jobs:
        if perf_counter() - started >= cfg["time_limit_seconds"]:
            break
        path = args.output / "attempts" / f"{job['index']:04}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            t = perf_counter()
            cuts = [
                PendantCut(c["anchor"], c["root"], tuple(c["component"]))
                for c in (job["parent_cut"], job["donor_cut"])
            ]
            result = compile_transplant(graphs[job["parent"]], graphs[job["donor"]], *cuts)
            record = {**job, "result": result, "proposal_seconds": perf_counter() - t}
            publish_json(path, record)
        result = record["result"]
        if result["status"] == "compiled":
            smiles = result["smiles"]
            if smiles not in observed:
                score, seconds = evaluate(smiles, "new_endpoint", len(new_rows))
                observed[smiles] = score
                new_rows.append(
                    {"smiles": smiles, "score": score, "attempt": job["index"], "seconds": seconds}
                )
            record = {**record, "score": observed[smiles]}
        attempts.append(
            {k: v for k, v in record.items() if k != "result"}
            | {
                "status": result["status"],
                "smiles": result.get("smiles"),
                "primitive_steps": result.get("primitive_steps"),
                "released_fraction": result.get("released_fraction"),
                "removed_atoms": result.get("removed_atoms"),
                "added_atoms": result.get("added_atoms"),
            }
        )
        print(
            json.dumps(
                {
                    "completed": len(attempts),
                    "planned": len(jobs),
                    "status": result["status"],
                    "new_calls": len(new_rows),
                    "best": max(observed.values()),
                    "seconds": round(perf_counter() - started, 3),
                }
            ),
            flush=True,
        )
    ordered = sorted(observed.items(), key=lambda x: (-x[1], x[0]))
    report = {
        "schema_version": "donor_probe_result_v1",
        "configuration": cfg,
        "configuration_sha256": identity(cfg),
        "batch_sha256": sha256_file(args.output / "batch.json"),
        "status": "complete_development"
        if len(attempts) == len(jobs)
        else "time_limited_inconclusive",
        "at": _stamp(),
        "initial_best": max(initial.values()),
        "best": ordered[0][1],
        "initial_top10_mean": float(np.mean(sorted(initial.values(), reverse=True)[:10])),
        "top10_mean": float(np.mean([v for _, v in ordered[:10]])),
        "historical_prescreen_reported_calls": bank["uncounted_calls"],
        "historical_provenance_limit": "legacy bank lacks complete modern producer/input chain",
        "initial_supported_unique": len(parents),
        "initial_exclusions": excluded,
        "historical_parity_repeat_calls": len(parity),
        "new_endpoint_calls": len(new_rows),
        "physical_calls_this_probe": len(parity) + len(new_rows),
        "attempted": len(attempts),
        "status_counts": dict(Counter(r["status"] for r in attempts)),
        "attempts": attempts,
        "new_rows": new_rows,
        "top10": ordered[:10],
        "parity": parity,
        "seconds": perf_counter() - started,
        "proposal_seconds": sum(r["proposal_seconds"] for r in attempts),
        "oracle_seconds": sum(r["seconds"] for r in parity + new_rows),
        "precision": "every reported new endpoint is executor-replayed; learned-law support unverified",
        "decision": "integrate_and_compare"
        if ordered[0][1] > max(initial.values())
        else "do_not_extend_unchanged",
    }
    publish_json(args.output / "result.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in ("configuration", "attempts", "new_rows", "parity", "initial_exclusions")
            }
        )
    )


if __name__ == "__main__":
    main()
