"""Prepare and fit a small winner-imitation ranker; reconstruct without a cursor."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.winner_imitation import ImitationRanker, features, proposals, rollout
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.winner_paths import replay
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/pmo_winner_imitation"
SEED = 20260919


def sources():
    directory = ROOT / "diagnostics/pmo_public_winner_recovery"
    manifest = json.loads((directory / "result.json").read_text())
    rows, hashes = [], {str(directory / "result.json"): sha256_file(directory / "result.json")}
    for item in manifest["paths"]:
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"demonstration hash mismatch: {path}")
        hashes[str(path)] = item["sha256"]
        r = json.loads(path.read_text())["result"]
        attempt = next(a for a in r["attempts"] if a["status"] == "witness_found")
        source = decode_state(r["source_state"])
        matched = {a for a, _ in attempt["mapping"]["persistent_slot_mapping"]}
        pending = frozenset(int(i) for i in np.flatnonzero(is_element(source.atom_types))) - matched
        rows.append({"id": item["source_id"], "route": r, "pending": pending})
    return rows, hashes


def prepare():
    start = perf_counter()
    rows, hashes = sources()
    system = editing_v2_semantic_rewrite_system()
    tensors, labels, metadata = [], [], []
    for row in rows:
        r, pending = row["route"], row["pending"]
        goal = decode_state(r["states"][-1])
        for index, mark in enumerate(r["actions"]):
            state = decode_state(r["states"][index])
            target = decode_state(r["states"][index + 1])
            family, action = decode_action(mark)
            expected_pending = pending - {action.v} if family == "atom_delete" else pending
            candidates = proposals(state, goal, pending, system)
            matches = [
                j
                for j, c in enumerate(candidates)
                if exact_graph_key(c.graph) == exact_graph_key(target)
                and c.pending == expected_pending
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"teacher absent or ambiguous: {row['id']} step {index}: {len(matches)}"
                )
            tensors.append(torch.from_numpy(features(state, goal, pending, candidates)))
            labels.append(matches[0])
            metadata.append(
                {
                    "source": row["id"],
                    "step": index,
                    "family": family,
                    "candidates": len(candidates),
                }
            )
            pending = expected_pending
        print(f"[imitation] prepared {row['id']}: {len(r['actions'])} rows", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Numerical artifact generation, never a source-state reconstruction from SMILES.
    torch.save({"features": tensors, "labels": labels, "metadata": metadata}, OUT / "panel.pt")
    config = {
        "schema_version": "winner_imitation_v1",
        "seed": SEED,
        "hidden": 128,
        "learning_rate": 0.003,
        "batch_rows": 16,
        "max_updates": 600,
        "fit_seconds_limit": 120,
        "check_interval": 50,
        "max_rollout_steps": 64,
        "input_sha256": hashes,
        "panel_sha256": sha256_file(OUT / "panel.pt"),
        "protocol_sha256": sha256_file(ROOT / "docs/PMO_WINNER_IMITATION.md"),
        "role": "all five routes are training/reconstruction; no held-out evaluation",
        "source_count": len(rows),
        "rows": len(tensors),
        "family_counts": dict(Counter(r["family"] for r in metadata)),
        "prepare_seconds": perf_counter() - start,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    publish_json(OUT / "config.json", config)
    print(json.dumps(config), flush=True)


def train():
    start = perf_counter()
    cfg = json.loads((OUT / "config.json").read_text())
    rows, hashes = sources()
    if hashes != cfg["input_sha256"] or sha256_file(OUT / "panel.pt") != cfg["panel_sha256"]:
        raise ValueError("prepared training inputs changed")
    if sha256_file(ROOT / "docs/PMO_WINNER_IMITATION.md") != cfg["protocol_sha256"]:
        raise ValueError("imitation protocol changed after preparation")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    panel = torch.load(OUT / "panel.pt", weights_only=False, map_location="cpu")
    xs, ys, meta = panel["features"], panel["labels"], panel["metadata"]
    model = ImitationRanker(xs[0].shape[1], cfg["hidden"])
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    groups = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        groups[m["family"]][m["source"]].append(i)
    families = sorted(groups)

    def audit():
        with torch.inference_mode():
            logits = [model(x) for x in xs]
            ranks = [int((v > v[y]).sum()) + 1 for v, y in zip(logits, ys, strict=True)]
            correct = [int(v.argmax()) == y for v, y in zip(logits, ys, strict=True)]
        return {
            "correct": sum(correct),
            "total": len(ranks),
            "ranks": ranks,
            "by_family": {
                f: {
                    "correct": sum(correct[i] for i, m in enumerate(meta) if m["family"] == f),
                    "total": sum(m["family"] == f for m in meta),
                }
                for f in families
            },
        }

    initial_audit = audit()
    fit_start, checkpoints = perf_counter(), []
    for update in range(1, cfg["max_updates"] + 1):
        indices = []
        for _ in range(cfg["batch_rows"]):
            family = families[int(rng.integers(len(families)))]
            route_names = sorted(groups[family])
            ids = groups[family][route_names[int(rng.integers(len(route_names)))]]
            indices.append(ids[int(rng.integers(len(ids)))])
        sizes = [len(xs[i]) for i in indices]
        logits = model(torch.cat([xs[i] for i in indices])).split(sizes)
        loss = torch.stack(
            [
                torch.nn.functional.cross_entropy(v[None], torch.tensor([ys[i]]))
                for v, i in zip(logits, indices, strict=True)
            ]
        ).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite imitation loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % cfg["check_interval"] == 0:
            measured = audit()
            checkpoints.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "correct": measured["correct"],
                    "seconds": perf_counter() - fit_start,
                }
            )
            print(
                f"[imitation] update={update} correct={measured['correct']}/{len(xs)} seconds={perf_counter() - fit_start:.1f}",
                flush=True,
            )
            if (
                measured["correct"] == len(xs)
                or perf_counter() - fit_start >= cfg["fit_seconds_limit"]
            ):
                break
    fit_seconds = perf_counter() - fit_start
    final_audit = audit()
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": rng.bit_generator.state,
            "configuration": cfg,
            "update": update,
        },
        OUT / "model.pt",
    )
    learned = {k: v.clone() for k, v in model.state_dict().items()}
    outputs = {}
    system = editing_v2_semantic_rewrite_system()
    for arm, weights in (("untrained", initial), ("learned", learned)):
        model.load_state_dict(weights)
        outputs[arm] = []
        for row in rows:
            r = row["route"]
            result = rollout(
                decode_state(r["source_state"]),
                decode_state(r["states"][-1]),
                row["pending"],
                system,
                model,
                cfg["max_rollout_steps"],
            )
            if result["status"] == "reconstructed":
                result["states"] = replay(r["source_state"], result["actions"], r["target_2d"])
            outputs[arm].append({"source": row["id"], **result})
            print(
                f"[imitation] {arm} {row['id']}: {result['status']} {len(result['actions'])} edits",
                flush=True,
            )
    result = {
        "schema_version": "winner_imitation_result_v1",
        "configuration": cfg,
        "initial_panel": initial_audit,
        "final_panel": final_audit,
        "training": checkpoints,
        "fit_seconds": fit_seconds,
        "seconds": perf_counter() - start,
        "rollouts": outputs,
        "model_sha256": sha256_file(OUT / "model.pt"),
        "new_oracle_calls": 0,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "implementation_sha256": {
            p: sha256_file(ROOT / p)
            for p in (
                "tools/pmo_winner_imitation.py",
                "src/compose_v4/control/winner_imitation.py",
                "src/compose_v4/experiments/winner_paths.py",
                "src/compose_v4/rewrite/kernel.py",
            )
        },
        "hardware": {
            "device": "cpu",
            "threads": 1,
            "precision": "float32",
            "platform": platform.platform(),
        },
        "exclusions": [],
        "interpretation": "answer-known training reconstruction with privileged goal alignment; no unknown-goal discovery, policy integration or generalization claim",
    }
    publish_json(OUT / "result.json", result)
    print(
        json.dumps(
            {
                "recovery": {
                    a: sum(r["status"] == "reconstructed" for r in rr) for a, rr in outputs.items()
                },
                "fit_seconds": fit_seconds,
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepare() if args.mode == "prepare" else train()
