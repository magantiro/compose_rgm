#!/usr/bin/env python3
"""R1xM4 cell: masked multitask MLP over the canonical pan-lung manifest.

Tests the shared-representation hypothesis: does a single molecular trunk with
typed per-head outputs (masked loss) rescue the small lung heads that fail as
isolated single-task predictors?  Trains multitask and single-task MLPs under
identical held-lipid GroupKFolds and identical R1 features (Morgan + RDKit
descriptors), so the only difference is representation sharing.

LuT is component-only and excluded from this molecular cell (it stays in the R2
component lane).  Held-lipid grouping is global (by canonical structure), so a
lipid never leaks across train/test even between heads.

Run (module mode; the script imports the sibling baseline package):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
        python3 -m scripts.train_pan_lung_multitask_mlp
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.train_lumi_oracle_baselines import molecular_features

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "artifacts/oracles/pan_lung_canonical_v1/canonical_rows.csv"

FULL_STRUCTURE_HEADS = [
    "airway_a549_in_vitro_expression",
    "airway_hbe_in_vitro_expression",
    "airway_hbec_ali_in_vitro_expression",
    "local_intratracheal_functional_expression",
    "systemic_iv_barcoded_lung_uptake",
]


def load_rows() -> list[dict]:
    with CANONICAL.open() as handle:
        return [r for r in csv.DictReader(handle)
                if r["has_full_structure"] == "True" and r["typed_head"] in FULL_STRUCTURE_HEADS]


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    count = len(target)
    if count < 3 or np.std(prediction) == 0:
        return {"spearman": 0.0, "mae": float(np.mean(np.abs(target - prediction))),
                "top_decile_enrichment": 0.0, "n": count}
    top = max(1, int(np.ceil(0.1 * count)))
    true_top = set(np.argsort(target)[-top:].tolist())
    pred_top = set(np.argsort(prediction)[-top:].tolist())
    overlap = len(true_top & pred_top)
    prevalence = top / count
    rank = spearmanr(target, prediction).statistic
    return {
        "spearman": float(rank) if np.isfinite(rank) else 0.0,
        "mae": float(np.mean(np.abs(target - prediction))),
        "top_decile_enrichment": float((overlap / top) / prevalence),
        "n": count,
    }


class MaskedMultitaskMLP(torch.nn.Module):
    def __init__(self, in_dim: int, n_heads: int, hidden: tuple[int, int] = (256, 128),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden[0]), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(), torch.nn.Dropout(dropout),
        )
        self.heads = torch.nn.Linear(hidden[1], n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(self.trunk(x))


def train_model(
    features: np.ndarray, targets_z: np.ndarray, head_idx: np.ndarray,
    n_heads: int, active_heads: set[int], *, seed: int, epochs: int, lr: float,
) -> MaskedMultitaskMLP:
    torch.manual_seed(seed)
    model = MaskedMultitaskMLP(features.shape[1], n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(targets_z.astype(np.float32))
    h = torch.from_numpy(head_idx.astype(np.int64))
    mask = torch.zeros((len(x), n_heads), dtype=torch.float32)
    mask[torch.arange(len(x)), h] = 1.0
    y_full = torch.zeros((len(x), n_heads), dtype=torch.float32)
    y_full[torch.arange(len(x)), h] = y
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        se = ((pred - y_full) ** 2) * mask
        loss = se.sum() / mask.sum()
        loss.backward()
        optimizer.step()
    return model


def predict(model: MaskedMultitaskMLP, features: np.ndarray, head_idx: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(features.astype(np.float32))).numpy()
    return pred[np.arange(len(features)), head_idx]


def run(mode: str, features: np.ndarray, raw_targets: np.ndarray, head_idx: np.ndarray,
        groups: np.ndarray, n_heads: int, *, seed: int, epochs: int, lr: float, folds: int) -> dict:
    """mode='multitask' trains one model on all heads; 'single' trains per head."""
    oof = np.full(len(features), np.nan)
    gkf = GroupKFold(n_splits=folds)
    for train_idx, test_idx in gkf.split(features, raw_targets, groups):
        scaler = StandardScaler().fit(features[train_idx])
        f_train = scaler.transform(features[train_idx])
        f_test = scaler.transform(features[test_idx])
        # per-head target standardization on train
        z = np.zeros(len(train_idx)); centers = {}; scales = {}
        for head in range(n_heads):
            sel = head_idx[train_idx] == head
            if sel.sum() == 0:
                continue
            mu = raw_targets[train_idx][sel].mean()
            sd = raw_targets[train_idx][sel].std() or 1.0
            centers[head] = mu; scales[head] = sd
            z[sel] = (raw_targets[train_idx][sel] - mu) / sd
        if mode == "multitask":
            model = train_model(f_train, z, head_idx[train_idx], n_heads,
                                set(range(n_heads)), seed=seed, epochs=epochs, lr=lr)
            pred_z = predict(model, f_test, head_idx[test_idx])
        else:  # single-task: train a separate model per head on its own rows
            pred_z = np.zeros(len(test_idx))
            for head in range(n_heads):
                tr = head_idx[train_idx] == head
                te = head_idx[test_idx] == head
                if tr.sum() < 3 or te.sum() == 0:
                    continue
                model = train_model(f_train[tr], z[tr], np.zeros(int(tr.sum()), dtype=int),
                                    1, {0}, seed=seed, epochs=epochs, lr=lr)
                pred_z[te] = predict(model, f_test[te], np.zeros(int(te.sum()), dtype=int))
        # invert per-head standardization
        for head in centers:
            te = head_idx[test_idx] == head
            oof[test_idx[te]] = pred_z[te] * scales[head] + centers[head]
    per_head = {}
    for head, name in enumerate(FULL_STRUCTURE_HEADS):
        sel = head_idx == head
        if sel.sum() == 0 or np.isnan(oof[sel]).any():
            continue
        per_head[name] = metrics(raw_targets[sel], oof[sel])
    return per_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/pan_lung_multitask_mlp_R1M4.json")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = load_rows()
    smiles = [r["canonical_lipid_id"] for r in rows]
    raw_targets = np.array([float(r["raw_value"]) for r in rows], dtype=np.float64)
    head_idx = np.array([FULL_STRUCTURE_HEADS.index(r["typed_head"]) for r in rows], dtype=np.int64)
    groups = np.array(smiles)
    features = np.asarray(molecular_features(smiles)["combined"], dtype=np.float64)
    n_heads = len(FULL_STRUCTURE_HEADS)

    multitask = run("multitask", features, raw_targets, head_idx, groups, n_heads,
                    seed=args.seed, epochs=args.epochs, lr=args.lr, folds=args.folds)
    single = run("single", features, raw_targets, head_idx, groups, n_heads,
                 seed=args.seed, epochs=args.epochs, lr=args.lr, folds=args.folds)

    comparison = {}
    for head in FULL_STRUCTURE_HEADS:
        if head in multitask and head in single:
            comparison[head] = {
                "rows": multitask[head]["n"],
                "multitask_spearman": round(multitask[head]["spearman"], 4),
                "single_task_spearman": round(single[head]["spearman"], 4),
                "multitask_minus_single": round(
                    multitask[head]["spearman"] - single[head]["spearman"], 4),
                "multitask_top_decile_enrichment": round(multitask[head]["top_decile_enrichment"], 3),
            }

    output = {
        "format": "compose_pan_lung_multitask_mlp_R1M4_v1",
        "cell_id": "R1xM4",
        "representation": "R1_morgan2048_plus_rdkit_descriptors",
        "model": "M4_shallow_masked_multitask_mlp",
        "split": "held_lipid_groupkfold_global",
        "heads": FULL_STRUCTURE_HEADS,
        "config": {"seed": args.seed, "epochs": args.epochs, "lr": args.lr, "folds": args.folds,
                   "hidden": [256, 128], "dropout": 0.1},
        "comparison_multitask_vs_single": comparison,
        "multitask_per_head": multitask,
        "single_task_per_head": single,
        "boundary": "LuT excluded (component-only, R2 lane). Held-lipid grouping is global by canonical structure.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(f"{'head':44s} {'rows':>5} {'multi':>7} {'single':>7} {'delta':>7}")
    for head, c in comparison.items():
        print(f"{head:44s} {c['rows']:>5} {c['multitask_spearman']:>7.3f} "
              f"{c['single_task_spearman']:>7.3f} {c['multitask_minus_single']:>+7.3f}")
    print(f"\nwritten: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
