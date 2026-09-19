"""Budget-conditioned achieved-path value, not an exact or calibrated committor."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from compose_v4.control.continuation import _tilted
from compose_v4.control.winner_imitation import ImitationRanker

BUDGETS = (1, 3, 6, 11, 32, 64)
SEED = 20260920


@lru_cache(maxsize=8192)
def molecule_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid feature molecule: {smiles}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    bits = fp.GetFingerprintAsNumPy(mol).astype(np.float32)
    counts = np.array(
        [
            mol.GetNumHeavyAtoms() / 40,
            (mol.GetNumBonds() - mol.GetNumAtoms() + 1) / 8,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 8,
            rdMolDescriptors.CalcNumHBD(mol) / 8,
            rdMolDescriptors.CalcNumHBA(mol) / 16,
        ],
        dtype=np.float32,
    )
    return np.concatenate((bits, counts))


def features(smiles: str, budget: int) -> np.ndarray:
    if type(budget) is not int or not 0 <= budget <= 64:
        raise ValueError("value budget must be an integer in 0..64")
    return np.append(molecule_features(smiles), np.float32(budget / 64)).astype(np.float32)


def achieved_returns(values, budgets=BUDGETS):
    """Only actually witnessed suffixes supply future labels; stop is available."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("invalid achieved score sequence")
    return [[float(max(values[i : i + b + 1])) for b in budgets] for i in range(len(values))]


def fit(panel):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)
    rows = panel["rows"]
    x = torch.from_numpy(np.stack([features(r["smiles"], r["budget"]) for r in rows]))
    y = torch.tensor([r["return"] for r in rows], dtype=torch.float32)
    model = ImitationRanker(x.shape[1], hidden=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    immediate = [i for i, r in enumerate(rows) if r["route"] is None]
    names = sorted({r["route"] for r in rows if r["route"] is not None})
    routes = {name: [i for i, r in enumerate(rows) if r["route"] == name] for name in names}
    if not immediate or not routes:
        raise ValueError("value fitting requires immediate labels and witnessed routes")
    history = []
    for step in range(600):
        indices = list(rng.choice(immediate, size=32))
        for _ in range(32):
            group = routes[names[int(rng.integers(len(names)))]]
            indices.append(group[int(rng.integers(len(group)))])
        loss = ((torch.sigmoid(model(x[indices])) - y[indices]) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite trajectory value loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            history.append({"update": step + 1, "loss": float(loss.detach())})
    model.eval()
    with torch.inference_mode():
        mse = float(((torch.sigmoid(model(x)) - y) ** 2).mean())
    return (
        model,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": rng.bit_generator.state,
            "updates": 600,
        },
        {"history": history, "training_mse": mse, "rows": len(rows)},
    )


def predict(model, smiles, budget):
    with torch.inference_mode():
        return torch.sigmoid(
            model(torch.from_numpy(np.stack([features(s, budget) for s in smiles])))
        ).numpy()


def choices(candidates, future_values, *, seed):
    """Reference draw multiplicity is conserved across canonical aggregation."""
    grouped = {}
    for c, value in zip(candidates, future_values, strict=True):
        if c is None:
            continue
        s = c["smiles"]
        if s not in grouped:
            grouped[s] = {"candidate": c, "count": 0, "future": float(value)}
        elif (
            abs(grouped[s]["future"] - value) > 1e-8
            or grouped[s]["candidate"]["score"] != c["score"]
        ):
            raise ValueError("canonical aliases disagree on state values")
        grouped[s]["count"] += 1
    if not grouped:
        return {
            a: {"status": "abstained", "candidate": None}
            for a in ("uniform", "immediate", "future")
        }
    groups = list(grouped.values())
    base = np.array([g["count"] for g in groups], dtype=float)
    base /= base.sum()
    now = np.array([g["candidate"]["score"] for g in groups])
    future = np.maximum(now, [g["future"] for g in groups])
    u = float(np.random.default_rng(seed).random())
    result = {}
    for arm, values in (("uniform", np.ones(len(groups))), ("immediate", now), ("future", future)):
        q, eta, kl = (
            (base.copy(), 0.0, 0.0)
            if arm == "uniform"
            else _tilted(base, values, kappa=1.0, exploration=0.1)
        )
        index = min(int(np.searchsorted(np.cumsum(q), u, side="right")), len(q) - 1)
        result[arm] = {
            "status": "selected",
            "candidate": groups[index]["candidate"],
            "selected": index,
            "probabilities": q.tolist(),
            "reference": base.tolist(),
            "values": np.asarray(values).tolist(),
            "eta": eta,
            "kl": kl,
            "u": u,
        }
    return result
