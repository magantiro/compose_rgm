#!/usr/bin/env python3
"""V1 learned twist / Doob value for the value-guided SMC controller.

Phase-2 of docs/CONDITIONAL_CONTROLLER_DESIGN.md. The Feynman-Kac target is
`pi(x) ~ p_B(x) exp(r(x)/alpha) 1[x in F]`; the exact steered generator uses the
Doob value `h(x) = E_B[exp(r(X_T)/alpha) 1_F | X_t = x]`. V0 approximated the
per-step potential with the immediate reward difference `r(y)-r(x)`. Here we
amortize the *future* value with a small net `V_theta(x) ~ log h(x)` trained
(DEFT-style) to regress toward the best feasible reward reachable downstream of a
state, using trajectories rolled from the frozen base. At inference the potential
becomes `exp((V_theta(y) - V_theta(x)) / alpha)`, steering toward states from
which high-reward feasible molecules are reachable -- not just locally greedy.

Trains on Morgan fingerprints (no dependence on the base model's internals), so
it is a drop-in twist for any base. The controller loads it via
`--value-twist-checkpoint`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch import nn

RDLogger.DisableLog("rdApp.*")

_FP_BITS = 2048
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=_FP_BITS)


def fingerprint_array(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = _FP.GetFingerprint(mol)
    out = np.zeros((_FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, out)
    return out


class ValueTwistNet(nn.Module):
    """Fingerprint -> scalar value V_theta(x) ~ log h(x)."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_FP_BITS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.net(fingerprints).squeeze(-1)

    @torch.no_grad()
    def value(self, smiles: str) -> float:
        fp = fingerprint_array(smiles)
        if fp is None:
            return 0.0
        return float(self(torch.from_numpy(fp).unsqueeze(0))[0])


def build_reward_to_go_labels(
    trajectories: list[list[tuple[str, float, bool]]],
) -> tuple[np.ndarray, np.ndarray]:
    """From rollout trajectories, build (fingerprint, best-feasible-reward-to-go).

    Each trajectory is a list of (canonical_smiles, qed, feasible). The training
    target for a state is the maximum feasible qed reached at or after it -- a
    bootstrap-free estimate of `log h` up to the monotone exp(./alpha) map, which
    is monotone so ranking is preserved.
    """
    features: list[np.ndarray] = []
    targets: list[float] = []
    for trajectory in trajectories:
        best_ahead = 0.0
        for smiles, qed, feasible in reversed(trajectory):
            if feasible:
                best_ahead = max(best_ahead, qed)
            fp = fingerprint_array(smiles)
            if fp is None:
                continue
            features.append(fp)
            targets.append(best_ahead)
    if not features:
        return np.empty((0, _FP_BITS), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(features), np.asarray(targets, dtype=np.float32)


def train_value_twist(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden: int = 256,
    seed: int = 0,
) -> tuple[ValueTwistNet, float]:
    torch.manual_seed(seed)
    net = ValueTwistNet(hidden=hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    n = len(x)
    rng = np.random.default_rng(seed)
    last = float("nan")
    for _ in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch_x, batch_y = x[idx], y[idx]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        last = total / max(n, 1)
    return net, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="JSON list of trajectories; each is a list of [smiles, qed, feasible].",
    )
    parser.add_argument("--output", type=Path, required=True, help="ValueTwistNet .pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trajectories = json.loads(args.trajectories.read_text())
    features, targets = build_reward_to_go_labels(trajectories)
    if len(features) == 0:
        raise SystemExit("no usable states in trajectories")
    net, final_loss = train_value_twist(
        features,
        targets,
        epochs=args.epochs,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "fp_bits": _FP_BITS}, args.output)
    print(json.dumps({"states": int(len(features)), "final_train_mse": final_loss, "output": str(args.output)}))


def load_value_twist(path: str) -> ValueTwistNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = ValueTwistNet(hidden=int(payload.get("hidden", 256)))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


if __name__ == "__main__":
    main()
