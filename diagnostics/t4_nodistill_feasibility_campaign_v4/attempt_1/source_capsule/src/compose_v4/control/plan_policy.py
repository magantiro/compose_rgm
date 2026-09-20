"""Learned complete-plan proposals; neither a committor nor a new executor.

The endpoint network initializes a conditional proposal. Online PPO updates only
the context residual, on recorded finite pools and real completed-edit rewards.
All probabilities are conditional on that sampled pool, not original-R_theta
molecular probabilities or claimed representation-invariant path densities.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn

from compose_v4.control.docking_value import identity

RECIPE = {
    "schema_version": "neural_complete_plan_policy_v1",
    "fingerprint_bits": 2048,
    "encoder_width": 64,
    "endpoint_epochs": 60,
    "endpoint_batch": 128,
    "endpoint_lr": 0.001,
    "proposal_temperature": 0.1,
    "exploration": 0.2,
    "plans_per_draw": 64,
    "online_steps": 8,
    "online_lr": 0.003,
    "ppo_clip": 0.2,
    "forward_kl_penalty": 0.01,
    "advantage_scale": 0.1,
    "advantage_clip": 5.0,
    "dtype": "float64",
    "seed": 20261011,
}


@lru_cache(maxsize=16384)
def molecular_features(smiles: str) -> tuple[float, ...]:
    # SMILES is used for invariant features, never reconstruction of exact states.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecular feature input: {smiles}")
    fp = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=RECIPE["fingerprint_bits"], includeChirality=False
    ).GetFingerprintAsNumPy(mol)
    return tuple(map(float, fp)) + (
        mol.GetNumHeavyAtoms() / 40,
        (mol.GetNumBonds() - mol.GetNumAtoms() + 1) / 10,
        sum(a.GetIsAromatic() for a in mol.GetAtoms()) / 40,
        mol.GetNumBonds() / 50,
    )


def feature_tensor(smiles):
    return torch.tensor(np.asarray([molecular_features(s) for s in smiles]), dtype=torch.float64)


class EndpointNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        w = RECIPE["encoder_width"]
        self.encoder = nn.Sequential(nn.Linear(2052, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU())
        self.value = nn.Sequential(nn.Linear(w, 1), nn.Sigmoid())

    def forward(self, x):
        return self.value(self.encoder(x)).squeeze(-1)


def actor_network():
    return nn.Sequential(nn.Linear(64 * 3 + 2, 64), nn.Tanh(), nn.Linear(64, 1)).double()


def weights(module):
    return {k: v.detach().cpu().tolist() for k, v in sorted(module.state_dict().items())}


def restore(module, payload):
    state = {k: torch.tensor(v, dtype=torch.float64) for k, v in payload.items()}
    module.load_state_dict(state, strict=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite plan-policy parameters")
    return module


def policy_snapshot(actor, *, parent=None, observations=()):
    body = {"actor": weights(actor), "parent": parent, "observations": list(observations)}
    return {**body, "policy_id": identity(body)}


def check_snapshot(snapshot):
    if identity({k: v for k, v in snapshot.items() if k != "policy_id"}) != snapshot["policy_id"]:
        raise ValueError("plan-policy snapshot identity mismatch")


def fit_endpoint(observed):
    """Fixed recipe, grouped diagnostic split before any fitting; no oracle calls."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RECIPE["seed"])
    smiles = sorted(observed)
    if not smiles or any(not np.isfinite(observed[s]) or not 0 <= observed[s] <= 1 for s in smiles):
        raise ValueError("endpoint initialization requires finite paid normalized labels")
    groups = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "acyclic" for s in smiles]
    train = [
        i for i, group in enumerate(groups) if int(identity(["plan-split", group])[:8], 16) % 5
    ]
    calibration = sorted(set(range(len(smiles))) - set(train))
    if not train or not calibration:
        raise ValueError("plan-policy grouped diagnostic split is empty")
    x, y = feature_tensor(smiles), torch.tensor([observed[s] for s in smiles], dtype=torch.float64)
    counts = Counter(groups[i] for i in train)
    row_weights = torch.tensor([1 / counts[groups[i]] for i in train], dtype=torch.float64)
    row_weights /= row_weights.mean()
    net = EndpointNetwork().double()
    optimizer = torch.optim.Adam(net.parameters(), lr=RECIPE["endpoint_lr"])
    generator = torch.Generator().manual_seed(RECIPE["seed"])
    losses = []
    for _ in range(RECIPE["endpoint_epochs"]):
        order = torch.randperm(len(train), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train), RECIPE["endpoint_batch"]):
            positions = order[start : start + RECIPE["endpoint_batch"]]
            indices = torch.tensor(train)[positions]
            error = net(x[indices]) - y[indices]
            loss = (row_weights[positions] * error.square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(positions) / len(train)
        losses.append(epoch_loss)
    with torch.no_grad():
        prediction = net(x[calibration]).numpy()
    actor = actor_network()
    nn.init.zeros_(actor[-1].weight)
    nn.init.zeros_(actor[-1].bias)
    body = {
        "recipe": RECIPE,
        "weights": weights(net),
        "training_smiles": [smiles[i] for i in train],
    }
    encoder = {**body, "encoder_id": identity(body)}
    report = {
        "split": {
            "train": [smiles[i] for i in train],
            "calibration": [smiles[i] for i in calibration],
        },
        "training_groups": len(counts),
        "calibration_groups": len({groups[i] for i in calibration}),
        "calibration_rmse": float(np.sqrt(np.mean((prediction - y[calibration].numpy()) ** 2))),
        "constant_rmse": float(torch.sqrt(((y[calibration] - y[train].mean()) ** 2).mean())),
        "loss": losses,
        "role": "exposed grouped endpoint prediction diagnostic; not future value or online success",
    }
    return encoder, policy_snapshot(actor), report


class PlanPolicy:
    def __init__(self, encoder, snapshot):
        check_snapshot(snapshot)
        if (
            encoder["recipe"] != RECIPE
            or identity({k: v for k, v in encoder.items() if k != "encoder_id"})
            != encoder["encoder_id"]
        ):
            raise ValueError("plan-policy encoder identity mismatch")
        self.net = restore(EndpointNetwork().double(), encoder["weights"]).eval()
        self.net.requires_grad_(False)
        self.actor = restore(actor_network(), snapshot["actor"])
        self.snapshot = snapshot

    def inputs(self, source, products, release):
        if not products or len(products) != len(release) or len(set(products)) != len(products):
            raise ValueError("plan pool requires distinct aligned products and release fractions")
        r = torch.tensor(release, dtype=torch.float64)
        if not torch.isfinite(r).all() or (r < 0).any() or (r > 1).any():
            raise ValueError("invalid plan region scale")
        with torch.no_grad():
            features = feature_tensor([source, *products])
            embedding = self.net.encoder(features)
            source_embedding = embedding[:1].expand(len(products), -1)
            context = torch.cat(
                (
                    source_embedding,
                    embedding[1:],
                    embedding[1:] - source_embedding,
                    r[:, None],
                    (features[1:, -4] - features[0, -4])[:, None],
                ),
                dim=1,
            )
            base_logits = (
                self.net.value(embedding[1:]).squeeze(-1) - self.net.value(embedding[:1]).squeeze()
            ) / RECIPE["proposal_temperature"]
        return context, base_logits

    def distribution_tensor(self, context, base_logits):
        logits = base_logits + self.actor(context).squeeze(-1)
        epsilon = RECIPE["exploration"]
        return epsilon / len(logits) + (1 - epsilon) * torch.softmax(logits, dim=0)

    def distribution(self, source, products, release):
        with torch.no_grad():
            q = self.distribution_tensor(*self.inputs(source, products, release)).numpy()
        if (
            not np.isfinite(q).all()
            or not np.isclose(q.sum(), 1)
            or (q < RECIPE["exploration"] / len(q) - 1e-12).any()
        ):
            raise RuntimeError("invalid complete-plan proposal law")
        return q

    def update(self, decisions):
        """On-policy clipped update on this snapshot, never labels from the other arm."""
        if not decisions:
            return self.snapshot, {"scored_decisions": 0, "changed": False}
        examples = []
        for row in decisions:
            if row["policy_id"] != self.snapshot["policy_id"]:
                raise ValueError("online update used a different behavior-policy snapshot")
            context, logits = self.inputs(row["source"], row["products"], row["release"])
            old = torch.tensor(row["probabilities"], dtype=torch.float64)
            expected = self.distribution_tensor(context, logits).detach()
            if old.shape != expected.shape or not torch.allclose(
                old, expected, atol=1e-10, rtol=1e-9
            ):
                raise ValueError("recorded behavior probabilities do not match policy")
            if not 0 <= row["selected"] < len(old) or not np.isfinite(row["gain"]):
                raise ValueError("invalid paid decision index or advantage")
            advantage = float(
                np.clip(
                    row["gain"] / RECIPE["advantage_scale"],
                    -RECIPE["advantage_clip"],
                    RECIPE["advantage_clip"],
                )
            )
            examples.append((context, logits, old, row["selected"], advantage))
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=RECIPE["online_lr"])
        for _ in range(RECIPE["online_steps"]):
            losses = []
            for context, logits, old, selected, advantage in examples:
                q = self.distribution_tensor(context, logits)
                ratio = q[selected] / old[selected]
                clipped = ratio.clamp(1 - RECIPE["ppo_clip"], 1 + RECIPE["ppo_clip"])
                losses.append(
                    -torch.minimum(ratio * advantage, clipped * advantage)
                    + RECIPE["forward_kl_penalty"] * (old * (old.log() - q.log())).sum()
                )
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite plan-policy update")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            after = [self.distribution_tensor(x, l).numpy() for x, l, *_ in examples]
        snapshot = policy_snapshot(
            self.actor,
            parent=self.snapshot["policy_id"],
            observations=[d["decision_id"] for d in decisions],
        )
        return snapshot, {
            "scored_decisions": len(decisions),
            "improving_decisions": sum(d["gain"] > 0 for d in decisions),
            "changed": snapshot["actor"] != self.snapshot["actor"],
            "mean_pool_total_variation": float(
                np.mean(
                    [
                        np.abs(q - e[2].numpy()).sum() / 2
                        for q, e in zip(after, examples, strict=True)
                    ]
                )
            ),
            "mean_selected_probability_change": float(
                np.mean([q[e[3]] - e[2][e[3]].item() for q, e in zip(after, examples, strict=True)])
            ),
        }
