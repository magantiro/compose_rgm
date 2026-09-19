"""Round-frozen chemistry-aware value, never a calibrated docking oracle."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

COEFFICIENT_DECIMAL_PLACES = 14
LEGACY_RECIPE = {
    "schema_version": "docking_value_recipe_v1",
    "ridge": 1.0,
    "warmup": 16,
    "morgan_radius": 2,
    "fp_size": 2048,
    "kernel": "mean_morgan_atompair_tanimoto",
    "center_targets": True,
    "uncertainty": "not_estimated",
}
RECIPE = {
    **LEGACY_RECIPE,
    # ``numpy.linalg.solve`` can differ at the final binary ulp across BLAS
    # builds/hosts.  Serialize below that operational noise so identical
    # archives produce one content identity without changing useful precision.
    "coefficient_decimal_places": COEFFICIENT_DECIMAL_PLACES,
}
SNAPSHOT_EQUIVALENCE_ATOL = 1e-14


def identity(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def molecular_features(smiles: str) -> tuple[str, tuple[tuple[int, ...], ...]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid task-value SMILES: {smiles!r}")
    generators = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=RECIPE["morgan_radius"], fpSize=RECIPE["fp_size"]
        ),
        rdFingerprintGenerator.GetAtomPairGenerator(fpSize=RECIPE["fp_size"]),
    )
    return Chem.MolToSmiles(molecule), tuple(
        tuple(g.GetFingerprint(molecule).GetOnBits()) for g in generators
    )


def graph_kernel(left, right) -> np.ndarray:
    """Average of two fixed graph-fingerprint kernels, no fitted preprocessing."""

    def masks(rows):
        # Build each bit set once, not once per molecule pair. Integer popcounts
        # compute exactly the same intersection/union cardinalities as sets.
        encoded = []
        for row in rows:
            values = []
            for bits in row:
                value = 0
                for bit in bits:
                    if (
                        not isinstance(bit, (int, np.integer))
                        or not 0 <= bit < RECIPE["fp_size"]
                    ):
                        raise ValueError(
                            f"fingerprint bit outside declared width: {bit!r}"
                        )
                    value |= 1 << int(bit)
                values.append(value)
            encoded.append(values)
        return encoded

    def tanimoto(a, b):
        union = a | b
        return (a & b).bit_count() / union.bit_count() if union else 1.0

    masks_left, masks_right = masks(left), masks(right)
    return np.asarray(
        [
            [
                sum(tanimoto(a, b) for a, b in zip(x, y, strict=True)) / 2
                for y in masks_right
            ]
            for x in masks_left
        ],
        dtype=np.float64,
    ).reshape(len(left), len(right))


def validate_archive(archive: list[dict]) -> None:
    """Validate the entire stream, including the final scored round."""
    seen = set()
    for index, row in enumerate(archive):
        rd, score = row.get("round"), row.get("ds")
        if type(rd) is not int or rd < 0:
            raise ValueError(f"archive row {index}: invalid round {rd!r}")
        key, _ = molecular_features(row["smiles"])
        if key in seen:
            raise ValueError(f"archive row {index}: duplicate canonical identity {key}")
        seen.add(key)
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(
                f"archive row {index}: nonfinite or malformed docking score {score!r}"
            )


@dataclass(frozen=True)
class DockingValue:
    payload: dict

    @classmethod
    def fit(cls, archive: list[dict], *, before_round: int, source_sha256: str):
        if type(before_round) is not int or before_round < 1:
            raise ValueError("before_round must be a positive integer")
        if len(source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in source_sha256
        ):
            raise ValueError("task-value archive requires a physical SHA-256 identity")
        rows, excluded, features, seen = [], [], [], set()
        for index, row in enumerate(archive):
            rd = row.get("round")
            if type(rd) is not int or not 0 <= rd < before_round:
                raise ValueError(
                    f"archive row {index}: round {rd!r} is outside the prior-round prefix"
                )
            key, feature = molecular_features(row["smiles"])
            if key in seen:
                raise ValueError(
                    f"archive row {index}: duplicate canonical identity {key}"
                )
            seen.add(key)
            score = row.get("ds")
            if score is None:
                excluded.append(
                    {
                        "row": index,
                        "smiles": key,
                        "round": rd,
                        "reason": (
                            "undocked_source" if rd == 0 else "missing_oracle_label"
                        ),
                    }
                )
                continue
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(
                    f"archive row {index}: nonfinite or malformed docking score {score!r}"
                )
            rows.append(
                {"smiles": key, "ds": float(score), "round": rd, "source_row": index}
            )
            features.append(feature)
        if len(rows) < RECIPE["warmup"]:
            raise ValueError(
                f"task value requires {RECIPE['warmup']} prior labels; got {len(rows)}"
            )
        scores = np.asarray([r["ds"] for r in rows])
        mean = float(scores.mean())
        coefficients = np.round(
            np.linalg.solve(
                graph_kernel(features, features) + RECIPE["ridge"] * np.eye(len(rows)),
                scores - mean,
            ),
            decimals=COEFFICIENT_DECIMAL_PLACES,
        )
        payload = {
            "schema_version": "docking_value_snapshot_v1",
            "recipe": RECIPE,
            "before_round": before_round,
            "source_sha256": source_sha256,
            "training_rows": rows,
            "excluded": excluded,
            "features": features,
            "coefficients": coefficients.tolist(),
            "mean": mean,
            "best": float(scores.min()),
            "scale": max(1.0, float(scores.std())),
        }
        # JSON round-trip detaches all mutable input/registry references.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        return cls({**payload, "snapshot_sha256": identity(payload)})

    @classmethod
    def from_payload(cls, payload: dict):
        snapshot = {k: v for k, v in payload.items() if k != "snapshot_sha256"}
        if (
            payload.get("schema_version") != "docking_value_snapshot_v1"
            or payload.get("recipe") != RECIPE
            or identity(snapshot) != payload.get("snapshot_sha256")
        ):
            raise ValueError("task-value snapshot schema, recipe or hash mismatch")
        return cls(json.loads(json.dumps(payload, allow_nan=False)))

    def predict(self, smiles: list[str]) -> np.ndarray:
        features = [molecular_features(s)[1] for s in smiles]
        return self.payload["mean"] + graph_kernel(
            features, self.payload["features"]
        ) @ np.asarray(self.payload["coefficients"], dtype=np.float64)

    def desirability(self, smiles: str, feasible: bool) -> float:
        if type(feasible) is not bool:
            raise ValueError("terminal feasibility must be an explicit boolean")
        if not feasible:
            return 0.0
        penalty = max(0.0, float(self.predict([smiles])[0]) - self.payload["best"])
        return math.exp(-min(700.0, penalty / self.payload["scale"]))


def snapshot_equivalence(left: dict, right: dict, *, probe_smiles=()) -> dict:
    """Prove bounded equivalence of legacy/current round-frozen snapshots.

    This is deliberately narrower than accepting arbitrary close predictors.
    Both payloads must be authentic, use the same fixed recipe apart from the
    declared coefficient-serialization field, and agree exactly on all fitted
    inputs and summaries.  The coefficient bound then provides a global score
    bound because every graph-kernel entry lies in ``[0, 1]``.
    """

    def validate(payload):
        body = {k: v for k, v in payload.items() if k != "snapshot_sha256"}
        if payload.get("schema_version") != "docking_value_snapshot_v1" or identity(
            body
        ) != payload.get("snapshot_sha256"):
            raise ValueError("task-value snapshot hash or schema mismatch")
        recipe = dict(payload.get("recipe", {}))
        decimals = recipe.pop("coefficient_decimal_places", None)
        if recipe != LEGACY_RECIPE or decimals not in (
            None,
            COEFFICIENT_DECIMAL_PLACES,
        ):
            raise ValueError("task-value snapshots use nonequivalent recipes")

    validate(left)
    validate(right)
    ignored = {"snapshot_sha256", "coefficients", "recipe"}
    if {k: v for k, v in left.items() if k not in ignored} != {
        k: v for k, v in right.items() if k not in ignored
    }:
        raise ValueError(
            "task-value snapshots differ outside coefficient serialization"
        )
    a = np.asarray(left["coefficients"], dtype=np.float64)
    b = np.asarray(right["coefficients"], dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("task-value snapshot coefficient shape or values differ")
    maximum = float(np.max(np.abs(a - b), initial=0.0))
    if maximum > SNAPSHOT_EQUIVALENCE_ATOL:
        raise ValueError(
            "task-value snapshot coefficient drift exceeds equivalence bound"
        )
    probes = tuple(sorted(set(probe_smiles)))
    prediction_drift = 0.0
    if probes:
        prediction_drift = float(
            np.max(
                np.abs(
                    DockingValue(left).predict(list(probes))
                    - DockingValue(right).predict(list(probes))
                ),
                initial=0.0,
            )
        )
    bound = len(a) * maximum
    if prediction_drift > bound + np.finfo(np.float64).eps:
        raise RuntimeError(
            "observed task-value prediction drift exceeds its kernel bound"
        )
    return {
        "left_snapshot_sha256": left["snapshot_sha256"],
        "right_snapshot_sha256": right["snapshot_sha256"],
        "coefficient_count": len(a),
        "max_abs_coefficient_difference": maximum,
        "global_prediction_difference_bound": float(bound),
        "probe_count": len(probes),
        "max_abs_probe_prediction_difference": prediction_drift,
        "equivalent": True,
    }


def chronological_check(archive: list[dict], *, source_sha256: str) -> dict:
    """Whole next rounds, no random split or recipe selection on inspected labels."""
    validate_archive(archive)
    predictions, skipped, snapshots = [], [], []
    for rd in sorted({row["round"] for row in archive if row["round"] > 0}):
        prefix = [row for row in archive if row["round"] < rd]
        current = [row for row in archive if row["round"] == rd]
        if sum(row.get("ds") is not None for row in prefix) < RECIPE["warmup"]:
            skipped.extend(
                {"smiles": row["smiles"], "round": rd, "reason": "prior_warmup"}
                for row in current
            )
            continue
        model = DockingValue.fit(prefix, before_round=rd, source_sha256=source_sha256)
        snapshots.append(model.payload)
        for row in current:
            if row.get("ds") is None:
                skipped.append(
                    {
                        "smiles": row["smiles"],
                        "round": rd,
                        "reason": "missing_oracle_label",
                    }
                )
                continue
            if not math.isfinite(row["ds"]):
                raise ValueError("nonfinite validation docking label")
            predictions.append(
                {
                    "smiles": row["smiles"],
                    "round": rd,
                    "observed": row["ds"],
                    "predicted": float(model.predict([row["smiles"]])[0]),
                    "baseline": model.payload["mean"],
                    "snapshot_sha256": model.payload["snapshot_sha256"],
                }
            )
    pairs, correct = 0, 0.0
    for i, a in enumerate(predictions):
        for b in predictions[i + 1 :]:
            if a["round"] != b["round"] or a["observed"] == b["observed"]:
                continue
            pairs += 1
            product = (a["observed"] - b["observed"]) * (
                a["predicted"] - b["predicted"]
            )
            correct += 1 if product > 0 else 0.5 if product == 0 else 0
    mae = (
        float(np.mean([abs(r["predicted"] - r["observed"]) for r in predictions]))
        if predictions
        else None
    )
    baseline = (
        float(np.mean([abs(r["baseline"] - r["observed"]) for r in predictions]))
        if predictions
        else None
    )
    concordance = correct / pairs if pairs else None
    passes = (
        len(predictions) >= 20
        and mae < baseline
        and concordance is not None
        and concordance > 0.5
    )
    return {
        "schema_version": "docking_value_chronological_v1",
        "source_sha256": source_sha256,
        "recipe": RECIPE,
        "predictions": predictions,
        "excluded": skipped,
        "snapshots": snapshots,
        "n_scored": len(predictions),
        "n_pairs": pairs,
        "mae": mae,
        "baseline_mae": baseline,
        "concordance": concordance,
        "decision": (
            "development_signal_present"
            if passes
            else "do_not_launch_on_this_predictor"
        ),
        "new_oracle_calls": 0,
        "uncertainty_calibrated": False,
    }
