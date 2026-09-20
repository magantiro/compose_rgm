"""Portable applicability-domain gates and filtering-only oracle inference."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 2048
HEAD_PATTERN = re.compile(r"^(\d+A)(\d+)$")
TAIL_PATTERN = re.compile(r"^B(\d+)$")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_molecule(smiles: str) -> tuple[str, Chem.Mol]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"unparsable molecular structure: {smiles!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True), molecule


def _descriptor_vector(molecule: Chem.Mol) -> np.ndarray:
    return np.asarray(
        (
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            Lipinski.FractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
            molecule.GetNumHeavyAtoms(),
        ),
        dtype=np.float32,
    )


def molecular_features(smiles: Iterable[str]) -> dict[str, object]:
    canonical: list[str] = []
    molecules: list[Chem.Mol] = []
    for value in smiles:
        canonical_value, molecule = _canonical_molecule(str(value))
        canonical.append(canonical_value)
        molecules.append(molecule)
    descriptors = np.stack(tuple(_descriptor_vector(molecule) for molecule in molecules))
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_SIZE
    )
    fingerprints = np.zeros((len(molecules), FINGERPRINT_SIZE), dtype=np.uint8)
    for row, molecule in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), fingerprints[row])
    combined = np.concatenate(
        (descriptors.astype(np.float32), fingerprints.astype(np.float32)), axis=1
    )
    return {
        "canonical_smiles": canonical,
        "descriptors": descriptors,
        "fingerprints": fingerprints,
        "combined": combined,
    }


def fit_molecular_applicability_domain(smiles: Iterable[str]) -> dict[str, object]:
    canonical = sorted(set(molecular_features(smiles)["canonical_smiles"]))
    features = molecular_features(canonical)
    descriptors = np.asarray(features["descriptors"], dtype=np.float64)
    fingerprints = np.asarray(features["fingerprints"], dtype=np.uint8)
    fp_float = fingerprints.astype(np.float32)
    intersections = fp_float @ fp_float.T
    counts = fp_float.sum(axis=1)
    unions = counts[:, None] + counts[None, :] - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    np.fill_diagonal(similarities, -np.inf)
    nearest_neighbor = np.max(similarities, axis=1)
    center = np.median(descriptors, axis=0)
    mad = np.median(np.abs(descriptors - center), axis=0)
    robust_scale = 1.4826 * mad
    fallback = np.std(descriptors, axis=0)
    robust_scale = np.where(robust_scale > 1e-8, robust_scale, fallback)
    robust_scale = np.where(robust_scale > 1e-8, robust_scale, 1.0)
    robust_distance = np.sqrt(
        np.mean(np.square((descriptors - center) / robust_scale), axis=1)
    )
    return {
        "format": "compose_molecular_applicability_domain_v1",
        "canonical_training_smiles": canonical,
        "training_fingerprints": fingerprints,
        "descriptor_center": center,
        "descriptor_scale": robust_scale,
        "similarity_threshold": float(np.quantile(nearest_neighbor, 0.05)),
        "robust_distance_threshold": float(np.quantile(robust_distance, 0.995)),
        "threshold_contract": {
            "similarity": "5th percentile of leave-one-out maximum Morgan Tanimoto similarity",
            "descriptor_distance": "99.5th percentile of robust RMS descriptor distance",
            "admission": "both thresholds must pass",
        },
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_size": FINGERPRINT_SIZE,
        "unique_training_structures": int(len(canonical)),
    }


def molecular_admission(ad: dict[str, object], smiles: str) -> dict[str, object]:
    try:
        features = molecular_features([smiles])
    except ValueError as exc:
        return {
            "admitted": False,
            "canonical_smiles": None,
            "maximum_training_tanimoto": None,
            "robust_descriptor_distance": None,
            "reasons": [str(exc)],
        }
    query_fp = np.asarray(features["fingerprints"], dtype=np.float32)[0]
    training_fp = np.asarray(ad["training_fingerprints"], dtype=np.float32)
    intersections = training_fp @ query_fp
    unions = training_fp.sum(axis=1) + query_fp.sum() - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    maximum_similarity = float(np.max(similarities))
    descriptor = np.asarray(features["descriptors"], dtype=np.float64)[0]
    distance = float(
        np.sqrt(
            np.mean(
                np.square(
                    (descriptor - np.asarray(ad["descriptor_center"]))
                    / np.asarray(ad["descriptor_scale"])
                )
            )
        )
    )
    reasons = []
    if maximum_similarity < float(ad["similarity_threshold"]):
        reasons.append("nearest-training-structure similarity below threshold")
    if distance > float(ad["robust_distance_threshold"]):
        reasons.append("robust descriptor distance above threshold")
    return {
        "admitted": not reasons,
        "canonical_smiles": features["canonical_smiles"][0],
        "maximum_training_tanimoto": maximum_similarity,
        "similarity_threshold": float(ad["similarity_threshold"]),
        "robust_descriptor_distance": distance,
        "robust_distance_threshold": float(ad["robust_distance_threshold"]),
        "reasons": reasons,
    }


def lut_candidate_frame(head: str, tail: str, round_id: int) -> pd.DataFrame:
    head_match = HEAD_PATTERN.fullmatch(str(head))
    tail_match = TAIL_PATTERN.fullmatch(str(tail))
    if head_match is None or tail_match is None:
        raise ValueError("LuT candidates require head '<family>A<index>' and tail 'B<index>'")
    head_family, head_index_text = head_match.groups()
    head_index = int(head_index_text)
    tail_index = int(tail_match.group(1))
    return pd.DataFrame.from_records(
        [
            {
                "round": int(round_id),
                "head": str(head),
                "tail": str(tail),
                "head_family": head_family,
                "head_index": head_index,
                "tail_index": tail_index,
                "tripod_compatible_head": int(
                    str(head) in {"1A1", "1A2", "1A3", "1A6", "1A7", "1A8"}
                ),
                "tripod_compatible_tail": int(tail_index >= 4),
                "branched_tail": int(13 <= tail_index <= 18),
                "unsaturated_tail": int(19 <= tail_index <= 25),
                "near_head_ester": int(tail_index in {11, 16, 20}),
            }
        ]
    )


def fit_lut_applicability_domain(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "format": "compose_lut_component_applicability_domain_v1",
        "known_heads": sorted(frame["head"].astype(str).unique().tolist()),
        "known_tails": sorted(
            frame["tail"].astype(str).unique().tolist(), key=lambda value: int(value[1:])
        ),
        "allowed_rounds": sorted(frame["round"].astype(int).unique().tolist()),
        "known_pairs": sorted(
            (str(head), str(tail))
            for head, tail in frame[["head", "tail"]].itertuples(index=False, name=None)
        ),
        "admission": (
            "Both components and round context must be in the released vocabulary. "
            "New head-tail combinations are allowed; unseen components are rejected."
        ),
    }


def lut_admission(
    ad: dict[str, object], head: str, tail: str, round_id: int
) -> dict[str, object]:
    reasons = []
    if str(head) not in set(ad["known_heads"]):
        reasons.append("unseen LuT head component")
    if str(tail) not in set(ad["known_tails"]):
        reasons.append("unseen LuT tail component")
    if int(round_id) not in set(ad["allowed_rounds"]):
        reasons.append("unsupported LuT experimental-round context")
    known_pair = [str(head), str(tail)] in ad["known_pairs"] or (
        str(head), str(tail)
    ) in ad["known_pairs"]
    return {
        "admitted": not reasons,
        "head": str(head),
        "tail": str(tail),
        "round": int(round_id),
        "known_pair": bool(known_pair),
        "novel_combination_of_known_components": bool(not known_pair and not reasons),
        "reasons": reasons,
    }


def _load_bundle(manifest_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    repo_root = (manifest_path.parent / manifest["repo_root_relative_to_manifest"]).resolve()
    contract_path = repo_root / manifest["ensemble_contract"]["path"]
    if _sha256(contract_path) != manifest["ensemble_contract"]["sha256"]:
        raise ValueError("ensemble contract hash does not match filtering manifest")
    contract = json.loads(contract_path.read_text())
    return manifest, repo_root, contract


def _load_verified_joblib(repo_root: Path, artifact: dict[str, object]) -> object:
    path = repo_root / artifact["path"]
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"serialized artifact hash mismatch: {path}")
    return joblib.load(path)


def _score_members(
    *,
    manifest: dict[str, object],
    repo_root: Path,
    contract: dict[str, object],
    domain_id: str,
    features: object,
) -> list[dict[str, object]]:
    domain = manifest["domains"][domain_id]
    raw_predictions = {
        model: float(np.asarray(_load_verified_joblib(repo_root, artifact).predict(features))[0])
        for model, artifact in domain["members"].items()
    }
    deployment = {
        item["ensemble_id"]: item for item in contract["deployment_policies"]
    }
    results = []
    for ensemble_id in domain["qualified_ensemble_ids"]:
        policy = deployment[ensemble_id]
        calibrated = []
        member_output = {}
        for model in policy["member_models"]:
            raw = raw_predictions[model]
            calibrator = policy["final_oof_affine_calibrators"][model]
            value = float(calibrator["intercept"] + calibrator["slope"] * raw)
            calibrated.append(value)
            member_output[model] = {"raw": raw, "calibrated": value}
        mean = float(np.mean(calibrated))
        disagreement = float(np.std(calibrated, ddof=0))
        radius = float(policy["final_oof_conformal_radius"])
        results.append(
            {
                "ensemble_id": ensemble_id,
                "members": member_output,
                "ensemble_mean": mean,
                "model_disagreement": disagreement,
                "conformal_radius": radius,
                "total_uncertainty": radius + disagreement,
                "pessimistic_score": mean - radius - disagreement,
            }
        )
    return results


def score_molecular_filter(
    manifest_path: Path, domain_id: str, smiles: str
) -> dict[str, object]:
    if domain_id not in {"a549", "lumi"}:
        raise ValueError("molecular filtering domain must be 'a549' or 'lumi'")
    manifest, repo_root, contract = _load_bundle(Path(manifest_path))
    domain = manifest["domains"][domain_id]
    ad = _load_verified_joblib(repo_root, domain["applicability_domain"])
    admission = molecular_admission(ad, smiles)
    result = {
        "format": "compose_pan_lung_filtering_score_v1",
        "domain_id": domain_id,
        "filtering_only": True,
        "reward_fine_tuning_enabled": False,
        "admission": admission,
        "scores": [],
        "conservative_pessimistic_score": None,
    }
    if not admission["admitted"]:
        return result
    features = molecular_features([admission["canonical_smiles"]])["combined"]
    scores = _score_members(
        manifest=manifest,
        repo_root=repo_root,
        contract=contract,
        domain_id=domain_id,
        features=features,
    )
    result["scores"] = scores
    result["conservative_pessimistic_score"] = float(
        min(item["pessimistic_score"] for item in scores)
    )
    return result


def score_lut_filter(
    manifest_path: Path, head: str, tail: str, round_id: int
) -> dict[str, object]:
    manifest, repo_root, contract = _load_bundle(Path(manifest_path))
    domain = manifest["domains"]["lut_selectivity"]
    ad = _load_verified_joblib(repo_root, domain["applicability_domain"])
    admission = lut_admission(ad, head, tail, round_id)
    result = {
        "format": "compose_pan_lung_filtering_score_v1",
        "domain_id": "lut_selectivity",
        "filtering_only": True,
        "reward_fine_tuning_enabled": False,
        "admission": admission,
        "scores": [],
        "conservative_pessimistic_score": None,
    }
    if not admission["admitted"]:
        return result
    features = lut_candidate_frame(head, tail, round_id)
    scores = _score_members(
        manifest=manifest,
        repo_root=repo_root,
        contract=contract,
        domain_id="lut_selectivity",
        features=features,
    )
    result["scores"] = scores
    result["conservative_pessimistic_score"] = float(
        min(item["pessimistic_score"] for item in scores)
    )
    return result
