"""Verify and reduce complete source-level QED benchmark results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.experiments.qed_shared_smc import QEDSMCConfig, candidate_seed
from compose_v4.experiments.qed_source_support import audit_qed_source


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - half, center + half


def reduce_qed_sources(
    directory: Path,
    test_sources: tuple[str, ...],
    *,
    source_split_sha256: str,
    checkpoint_sha256: str,
    config: QEDSMCConfig,
) -> dict:
    """Require every test source and recompute each returned benchmark metric."""
    expected = {f"source_{index:04d}.json" for index in range(len(test_sources))}
    observed = {path.name for path in directory.glob("*.json")}
    if expected != observed:
        raise ValueError(
            "QED result set incomplete or mixed: "
            f"missing={sorted(expected - observed)[:5]}, extra={sorted(observed - expected)[:5]}"
        )
    if not test_sources:
        raise ValueError("QED reduction needs at least one test source")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    identity = None
    inputs = {}
    successful = 0
    failed_slots = 0
    for index, source in enumerate(test_sources):
        path = directory / f"source_{index:04d}.json"
        raw = path.read_bytes()
        inputs[path.name] = hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        result = record.get("result")
        if not isinstance(result, dict):
            raise TypeError(f"QED source result lacks a result object: {path}")
        if (
            record.get("schema_version") != "compose.qed.shared_result.v1"
            or record.get("test_index") != index
            or record.get("source_split_sha256") != source_split_sha256
            or result.get("source_original") != source
            or result.get("value_head_source_split_sha256") != source_split_sha256
            or result.get("configuration")
            != {
                "horizon": config.horizon,
                "particles": config.particles,
                "candidates": config.candidates,
                "qed_minimum": config.qed_minimum,
                "similarity_minimum": config.similarity_minimum,
            }
        ):
            raise ValueError(f"QED source result has mismatched source or protocol: {path}")
        reference = result.get("reference")
        if (
            not isinstance(reference, dict)
            or reference.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError(f"QED source result uses another reference checkpoint: {path}")
        row_identity = {
            "reference": reference,
            "reference_manifest_sha256": record.get("reference_manifest_sha256"),
            "value_assets_manifest_sha256": record.get("value_assets_manifest_sha256"),
            "value_metadata_sha256": record.get("value_metadata_sha256"),
            "code_sha256": record.get("code_sha256"),
        }
        if any(value is None for value in row_identity.values()):
            raise ValueError(f"QED source result lacks model or code identity: {path}")
        if identity is None:
            identity = row_identity
        elif row_identity != identity:
            raise ValueError(f"QED source results mix model or code identities: {path}")
        support = audit_qed_source(source, max_active_atoms=40)
        if not support.supported or not support.benchmark_equivalent:
            raise ValueError(f"QED source is outside the declared graph support: {path}")
        if result.get("source_represented") != support.represented:
            raise ValueError(f"QED source representation drift: {path}")
        molecule = Chem.MolFromSmiles(source)
        if molecule is None:
            raise ValueError(f"QED source cannot be parsed: {path}")
        source_fp = fingerprint.GetFingerprint(molecule)
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != config.candidates:
            raise ValueError(f"QED source result has the wrong candidate count: {path}")
        source_success = False
        for candidate_index, candidate in enumerate(candidates):
            if candidate.get("index") != candidate_index or candidate.get("seed") != candidate_seed(
                support.represented, candidate_index
            ):
                raise ValueError(f"QED candidate identity or seed mismatch: {path}")
            status = candidate.get("status")
            if status not in ("OK", "EXTINCT_NO_HIT"):
                raise ValueError(f"QED candidate has unknown status: {path}")
            smiles = candidate.get("smiles")
            if not isinstance(smiles, str):
                raise TypeError(f"QED candidate has no molecule: {path}")
            returned = Chem.MolFromSmiles(smiles)
            if returned is None:
                raise ValueError(f"QED candidate cannot be parsed: {path}")
            quality = float(QED.qed(returned))
            similarity = float(
                DataStructs.TanimotoSimilarity(source_fp, fingerprint.GetFingerprint(returned))
            )
            reported_quality = candidate.get("qed")
            reported_similarity = candidate.get("similarity_to_source")
            if (
                not isinstance(reported_quality, (int, float))
                or not isinstance(reported_similarity, (int, float))
                or not math.isfinite(reported_quality)
                or not math.isfinite(reported_similarity)
            ):
                raise ValueError(f"QED candidate lacks finite benchmark metrics: {path}")
            if (
                abs(quality - reported_quality) > 1e-8
                or abs(similarity - reported_similarity) > 1e-8
            ):
                raise ValueError(f"QED candidate benchmark metrics differ from its receipt: {path}")
            success = (
                status == "OK"
                and quality >= config.qed_minimum
                and similarity >= config.similarity_minimum
            )
            if candidate.get("success") is not success:
                raise ValueError(
                    f"QED candidate success flag differs from benchmark criteria: {path}"
                )
            if status == "EXTINCT_NO_HIT":
                failed_slots += 1
                if smiles != source:
                    raise ValueError(f"QED extinct slot did not return its original source: {path}")
            source_success |= success
        if result.get("success") is not source_success:
            raise ValueError(f"QED source success flag differs from its candidates: {path}")
        successful += int(source_success)
    low, high = _wilson(successful, len(test_sources))
    return {
        "schema_version": "compose.qed.shared_reduction.v1",
        "sources": len(test_sources),
        "successful_sources": successful,
        "success_fraction": successful / len(test_sources),
        "wilson_95": [low, high],
        "failed_output_slots": failed_slots,
        "configuration": {
            "horizon": config.horizon,
            "particles": config.particles,
            "candidates": config.candidates,
            "qed_minimum": config.qed_minimum,
            "similarity_minimum": config.similarity_minimum,
        },
        "source_split_sha256": source_split_sha256,
        "identity": identity,
        "input_sha256": inputs,
    }
