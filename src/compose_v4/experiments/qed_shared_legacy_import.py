"""Validate and relabel the frozen H24 reference rollouts for terminal QED training."""

from __future__ import annotations

import hashlib

from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.experiments.qed_shared_sources import QEDSourceRoles
from compose_v4.experiments.qed_shared_training import LEGACY_H24_CORPUS_SHA256
from compose_v4.experiments.qed_source_support import audit_qed_source


def imported_train_rollouts(
    corpus: dict, roles: QEDSourceRoles, train_input: tuple[str, ...], reference: dict
) -> tuple[dict, ...]:
    """Keep only the frozen train role and preserve each historical seed."""
    if (
        corpus.get("schema") != "compose.hphi.rollout_corpus"
        or corpus.get("corpus_version") != "hphi-corpus-v1"
        or corpus.get("n") != len(train_input)
        or len(train_input) != len(roles.train) + len(roles.excluded_train_indices)
        or tuple(train_input[i] for i in roles.train_input_indices) != roles.train
        or len(corpus.get("results", ())) != len(train_input)
    ):
        raise ValueError("H24 corpus structure or source count differs from the frozen input")
    rows = {row.get("index"): row for row in corpus["results"]}
    if set(rows) != set(range(len(train_input))):
        raise ValueError("H24 corpus has missing or repeated source indices")
    for input_index, source in enumerate(train_input):
        row = rows[input_index]
        if row.get("status") != "OK" or row.get("source") != source:
            raise ValueError(f"H24 source {input_index} differs from the frozen input")

    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    imported = []
    for index, (input_index, source) in enumerate(zip(roles.train_input_indices, roles.train)):
        row = rows[input_index]
        molecule = Chem.MolFromSmiles(source)
        if molecule is None or row.get("canonical_source") != Chem.MolToSmiles(molecule):
            raise ValueError(f"H24 source {input_index} has a different canonical identity")
        support = audit_qed_source(source, max_active_atoms=reference["max_active_atoms"])
        if not support.supported or not support.benchmark_equivalent or support.represented is None:
            raise ValueError(f"H24 source {input_index} is outside benchmark-compatible support")
        if row.get("horizon") != 24 or len(row.get("trajectories", ())) != 2:
            raise ValueError(
                f"H24 source {input_index} has an unexpected horizon or replicate count"
            )
        source_fp = fingerprint.GetFingerprint(molecule)
        trajectories = []
        for replicate, old in enumerate(sorted(row["trajectories"], key=lambda t: t["replicate"])):
            seed_payload = f"hphi-corpus-v1|{row['canonical_source']}|{replicate}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
            paths = old.get("path", ())
            if (
                old.get("replicate") != replicate
                or old.get("seed") != seed
                or old.get("termination") != {"kind": "complete", "edits": 24, "reason": None}
                or len(paths) != 25
                or paths[0] != source
                or len(old.get("qed", ())) != 25
                or len(old.get("similarity_to_source", ())) != 25
            ):
                raise ValueError(f"H24 source {input_index} replicate {replicate} is incomplete")
            path = []
            for step, smiles in enumerate(paths):
                represented = support.represented if step == 0 else smiles
                state = Chem.MolFromSmiles(represented)
                if state is None or state.GetNumHeavyAtoms() > reference["max_active_atoms"]:
                    raise ValueError(
                        f"H24 source {input_index} replicate {replicate} step {step} is unsupported"
                    )
                quality = float(QED.qed(state))
                similarity = float(
                    DataStructs.TanimotoSimilarity(source_fp, fingerprint.GetFingerprint(state))
                )
                if (
                    abs(quality - float(old["qed"][step])) > 5.1e-7
                    or abs(similarity - float(old["similarity_to_source"][step])) > 5.1e-7
                ):
                    raise ValueError(
                        f"H24 source {input_index} replicate {replicate} step {step} metric drift"
                    )
                path.append(
                    {"smiles": represented, "qed": quality, "similarity_to_source": similarity}
                )
            trajectories.append(
                {"replicate": replicate, "seed": seed, "status": "HORIZON", "path": path}
            )
        imported.append(
            {
                "schema_version": "compose.qed.imported_h24_rollout.v1",
                "source_role": "train",
                "source_index": index,
                "source_input_row_index": input_index,
                "source_original": source,
                "source_represented": support.represented,
                "source_split_sha256": roles.manifest_sha256,
                "reference": reference,
                "configuration": {"horizon": 24, "replicates": 2},
                "import_provenance": {
                    "corpus_sha256": LEGACY_H24_CORPUS_SHA256,
                    "seed_scheme": "hphi-corpus-v1",
                },
                "trajectories": trajectories,
            }
        )
    return tuple(imported)
