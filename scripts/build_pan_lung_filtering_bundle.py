#!/usr/bin/env python3
"""Train and serialize filtering-only pan-lung oracle ensemble members."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from compose_v4.oracles.pan_lung_filtering import (  # noqa: E402
    fit_lut_applicability_domain,
    fit_molecular_applicability_domain,
    molecular_features as inference_molecular_features,
    molecular_admission,
    score_lut_filter,
    score_molecular_filter,
)
from scripts.export_pan_lung_matrix_cell_predictions import (  # noqa: E402
    REPO_ROOT,
    canonical_smiles,
    extra_trees,
    file_sha256,
    manifest_source,
    resolve_path,
)
from scripts.export_pan_lung_qualified_matrix_comparison import (  # noqa: E402
    ridge_estimator,
    xgboost_estimator,
)
from scripts.train_lumi_oracle_baselines import (  # noqa: E402
    molecular_features as training_molecular_features,
)
from scripts.train_lut_oracle_baselines import make_preprocessor  # noqa: E402


SEED = 20260720


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _dump_verified(
    value: object, path: Path, probe_features: object | None = None
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    before = None
    if probe_features is not None:
        before = np.asarray(value.predict(probe_features), dtype=np.float64)
    joblib.dump(value, path, compress=3)
    restored = joblib.load(path)
    if probe_features is not None:
        after = np.asarray(restored.predict(probe_features), dtype=np.float64)
        if not np.allclose(before, after, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"serialized model prediction mismatch: {path}")
    return {
        "path": _relative(path),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _source_paths(manifest: dict[str, object]) -> tuple[dict[str, Path], dict[str, str]]:
    sources = {
        source_id: manifest_source(manifest, source_id)
        for source_id in ("lnpdb_2026", "lumi_lab_4cr1920", "lut_444_2026")
    }
    lut_table = next(
        value
        for value in sources["lut_444_2026"]["local_paths"]
        if str(value).endswith(".csv")
    )
    paths = {
        "lnpdb": resolve_path(Path(sources["lnpdb_2026"]["local_path"])),
        "lumi": resolve_path(Path(sources["lumi_lab_4cr1920"]["local_path"])),
        "lut": resolve_path(Path(lut_table)),
    }
    expected = {
        "lnpdb": sources["lnpdb_2026"]["sha256"],
        "lumi": sources["lumi_lab_4cr1920"]["sha256"],
        "lut": sources["lut_444_2026"]["sha256"]["extracted_table"],
    }
    observed = {name: file_sha256(path) for name, path in paths.items()}
    if observed != expected:
        raise ValueError(
            f"source artifacts do not match frozen manifest: expected={expected}, observed={observed}"
        )
    return paths, observed


def _qualified_policies(
    contract: dict[str, object], qualification: pd.DataFrame
) -> dict[str, list[str]]:
    qualified = set(
        qualification.loc[
            qualification["qualification_status"]
            == "qualified_for_domain_bounded_filtering",
            "ensemble_id",
        ].tolist()
    )
    if qualified != set(contract["qualified_ensemble_ids"]):
        raise ValueError("qualification table and ensemble contract disagree")
    return {
        "a549": sorted(value for value in qualified if value.startswith("lnpdb_2026__")),
        "lumi": sorted(value for value in qualified if value.startswith("lumi_lab_4cr1920__")),
        "lut_selectivity": sorted(
            value for value in qualified if value.startswith("lut_444_2026__")
        ),
    }


def _first_rejected_molecule(manifest_path: Path, domain_id: str) -> dict[str, object]:
    candidates = (
        "C",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    )
    for smiles in candidates:
        result = score_molecular_filter(manifest_path, domain_id, smiles)
        if not result["admission"]["admitted"]:
            return result
    raise RuntimeError(f"no OOD control was rejected by {domain_id} gate")


def _leave_one_structure_out_control(ad: dict[str, object]) -> dict[str, object]:
    smiles = list(ad["canonical_training_smiles"])
    fingerprints = np.asarray(ad["training_fingerprints"])
    for index, value in enumerate(smiles):
        reduced = dict(ad)
        reduced["canonical_training_smiles"] = smiles[:index] + smiles[index + 1 :]
        reduced["training_fingerprints"] = np.delete(fingerprints, index, axis=0)
        decision = molecular_admission(reduced, value)
        if decision["admitted"]:
            return {
                "test_type": "leave_one_structure_out_applicability_control",
                "removed_structure": value,
                "admission": decision,
            }
    raise RuntimeError("no training structure passed the leave-one-structure-out AD control")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=REPO_ROOT / "configs/pan_lung_corpus_manifest_v1.json",
    )
    parser.add_argument(
        "--ensemble-contract",
        type=Path,
        default=REPO_ROOT / "configs/pan_lung_qualified_ensemble_contract_v1.json",
    )
    parser.add_argument(
        "--qualification-table",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_qualified_ensemble_qualification.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts/oracles/pan_lung_filtering_v1",
    )
    parser.add_argument(
        "--admission-test",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_filtering_admission_test.json",
    )
    args = parser.parse_args()

    corpus_path = resolve_path(args.corpus_manifest)
    contract_path = resolve_path(args.ensemble_contract)
    qualification_path = resolve_path(args.qualification_table)
    output_root = resolve_path(args.output_root)
    admission_path = resolve_path(args.admission_test)
    corpus = json.loads(corpus_path.read_text())
    contract = json.loads(contract_path.read_text())
    qualification = pd.read_csv(qualification_path)
    policies = _qualified_policies(contract, qualification)
    paths, source_hashes = _source_paths(corpus)

    lnpdb = pd.read_csv(paths["lnpdb"], low_memory=False)
    a549 = lnpdb[
        (lnpdb["Experiment_ID"] == "JW_2024")
        & (lnpdb["Model_type"] == "A549")
        & lnpdb["Model_target"].astype(str).str.contains("lung", case=False, na=False)
    ].copy()
    if len(a549) != 1801:
        raise ValueError(f"expected 1,801 A549 rows, observed {len(a549)}")
    a549_smiles = np.asarray([canonical_smiles(value) for value in a549["IL_SMILES"]])
    a549_features = training_molecular_features(a549_smiles)["combined"]
    if not np.array_equal(
        a549_features[:8], inference_molecular_features(a549_smiles[:8])["combined"]
    ):
        raise RuntimeError("A549 training and filtering feature implementations disagree")
    a549_target = a549["Experiment_value"].to_numpy(dtype=np.float64)
    a549_models = {
        "ridge_v1": ridge_estimator(),
        "extra_trees_v1": extra_trees(SEED, 300),
        "xgboost_v1": xgboost_estimator(SEED),
    }
    for estimator in a549_models.values():
        estimator.fit(a549_features, a549_target)

    lumi = pd.read_csv(paths["lumi"])
    if len(lumi) != 1920:
        raise ValueError(f"expected 1,920 LUMI rows, observed {len(lumi)}")
    lumi_smiles = np.asarray([canonical_smiles(value) for value in lumi["mol"]])
    lumi_features = training_molecular_features(lumi_smiles)["combined"]
    if not np.array_equal(
        lumi_features[:8], inference_molecular_features(lumi_smiles[:8])["combined"]
    ):
        raise RuntimeError("LUMI training and filtering feature implementations disagree")
    lumi_target = lumi["RLU (log2)"].to_numpy(dtype=np.float64)
    lumi_models = {
        "extra_trees_v1": extra_trees(SEED, 300),
        "xgboost_v1": xgboost_estimator(SEED),
    }
    for estimator in lumi_models.values():
        estimator.fit(lumi_features, lumi_target)

    lut = pd.read_csv(paths["lut"])
    if len(lut) != 444:
        raise ValueError(f"expected 444 LuT rows, observed {len(lut)}")
    lut_target = lut["lung_selectivity_fraction"].to_numpy(dtype=np.float64)
    lut_models = {
        "ridge_v1": make_pipeline(
            make_preprocessor(lut, "combined"), Ridge(alpha=1.0, solver="lsqr")
        ),
        "extra_trees_v1": make_pipeline(
            make_preprocessor(lut, "combined"), extra_trees(SEED, 300)
        ),
        "xgboost_v1": make_pipeline(
            make_preprocessor(lut, "combined"), xgboost_estimator(SEED)
        ),
    }
    for estimator in lut_models.values():
        estimator.fit(lut, lut_target)

    models_dir = output_root / "models"
    ad_dir = output_root / "applicability_domains"
    domains: dict[str, dict[str, object]] = {}
    domain_inputs = {
        "a549": (a549_models, a549_features[:3]),
        "lumi": (lumi_models, lumi_features[:3]),
        "lut_selectivity": (lut_models, lut.iloc[:3]),
    }
    for domain_id, (models, probe) in domain_inputs.items():
        members = {}
        for model_id, estimator in models.items():
            members[model_id] = _dump_verified(
                estimator, models_dir / f"{domain_id}__{model_id}.joblib", probe
            )
        domains[domain_id] = {
            "members": members,
            "qualified_ensemble_ids": policies[domain_id],
        }

    a549_ad_path = ad_dir / "a549_molecular_ad.joblib"
    lumi_ad_path = ad_dir / "lumi_molecular_ad.joblib"
    lut_ad_path = ad_dir / "lut_component_ad.joblib"
    a549_ad = fit_molecular_applicability_domain(a549_smiles)
    lumi_ad = fit_molecular_applicability_domain(lumi_smiles)
    lut_ad = fit_lut_applicability_domain(lut)
    domains["a549"]["applicability_domain"] = _dump_verified(a549_ad, a549_ad_path)
    domains["lumi"]["applicability_domain"] = _dump_verified(lumi_ad, lumi_ad_path)
    domains["lut_selectivity"]["applicability_domain"] = _dump_verified(
        lut_ad, lut_ad_path
    )
    domains["a549"].update(
        {
            "training_rows": 1801,
            "unique_training_structures": int(len(set(a549_smiles))),
            "representation": "morgan_descriptors_plus_typed_context_v1",
        }
    )
    domains["lumi"].update(
        {
            "training_rows": 1920,
            "unique_training_structures": int(len(set(lumi_smiles))),
            "representation": "morgan_descriptors_plus_typed_context_v1",
            "policy_boundary": (
                "R1 is diagnostic-only. If the novel component axis is unknown, "
                "use the minimum pessimistic score across qualified R2/R3/R4 policies."
            ),
        }
    )
    domains["lut_selectivity"].update(
        {
            "training_rows": 444,
            "representation": "lut_component_sar_plus_typed_context_v1",
            "policy_boundary": (
                "Filtering admits new combinations of released components only; "
                "unseen heads/tails and round-transfer rewards are rejected."
            ),
        }
    )

    manifest_path = output_root / "manifest.json"
    manifest = {
        "format": "compose_pan_lung_filtering_bundle_v1",
        "status": "qualified_for_filtering_only",
        "reward_fine_tuning_enabled": False,
        "repo_root_relative_to_manifest": "../../..",
        "ensemble_contract": {
            "path": _relative(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "qualification_table": {
            "path": _relative(qualification_path),
            "sha256": file_sha256(qualification_path),
        },
        "corpus_manifest": {
            "path": _relative(corpus_path),
            "sha256": file_sha256(corpus_path),
        },
        "source_hashes": source_hashes,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "rdkit": rdkit.__version__,
            "joblib": joblib.__version__,
        },
        "seed": SEED,
        "domains": domains,
        "filtering_contract": {
            "admission_precedes_scoring": True,
            "rejected_candidates_receive_no_score": True,
            "default_selection_score": "conservative_pessimistic_score",
            "scores_are_not_comparable_across_typed_heads": True,
        },
        "clean_runtime_verification": "diagnostics/pan_lung_filtering_clean_runtime_verification.json",
        "reward_guard": "src/compose_v4/oracles/reward_guard.py",
        "preregistration_template": "configs/pan_lung_reward_preregistration.template.json",
        "reward_fine_tuning_blockers": [
            "A passed clean-runtime verification artifact for this exact manifest hash.",
            "A valid frozen preregistration bound to this exact manifest hash.",
            "An explicit future bundle authorization; the released bundle flag is false.",
            "Pre-registered applicability thresholds and candidate budgets.",
            "The hard reward guard must remain in the optimization path.",
            "Prospective monitoring for score hacking and domain drift."
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    a549_in = score_molecular_filter(manifest_path, "a549", a549_smiles[0])
    a549_ood = _first_rejected_molecule(manifest_path, "a549")
    lumi_in = score_molecular_filter(manifest_path, "lumi", lumi_smiles[0])
    lumi_ood = _first_rejected_molecule(manifest_path, "lumi")
    lut_known = score_lut_filter(
        manifest_path, str(lut.iloc[0]["head"]), str(lut.iloc[0]["tail"]), int(lut.iloc[0]["round"])
    )
    known_pairs = set(lut[["head", "tail"]].itertuples(index=False, name=None))
    novel_pair = next(
        (str(head), str(tail))
        for head in lut["head"].unique()
        for tail in lut["tail"].unique()
        if (head, tail) not in known_pairs
    )
    lut_novel_pair = score_lut_filter(manifest_path, novel_pair[0], novel_pair[1], 2)
    lut_ood = score_lut_filter(
        manifest_path, "9A999", str(lut.iloc[0]["tail"]), 2
    )
    tests = {
        "format": "compose_pan_lung_filtering_admission_test_v1",
        "manifest": _relative(manifest_path),
        "cases": {
            "a549_in_domain": a549_in,
            "a549_ood": a549_ood,
            "lumi_in_domain": lumi_in,
            "lumi_ood": lumi_ood,
            "lut_known_pair": lut_known,
            "lut_novel_pair_known_components": lut_novel_pair,
            "lut_unseen_component_ood": lut_ood,
        },
        "gate_only_cases": {
            "a549_leave_one_structure_out": _leave_one_structure_out_control(a549_ad),
            "lumi_leave_one_structure_out": _leave_one_structure_out_control(lumi_ad),
        },
    }
    expected = {
        "a549_in_domain": True,
        "a549_ood": False,
        "lumi_in_domain": True,
        "lumi_ood": False,
        "lut_known_pair": True,
        "lut_novel_pair_known_components": True,
        "lut_unseen_component_ood": False,
    }
    observed = {
        name: bool(case["admission"]["admitted"])
        for name, case in tests["cases"].items()
    }
    if observed != expected:
        raise RuntimeError(f"applicability admission test failed: {observed}")
    if not all(
        case["admission"]["admitted"] for case in tests["gate_only_cases"].values()
    ):
        raise RuntimeError("a molecular leave-one-structure-out AD control failed")
    for name, admitted in observed.items():
        case = tests["cases"][name]
        if admitted and (
            not case["scores"] or case["conservative_pessimistic_score"] is None
        ):
            raise RuntimeError(f"admitted case has no filtering score: {name}")
        if not admitted and (
            case["scores"] or case["conservative_pessimistic_score"] is not None
        ):
            raise RuntimeError(f"rejected case leaked an oracle score: {name}")
    admission_path.parent.mkdir(parents=True, exist_ok=True)
    admission_path.write_text(json.dumps(tests, indent=2, sort_keys=True) + "\n")
    manifest["admission_test"] = {
        "path": _relative(admission_path),
        "sha256": file_sha256(admission_path),
        "observed": observed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Re-run one admission and one rejection through the final manifest.
    if not score_molecular_filter(manifest_path, "a549", a549_smiles[0])["admission"][
        "admitted"
    ]:
        raise RuntimeError("final filtering manifest rejected its in-domain control")
    if score_lut_filter(manifest_path, "9A999", str(lut.iloc[0]["tail"]), 2)[
        "admission"
    ]["admitted"]:
        raise RuntimeError("final filtering manifest admitted an unseen LuT component")

    print(
        json.dumps(
            {
                "manifest": _relative(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "admission_test": _relative(admission_path),
                "admission_test_sha256": file_sha256(admission_path),
                "serialized_members": int(
                    sum(len(domain["members"]) for domain in domains.values())
                ),
                "domains": sorted(domains),
                "reward_fine_tuning_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
