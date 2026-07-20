#!/usr/bin/env python3
"""Verify the filtering bundle after relocation in a fresh Python environment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv


REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_artifact(source_root: Path, target_root: Path, relative: str) -> None:
    source = source_root / relative
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/oracles/pan_lung_filtering_v1/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "diagnostics/pan_lung_filtering_clean_runtime_verification.json",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())

    with tempfile.TemporaryDirectory(prefix="compose_filtering_clean_runtime_") as temp:
        temp_root = Path(temp)
        relocated_root = temp_root / "relocated_repo"
        relative_bundle = str(manifest_path.parent.relative_to(REPO_ROOT))
        copy_artifact(REPO_ROOT, relocated_root, relative_bundle)
        for entry in (
            manifest["ensemble_contract"],
            manifest["qualification_table"],
            manifest["corpus_manifest"],
            manifest["admission_test"],
        ):
            copy_artifact(REPO_ROOT, relocated_root, entry["path"])
        copy_artifact(
            REPO_ROOT,
            relocated_root,
            "configs/pan_lung_reward_preregistration.template.json",
        )
        relocated_manifest = relocated_root / relative_bundle / "manifest.json"

        hash_checks = {}
        relocated = json.loads(relocated_manifest.read_text())
        for label, entry in (
            ("ensemble_contract", relocated["ensemble_contract"]),
            ("qualification_table", relocated["qualification_table"]),
            ("corpus_manifest", relocated["corpus_manifest"]),
            ("admission_test", relocated["admission_test"]),
        ):
            observed = file_sha256(relocated_root / entry["path"])
            hash_checks[label] = observed == entry["sha256"]
        for domain_id, domain in relocated["domains"].items():
            entries = {"ad": domain["applicability_domain"], **domain["members"]}
            for label, entry in entries.items():
                key = f"{domain_id}:{label}"
                hash_checks[key] = (
                    file_sha256(relocated_root / entry["path"]) == entry["sha256"]
                )
        if not all(hash_checks.values()):
            raise RuntimeError(f"relocated artifact hash verification failed: {hash_checks}")

        environment = temp_root / "runtime"
        venv.EnvBuilder(
            system_site_packages=True,
            clear=True,
            with_pip=False,
        ).create(environment)
        python = environment / "bin/python"
        clean_env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
        site_packages = Path(
            subprocess.run(
                [
                    str(python),
                    "-c",
                    "import site; print(site.getsitepackages()[0])",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=clean_env,
            ).stdout.strip()
        )
        shutil.copytree(
            REPO_ROOT / "src/compose_v4",
            site_packages / "compose_v4",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import compose_v4; print(compose_v4.__file__)",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=clean_env,
        )

        verifier = r'''
import json
from pathlib import Path
import sys
import joblib, numpy, pandas, rdkit, sklearn, xgboost
from compose_v4.oracles.reward_guard import FilteringRewardGuard, RewardFineTuningDisabledError

manifest_path = Path(sys.argv[1])
admission_path = Path(sys.argv[2])
template_path = Path(sys.argv[3])
admission = json.loads(admission_path.read_text())
known_smiles = admission["cases"]["a549_in_domain"]["admission"]["canonical_smiles"]
guard = FilteringRewardGuard(manifest_path)
filtering = guard.filter_molecular("a549", known_smiles)
ood_reward = guard.reward_molecular("a549", "C")
disabled_exception = None
try:
    guard.reward_molecular("a549", known_smiles)
except RewardFineTuningDisabledError as exc:
    disabled_exception = str(exc)
template_guard = FilteringRewardGuard(manifest_path, template_path)
lut_filter = guard.filter_lut("1A1", "B10", 2)
default_status = guard.authorization_status()
template_status = template_guard.authorization_status()
result = {
    "python": sys.version,
    "versions": {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "rdkit": rdkit.__version__,
        "joblib": joblib.__version__,
    },
    "default_authorization": {
        "reward_fine_tuning_authorized": default_status["reward_fine_tuning_authorized"],
        "blocking_reasons": default_status["blocking_reasons"],
    },
    "template_authorization": {
        "reward_fine_tuning_authorized": template_status["reward_fine_tuning_authorized"],
        "blocking_reasons": template_status["blocking_reasons"],
    },
    "admitted_filtering": {
        "admitted": filtering["admitted"],
        "filtering_score_present": filtering["filtering_score"] is not None,
        "reward": filtering["reward"],
    },
    "ood_reward_request": {
        "admitted": ood_reward["admitted"],
        "filtering_score": ood_reward["filtering_score"],
        "reward": ood_reward["reward"],
    },
    "admitted_reward_request_blocked": disabled_exception is not None,
    "admitted_reward_block_message": disabled_exception,
    "lut_novel_known_component_pair": {
        "admitted": lut_filter["admitted"],
        "filtering_score_present": lut_filter["filtering_score"] is not None,
        "reward": lut_filter["reward"],
    },
}
assert result["admitted_filtering"] == {"admitted": True, "filtering_score_present": True, "reward": None}
assert result["ood_reward_request"] == {"admitted": False, "filtering_score": None, "reward": None}
assert result["admitted_reward_request_blocked"]
assert not result["default_authorization"]["reward_fine_tuning_authorized"]
assert not result["template_authorization"]["reward_fine_tuning_authorized"]
assert result["lut_novel_known_component_pair"] == {"admitted": True, "filtering_score_present": True, "reward": None}
print(json.dumps(result, sort_keys=True))
'''
        completed = subprocess.run(
            [
                str(python),
                "-c",
                verifier,
                str(relocated_manifest),
                str(relocated_root / relocated["admission_test"]["path"]),
                str(
                    relocated_root
                    / "configs/pan_lung_reward_preregistration.template.json"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=clean_env,
        )
        runtime = json.loads(completed.stdout)
        if runtime["versions"] != manifest["software"]:
            raise RuntimeError(
                f"clean runtime versions differ from bundle: {runtime['versions']} "
                f"!= {manifest['software']}"
            )

    output = {
        "format": "compose_pan_lung_filtering_clean_runtime_verification_v1",
        "status": "passed",
        "source_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "source_manifest_sha256": file_sha256(manifest_path),
        "runtime_isolation": {
            "fresh_virtual_environment": True,
            "project_package_copied_into_isolated_site_packages": True,
            "project_import_path": "fresh_venv/site-packages/compose_v4/__init__.py",
            "system_site_packages": True,
            "reason": (
                "The fresh environment reuses the exact preinstalled binary dependency "
                "versions recorded by the bundle while isolating project installation, "
                "process state, user site packages, and artifact location."
            ),
            "artifact_relocated_outside_repository": True,
            "temporary_runtime_removed_after_test": True,
        },
        "relocated_hash_checks": hash_checks,
        "runtime": runtime,
        "reward_fine_tuning_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(args.output.relative_to(REPO_ROOT)),
                "output_sha256": file_sha256(args.output),
                "hash_checks": len(hash_checks),
                "reward_fine_tuning_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
