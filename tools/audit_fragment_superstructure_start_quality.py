"""Compare superstructure output quality with the supplied starting cores.

This is a post-hoc development diagnostic, not a sampler-selection result.
It uses the pinned official property implementation for both populations and
does not modify a prompt, sample, quality threshold, or official denominator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import rdkit
from fetch_official_fragment_evaluator import OFFICIAL_BLOBS, verify_only

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import _official_module_path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "diagnostics/fragment_initial_family_dev_v1"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
QUALITY = FOLDER / "quality_seed10_n20_all10.json"
OUTPUT = FOLDER / "start_quality_seed10_n20_all10.json"
ARMS = ("baseline", "conditioned", "conditioned_strict")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    verified = verify_only()
    property_hash = OFFICIAL_BLOBS["in_virtuo_gen/utils/mol.py"][1]
    if property_hash not in verified.values():
        raise RuntimeError("official property implementation hash not verified")
    _official_module_path()
    from in_virtuo_gen.utils.mol import compute_single_property

    quality = json.loads(QUALITY.read_text())
    if quality["schema"] != "fragment_initial_family_quality_audit_v1":
        raise RuntimeError(f"unexpected quality artifact schema: {QUALITY}")
    prompts = {
        prompt.drug_name: prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task == FragmentTask.SUPERSTRUCTURE_GENERATION
    }
    if len(prompts) != 10:
        raise RuntimeError(f"expected ten superstructure prompts in {PROMPTS}")
    molecules = defaultdict(list)
    for row in quality["molecules"]:
        key = (row["arm"], row["drug"])
        if row["arm"] not in ARMS or row["drug"] not in prompts:
            raise RuntimeError(f"unexpected molecule group: {key}")
        molecules[key].append(row)
    if set(molecules) != {(arm, drug) for arm in ARMS for drug in prompts}:
        raise RuntimeError("missing arm/prompt molecule group")

    rows = []
    for drug, prompt in sorted(prompts.items()):
        start = build_prompt_context(prompt)
        if start.start_smiles is None:
            raise RuntimeError(f"starting core did not serialize: {drug}")
        start_sa, start_qed = compute_single_property(start.start_smiles)
        for arm in ARMS:
            cohort = molecules[(arm, drug)]
            rows.append(
                {
                    "arm": arm,
                    "drug": drug,
                    "start_smiles": start.start_smiles,
                    "start_sa": float(start_sa),
                    "start_qed": float(start_qed),
                    "start_quality_pass": bool(start_qed >= 0.6 and start_sa <= 4.0),
                    "unique_emitted": len(cohort),
                    "quality_pass": sum(bool(item["quality_pass"]) for item in cohort),
                    "sa_improved_from_start": sum(item["sa"] < start_sa for item in cohort),
                    "qed_improved_from_start": sum(item["qed"] > start_qed for item in cohort),
                    "mean_sa_delta": sum(item["sa"] - start_sa for item in cohort) / len(cohort),
                    "mean_qed_delta": sum(item["qed"] - start_qed for item in cohort) / len(cohort),
                }
            )

    result = {
        "schema": "fragment_superstructure_start_quality_audit_v1",
        "evidence_role": "Post-hoc development diagnosis; not controller selection or official comparison",
        "input_sha256": {
            str(PROMPTS.relative_to(ROOT)): sha256(PROMPTS),
            str(QUALITY.relative_to(ROOT)): sha256(QUALITY),
        },
        "official_property_sha256": property_hash,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
                ROOT / "src/compose_v4/benchmark/fragment_constrained.py",
            )
        },
        "auditor_sha256": sha256(Path(__file__)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {"python": sys.version.split()[0], "rdkit": rdkit.__version__},
        "rows": rows,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=FOLDER, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
