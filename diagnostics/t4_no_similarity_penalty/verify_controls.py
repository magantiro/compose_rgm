"""Prove old arms are unchanged outside the contract-opt-in score addition."""

import ast
import json
import subprocess
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    contract_path = ROOT / "configs/t4_no_similarity_penalty.json"
    contract = json.loads(contract_path.read_text())
    old = contract["controls"]["implementation_revision"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changes = subprocess.check_output(
        ["git", "diff", "--name-only", old, head, "--", "src"], cwd=ROOT, text=True
    ).splitlines()
    beam = "src/compose_v4/experiments/t4_macro_beam.py"
    # Repair-neighbors was added between controls and this run but is not imported
    # by the beam or its dependencies. No existing scientific module may change.
    assert set(changes) <= {beam, "src/compose_v4/experiments/t4_repair_neighbors.py"}
    before = ast.parse(subprocess.check_output(["git", "show", f"{old}:{beam}"], cwd=ROOT))
    after = ast.parse((ROOT / beam).read_text())
    added = [n for n in after.body if getattr(n, "name", "") == "no_similarity_desirability"]
    assert len(added) == 1
    after.body.remove(added[0])
    additions = 0
    for node in ast.walk(after):
        if isinstance(node, ast.Tuple):
            new = [
                n
                for n in node.elts
                if isinstance(n, ast.Constant) and n.value == "no_similarity_desirability"
            ]
            for value in new:
                node.elts.remove(value)
                additions += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "score":
            branch = [
                n
                for n in node.body
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "'no_similarity_guidance' in contract"
            ]
            assert len(branch) == 1
            node.body.remove(branch[0])
            additions += 1
    assert additions == 2 and ast.dump(before) == ast.dump(after)
    prior = json.loads((ROOT / contract["controls"]["contract"]).read_text())
    assert prior["contract_sha256"] == contract["controls"]["contract_sha256"]
    for key in (
        "task",
        "search",
        "reference",
        "archive",
        "value_snapshot",
        "expected_input_sha256",
        "recovery_scale",
    ):
        assert contract[key] == prior[key], key
    assert contract["cases"][0] == {
        **prior["cases"][2],
        "retention_score": "no_similarity_desirability",
    }
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", old, "--", "src"], cwd=ROOT, text=True
    ).splitlines()
    publish_json(
        HERE / "control_reuse.json",
        {
            "schema_version": "t4_no_similarity_control_reuse_v1",
            "old_revision": old,
            "new_revision": head,
            "contract_sha256": sha256_file(contract_path),
            "verifier_sha256": sha256_file(Path(__file__)),
            "old_arm_ast_unchanged_after_removing_opt_in_additions": True,
            "unchanged_scientific_files": {p: sha256_file(ROOT / p) for p in files if p != beam},
            "scope": "old three cases only; new arm is an ablation, not equivalent",
        },
    )
    print("control reuse verified: existing scientific files and legacy beam paths unchanged")


if __name__ == "__main__":
    main()
