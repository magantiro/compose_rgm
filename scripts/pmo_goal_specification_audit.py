"""What does each PMO task's PUBLIC definition declare? Zero oracle calls.

WHY THIS EXISTS
---------------
The transport controller turns a DECLARED GOAL into executable COMPOSE programs. Which
goals exist is therefore a fact about the benchmark, and it has to be read from the
benchmark source rather than assumed or hand-written -- a hand-written target list is
exactly the per-task content the architecture forbids.

INFORMATION BOUNDARY, and it moved on evidence. A target structure written as a literal
in the PUBLIC task definition is a task INPUT, in the same sense that delta and the start
molecule are inputs in T4. VERIFIED in the pinned PyTDC 1.1.15 this repository already
pins:

    celecoxib_rediscovery = rediscovery_meta(
        target_smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", ...)

(An earlier note placed this at line 748; in the pinned version it is line 921. The claim
holds, the line number did not -- which is why this reads the pinned source itself.)

"No prescreen" is about not pre-scoring the ~250k ZINC set to build task-specific
vocabularies, NOT about ignoring declared task structure. What must stay general is the
CONTROLLER that turns an arbitrary declared goal into programs.

This audit does NOT execute TDC. It parses the module with `ast`, so no predictor pickle
is downloaded and no oracle is constructed -- which also means a near-miss task name
cannot silently resolve to a different oracle through TDC's fuzzy matcher.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

#: Goal kinds the transport planner must be able to consume.
DECLARED_TARGET = "declared_target_structure"
DECLARED_PAIR = "declared_target_pair"
DECLARED_FORMULA = "declared_molecular_formula"
DECLARED_SMARTS = "declared_smarts"
DECLARED_REFERENCE = "declared_reference_structure_in_composite"
BLACK_BOX = "black_box_scalar_only"
UNRESOLVED = "unresolved"

#: Constructors whose keyword arguments carry the declared goal.
TARGET_CALLS = {
    "rediscovery_meta": DECLARED_TARGET,
    "similarity_meta": DECLARED_TARGET,
    "isomer_meta": DECLARED_FORMULA,
    "isomer_meta_prev": DECLARED_FORMULA,
    "median_meta": DECLARED_PAIR,
}

_FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def _classify_structure(name: str, value: str) -> str | None:
    """SMARTS, formula or SMILES. The VARIABLE NAME is the primary signal.

    A SMARTS such as "CN(C=O)Cc1ccc(c2ccccc2)cc1" also parses as SMILES, so RDKit alone
    cannot separate them; TDC's own naming does, and RDKit then validates.
    """
    lowered = name.lower()
    if "smarts" in lowered:
        return "smarts" if Chem.MolFromSmarts(value) is not None else None
    if _FORMULA.match(value) and not Chem.MolFromSmiles(value):
        return "formula"
    if Chem.MolFromSmiles(value) is not None and len(value) > 4:
        return "smiles"
    return None


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level string constants, so a kwarg naming one can be resolved.

    `median1` passes `target_smiles_1=camphor_smiles` -- a NAME, not a literal. Reading
    only literals reported median1 as carrying no declared content, which is false.
    """
    constants: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


def _declared_in_body(node: ast.AST, constants: dict[str, str]) -> list[dict]:
    """Every declared structure inside a function or class body.

    TDC's MPO tasks assign their reference molecule to a LOCAL variable
    (`osimertinib_smiles = "COc1cc(...)"`) rather than passing it as a keyword, and
    `valsartan_smarts` does the same with its SMARTS. Scanning keywords only classified
    all of them as black-box, which is wrong: they declare a reference structure.
    """
    found: list[dict] = []
    for child in ast.walk(node):
        pairs: list[tuple[str, str]] = []
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Constant):
            if isinstance(child.value.value, str):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        pairs.append((target.id, child.value.value))
        elif isinstance(child, ast.Call):
            for keyword in child.keywords:
                if not keyword.arg:
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    pairs.append((keyword.arg, keyword.value.value))
                elif isinstance(keyword.value, ast.Name) and keyword.value.id in constants:
                    pairs.append((keyword.arg, constants[keyword.value.id]))
        for name, value in pairs:
            kind = _classify_structure(name, value)
            if kind:
                found.append({"name": name, "value": value, "kind": kind})
    # Deduplicate, preserving order.
    seen, unique = set(), []
    for row in found:
        key = (row["kind"], row["value"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def audit(oracle_source: Path, tasks: list[str]) -> dict:
    tree = ast.parse(oracle_source.read_text())
    constants = _module_constants(tree)
    assignments: dict[str, ast.AST] = {}
    bodies: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # ClassDef matters: `jnk3` is a CLASS, and scanning only Assign and
            # FunctionDef reported it unresolved.
            bodies[node.name] = node

    rows = []
    for task in sorted(tasks):
        row: dict = {"task": task}
        value = assignments.get(task)
        if isinstance(value, ast.Call):
            callee = (
                value.func.id if isinstance(value.func, ast.Name)
                else getattr(value.func, "attr", "?")
            )
            row["constructor"] = callee
            declared = _declared_in_body(value, constants)
            row["declared"] = declared
            row["goal_kind"] = TARGET_CALLS.get(
                callee, DECLARED_REFERENCE if declared else BLACK_BOX
            )
        elif task in bodies:
            node = bodies[task]
            row["constructor"] = type(node).__name__
            declared = _declared_in_body(node, constants)
            row["declared"] = declared
            kinds = {item["kind"] for item in declared}
            if "smarts" in kinds:
                row["goal_kind"] = DECLARED_SMARTS
            elif "formula" in kinds:
                row["goal_kind"] = DECLARED_FORMULA
            elif "smiles" in kinds:
                row["goal_kind"] = DECLARED_REFERENCE
            else:
                row["goal_kind"] = BLACK_BOX
        else:
            row["declared"] = []
            row["goal_kind"] = UNRESOLVED
        rows.append(row)
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle-source",
        type=Path,
        default=Path(
            ROOT.parent
            / "compose_rgm_git/.worktrees/t4-objective-dynamic-reset-20260916/.uv-cache"
            / "sdists-v7/pypi/pytdc/1.1.15/1BbE1CByMlrf4ye7noADs/src/tdc/chem_utils/oracle/oracle.py"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.oracle_source.exists():
        print(f"ABORT: pinned TDC oracle source not found at {args.oracle_source}")
        return 2
    targets = json.loads((ROOT / "docs/invirtuogen_pmo_targets.json").read_text())["targets"]
    tasks = sorted(targets)
    result = audit(args.oracle_source, tasks)

    by_kind: dict[str, list[str]] = {}
    for row in result["rows"]:
        by_kind.setdefault(row["goal_kind"], []).append(row["task"])
    for row in result["rows"]:
        declared = row.get("declared") or []
        detail = f"{len(declared):>2} declared  " + (
            declared[0]["value"][:44] if declared else ""
        )
        print(f"{row['task']:<30} {row['goal_kind']:<40} {detail}")
    print()
    for kind, names in sorted(by_kind.items()):
        print(f"{kind:<28} {len(names):>2}  {', '.join(names)}")

    report = {
        "schema_version": "pmo_goal_specification_audit_v1",
        "oracle_calls": 0,
        "tdc_version": "1.1.15 (pinned)",
        "oracle_source": str(args.oracle_source),
        "tasks": len(tasks),
        "by_kind": {kind: sorted(names) for kind, names in sorted(by_kind.items())},
        "rows": result["rows"],
        "note": (
            "Parsed with ast; TDC is never imported and no oracle is constructed, so no "
            "predictor pickle is downloaded and no task name can resolve through TDC's "
            "fuzzy matcher to a different oracle."
        ),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
