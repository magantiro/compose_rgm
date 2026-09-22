"""Kernel parity for the PMO transport path: the INTERMEDIATES, not just the endpoints.

The PMO production image pins **rdkit 2023.9.6** (``modal_apps/pmo_population_v1_app.py``
-- PyTDC 1.1.15 requires ``rdkit>=2023.9.5,<2024.3.1``, so the 2024.3.5 kernel that T4 and
the editing corpus pin is flatly unsatisfiable there).  The transport correspondence,
stage splitter and ordering selector were all first measured under 2024.3.5.

This repository has already measured that two rdkit versions can write DIFFERENT canonical
SMILES for the same molecule, and that a canonical SMILES is used as a CACHE KEY -- so a
spelling difference is not cosmetic in a procedure that dedupes on the string.  It has also
measured that parity on drug-like molecules says nothing about parity on the exotic
intermediates a search constructs.

So this probe does not sample endpoints.  It walks the FULL prefix lattice of
``build_intermediate`` -- every (deletions, bond changes, installations) prefix of every
correspondence -- which is precisely where the partially-installed rings, bare heteroatom
fragments and unkekulizable pieces live, and records for each:

  * the canonical SMILES, or ``null`` when the intermediate does not sanitize (the
    sanitize-or-not verdict is itself a kernel-sensitive observable and is compared)

Run once per kernel with ``--out``, then ``--compare A B``.  The comparator refuses two
dumps produced by the same rdkit build, because a parity check between one kernel and
itself cannot fail and would report a vacuous PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.pmo_transport_correspondence import correspondences
from compose_v4.control.pmo_transport_staging import build_intermediate, split

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

#: Strings whose canonicalization is known or suspected to move between rdkit builds --
#: the Kekule-degenerate hypervalent-sulfur class this repo measured diverging between
#: 2024.3.5 and 2026.03.6, plus charged, radical and strained forms.  Present so the
#: probe carries observables capable of DIFFERING; a parity report over a canary set
#: that cannot vary is not a measurement.
CANARY = (
    "C1CC[SH4]CC1",
    "O=C=[SH4]",
    "CC(=O)[O-]",
    "C[N+](C)(C)CC(=O)[O-]",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "C1=CC2=NC=C(CCCc3ccccc3)C2C=C1n1cnnc1",
    "N#CCNOCO",
    "C1COOCOOCCC2OOOO1",
    "O=C1NC(O)C2CCCCC12",
    "[CH]c1ccccc1",
    "c1ccsc1",
    "c1cc[se]c1",
    "C1CC1",
    "C1CN1",
    "OO",
)


def best_start(bank: list[str], target: str, *, top: int) -> list[str]:
    mol = Chem.MolFromSmiles(target)
    fingerprint = _GEN.GetFingerprint(mol)
    scored = []
    for smiles in bank:
        candidate = Chem.MolFromSmiles(smiles)
        if candidate is None:
            continue
        scored.append(
            (DataStructs.TanimotoSimilarity(fingerprint, _GEN.GetFingerprint(candidate)), smiles)
        )
    scored.sort(reverse=True)
    return [smiles for _, smiles in scored[:top]]


def lattice(correspondence, *, stride: int) -> list[tuple[int, int, int]]:
    """Every (deleted, changed, installed) prefix, strided so a large transport does not
    dominate the budget.  The endpoints of every axis are always included."""

    def axis(size: int) -> list[int]:
        points = list(range(0, size + 1, stride))
        if points[-1] != size:
            points.append(size)
        return points

    return [
        (d, c, i)
        for d in axis(len(correspondence.r_delete))
        for c in axis(len(correspondence.core_bond_changes))
        for i in axis(len(correspondence.h_install))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--stage-endpoints",
        action="store_true",
        help="dump only the stage endpoints the staged path actually visits, "
             "instead of the full off-path lattice",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare", type=Path, nargs=2, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--artifact-parity",
        type=Path,
        nargs=2,
        default=None,
        metavar=("PYTHON_A", "PYTHON_B"),
        help="run the three derived-artifact scripts under two interpreters and diff",
    )
    args = parser.parse_args()

    if args.artifact_parity:
        return artifact_parity(*args.artifact_parity, report=args.report)
    if args.compare:
        return compare(*args.compare, report=args.report)

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    if isinstance(bank, dict):
        bank = bank.get("smiles") or bank.get("molecules")

    records: dict[str, str | None] = {}
    alignments = 0
    for entry in audit["rows"]:
        declared = [d for d in entry.get("declared", []) if d["kind"] == "smiles"]
        if not declared:
            continue
        target = declared[0]["value"]
        for source in best_start(bank, target, top=args.starts):
            for index, correspondence in enumerate(correspondences(source, target)):
                alignments += 1
                if args.stage_endpoints:
                    for interleave in (False, True):
                        for stage in split(correspondence, interleave=interleave):
                            key = (
                                f"{entry['task']}|{source}|{index}"
                                f"|{int(interleave)}|{stage.index}"
                            )
                            records[key] = stage.endpoint_smiles
                    continue
                for deleted, changed, installed in lattice(correspondence, stride=args.stride):
                    key = f"{entry['task']}|{source}|{index}|{deleted}|{changed}|{installed}"
                    records[key] = build_intermediate(
                        correspondence,
                        deleted=deleted,
                        changed=changed,
                        installed=installed,
                    )

    canary = {smiles: _canonical(smiles) for smiles in CANARY}
    non_sanitizing = sum(1 for value in records.values() if value is None)
    payload = {
        "schema_version": "pmo_transport_kernel_parity_v1",
        "rdkit_version": rdkit.__version__,
        "mode": "stage_endpoints" if args.stage_endpoints else "offpath_lattice",
        "oracle_calls": 0,
        "alignments": alignments,
        "intermediates": len(records),
        "non_sanitizing": non_sanitizing,
        "distinct_canonical_smiles": len({v for v in records.values() if v is not None}),
        "canary": canary,
        "records": records,
    }
    payload["records_sha256"] = _digest(records)
    print(
        f"rdkit {rdkit.__version__}: alignments={alignments} "
        f"intermediates={len(records)} non_sanitizing={non_sanitizing} "
        f"distinct={payload['distinct_canonical_smiles']} "
        f"records_sha256={payload['records_sha256'][:16]}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


#: The derived artifacts whose numbers the branch reports. Each is deterministic and
#: takes ``--out``, so "does this kernel change the finding" is a byte comparison.
DERIVED_SCRIPTS = (
    "scripts/pmo_transport_correspondence_validation.py",
    "scripts/pmo_transport_staging_validation.py",
    "scripts/pmo_transport_ordering_policy.py",
)


def artifact_parity(python_a: Path, python_b: Path, *, report: Path | None = None) -> int:
    """Re-derive every reported artifact under two interpreters and compare bytes.

    The interpreters are compared by their rdkit build, not by their path: two paths that
    resolve to the same build would make the check vacuous, so that case is refused.
    """
    import subprocess
    import tempfile

    versions = {
        str(path): subprocess.run(
            [str(path), "-c", "import rdkit; print(rdkit.__version__)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for path in (python_a, python_b)
    }
    if versions[str(python_a)] == versions[str(python_b)]:
        print(f"REFUSED: both interpreters are rdkit {versions[str(python_a)]}.")
        return 2

    rows = []
    agreed = True
    with tempfile.TemporaryDirectory() as scratch:
        for script in DERIVED_SCRIPTS:
            digests = {}
            for path in (python_a, python_b):
                out = Path(scratch) / f"{Path(script).stem}.{versions[str(path)]}.json"
                subprocess.run([str(path), script, "--out", str(out)],
                               cwd=ROOT, capture_output=True, text=True, check=False)
                digests[versions[str(path)]] = (
                    hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None
                )
            identical = len(set(digests.values())) == 1 and None not in digests.values()
            agreed &= identical
            rows.append({"script": script, "sha256_by_rdkit": digests, "identical": identical})
            print(f"  {'OK  ' if identical else 'DIFF'} {script}")

    outcome = "DERIVED_ARTIFACT_PARITY" if agreed else "DERIVED_ARTIFACT_DIVERGENCE"
    print("VERDICT:", outcome)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({
            "schema_version": "pmo_transport_kernel_artifact_parity_v1",
            "oracle_calls": 0,
            "rdkit_versions": sorted(set(versions.values())),
            "rows": rows,
            "verdict": outcome,
        }, indent=2, sort_keys=True) + "\n")
        print(f"wrote {report}")
    return 0 if agreed else 1


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def _digest(records: dict[str, str | None]) -> str:
    stream = hashlib.sha256()
    for key in sorted(records):
        stream.update(key.encode())
        stream.update(b"\x00")
        stream.update((records[key] or "\x01NONE").encode())
        stream.update(b"\x00")
    return stream.hexdigest()


def compare(left_path: Path, right_path: Path, *, report: Path | None = None) -> int:
    left = json.loads(left_path.read_text())
    right = json.loads(right_path.read_text())
    if left.get("mode") != right.get("mode"):
        print(
            f"REFUSED: dumps are different modes "
            f"({left.get('mode')} vs {right.get('mode')})."
        )
        return 2
    if left["rdkit_version"] == right["rdkit_version"]:
        print(
            "REFUSED: both dumps were produced by rdkit "
            f"{left['rdkit_version']}. A parity check against the same kernel cannot fail."
        )
        return 2

    left_records, right_records = left["records"], right["records"]
    only_left = sorted(set(left_records) - set(right_records))
    only_right = sorted(set(right_records) - set(left_records))
    shared = sorted(set(left_records) & set(right_records))

    spelling, verdict = [], []
    for key in shared:
        lhs, rhs = left_records[key], right_records[key]
        if lhs == rhs:
            continue
        (verdict if (lhs is None) != (rhs is None) else spelling).append((key, lhs, rhs))

    canary_moves = [
        (smiles, left["canary"][smiles], right["canary"][smiles])
        for smiles in sorted(set(left["canary"]) & set(right["canary"]))
        if left["canary"][smiles] != right["canary"][smiles]
    ]

    print(f"{left['rdkit_version']}  vs  {right['rdkit_version']}")
    print(f"  intermediates compared      {len(shared)}")
    print(f"  keys only in one dump       {len(only_left)} / {len(only_right)}")
    print(f"  CANONICAL SMILES DISAGREE   {len(spelling)}")
    print(f"  SANITIZE VERDICT DISAGREE   {len(verdict)}")
    print(f"  canary strings disagreeing  {len(canary_moves)} of {len(left['canary'])}")
    for key, lhs, rhs in (spelling + verdict)[:20]:
        print(f"    {key}\n      {lhs}\n      {rhs}")
    for smiles, lhs, rhs in canary_moves[:20]:
        print(f"    canary {smiles}\n      {lhs}\n      {rhs}")
    agreed = not (spelling or verdict or only_left or only_right)
    outcome = "KERNEL_PARITY" if agreed else "KERNEL_DIVERGENCE"
    print("VERDICT:", outcome)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "schema_version": "pmo_transport_kernel_parity_report_v1",
                    "oracle_calls": 0,
                    "left": _side(left),
                    "right": _side(right),
                    "intermediates_compared": len(shared),
                    "keys_only_left": len(only_left),
                    "keys_only_right": len(only_right),
                    "canonical_smiles_disagreements": len(spelling),
                    "sanitize_verdict_disagreements": len(verdict),
                    "canary_strings": len(left["canary"]),
                    "canary_disagreements": len(canary_moves),
                    "disagreement_examples": [
                        {"key": key, "left": lhs, "right": rhs}
                        for key, lhs, rhs in (spelling + verdict)[:20]
                    ],
                    "verdict": outcome,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {report}")
    return 0 if agreed else 1


def _side(dump: dict) -> dict:
    return {
        "rdkit_version": dump["rdkit_version"],
        "mode": dump.get("mode"),
        "alignments": dump["alignments"],
        "intermediates": dump["intermediates"],
        "non_sanitizing": dump["non_sanitizing"],
        "distinct_canonical_smiles": dump["distinct_canonical_smiles"],
        "records_sha256": dump["records_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
