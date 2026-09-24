"""Audit already-saved de novo endpoints without sampling a new molecule.

The input is one or more directories of the immutable ring-marginal JSON shards.
This diagnostic never filters an endpoint or changes the production process.
Its RDKit version is recorded and its numbers remain exploratory until checked
against the pinned evaluation environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

SCHEMA = "denovo_postring_quality_audit_v1"
QED_FLOOR = 0.6
SA_CEILING = 4.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sa_components(mol: Chem.Mol) -> dict[str, float | int]:
    """Expose the installed RDKit SA scorer's exact additive terms.

    The reconstructed score is checked against ``calculateScore`` for every
    molecule, so this decomposition fails if the installed implementation moves.
    """

    if sascorer._fscores is None:
        sascorer.readFragmentScores()
    fingerprint = sascorer.mfpgen.GetSparseCountFingerprint(mol)
    nonzero = fingerprint.GetNonzeroElements()
    observations = sum(nonzero.values())
    fragment = (
        sum(sascorer._fscores.get(bit, -4) * count for bit, count in nonzero.items()) / observations
    )
    atoms = mol.GetNumAtoms()
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    bridgeheads, spiro = sascorer.numBridgeheadsAndSpiro(mol, mol.GetRingInfo())
    macrocycles = sum(len(ring) > 8 for ring in mol.GetRingInfo().AtomRings())
    penalties = {
        "size": atoms**1.005 - atoms,
        "stereo": math.log10(chiral + 1),
        "spiro": math.log10(spiro + 1),
        "bridgehead": math.log10(bridgeheads + 1),
        "macrocycle": math.log10(2) if macrocycles else 0.0,
    }
    density = math.log(atoms / len(nonzero)) * 0.5 if atoms > len(nonzero) else 0.0
    raw = fragment - sum(penalties.values()) + density
    reconstructed = 11.0 - (raw + 5.0) / 6.5 * 9.0
    if reconstructed > 8.0:
        reconstructed = 8.0 + math.log(reconstructed - 8.0)
    reconstructed = min(10.0, max(1.0, reconstructed))
    official = float(sascorer.calculateScore(mol))
    if not math.isclose(reconstructed, official, abs_tol=1e-10):
        raise RuntimeError(f"SA decomposition mismatch {reconstructed} != {official}")
    return {
        "fragment_score": float(fragment),
        "fingerprint_density_bonus": float(density),
        "size_penalty": float(penalties["size"]),
        "stereo_penalty": float(penalties["stereo"]),
        "spiro_penalty": float(penalties["spiro"]),
        "bridgehead_penalty": float(penalties["bridgehead"]),
        "macrocycle_penalty": float(penalties["macrocycle"]),
        "chiral_centers": chiral,
        "bridgeheads": bridgeheads,
        "spiro_centers": spiro,
        "macrocycles": macrocycles,
        "score": official,
    }


def characterize(record: dict, arm: str, shard: Path) -> dict:
    text = str(record.get("smiles") or "")
    mol = Chem.MolFromSmiles(text) if text else None
    row = {
        "arm": arm,
        "shard": shard.name,
        "index": record["index"],
        "trajectory_seed": record["trajectory_seed"],
        "smiles": text,
        "events": record["events"],
        "event_rule_counts": dict(sorted(Counter(record["event_rules"]).items())),
        "valid_state": record["valid_state"],
        "connected": record["connected"],
        "ring_plan": record.get("ring_plan"),
    }
    if mol is None:
        return {**row, "group": "NO_ENDPOINT"}
    canonical = Chem.MolToSmiles(mol)
    qed = float(QED.qed(mol))
    sa = sa_components(mol)
    qpass, spass = qed >= QED_FLOOR, sa["score"] <= SA_CEILING
    group = ("QED_PASS" if qpass else "QED_FAIL") + "_" + ("SA_PASS" if spass else "SA_FAIL")
    atom_rings = mol.GetRingInfo().AtomRings()
    rings = [len(ring) for ring in atom_rings]
    nonaromatic_unsaturated_5_6 = 0
    hetero_nonaromatic_unsaturated_5_6 = 0
    ring_bond_patterns = []
    for ring in atom_rings:
        bonds = [
            mol.GetBondBetweenAtoms(ring[index], ring[(index + 1) % len(ring)])
            for index in range(len(ring))
        ]
        aromatic = all(bond.GetIsAromatic() for bond in bonds)
        doubles = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds)
        hetero = any(mol.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in ring)
        ring_bond_patterns.append(
            {
                "size": len(ring),
                "aromatic": aromatic,
                "double_bonds": doubles,
                "heteroatoms": hetero,
            }
        )
        if len(ring) in (5, 6) and doubles and not aromatic:
            nonaromatic_unsaturated_5_6 += 1
            hetero_nonaromatic_unsaturated_5_6 += int(hetero)
    symbols = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    return {
        **row,
        "canonical_smiles": canonical,
        "group": group,
        "qed": qed,
        "sa": sa,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rings": rings,
        "ring_bond_patterns": ring_bond_patterns,
        "nonaromatic_unsaturated_5_6_rings": nonaromatic_unsaturated_5_6,
        "hetero_nonaromatic_unsaturated_5_6_rings": hetero_nonaromatic_unsaturated_5_6,
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "elements": dict(sorted(symbols.items())),
    }


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else None


def _group_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = (
        "qed",
        "heavy_atoms",
        "molecular_weight",
        "clogp",
        "tpsa",
        "rotatable_bonds",
        "aromatic_rings",
        "formal_charge",
    )
    summary = {"n": len(rows), **{f"mean_{key}": _mean(rows, key) for key in keys}}
    summary["mean_sa"] = sum(row["sa"]["score"] for row in rows) / len(rows)
    for key in (
        "fragment_score",
        "size_penalty",
        "stereo_penalty",
        "spiro_penalty",
        "bridgehead_penalty",
        "macrocycle_penalty",
        "fingerprint_density_bonus",
    ):
        summary[f"mean_sa_{key}"] = sum(row["sa"][key] for row in rows) / len(rows)
    summary["fraction_with_strained_ring"] = sum(
        3 in row["rings"] or 4 in row["rings"] for row in rows
    ) / len(rows)
    summary["fraction_with_bridgehead"] = sum(row["sa"]["bridgeheads"] > 0 for row in rows) / len(
        rows
    )
    summary["fraction_with_spiro"] = sum(row["sa"]["spiro_centers"] > 0 for row in rows) / len(rows)
    summary["mean_nonaromatic_unsaturated_5_6_rings"] = _mean(
        rows, "nonaromatic_unsaturated_5_6_rings"
    )
    summary["fraction_with_hetero_nonaromatic_unsaturated_5_6_ring"] = sum(
        row["hetero_nonaromatic_unsaturated_5_6_rings"] > 0 for row in rows
    ) / len(rows)
    # Element marginals diagnose (rather than presume) halogen and heteroatom excess.
    # Count atoms and affected molecules separately; one heavily fluorinated outlier
    # must not be mistaken for broad fluorination of the generated distribution.
    for element in ("F", "Cl", "Br", "I", "N", "O", "S", "P"):
        counts = [row["elements"].get(element, 0) for row in rows]
        summary[f"mean_{element}_atoms"] = sum(counts) / len(rows)
        summary[f"fraction_with_{element}"] = sum(count > 0 for count in counts) / len(rows)
    rules = Counter()
    for row in rows:
        rules.update(row.get("event_rule_counts", {}))
    summary["mean_event_rules"] = {name: count / len(rows) for name, count in sorted(rules.items())}
    return summary


def _grid(rows: list[dict], path: Path, *, maximum: int = 50) -> int:
    selected = sorted(
        rows,
        key=lambda row: hashlib.blake2b(
            f"{row['arm']}:{row['trajectory_seed']}".encode(), digest_size=8
        ).digest(),
    )[:maximum]
    if not selected:
        return 0
    mols = [Chem.MolFromSmiles(row["canonical_smiles"]) for row in selected]
    legends = [
        f"{row['arm']} #{row['index']} Q{row['qed']:.2f} SA{row['sa']['score']:.2f}"
        for row in selected
    ]
    drawing = Draw.MolsToGridImage(
        mols, molsPerRow=5, subImgSize=(330, 245), legends=legends, useSVG=False
    )
    drawing.save(str(path))
    return len(selected)


def audit(shard_dirs: list[Path], output: Path, official_report: Path) -> dict:
    shards = sorted({path for directory in shard_dirs for path in directory.rglob("*.json")})
    if not shards:
        raise ValueError("no JSON shards found")
    rows: list[dict] = []
    inputs = []
    designs: dict[str, str] = {}
    for path in shards:
        payload = json.loads(path.read_text())
        arm = payload["arm"]
        design = payload["design"]
        if arm in designs and designs[arm] != design:
            raise ValueError(f"mixed design for {arm}: {design} != {designs[arm]}")
        designs[arm] = design
        if len(payload["records"]) != payload["stop"] - payload["start"]:
            raise ValueError(f"incomplete shard {path}")
        inputs.append(
            {"path": str(path), "sha256": sha256(path), "records": len(payload["records"])}
        )
        rows.extend(characterize(record, arm, path) for record in payload["records"])
    identities = [(row["arm"], row["index"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate arm/index across shards")
    if any(
        row["group"] != "NO_ENDPOINT" and not (row["valid_state"] and row["connected"])
        for row in rows
    ):
        raise ValueError("saved committed endpoint violates exact-validity invariant")
    official = json.loads(official_report.read_text())

    output.mkdir(parents=True, exist_ok=True)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
        by_group[f"{row['arm']}:{row['group']}"].append(row)
    summaries = {}
    grids = {}
    for arm, members in sorted(by_arm.items()):
        valid = [row for row in members if row["group"] != "NO_ENDPOINT"]
        quality_count = sum(row["group"] == "QED_PASS_SA_PASS" for row in valid)
        expected = official["arms"][arm]["published_metrics"]
        observed = {
            "attempted": len(members),
            "valid": len(valid),
            "unique": len({row["canonical_smiles"] for row in valid}),
            "high_quality": quality_count,
            "mean_qed": _mean(valid, "qed"),
            "mean_sa": sum(row["sa"]["score"] for row in valid) / len(valid) if valid else None,
        }
        for key, value in observed.items():
            if not math.isclose(value, expected[key], abs_tol=1e-10):
                raise ValueError(
                    f"official report parity failed for {arm}/{key}: {value} != {expected[key]}"
                )
        plan_rows = [row for row in valid if row["ring_plan"] is not None]
        summaries[arm] = {
            "attempted": len(members),
            "endpoint_count": len(valid),
            "unique_endpoints": len({row["canonical_smiles"] for row in valid}),
            "groups": dict(sorted(Counter(row["group"] for row in members).items())),
            "mean_qed": _mean(valid, "qed"),
            "mean_sa": sum(row["sa"]["score"] for row in valid) / len(valid) if valid else None,
            "mean_heavy_atoms": _mean(valid, "heavy_atoms"),
            "mean_aromatic_rings": _mean(valid, "aromatic_rings"),
            "mean_sa_fragment_score": sum(row["sa"]["fragment_score"] for row in valid) / len(valid)
            if valid
            else None,
            "mean_sa_bridgehead_penalty": sum(row["sa"]["bridgehead_penalty"] for row in valid)
            / len(valid)
            if valid
            else None,
            "mean_sa_spiro_penalty": sum(row["sa"]["spiro_penalty"] for row in valid) / len(valid)
            if valid
            else None,
            "official_metric_parity": observed,
            "by_plan_completion": {
                status: _group_summary(
                    [
                        row
                        for row in plan_rows
                        if row["ring_plan"]["fully_realized"] == (status == "complete")
                    ]
                )
                for status in ("complete", "incomplete")
            },
            "by_quality_group": {
                group: _group_summary([row for row in valid if row["group"] == group])
                for group in (
                    "QED_PASS_SA_PASS",
                    "QED_PASS_SA_FAIL",
                    "QED_FAIL_SA_PASS",
                    "QED_FAIL_SA_FAIL",
                )
            },
        }
        grids[f"{arm}:random"] = _grid(valid, output / f"{arm.lower()}_random.png", maximum=100)
    for label, members in sorted(by_group.items()):
        if label.endswith("NO_ENDPOINT"):
            continue
        grids[label] = _grid(members, output / f"{label.lower()}.png")

    by_arm_seed = {
        arm: {row["trajectory_seed"]: row for row in members} for arm, members in by_arm.items()
    }
    paired = None
    if {"A", "C1"}.issubset(by_arm_seed):
        seeds = sorted(set(by_arm_seed["A"]) & set(by_arm_seed["C1"]))
        if not seeds:
            raise ValueError("A and C1 have no shared trajectory seeds")
        a_rows = [by_arm_seed["A"][seed] for seed in seeds]
        c_rows = [by_arm_seed["C1"][seed] for seed in seeds]
        if any(row["group"] == "NO_ENDPOINT" for row in a_rows + c_rows):
            raise ValueError("paired A/C1 diagnostic requires committed endpoints")
        paired = {
            "n_shared_seeds": len(seeds),
            "a": _group_summary(a_rows),
            "c1": _group_summary(c_rows),
            "c1_minus_a_mean_F_atoms": sum(
                c["elements"].get("F", 0) - a["elements"].get("F", 0)
                for a, c in zip(a_rows, c_rows)
            )
            / len(seeds),
            "F_atom_count_change": dict(
                sorted(
                    Counter(
                        c["elements"].get("F", 0) - a["elements"].get("F", 0)
                        for a, c in zip(a_rows, c_rows)
                    ).items()
                )
            ),
        }

    manifest = {
        "schema_version": SCHEMA,
        "evidence_class": "computed_exploratory_local_rdkit",
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_script_sha256": sha256(Path(__file__)),
        "inputs": inputs,
        "official_report": {"path": str(official_report), "sha256": sha256(official_report)},
        "designs": dict(sorted(designs.items())),
        "configuration": {
            "qed_floor": QED_FLOOR,
            "sa_ceiling": SA_CEILING,
            "grid_selection": "BLAKE2b-8 of arm:trajectory_seed",
        },
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "platform": platform.platform(),
        },
        "summaries": summaries,
        "paired_A_C1": paired,
        "grids": grids,
    }
    (output / "records.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    (output / "report.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.shards, args.output, args.official_report)
    print(json.dumps(report["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
