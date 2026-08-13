"""The shared canonicalizer/evaluator, run under the PRODUCTION-PINNED RDKit.

    baseline-native environment
            |  raw SMILES
    THIS SERVER, pinned to production RDKit 2024.3.5
            |  canonical key + validity + oracle score

Baselines keep whatever RDKit their implementation requires -- forcing them all
into one environment breaks the old ones. Every molecule they return crosses the
boundary as a raw SMILES string and is canonicalized and scored exactly once,
here, so ``unique_valid_canonical_evaluations`` is genuinely comparable across
methods and canonicalization differences cannot silently alter a budget.

Protocol: line-delimited JSON on stdin/stdout. One request per line::

    {"smiles": "c1ccccc1"}            -> {"canonical": "c1ccccc1", "valid": true,
                                          "score": 0.42}
    {"smiles": "C1CC"}                -> {"canonical": null, "valid": false,
                                          "score": null}
    {"cmd": "identity"}               -> {"rdkit": "2024.03.5", ...}

Run it under the pinned interpreter, never the project one:

    /tmp/baseline_envs/rdkit-prod-2024_3_5/bin/python \\
        scripts/compose_shared_evaluator_server.py --objective developability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
NORMALIZERS = REPO / "diagnostics" / "retarget_goal_language_normalizers.json"

#: The pin this server exists to enforce. Refusing to start under anything else
#: is the point: a server that silently ran under a different RDKit would
#: reintroduce exactly the drift it was built to remove.
REQUIRED_RDKIT = "2024.03.5"


def developability_objective():
    normalizers = json.loads(NORMALIZERS.read_text())["normalizers"]
    qed_scale = normalizers["qed"]["iqr"]
    clogp_scale = normalizers["clogp"]["iqr"]
    qed_floor = normalizers["qed"]["median"]
    clogp_low = normalizers["clogp"]["p25"]
    clogp_high = normalizers["clogp"]["p75"]

    def evaluate(molecule) -> float:
        qed_margin = (QED.qed(molecule) - qed_floor) / qed_scale
        clogp = Crippen.MolLogP(molecule)
        clogp_margin = min(clogp - clogp_low, clogp_high - clogp) / clogp_scale
        return float(min(qed_margin, clogp_margin))

    return evaluate


def logp_objective():
    return lambda molecule: float(Crippen.MolLogP(molecule))


OBJECTIVES = {
    "developability": developability_objective,
    "logp": logp_objective,
    "none": lambda: (lambda molecule: 0.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="none")
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="diagnostic only; NEVER use for a run that produces a number",
    )
    args = parser.parse_args()

    if rdkit.__version__ != REQUIRED_RDKIT and not args.allow_unpinned:
        sys.stderr.write(
            f"REFUSING TO START: shared evaluator requires RDKit {REQUIRED_RDKIT}, "
            f"found {rdkit.__version__}. Run it under the pinned interpreter.\n"
        )
        return 2

    objective = OBJECTIVES[args.objective]()
    identity = {
        "rdkit": rdkit.__version__,
        "pinned": rdkit.__version__ == REQUIRED_RDKIT,
        "objective": args.objective,
        "required_rdkit": REQUIRED_RDKIT,
    }

    sys.stdout.write(json.dumps({"ready": True, **identity}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"error": "bad_json"}) + "\n")
            sys.stdout.flush()
            continue

        if request.get("cmd") == "identity":
            sys.stdout.write(json.dumps(identity) + "\n")
            sys.stdout.flush()
            continue
        if request.get("cmd") == "quit":
            break

        smiles = request.get("smiles")
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
        if molecule is None:
            reply = {"canonical": None, "valid": False, "score": None}
        else:
            canonical = Chem.MolToSmiles(molecule)
            try:
                score = objective(molecule)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the server
                reply = {
                    "canonical": canonical,
                    "valid": True,
                    "score": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                reply = {"canonical": canonical, "valid": True, "score": score}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
