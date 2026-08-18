"""Does adding TDC disturb the frozen numerical environment?

Everything banked today rests on bit-level reproducibility: 64/64 slice parity,
4/4 candidate reproduction, a byte-identical H24 head. The QED image pins
torch 2.4.0 / numpy 1.26.4 / scipy 1.13.1 / rdkit 2024.3.5, and PyTDC drags in
pandas, scikit-learn and friends. If pip resolves numpy or scipy upward, R_theta
computes slightly different numbers and every downstream MOLLEO result is built
on a silently different kernel -- with no error to notice.

So TDC is installed and then the four pins are FORCED BACK, and this job reports
what actually landed plus whether the TDC oracles still import and score. It is
the cheap half of the gate; the expensive half re-runs a banked QED candidate
and demands the identical molecule back.
"""

from __future__ import annotations

from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

# TDC first, then the frozen pins reasserted. Order matters: the last
# pip_install wins, and R_theta reproducibility outranks TDC's preferences.
image = (
    _base_image
    .pip_install("PyTDC")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "rdkit==2024.3.5",
        # TDC's JNK3/GSK3B forests were pickled by scikit-learn < 1.3: their
        # node arrays carry 7 fields, and 1.3 added `missing_go_to_left` as an
        # 8th, so a modern sklearn refuses to unpickle them. This is the
        # benchmark's own frozen artifact -- we pin the reader to it rather
        # than regenerate the oracle, which would redefine the benchmark.
        "scikit-learn==1.2.2",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
)

app = modal.App("molleo-env-gate")

#: MOLLEO Task 3's five objectives, as named by the official implementation.
OBJECTIVES = ("JNK3", "GSK3B", "DRD2", "QED", "SA")


@app.function(image=image, cpu=(2.0, 2.0), memory=8192, timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def check() -> dict[str, Any]:
    import numpy as np
    import scipy
    import torch

    out: dict[str, Any] = {
        "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    import rdkit
    out["rdkit"] = rdkit.__version__
    print("VERSIONS AFTER ADDING TDC")
    for k in ("torch", "numpy", "scipy", "rdkit"):
        print(f"  {k:<8} {out[k]}")
    want = {"torch": "2.4.0", "numpy": "1.26.4", "scipy": "1.13.1",
            "rdkit": "2024.03.5"}
    drift = {k: (out[k], v) for k, v in want.items()
             if not out[k].startswith(v.split("+")[0])}
    out["pins_held"] = not drift
    out["drift"] = drift
    print(f"  pins held: {not drift}" + (f"   DRIFT {drift}" if drift else ""))

    # COMPAT SHIM, and the reason it is the only option here.
    # Every current PyTDC (through 1.1.15) does `from rdkit.six import
    # iteritems` inside its oracle module. rdkit.six was a Python-2
    # compatibility shim removed from RDKit years ago, so TDC's oracles cannot
    # import against ANY modern RDKit. Downgrading RDKit is not available to
    # us: the chemistry kernel, every canonical SMILES in the banked results
    # and R_theta's own behaviour are tied to 2024.03.5.
    #
    # iteritems(d) is literally iter(d.items()). Supplying it changes no
    # oracle arithmetic -- but it IS a deviation from the official environment,
    # so the deterministic oracles are parity-checked against our own RDKit
    # below rather than trusted.
    import sys as _sys
    import types as _types
    import rdkit as _rd
    _six = _types.ModuleType("rdkit.six")
    _six.iteritems = lambda d: iter(d.items())
    _six.itervalues = lambda d: iter(d.values())
    _six.iterkeys = lambda d: iter(d.keys())
    _sys.modules["rdkit.six"] = _six
    _rd.six = _six
    out["rdkit_six_shim"] = True
    import sklearn
    out["sklearn"] = sklearn.__version__
    print(f"\nrdkit.six shim installed; sklearn {sklearn.__version__}")

    try:
        import tdc
        out["tdc"] = getattr(tdc, "__version__", "unknown")
        print(f"\nTDC imported: {out['tdc']}")
    except Exception as exc:  # noqa: BLE001
        out["tdc_error"] = f"{type(exc).__name__}: {exc}"
        print(f"\nTDC IMPORT FAILED: {out['tdc_error']}")
        return out

    # Instantiate each objective and score two known molecules. Downloading the
    # oracle payloads is part of what is being tested -- a container that
    # cannot fetch them is a container that cannot run MOLLEO.
    # TDC re-raises a FIXED "install rdkit" string from a bare except, so the
    # real failure is invisible. Import its oracle module's dependencies
    # directly to surface the actual exception.
    # Read TDC's own try block and run each import separately, so the real
    # failure is named instead of masked by its bare except.
    import inspect, pathlib, traceback
    import tdc.chem_utils.oracle as _pkg
    src = pathlib.Path(_pkg.__file__).parent / "oracle.py"
    head = src.read_text().split("\n")[:30]
    print("\n--- tdc oracle.py lines 1-30 ---")
    for j, ln in enumerate(head, 1):
        print(f"{j:>3}| {ln}")
    print("\n--- executing each import from that block ---")
    for ln in head:
        t = ln.strip()
        if not (t.startswith("from ") or t.startswith("import ")):
            continue
        try:
            exec(t, {"__package__": "tdc.chem_utils.oracle",
                     "__name__": "tdc.chem_utils.oracle.probe"})
            print(f"  ok    {t}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {t}")
            print(f"        {type(exc).__name__}: {exc}")
            out.setdefault("real_failures", {})[t] = f"{type(exc).__name__}: {exc}"
    print("\n--- what TDC's oracle module actually needs ---")
    for mod in ("rdkit.Chem", "rdkit.Chem.AllChem", "rdkit.Chem.Descriptors",
                "rdkit.Chem.QED", "rdkit.Chem.rdMolDescriptors",
                "sklearn", "sklearn.ensemble", "sklearn.svm", "pandas",
                "joblib", "pickle"):
        try:
            __import__(mod)
            print(f"  ok    {mod}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {mod}: {type(exc).__name__}: {exc}")
            out.setdefault("import_failures", {})[mod] = f"{type(exc).__name__}: {exc}"
    try:
        import traceback
        from tdc.chem_utils.oracle import oracle as _o  # noqa: F401
        print("  ok    tdc.chem_utils.oracle.oracle")
    except Exception:  # noqa: BLE001
        print("  FAIL  tdc.chem_utils.oracle.oracle -- real traceback:")
        traceback.print_exc()
        out["oracle_module_traceback"] = traceback.format_exc()[-1500:]

    smis = ["CC(C)CCNC(=O)CC1CCN(C(=O)C2CCCO2)CC1",
            "c1ccccc1O"]
    scores: dict[str, Any] = {}
    from tdc import Oracle
    for name in OBJECTIVES:
        try:
            o = Oracle(name=name)
            v = [float(o(s)) for s in smis]
            scores[name] = v
            print(f"  {name:<6} {v}")
        except Exception as exc:  # noqa: BLE001
            scores[name] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:<6} FAILED {scores[name]}")
    out["objective_scores"] = scores
    out["all_objectives_ok"] = all(isinstance(v, list) for v in scores.values())
    print(f"\nall five objectives scored: {out['all_objectives_ok']}")

    # PARITY on the deterministic oracle. TDC's QED must equal the RDKit QED we
    # already compute natively; if the shim had perturbed anything in the
    # oracle path this is where it would show.
    if isinstance(scores.get("QED"), list):
        from rdkit import Chem
        from rdkit.Chem import QED as _QED
        ours = [float(_QED.qed(Chem.MolFromSmiles(x))) for x in smis]
        diff = [abs(a - b) for a, b in zip(ours, scores["QED"])]
        out["qed_parity"] = {"ours": ours, "tdc": scores["QED"],
                             "max_abs_diff": max(diff)}
        print(f"\nQED parity  ours {ours}\n            tdc  {scores['QED']}"
              f"\n            max |diff| {max(diff):.3e}")
        out["qed_parity_ok"] = max(diff) < 1e-12
        print(f"            exact: {out['qed_parity_ok']}")
    return out


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    o = check.remote()
    Path("docs/MOLLEO_ENV_GATE.json").write_text(json.dumps(o, indent=2))
    print(f"\npins_held={o.get('pins_held')}  "
          f"objectives_ok={o.get('all_objectives_ok')}")
