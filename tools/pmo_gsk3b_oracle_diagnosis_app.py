"""Attribution probe for the all-zero PMO gsk3b ledger, in the production image.

The scored gsk3b task charged 250/250 calls and recorded a score of exactly 0.0 on
every one of 250 distinct endpoints.  This app decides between three explanations --
oracle/runtime defect, wrapper defect, and genuine discovery failure -- by measuring
inside the SAME image the scored run used.  It imports ``image`` from
``modal_apps.pmo_population_v1_app`` rather than defining one, for the reason
``pmo_environment_smoke_app`` gives: a probe that passes on a different image proves
nothing about the image that produced the ledger.

It charges no PMO budget.  The TDC gsk3b oracle is a local scikit-learn random forest
over a Morgan bit vector; calling it is arithmetic on a file already in the image, not
a request against any metered service, and these evaluations are outside every
contract ledger.

WHAT IT MEASURES
----------------
* oracle identity -- the name TDC resolved (fuzzy_search can silently land on a
  DIFFERENT oracle at Levenshtein 0.8), PyTDC version, the estimator's own
  ``n_features_in_``/``classes_``/``n_estimators``, and the sha256 of the asset file
  actually opened.
* the raw ``predict_proba`` ROW, both columns, before TDC or COMPOSE touches it, so a
  wrong-column or thresholding bug is visible rather than inferred.
* a positive control -- molecules the frozen bundle records as known gsk3b actives,
  which must not score 0.
* a fresh re-score of the ledger's own endpoints through a newly constructed oracle,
  compared molecule by molecule against what the ledger recorded.
* the featurized representation per molecule (on-bit count and a digest), to rule out
  the specific hidden failure of different SMILES collapsing to one feature vector.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_apps.pmo_population_v1_app import REMOTE_ROOT, _rdkit_six_shim, image

ASSET_DIR = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"

app = modal.App("compose-pmo-gsk3b-oracle-diagnosis")


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=30 * 60, retries=0)
def diagnose(payload: dict) -> dict:
    """Identity, positive control, fresh re-score and featurization, in one container."""

    import hashlib
    import inspect
    import os
    import sys
    from time import perf_counter

    import numpy as np

    report: dict = {"schema_version": "pmo_gsk3b_oracle_diagnosis_v1"}

    # ---- environment ----------------------------------------------------
    import rdkit
    import sklearn
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    report["environment"] = {
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
    }

    _rdkit_six_shim()
    import tdc
    from tdc import Oracle

    report["environment"]["pytdc"] = getattr(tdc, "__version__", "unknown")
    report["environment"]["tdc_module"] = sys.modules["tdc"].__file__

    root = REMOTE_ROOT
    asset_dir = root / ASSET_DIR

    # ---- asset identity, before anything opens it -------------------------
    assets = {}
    for path in sorted((asset_dir / "oracle").glob("*")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assets[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    report["asset_files"] = assets
    report["declared_gsk3b_sha256"] = payload["declared_gsk3b_sha256"]
    report["asset_sha256_matches_contract"] = any(
        item["sha256"] == payload["declared_gsk3b_sha256"] for item in assets.values()
    )

    # ---- name resolution: did fuzzy search move the name? -----------------
    resolution = {"requested": "gsk3b"}
    try:
        from tdc.metadata import download_oracle_names, oracle_names, trivial_oracle_names
        from tdc.utils import fuzzy_search

        every = list(oracle_names) + list(download_oracle_names) + list(trivial_oracle_names)
        resolution["exact_name_in_catalog"] = "gsk3b" in every
        resolution["fuzzy_search_result"] = fuzzy_search("gsk3b", every)
        resolution["catalog_near_names"] = sorted(
            n for n in every if "gsk" in n.lower() or "jnk" in n.lower()
        )
    except Exception as error:  # noqa: BLE001  # pragma: no cover - catalog layout varies by version
        resolution["catalog_probe_error"] = repr(error)

    previous = Path.cwd()
    os.chdir(asset_dir)
    try:
        oracle = Oracle(name="gsk3b")
    finally:
        os.chdir(previous)
    resolution["oracle_name_attribute"] = getattr(oracle, "name", None)
    resolution["oracle_type"] = type(oracle).__name__
    report["name_resolution"] = resolution

    # ---- what is actually behind the callable -----------------------------
    inner = getattr(oracle, "evaluator_func", None)
    wrapper: dict = {
        "evaluator_func_type": type(inner).__name__,
        "evaluator_func_module": getattr(type(inner), "__module__", None),
    }
    for attribute in ("gsk3_model", "model", "clf", "gsk3b_model"):
        estimator = getattr(inner, attribute, None)
        if estimator is not None:
            wrapper["estimator_attribute"] = attribute
            wrapper["estimator_type"] = type(estimator).__name__
            wrapper["estimator_module"] = type(estimator).__module__
            for field in ("n_features_in_", "n_estimators", "n_outputs_"):
                if hasattr(estimator, field):
                    wrapper[field] = int(getattr(estimator, field))
            if hasattr(estimator, "classes_"):
                wrapper["classes_"] = [float(c) for c in estimator.classes_]
            if hasattr(estimator, "estimators_"):
                wrapper["n_fitted_trees"] = len(estimator.estimators_)
            report["_estimator_handle"] = attribute
            break
    else:
        estimator = None
        wrapper["estimator_attribute"] = None
    try:
        wrapper["call_source"] = inspect.getsource(type(inner).__call__)
    except Exception as error:  # noqa: BLE001
        wrapper["call_source_error"] = repr(error)
    report["wrapper"] = wrapper
    report.pop("_estimator_handle", None)

    # ---- featurization + raw model output, per molecule -------------------
    def featurize(smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        row: dict = {"input_smiles": smiles, "parses": mol is not None}
        if mol is None:
            return row
        canonical = Chem.MolToSmiles(mol)
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        vector = np.zeros((2048,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(bits, vector)
        row.update({
            "canonical_smiles": canonical,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formal_charge": Chem.GetFormalCharge(mol),
            "on_bits": int(vector.sum()),
            "fingerprint_sha256": hashlib.sha256(
                vector.astype(np.uint8).tobytes()).hexdigest(),
        })
        if estimator is not None:
            raw = estimator.predict_proba(vector.reshape(1, -1))
            row["predict_proba_raw"] = [float(v) for v in raw[0]]
        return row

    def score(smiles: str) -> dict:
        row = featurize(smiles)
        begin = perf_counter()
        try:
            value = oracle(smiles)
            row["oracle_raw"] = value if isinstance(value, (int, float)) else repr(value)
            row["oracle_repr"] = repr(value)
            row["oracle_float"] = float(value)
        except Exception as error:  # noqa: BLE001
            row["oracle_error"] = repr(error)
        row["seconds"] = perf_counter() - begin
        return row

    # ---- 1. positive control ---------------------------------------------
    report["positive_control"] = [
        {**score(item["smiles"]), "group": item["group"],
         "frozen_reference_gsk3b": item["gsk3b"]}
        for item in payload["control_panel"]
    ]

    # ---- 2. fresh re-score of ledger molecules, NEW oracle instance -------
    os.chdir(asset_dir)
    try:
        fresh = Oracle(name="gsk3b")
    finally:
        os.chdir(previous)
    report["fresh_oracle_is_new_object"] = fresh is not oracle

    ledger_rows = []
    for item in payload["ledger_sample"]:
        row = featurize(item["endpoint"])
        try:
            row["fresh_oracle_score"] = float(fresh(item["endpoint"]))
        except Exception as error:  # noqa: BLE001
            row["fresh_oracle_error"] = repr(error)
        row.update({
            "index": item["index"],
            "role": item["role"],
            "recorded_score": item["score"],
            "recorded_seconds": item["seconds"],
            "reference_forest_score": item["reference_forest_score"],
        })
        ledger_rows.append(row)
    report["ledger_rescore"] = ledger_rows

    # ---- 3. whole-ledger sweep, summary only ------------------------------
    everything = [item["endpoint"] for item in payload["ledger_all"]]
    values, errors = [], 0
    for smiles in everything:
        try:
            values.append(float(fresh(smiles)))
        except Exception:  # noqa: BLE001
            errors += 1
            values.append(float("nan"))
    array = np.array(values, dtype=float)
    finite = array[np.isfinite(array)]
    report["ledger_sweep"] = {
        "n": len(everything),
        "errors": errors,
        "exact_zero": int((finite == 0.0).sum()),
        "nonzero": int((finite > 0.0).sum()),
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "distinct_values": len(set(finite.tolist())),
    }
    distinct_fp = {r["fingerprint_sha256"] for r in ledger_rows if "fingerprint_sha256" in r}
    report["distinct_fingerprints_in_sample"] = len(distinct_fp)
    report["sample_size"] = len(ledger_rows)
    report["charged_pmo_calls"] = 0
    return report


@app.local_entrypoint()
def main(payload_path: str, out: str) -> None:
    payload = json.loads(Path(payload_path).read_text())
    result = diagnose.remote(payload)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("positive_control", "ledger_rescore", "wrapper")},
                     indent=2, sort_keys=True))


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=30 * 60, retries=0)
def probe(payload: dict) -> dict:
    """Name the mechanism: what TDC bound as the gsk3b callable, and does the asset work?"""

    import inspect
    import os
    import pickle
    import sys

    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    out: dict = {"schema_version": "pmo_gsk3b_oracle_probe_v1"}

    try:
        from importlib.metadata import version
        out["pytdc_dist_version"] = version("PyTDC")
    except Exception as error:  # noqa: BLE001
        out["pytdc_dist_version_error"] = repr(error)

    _rdkit_six_shim()
    from tdc import Oracle

    root = REMOTE_ROOT
    asset_dir = root / ASSET_DIR
    previous = Path.cwd()

    def describe(name: str) -> dict:
        os.chdir(asset_dir)
        try:
            oracle = Oracle(name=name)
        finally:
            os.chdir(previous)
        inner = getattr(oracle, "evaluator_func", None)
        item = {
            "requested": name,
            "resolved_name": getattr(oracle, "name", None),
            "oracle_dict_keys": sorted(oracle.__dict__.keys()),
            "evaluator_repr": repr(inner)[:300],
            "evaluator_name": getattr(inner, "__name__", None),
            "evaluator_module": getattr(inner, "__module__", None),
            "evaluator_qualname": getattr(inner, "__qualname__", None),
        }
        code = getattr(inner, "__code__", None)
        if code is not None:
            item["evaluator_file"] = code.co_filename
            item["evaluator_firstlineno"] = code.co_firstlineno
            item["evaluator_names"] = list(code.co_names)[:40]
            item["evaluator_consts"] = [repr(c)[:80] for c in code.co_consts][:20]
        try:
            item["evaluator_source"] = inspect.getsource(inner)[:4000]
        except Exception as error:  # noqa: BLE001
            item["evaluator_source_error"] = repr(error)
        return item

    out["gsk3b"] = describe("gsk3b")
    out["perindopril_mpo"] = describe("perindopril_mpo")

    try:
        out["oracle_call_source"] = inspect.getsource(Oracle.__call__)[:4000]
    except Exception as error:  # noqa: BLE001
        out["oracle_call_source_error"] = repr(error)
    try:
        out["assign_evaluator_gsk_branch"] = [
            line for line in inspect.getsource(Oracle.assign_evaluator).splitlines()
            if "gsk" in line or "jnk" in line or "_current" in line
        ]
    except Exception as error:  # noqa: BLE001
        out["assign_evaluator_error"] = repr(error)

    # ---- does the pinned asset itself work when loaded directly? ----------
    asset = asset_dir / "oracle" / "gsk3b_current.pkl"
    direct: dict = {"path": str(asset), "exists": asset.exists()}
    try:
        with asset.open("rb") as handle:
            model = pickle.load(handle)
        direct["loaded_type"] = type(model).__name__
        direct["loaded_module"] = type(model).__module__
        for field in ("n_features_in_", "n_estimators", "n_outputs_"):
            if hasattr(model, field):
                direct[field] = int(getattr(model, field))
        if hasattr(model, "classes_"):
            direct["classes_"] = [float(c) for c in model.classes_]
        rows, labels = [], []
        for item in payload["control_panel"]:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is None:
                continue
            bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            vector = np.zeros((2048,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(bits, vector)
            rows.append(vector)
            labels.append((item["group"], item["gsk3b"]))
        proba = model.predict_proba(np.vstack(rows))
        direct["predictions"] = [
            {"group": group, "frozen_reference_gsk3b": reference,
             "direct_predict_proba": [float(v) for v in proba[i]]}
            for i, (group, reference) in enumerate(labels)
        ]
    except Exception as error:  # noqa: BLE001
        direct["error"] = repr(error)
        direct["error_type"] = type(error).__name__
    out["direct_asset_load"] = direct

    # ---- what files does the oracle directory hold, from TDC's view? ------
    out["cwd_at_construction"] = str(asset_dir)
    out["oracle_dir_listing"] = sorted(p.name for p in (asset_dir / "oracle").glob("*"))
    out["python"] = sys.version.split()[0]
    return out


@app.local_entrypoint()
def run_probe(payload_path: str, out: str) -> None:
    payload = json.loads(Path(payload_path).read_text())
    result = probe.remote(payload)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("written", out)


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=30 * 60, retries=0)
def dissect(payload: dict) -> dict:
    """Walk TDC's own five lines and find where the value becomes zero."""

    import inspect
    import os

    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    out: dict = {"schema_version": "pmo_gsk3b_oracle_dissect_v1"}
    actives = [item["smiles"] for item in payload["control_panel"]
               if item["group"] == "gsk3b_active"][:5]
    out["actives"] = actives

    _rdkit_six_shim()
    from tdc import Oracle
    from tdc.chem_utils.oracle import oracle as oracle_module

    root = REMOTE_ROOT
    asset_dir = root / ASSET_DIR
    previous = Path.cwd()

    for name in ("load_gsk3b_model", "smiles_to_rdkit_mol"):
        try:
            out[f"{name}_source"] = inspect.getsource(getattr(oracle_module, name))[:3000]
        except Exception as error:  # noqa: BLE001
            out[f"{name}_source_error"] = repr(error)

    # ---- TDC's exact featurization lines, instrumented -------------------
    mol = Chem.MolFromSmiles(actives[0])
    bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features = np.zeros((1,))
    before = {"shape": list(features.shape), "sum": float(features.sum())}
    DataStructs.ConvertToNumpyArray(bits, features)
    out["tdc_featurization"] = {
        "before": before,
        "after_shape": list(features.shape),
        "after_sum": float(features.sum()),
        "after_dtype": str(features.dtype),
        "bitvector_on_bits": int(bits.GetNumOnBits()),
        "reshaped_shape": list(features.reshape(1, -1).shape),
    }

    # ---- construct exactly as production does, then call from the restored cwd ----
    out["cwd_before_construction"] = str(previous)
    os.chdir(asset_dir)
    try:
        oracle = Oracle(name="gsk3b")
    finally:
        os.chdir(previous)
    out["cwd_at_call_time"] = str(Path.cwd())
    out["model_global_present_before_first_call"] = hasattr(oracle_module, "gsk3_model")

    out["production_pattern_scores"] = [float(oracle(s)) for s in actives]
    out["model_global_present_after_first_call"] = hasattr(oracle_module, "gsk3_model")

    model = getattr(oracle_module, "gsk3_model", None)
    handle: dict = {"type": type(model).__name__ if model is not None else None}
    if model is not None:
        handle["module"] = type(model).__module__
        for field in ("n_features_in_", "n_estimators", "n_outputs_"):
            if hasattr(model, field):
                handle[field] = int(getattr(model, field))
        if hasattr(model, "classes_"):
            handle["classes_"] = [float(c) for c in model.classes_]
        rows = []
        for smiles in actives:
            molecule = Chem.MolFromSmiles(smiles)
            vector = np.zeros((2048,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048), vector)
            rows.append(vector)
        handle["predict_proba_on_correct_fingerprints"] = [
            [float(v) for v in row] for row in model.predict_proba(np.vstack(rows))
        ]
        zero_row = np.zeros((1, 2048), dtype=np.float64)
        handle["predict_proba_on_all_zero_fingerprint"] = [
            float(v) for v in model.predict_proba(zero_row)[0]
        ]
    out["loaded_model_handle"] = handle

    # ---- the counterfactual: same oracle, called with cwd held at the assets ----
    os.chdir(asset_dir)
    try:
        out["scores_with_cwd_held_at_assets"] = [float(oracle(s)) for s in actives]
    except Exception as error:  # noqa: BLE001
        out["scores_with_cwd_held_error"] = repr(error)
    finally:
        os.chdir(previous)

    # ---- and a fully fresh interpreter-state variant: unload the global --------
    if hasattr(oracle_module, "gsk3_model"):
        delattr(oracle_module, "gsk3_model")
    os.chdir(asset_dir)
    try:
        fresh = Oracle(name="gsk3b")
        out["scores_constructed_and_called_inside_assets"] = [float(fresh(s)) for s in actives]
    except Exception as error:  # noqa: BLE001
        out["scores_constructed_and_called_inside_assets_error"] = repr(error)
    finally:
        os.chdir(previous)

    out["oracle_dir_relative_from_cwd_exists"] = (previous / "oracle" / "gsk3b_current.pkl").exists()
    out["files_named_gsk3b_under_root"] = sorted(
        str(p) for p in root.rglob("*gsk3b*") if p.is_file())[:20]
    return out


@app.local_entrypoint()
def run_dissect(payload_path: str, out: str) -> None:
    payload = json.loads(Path(payload_path).read_text())
    result = dissect.remote(payload)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("written", out)


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=30 * 60, retries=0)
def swallow() -> dict:
    """Name the exception and the fallback constant that turn a load failure into 0.0."""

    import inspect
    import os

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out: dict = {"schema_version": "pmo_gsk3b_oracle_swallow_v1"}

    _rdkit_six_shim()
    from tdc import Oracle
    from tdc.chem_utils.oracle import oracle as oracle_module

    source = inspect.getsource(Oracle.__call__)
    out["oracle_call_single_smiles_branch"] = source[source.rfind("else:  ### a string"):]
    try:
        out["load_pickled_model_source"] = inspect.getsource(
            oracle_module.load_pickled_model)
    except Exception as error:  # noqa: BLE001
        out["load_pickled_model_source_error"] = repr(error)
    try:
        out["oracle_init_default_property"] = [
            line.strip() for line in inspect.getsource(Oracle.__init__).splitlines()
            if "default_property" in line or "normalize" in line
        ]
    except Exception as error:  # noqa: BLE001
        out["oracle_init_error"] = repr(error)

    previous = Path.cwd()
    os.chdir(REMOTE_ROOT / ASSET_DIR)
    try:
        oracle = Oracle(name="gsk3b")
    finally:
        os.chdir(previous)
    out["default_property"] = oracle.default_property
    out["normalize"] = repr(oracle.normalize)
    out["num_called_before"] = oracle.num_called

    # The exception TDC swallows, raised in the open.
    try:
        oracle_module.load_gsk3b_model()
        out["load_from_production_cwd"] = "SUCCEEDED"
    except Exception as error:  # noqa: BLE001
        out["load_from_production_cwd_error_type"] = type(error).__name__
        out["load_from_production_cwd_error"] = str(error)[:400]

    smiles = "C1(C2=CN(CCCO)C=3C2=CC=CN3)=C(C(=O)NC1=O)C4=NC=CC=C4"
    out["score_from_production_cwd"] = float(oracle(smiles))
    out["num_called_after"] = oracle.num_called
    out["model_global_set"] = hasattr(oracle_module, "gsk3_model")
    out["cwd"] = str(Path.cwd())
    return out


@app.local_entrypoint()
def run_swallow(out: str) -> None:
    result = swallow.remote()
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("written", out)
