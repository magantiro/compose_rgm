"""Audit every PMO task for the lazy relative-path asset load, and CALL the ones that have it.

The gsk3b ledger defect was not a gsk3b bug: it is a property of any PyTDC oracle whose
score comes from a pickled estimator that is loaded **lazily, on the first call**, from a
**relative** path, inside ``Oracle.__call__``'s bare ``except``.  This app measures which
of the 23 PMO tasks have that shape, in the pinned production image, and then exercises
the ones whose asset is available against the frozen reference panel.

WHY IT RUNS TDC'S OWN DISPATCH
------------------------------
`Oracle.__init__` resolves a task name through `fuzzy_search` and then `assign_evaluator`
picks the evaluator by a long substring chain.  Reimplementing that chain here would audit
a transcription rather than the shipped bytes, so instead the real `Oracle` is constructed
with the DOWNLOAD stubbed out.  Construction never triggers the lazy load -- that is the
whole defect -- so this costs no network and needs no asset, and `oracle.evaluator_func`
is then the genuine bound evaluator whose source can be walked.

WHAT IS MEASURED vs INFERRED
----------------------------
MEASURED for every task: the resolved name, the bound evaluator, and whether the
evaluator's call graph within TDC's oracle module reaches a relative-path asset load
cached in a module global.
MEASURED for tasks whose asset is in hand: the cwd counterfactual (same oracle, same
asset, different working directory at CALL time) and the pinned positive control.
INFERRED for tasks whose asset cannot be fetched: exposure follows from the shared
idiom, and is recorded as UNVERIFIED_ASSET_ABSENT rather than as a pass.

It charges no PMO budget.  These are local scikit-learn evaluations on files in the
image, outside every contract ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_apps.pmo_population_v1_app import ASSET_DIR, REMOTE_ROOT, _rdkit_six_shim, image

app = modal.App("compose-pmo-oracle-asset-audit")

# The PMO suite, from the two published transcriptions that `tools/verify_pmo_task_registry.py`
# requires to agree exactly.
PMO_TASKS = (
    "albuterol_similarity", "amlodipine_mpo", "celecoxib_rediscovery", "deco_hop", "drd2",
    "fexofenadine_mpo", "gsk3b", "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl", "jnk3",
    "median1", "median2", "mestranol_similarity", "osimertinib_mpo", "perindopril_mpo",
    "qed", "ranolazine_mpo", "scaffold_hop", "sitagliptin_mpo", "thiothixene_rediscovery",
    "troglitazone_rediscovery", "valsartan_smarts", "zaleplon_mpo",
)

_ASSET_SUFFIXES = (".pkl", ".joblib", ".pickle", ".ckpt", ".pt", ".npz", ".txt.gz")


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=60 * 60, retries=0)
def audit() -> dict:
    """Classify all 23 tasks, then call every asset-backed oracle we can reach."""

    import ast
    import contextlib
    import inspect
    import io
    import os
    import sys
    import tempfile
    import textwrap
    import types

    import rdkit
    import sklearn
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    from compose_v4.experiments.pmo_oracle_assets import (
        POSITIVE_CONTROLS,
        AssetPinnedOracle,
        pinned_working_directory,
        run_positive_control,
    )

    report: dict = {
        "schema_version": "pmo_oracle_asset_audit_v1",
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "rdkit": rdkit.__version__,
        },
    }

    _rdkit_six_shim()
    import tdc
    from tdc import Oracle
    from tdc.chem_utils.oracle import oracle as omod

    report["environment"]["pytdc"] = getattr(tdc, "__version__", "unknown")
    report["environment"]["tdc_oracle_module"] = omod.__file__

    # ---- the swallow, measured once for the whole suite ------------------
    call_source = inspect.getsource(Oracle.__call__)
    report["swallow"] = {
        "oracle_call_has_bare_except": "except:" in call_source,
        "default_property_returned_on_failure": "default_property" in call_source,
        "consequence": (
            "any failure inside an evaluator -- including a FileNotFoundError on a "
            "relative asset path -- is returned as a plausible score, not raised"
        ),
    }

    # ---- static call-graph analysis --------------------------------------
    def _resolve(obj):
        """The object whose SOURCE defines `obj`'s behaviour.

        TDC's evaluators are a mix of plain functions and *instances* of callable
        classes (`rediscovery_meta`, `Isomers`, ...).  Taking `inspect.getsource` of
        an instance raises, which silently produced an empty call graph and would
        have classified every class-backed oracle as asset-free.
        """
        if isinstance(obj, types.MethodType):
            return obj.__func__
        if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
            return obj
        return type(obj)

    def _sources(obj) -> dict[str, str]:
        """The evaluator's source plus every oracle-module callable it reaches."""
        collected: dict[str, str] = {}
        pending = [obj]
        seen: set[int] = set()
        while pending:
            target = _resolve(pending.pop())
            if id(target) in seen:
                continue
            seen.add(id(target))
            try:
                source = textwrap.dedent(inspect.getsource(target))
            except (OSError, TypeError):
                continue
            name = getattr(target, "__qualname__", repr(target))
            if name in collected:
                continue
            collected[name] = source
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    candidate = getattr(omod, node.id, None)
                    # Only recurse into TDC's own oracle module; stop at rdkit/sklearn.
                    if getattr(candidate, "__module__", None) == omod.__name__:
                        pending.append(candidate)
        return collected

    def _mentions_globals(node: ast.AST) -> bool:
        """`globals()` or `globals().keys()` -- TDC uses the latter."""
        return any(
            isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "globals"
            for inner in ast.walk(node)
        )

    def _classify(sources: dict[str, str]) -> dict:
        relative_assets: list[str] = []
        loaders: list[str] = []
        lazy_globals: list[str] = []
        for name, source in sources.items():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.endswith(_ASSET_SUFFIXES) and not value.startswith("/"):
                        relative_assets.append(value)
                if isinstance(node, ast.Global):
                    lazy_globals.extend(node.names)
                if isinstance(node, ast.Call):
                    target = node.func
                    label = (
                        target.attr if isinstance(target, ast.Attribute)
                        else target.id if isinstance(target, ast.Name) else ""
                    )
                    if label in {"load_pickled_model", "load", "oracle_load", "open",
                                 "download_oracle", "load_gsk3b_model", "load_jnk3_model",
                                 "load_drd2_model"}:
                        loaders.append(f"{name}:{label}")
                # the `if "model" not in globals():` lazy-cache idiom
                if (
                    isinstance(node, ast.Compare)
                    and isinstance(node.left, ast.Constant)
                    and any(_mentions_globals(c) for c in node.comparators)
                ):
                    lazy_globals.append(str(node.left.value))
        asset_backed = bool(relative_assets) and bool(loaders)
        return {
            "relative_asset_paths": sorted(set(relative_assets)),
            "loader_calls": sorted(set(loaders)),
            "module_global_cache": sorted(set(lazy_globals)),
            "asset_backed": asset_backed,
            "lazy": asset_backed and bool(lazy_globals),
            "classification": (
                "ASSET_BACKED_LAZY_RELATIVE" if asset_backed and lazy_globals
                else "ASSET_BACKED" if asset_backed
                else "PURE_RDKIT_NO_ASSET"
            ),
        }

    # Stub only the NETWORK transfer.  `oracle_load` must keep running, because it is
    # what RESOLVES the exact asset name (gsk3b -> gsk3b_current under sklearn >= 0.24)
    # and returns it to `Oracle.__init__`; stubbing it out returns None and every
    # asset-backed task fails to construct, which reads exactly like "not affected".
    stubbed: list[tuple[object, str, object]] = []
    for module_name in ("tdc.utils.load", "tdc.chem_utils.oracle.oracle"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "dataverse_download"):
            stubbed.append((module, "dataverse_download", module.dataverse_download))
            module.dataverse_download = lambda *a, **k: None
    report["download_stubbed"] = [f"{m.__name__}.{a}" for m, a, _ in stubbed]

    # Construct inside a scratch directory so TDC's `os.mkdir("./oracle")` cannot
    # leave an empty `oracle/` beside the container root and blunt the counterfactual.
    classify_dir = Path(tempfile.mkdtemp(prefix="pmo_audit_classify_"))

    def _describe(task: str, row: dict) -> dict:
        with pinned_working_directory(classify_dir), contextlib.redirect_stdout(io.StringIO()):
            oracle = Oracle(name=task)
        row["resolved_name"] = oracle.name
        evaluator = oracle.evaluator_func
        row["evaluator"] = getattr(evaluator, "__qualname__", type(evaluator).__name__)
        row["evaluator_module"] = getattr(
            evaluator, "__module__", type(evaluator).__module__)
        sources = _sources(evaluator)
        row["call_graph_functions"] = sorted(sources)
        row.update(_classify(sources))
        return row

    tasks: dict[str, dict] = {}
    deferred: list[str] = []
    try:
        for task in PMO_TASKS:
            row: dict = {"requested": task, "construction_requires_asset": False}
            try:
                _describe(task, row)
            except Exception as error:  # noqa: BLE001 - reported, never swallowed
                # An asset error HERE means the evaluator loads its pickle EAGERLY, in
                # its constructor, rather than lazily on the first call.  That is a
                # materially different exposure: it raises where the lazy form is
                # swallowed into a score.  Retry with the download allowed so the task
                # is still classified rather than left unknown.
                row["construction_requires_asset"] = isinstance(error, (OSError, FileNotFoundError))
                row["construction_error_without_asset"] = f"{type(error).__name__}: {error}"
                deferred.append(task)
            tasks[task] = row
    finally:
        for module, attribute, value in stubbed:
            setattr(module, attribute, value)

    report["deferred_to_download"] = sorted(deferred)
    for task in deferred:
        row = tasks[task]
        try:
            _describe(task, row)
            row.pop("error", None)
        except Exception as error:  # noqa: BLE001
            row["error"] = f"{type(error).__name__}: {error}"
            row["classification"] = "UNRESOLVED"

    # An eager constructor load is still a relative-path asset dependency, so it
    # belongs in the audit's affected set even though its call graph caches on the
    # instance rather than in a module global.
    for task, row in tasks.items():
        if row.get("construction_requires_asset") and row.get("asset_backed"):
            row["classification"] = "ASSET_BACKED_EAGER_RELATIVE"

    report["tasks"] = tasks
    affected = sorted(
        name for name, row in tasks.items()
        if row.get("classification", "").startswith("ASSET_BACKED")
    )
    report["asset_backed_tasks"] = affected

    # ---- dynamic confirmation where the asset is reachable ---------------
    capsule = REMOTE_ROOT / ASSET_DIR
    report["capsule"] = {
        "path": str(capsule),
        "assets": sorted(p.name for p in (capsule / "oracle").glob("*")),
    }

    dynamic: dict[str, dict] = {}
    for task in affected:
        entry: dict = {"task": task}
        # Which asset file this task's evaluator wants, as the static pass saw it.
        wanted = tasks[task].get("relative_asset_paths") or []
        entry["relative_asset_paths"] = wanted
        resolved = tasks[task].get("resolved_name", task)
        expected_file = f"{resolved}.pkl"
        root: Path | None = None
        if (capsule / "oracle" / expected_file).exists():
            root, entry["asset_source"] = capsule, "pinned_capsule"
        else:
            # Try TDC's own download into a scratch directory; the capsule stays pristine.
            scratch = Path(tempfile.mkdtemp(prefix=f"oracle_{task}_"))
            (scratch / "oracle").mkdir(parents=True, exist_ok=True)
            try:
                from tdc.utils.load import oracle_load

                with pinned_working_directory(scratch), contextlib.redirect_stdout(io.StringIO()):
                    oracle_load(resolved)
                if (scratch / "oracle" / expected_file).exists():
                    root, entry["asset_source"] = scratch, "downloaded"
                else:
                    entry["asset_source"] = "download_produced_no_file"
            except Exception as error:  # noqa: BLE001
                entry["asset_source"] = "download_failed"
                entry["download_error"] = f"{type(error).__name__}: {str(error)[:300]}"

        if root is None:
            entry["status"] = "UNVERIFIED_ASSET_ABSENT"
            entry["scored_launch_authorized"] = False
            dynamic[task] = entry
            continue

        with pinned_working_directory(root), contextlib.redirect_stdout(io.StringIO()):
            live = Oracle(name=task)
        probe = POSITIVE_CONTROLS[task][0].smiles if task in POSITIVE_CONTROLS else "CCO"

        # The counterfactual: identical oracle and asset, differing ONLY in the working
        # directory at CALL time.  This is the defect, reproduced per task.  The
        # "elsewhere" directory is asserted to be free of an `oracle/` subdirectory,
        # so a pass here cannot come from an accidental local copy.
        elsewhere = Path(tempfile.mkdtemp(prefix=f"pmo_audit_elsewhere_{task}_"))
        entry["elsewhere_has_oracle_dir"] = (elsewhere / "oracle").exists()
        os.chdir(elsewhere)
        entry["score_from_unpinned_cwd"] = float(live(probe))
        pinned = AssetPinnedOracle(live, root, name=task)
        entry["score_with_cwd_pinned_at_call"] = pinned(probe)
        entry["cwd_dependent"] = (
            entry["score_from_unpinned_cwd"] != entry["score_with_cwd_pinned_at_call"]
        )

        if task in POSITIVE_CONTROLS:
            # A FRESH oracle, so the previous call's module-global cache cannot mask
            # a load failure the production path would have hit.
            with pinned_working_directory(root), contextlib.redirect_stdout(io.StringIO()):
                fresh = Oracle(name=task)
            control = run_positive_control(
                AssetPinnedOracle(fresh, root, name=task), task)
            entry["positive_control"] = control
            entry["status"] = "POSITIVE_CONTROL_PASSED" if control["passed"] else "POSITIVE_CONTROL_FAILED"
            entry["scored_launch_authorized"] = bool(control["passed"])
        else:
            entry["status"] = "NO_PINNED_REFERENCES"
            entry["scored_launch_authorized"] = False
        dynamic[task] = entry

    report["dynamic"] = dynamic
    report["cwd_at_end"] = str(Path.cwd())
    return report


@app.local_entrypoint()
def main(out: str) -> None:
    result = audit.remote()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("written", out)
    print("asset-backed:", result.get("asset_backed_tasks"))
    for task, row in sorted(result.get("dynamic", {}).items()):
        print(f"  {task:8s} {row.get('status'):28s} cwd_dependent={row.get('cwd_dependent')}")


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=30 * 60, retries=0)
def dump() -> dict:
    """Exploratory: the PyTDC bytes the audit's stubbing and call-graph walk target."""

    import inspect
    import sys

    _rdkit_six_shim()
    from tdc import Oracle
    from tdc.chem_utils.oracle import oracle as omod
    from tdc.utils import load as lmod

    out: dict = {}
    out["oracle_init"] = inspect.getsource(Oracle.__init__)
    out["assign_evaluator_head"] = inspect.getsource(Oracle.assign_evaluator)[:4000]
    out["oracle_load"] = inspect.getsource(lmod.oracle_load)
    for name in ("download_wrapper", "dataverse_download", "oracle_names"):
        try:
            value = getattr(lmod, name)
            out[name] = (inspect.getsource(value) if callable(value) else repr(value)[:2000])
        except Exception as error:  # noqa: BLE001
            out[name] = f"MISSING: {error}"
    for name in ("gsk3b", "jnk3", "drd2", "load_pickled_model", "rediscovery_meta"):
        try:
            out[f"omod.{name}"] = inspect.getsource(getattr(omod, name))
        except Exception as error:  # noqa: BLE001
            out[f"omod.{name}"] = f"MISSING: {error}"
    out["tdc_utils_load_module"] = lmod.__file__
    out["python"] = sys.version.split()[0]
    return out


@app.local_entrypoint()
def run_dump(out: str) -> None:
    Path(out).write_text(json.dumps(dump.remote(), indent=2, sort_keys=True) + "\n")
    print("written", out)
