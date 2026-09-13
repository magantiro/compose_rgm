"""Approved, matched T4/PMO learning cycles using the shared program runner."""

from __future__ import annotations

import inspect
import json
import platform
import shutil
import threading
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.parent_edit_model import (
    MUTATION_RECIPE,
    ParentEditFeatures,
    ParentEditModel,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "parent_edit_cycles"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared"
APP = f"modal_apps/{KIND}_app.py"
APP_NAME = "compose-parent-edit-cycles"
PMO_ORACLE_SHA256 = "0ea83059a50a730f55db3df7367ac642c8ac27b525acd0c8f240494524a1a014"


def load_contract(root):
    c = unseal(root / CONTRACT)
    if (c["max_docking_calls"], c["max_pmo_queries"], c["reserved_usd"], c["container_limit"]) != (
        108,
        24000,
        20,
        30,
    ):
        raise ValueError("learning-cycle allocation differs from user approval")
    units = c["units"]
    if len({r["unit_id"] for r in units}) != 32:
        raise ValueError("expected eight T4 and twenty-four PMO units")
    if sum(r["budget"] for r in units if r["kind"] == "t4") != 96:
        raise ValueError("T4 first-query reservation changed")
    if sum(r["budget"] for r in units if r["kind"] == "pmo") != 24000:
        raise ValueError("PMO initialization-inclusive reservation changed")
    for path, digest in c["inputs"].items():
        verify_file(root / path, digest)
    if "update_function_source" in c:
        expected = {
            "recipe": MUTATION_RECIPE,
            "function_source": inspect.getsource(fit_measured_edits),
        }
        if c["update_function_source"] != expected["function_source"] or c[
            "update_rule_id"
        ] != identity(expected):
            raise ValueError("identified model update rule differs from the running function")
    return c


def fit_measured_edits(search, task):
    """All accessible completed outcomes, balanced by endpoint; no future labels."""
    feature, labels, rows = ParentEditFeatures(), defaultdict(list), []
    for receipt, observation in search.observations.items():
        labels[observation["endpoint"]].append((receipt, task.utility(observation["score"])))
    for entry in search.entries.values():
        provenance = entry.get("provenance", {})
        parent_id = provenance.get("entry_id")
        parent = search.entries.get(parent_id) if parent_id else None
        if parent_id and parent is None:
            raise ValueError("training mutation lost its selected parent")
        parent_state = parent["trace"]["states"][-1] if parent else entry["source_state"]
        parent_endpoint = canonical_state_key(decode_state(parent_state))
        if "parent_measured_score" in provenance:
            scores = [task.utility(provenance["parent_measured_score"])]
        else:
            scores = [u for _, u in labels.get(parent_endpoint, [])]
        values = feature.with_mutation_context(
            entry, parent_state=parent_state, parent_scores=scores, parent_record=parent
        )
        rows.extend(
            {
                "endpoint": entry["endpoint"],
                "features": values,
                "receipt_id": receipt,
                "utility": utility,
                "oracle_protocol": task.oracle_protocol,
            }
            for receipt, utility in labels[entry["endpoint"]]
        )
    return ParentEditModel.fit(
        rows,
        oracle_protocol=task.oracle_protocol,
        input_sha256=identity({"entries": search.entries, "observations": search.observations}),
        recipe=MUTATION_RECIPE,
    )


def configured(unit):
    return replace(
        ProgramSearchConfig.parent_edit_recipe(
            seed=unit["seed"], score_direction="minimize" if unit["kind"] == "t4" else "maximize"
        ),
        candidates_per_batch=12 if unit["kind"] == "t4" else 48,
        attempts_per_batch=128,
        wall_seconds=45.0,
    )


def warm_archive(root, cell, config, hierarchy):
    payload = json.loads((root / PREPARED / f"{cell}_warm.json").read_text())
    search = ProgramOptimizer(
        config,
        source_group=payload["source_group"],
        oracle_protocol=payload["oracle_protocol"],
        hierarchy=hierarchy,
    )
    by_endpoint = defaultdict(list)
    for receipt, row in payload["observations"].items():
        by_endpoint[row["endpoint"]].append((receipt, row["score"]))
    for key, entry in payload["entries"].items():
        for receipt, score in by_endpoint[entry["endpoint"]]:
            if search.add_measured_program(entry, receipt_id=receipt, score=score) != key:
                raise ValueError("warm archive construction identity changed")
    return search.snapshot()


def validate_launch(task, root, validate_revision):
    validate_revision(task["image_revision"])
    base = {k: v for k, v in task.items() if k not in ("run_id", "unit_id", "group", "index")}
    if identity(base) != task["run_id"]:
        raise ValueError("learning-cycle deployment identity changed")
    for path, digest in task["files_sha256"].items():
        verify_file(root / path, digest)
    c = load_contract(root)
    if rdBase.rdkitVersion != c["rdkit"]:
        raise ValueError("learning-cycle chemistry version changed")
    return c


def hierarchy_for(root, artifacts, folder, save, read, progress):
    from compose_v4.control.molecular_task_search import MolecularHierarchy
    from compose_v4.control.option_continuation import (
        EXECUTABLE_PRODUCT_GATE,
        OptionContinuationKernel,
    )
    from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
    from compose_v4.experiments.t4_winner_refinement import qualified_runtime

    rt = qualified_runtime(root, artifacts)
    save("runtime_validation", rt["validation"])
    law = SavedMarkedLaw(
        rt["model"],
        folder,
        save,
        read,
        repo_root=root,
        artifact_root=artifacts,
        contract={},
        progress=progress,
    )
    kernel = OptionContinuationKernel(
        law, rt["system"], max_executor_applications=None, product_gate=EXECUTABLE_PRODUCT_GATE
    )
    return MolecularHierarchy(kernel, lazy_applicability=True, include_carbonyl_options=True), law


def dock_observation(smiles, target, seed, folder, tag, dock):
    """Unchanged preparation/search; preserve the actual ligand and pose files."""
    started = perf_counter()
    value = dock(smiles, target, tag, seed)
    hashes = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        source = Path("/tmp") / tag / name
        if source.exists():
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, folder / name)
            hashes[name] = sha256_file(folder / name)
    publish_json(
        folder / "preparation.json",
        {
            "endpoint": smiles,
            "target": target,
            "docking_seed": seed,
            "score": value,
            "pose_sha256": hashes,
            "seconds": perf_counter() - started,
        },
    )
    if value is None:
        raise RuntimeError("docking failed; charged observation is not retried")
    return value


def pmo_oracle(name, protocols):
    """Dispatch the frozen campaign's tasks, not an older probe's task subset.

    Constructing this adapter performs no scoring. Each later call is made only
    by ProgramQueryLedger. The compatibility shim is identical to the existing
    PyTDC adapter and does not replace the pinned oracle implementation.
    """
    if name not in protocols:
        raise ValueError(f"PMO task {name!r} is outside the frozen campaign")
    import sys
    import types

    import rdkit

    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda d, **kwargs: iter(d.items())
    shim.itervalues = lambda d, **kwargs: iter(d.values())
    shim.iterkeys = lambda d, **kwargs: iter(d.keys())
    shim.string_types = (str,)
    sys.modules["rdkit.six"] = rdkit.six = shim
    from tdc import Oracle

    oracle = Oracle(name=name)
    return lambda smiles: float(oracle(smiles))


def unit_reservations(folder):
    """Read charged reservations, retaining unresolved attempts without retry."""
    rows = []
    for reservation in sorted((folder / "oracle").glob("query_*/started.json")):
        receipt = reservation.with_name("result.json")
        started = json.loads(reservation.read_text())
        row = (
            json.loads(receipt.read_text())
            if receipt.exists()
            else {**started, "status": "started"}
        )
        if receipt.exists() and (
            identity({k: v for k, v in row.items() if k != "receipt_id"}) != row.get("receipt_id")
            or any(row.get(k) != v for k, v in started.items())
        ):
            raise ValueError(f"corrupt existing query receipt: {receipt}")
        if row["index"] != len(rows):
            raise ValueError(f"noncontiguous query reservations: {reservation}")
        rows.append(row)
    return rows


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    c = validate_launch(task, root, validate_revision)
    unit = next((r for r in c["units"] if r["unit_id"] == task["unit_id"]), None)
    if unit is None:
        raise ValueError("unit outside the frozen launch census")
    volume.reload()
    folder = artifacts / KIND / task["run_id"] / "units" / unit["unit_id"]
    if (folder / "result.json").exists():
        return unseal(folder / "result.json")
    if (folder / "started.json").exists():
        # A duplicate/ambiguous invocation must stop THIS unit, not cancel other
        # independent workers. Read reservations without evaluating or retrying.
        rows = unit_reservations(folder)
        failure = {
            "unit": unit,
            "error": "started unit requires explicit receipt-based recovery, not relaunch",
            "failure_kind": "restart_guard",
            "oracle_calls": len(rows),
            "rows": rows,
            "at": _stamp(),
            "automatic_retry": False,
            "accounting_complete": False,
        }
        seal(folder / "restart_guard.json", failure)
        volume.commit()
        return failure
    seal(folder / "started.json", {"unit": unit, "task": task, "at": _stamp()})
    volume.commit()
    started, progress = perf_counter(), {"phase": "runtime", "oracle_calls": 0}
    lock, stop = threading.RLock(), threading.Event()

    def flush():
        with lock:
            volume.commit()

    def save(name, value):
        with lock:
            seal(folder / f"{name}.json", value)

    def read(name):
        path = folder / f"{name}.json"
        return unseal(path) if path.exists() else None

    def report(row):
        with lock:
            progress.update(row)
            publish_json(folder / "progress.json", {**progress, "at": _stamp()})
            print(json.dumps({"unit": unit["unit_id"], **row}), flush=True)
            volume.commit()

    def heartbeat():
        while not stop.wait(30):
            report({"heartbeat_seconds": perf_counter() - started})

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    ledger = None
    try:
        hierarchy, law = hierarchy_for(root, artifacts, folder, save, read, progress)
        config = configured(unit)
        if unit["kind"] == "t4":
            domain = c["t4_protocols"][unit["name"]]
            for path, digest in c["docking_inputs"].items():
                verify_file(Path(path), digest)
            pt = ProgramTask(unit["name"], identity(domain), "t4", domain["original_seed"], 0.4)
            warm = warm_archive(root, unit["name"], config, hierarchy)
            init_body = {
                "candidates": [],
                "warm_start": warm["snapshot_id"],
                "accounting": "historical paid archive declared in frozen inputs",
            }
            initialized = {**init_body, "lock_sha256": identity(init_body)}

            def evaluate(smiles):
                index = len(ledger.rows) - 1
                tag = f"pec_{task['run_id'][:10]}_{unit['unit_id']}_{index}"
                return dock_observation(
                    smiles, domain["target"], 1701, folder / "poses" / f"{index:06d}", tag, dock
                )
        else:
            domain = c["pmo_protocols"][unit["name"]]
            pt = ProgramTask(unit["name"], identity(domain), "pmo")
            evaluate = pmo_oracle(unit["name"], c["pmo_protocols"])
            initialized = json.loads((root / PREPARED / f"init_{unit['seed']}.json").read_text())
            warm = None
            import importlib.metadata

            import tdc.chem_utils.oracle.oracle as oracle_module

            verify_file(Path(oracle_module.__file__), PMO_ORACLE_SHA256)
            if importlib.metadata.version("PyTDC") != "0.3.6":
                raise ValueError("PMO oracle package differs from the locked version")
            save(
                "oracle_runtime",
                {
                    "pytdc": importlib.metadata.version("PyTDC"),
                    "source_sha256": sha256_file(Path(oracle_module.__file__)),
                },
            )
        ledger = ProgramQueryLedger(
            folder / "oracle", pt, evaluate, budget=unit["budget"], flush=flush
        )
        library = tuple(
            ProgramEntry(EditProgram.from_payload(r["program"]), tuple(r["source_groups"]))
            for r in json.loads((root / c["library_path"]).read_text())
        )
        result = run_program_campaign(
            output=folder / "campaign",
            task=pt,
            config=config,
            initialization=initialized,
            library=library,
            ledger=ledger,
            rounds=unit["rounds"],
            queries_per_round=unit["batch_queries"],
            hierarchy=hierarchy,
            fit_model=fit_measured_edits if unit["arm"] == "learned" else None,
            fit_model_id=c["update_rule_id"] if unit["arm"] == "learned" else None,
            stagnation_rounds=None,
            warm_start=warm,
            bootstrap_rounds=1 if warm is not None else 4,
            progress=report,
            max_seconds=1800,
        )
        result.update(
            unit=unit,
            task=task,
            seconds=perf_counter() - started,
            rows=ledger.rows,
            law_work=law.counts,
            kernel_work=asdict(hierarchy.kernel.work),
            completed_at_utc=_stamp(),
            software={"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        )
        save("result", result)
        flush()
        return {
            k: result[k]
            for k in (
                "unit",
                "history",
                "oracle_calls",
                "seconds",
                "rows",
                "pmo_top10_auc_with_flat_tail",
            )
        }
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
        result = {
            "unit": unit,
            "error": repr(error),
            "seconds": perf_counter() - started,
            "oracle_calls": 0 if ledger is None else len(ledger.rows),
            "rows": [] if ledger is None else ledger.rows,
            "at": _stamp(),
        }
        save("failure", result)
        flush()
        return result  # First-class failure; never silently substituted or retried.
    finally:
        stop.set()
        thread.join(timeout=2)


def confirmation_selection(c, results):
    queries = {}
    for cell in ("braf_1", "jak2_1"):
        endpoints = [(c["incumbents"][cell]["smiles"], "incumbent")]
        for arm in ("score_blind", "learned"):
            eligible = [
                r
                for result in results
                if result["unit"]["name"] == cell and result["unit"]["arm"] == arm
                for r in result["rows"]
                if r["status"] == "complete"
            ]
            best = min(eligible, key=lambda r: (r["score"], r["endpoint"]), default=None)
            if best is not None:
                endpoints.append((best["endpoint"], arm))
        for endpoint, role in endpoints:
            for seed in (1702, 1703):
                row = {"cell": cell, "endpoint": endpoint, "seed": seed}
                key = identity(row)
                if key not in queries:
                    queries[key] = {**row, "query_id": key, "roles": []}
                queries[key]["roles"].append(role)
    if len(queries) > 12:
        raise ValueError("confirmation selection exceeds its twelve-call reservation")
    return [queries[k] for k in sorted(queries)]


def run_confirmation(task, root, artifacts, volume, validate_revision, dock):
    c = validate_launch(task, root, validate_revision)
    volume.reload()
    folder = artifacts / KIND / task["run_id"] / "t4"
    locked = unseal(folder / "confirmation_lock.json")
    first = unseal(folder / "first_result.json")
    if locked != confirmation_selection(c, first["results"]):
        raise ValueError("confirmation lock differs from predeclared selection")
    if type(task["index"]) is not int or not 0 <= task["index"] < len(locked):
        raise ValueError("confirmation index outside approved census")
    row = locked[task["index"]]
    output = folder / "confirmations" / row["query_id"]
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    if (output / "started.json").exists():
        raise RuntimeError("ambiguous prior confirmation; no retry")
    for path, digest in c["docking_inputs"].items():
        verify_file(Path(path), digest)
    domain = c["t4_protocols"][row["cell"]]
    pt = ProgramTask(row["cell"], identity(domain), "t4", domain["original_seed"], 0.4)
    if not pt.endpoint_evaluator()({"smiles": row["endpoint"]})["oracle_eligible"]:
        raise ValueError("confirmation lost endpoint eligibility")
    seal(output / "started.json", {**row, "at": _stamp()})
    volume.commit()
    try:
        score = dock_observation(
            row["endpoint"],
            domain["target"],
            row["seed"],
            output,
            f"pec_confirm_{task['run_id'][:12]}_{task['index']}",
            dock,
        )
        result = {**row, "score": score, "status": "complete", "at": _stamp()}
    except (RuntimeError, ValueError, OSError) as error:
        result = {**row, "score": None, "status": "failed", "error": repr(error), "at": _stamp()}
    seal(output / "result.json", result)
    volume.commit()
    return result


def zero_query_dispatch_failures(previous, contract_sha256):
    """Explicit recovery only for completed, identified pre-oracle dispatch failures.

    Scored, ambiguous, or differently failed units are never resubmitted by this
    path. The first attempt and all its compute remain part of the campaign.
    """
    if previous["group"] != "pmo" or previous["task"]["files_sha256"][CONTRACT] != contract_sha256:
        raise ValueError("recovery predecessor has a different group or frozen contract")
    units = []
    for row in previous["results"]:
        name = row["unit"]["name"]
        if row.get("error") != repr(ValueError(f"undeclared PMO oracle {name}")):
            continue
        if name not in ("isomers_c7h8n2o2", "scaffold_hop"):
            raise ValueError("unexpected dispatch failure outside the identified repair")
        if row["oracle_calls"] != 0 or row["rows"]:
            raise ValueError("dispatch recovery cannot retry a charged or ambiguous query")
        units.append(row["unit"]["unit_id"])
    if not units or len(set(units)) != len(units):
        raise ValueError("no distinct zero-query dispatch failures to recover")
    return sorted(units)


def run_group(task, root, artifacts, volume, validate_revision, parallel, confirm=None):
    c = validate_launch(task, root, validate_revision)
    group = task["group"]
    if group not in ("t4", "pmo"):
        raise ValueError("unknown campaign group")
    volume.reload()
    units = [r["unit_id"] for r in c["units"] if r["kind"] == group]
    if "recovery" in task:
        prior = task["recovery"]
        if (
            group != "pmo"
            or len(prior["run_id"]) != 64
            or any(ch not in "0123456789abcdef" for ch in prior["run_id"])
        ):
            raise ValueError("invalid recovery predecessor")
        previous = unseal(artifacts / KIND / prior["run_id"] / "pmo/result.json")
        if identity(previous) != prior["result_sha256"]:
            raise ValueError("recovery predecessor results changed")
        recovered = zero_query_dispatch_failures(previous, task["files_sha256"][CONTRACT])
        if recovered != prior["units"] or not set(recovered).issubset(units):
            raise ValueError("recovery unit census changed")
        units = recovered
        claim = artifacts / KIND / prior["run_id"] / "pmo/dispatch_recovery.json"
        if claim.exists() and unseal(claim)["run_id"] != task["run_id"]:
            raise RuntimeError("predecessor already has an explicit dispatch recovery")
        seal(claim, {"run_id": task["run_id"], "units": units, "at": _stamp()})
        volume.commit()
    folder = artifacts / KIND / task["run_id"] / group
    if (folder / "result.json").exists():
        return unseal(folder / "result.json")
    if (folder / "started.json").exists():
        raise RuntimeError("group already started; inspect unit receipts instead of respawning")
    seal(folder / "started.json", task)
    volume.commit()
    rows = []
    # Ordered exception-valued map results isolate failed workers. No restart.
    for unit_id, result in zip(
        units, parallel([{**task, "unit_id": unit} for unit in units]), strict=True
    ):
        if isinstance(result, BaseException):
            volume.reload()
            reserved = unit_reservations(artifacts / KIND / task["run_id"] / "units" / unit_id)
            result = {
                "unit": next(r for r in c["units"] if r["unit_id"] == unit_id),
                "error": repr(result),
                "failure_kind": "worker_exception",
                "oracle_calls": len(reserved),
                "rows": reserved,
                "automatic_retry": False,
                "accounting_complete": False,
                "at": _stamp(),
            }
            seal(folder / "worker_failures" / f"{unit_id}.json", result)
        if result["unit"]["unit_id"] != unit_id:
            raise ValueError("parallel adapter must retain input/result identities and order")
        rows.append(result)
        publish_json(
            folder / "progress.json",
            {"completed_units": len(rows), "results": rows, "at": _stamp()},
        )
        volume.commit()
    result = {
        "group": group,
        "results": sorted(rows, key=lambda r: r["unit"]["unit_id"]),
        "oracle_calls": sum(r["oracle_calls"] for r in rows),
        "accounting_complete": all(r.get("accounting_complete", True) for r in rows),
        "task": task,
        "at": _stamp(),
    }
    if group == "t4":
        if confirm is None:
            raise ValueError("T4 group requires its confirmation callback")
        seal(folder / "first_result.json", result)
        locked = confirmation_selection(c, result["results"])
        if result["oracle_calls"] + len(locked) > c["max_docking_calls"]:
            raise ValueError("T4 group exceeds total query ceiling")
        seal(folder / "confirmation_lock.json", locked)
        volume.commit()
        confirmed = list(confirm([{**task, "index": i} for i in range(len(locked))]))
        result["confirmations"] = confirmed
        result["oracle_calls"] += len(confirmed)
    seal(folder / "result.json", result)
    volume.commit()
    return result
