"""Workload adapters for the zero-oracle chemistry fan-out harness.

WHAT A WORKLOAD IS
------------------
Three pure functions plus a declared input list:

    plan(**parameters) -> list[dict]      the WORK LIST; one dict per independent unit
    execute(item)      -> dict            run ONE unit; must be pure in ``item``
    reduce(plan, rows) -> dict            reassemble the units into the artifact the
                                          equivalent local driver would have written

``execute`` is called identically by the local driver and by the Modal container, so the
equivalence proof drives the SAME function on both sides rather than a transcription of
it.  Nothing here imports ``modal``: this module is importable on a laptop with no Modal
credentials, which is what makes the local baseline arm possible.

ZERO ORACLE, BY CONSTRUCTION AND BY CHECK
-----------------------------------------
Every workload registered here computes only free chemistry -- RDKit sanitisation, QED,
SA, Tanimoto, and the COMPOSE fiber/executor.  ``assert_zero_oracle`` is called inside
``execute`` and refuses to proceed if a docking binary or a TDC oracle module has entered
the process, so a future edit that quietly adds one fails loudly instead of silently
spending budget.

The first registered workload is the T4-v2 feasibility grid.  It is decomposed at ARM
granularity: ``scripts/t4_v2_feasibility_gate.run_cell`` runs four arms plus four
component freezes plus one attribution for a single source, and those nine calls share no
state, so they are nine units.  ``reduce`` rebuilds ``run_cell``'s exact payload from
them.  A ``cell``-granularity mode calls ``run_cell`` verbatim and exists so the
arm-granularity reassembly can be proved against the production function itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

#: Resolved identically on a laptop (``<repo>/modal_apps/..``) and inside the container
#: (``/root/compose_v4/workloads/..``), because the harness copies this file one level
#: below the remote root.
ROOT = Path(__file__).resolve().parents[1]

#: Module names whose PRESENCE in ``sys.modules`` means an oracle path has been imported.
#: ``tdc`` transitively pulls the property-prediction oracles; ``vina``/``openbabel`` are
#: the docking path.  Importing is not the same as calling, but a zero-oracle workload has
#: no reason to import any of them, so presence is treated as the failure.
FORBIDDEN_MODULES = ("tdc", "vina", "openbabel", "pyscreener")

#: Docking binaries the production T4 image carries.  A zero-oracle container must not.
FORBIDDEN_BINARIES = ("/opt/dock/qvina02", "/opt/dock/qvina-w", "/usr/bin/obabel")


class OracleContactError(RuntimeError):
    """Raised when a zero-oracle unit finds an oracle or docking path in the process."""


def assert_zero_oracle() -> dict:
    """Refuse to run if anything oracle-shaped is reachable.  Returns the evidence."""

    imported = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    if imported:
        raise OracleContactError(f"oracle modules imported: {imported}")
    present = sorted(path for path in FORBIDDEN_BINARIES if Path(path).exists())
    if present:
        raise OracleContactError(f"docking binaries present in image: {present}")
    return {
        "forbidden_modules_imported": imported,
        "forbidden_binaries_present": present,
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def canonical_sha256(payload: object) -> str:
    """Key-order-independent identity for a JSON-shaped payload."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _unit_id(item: dict) -> str:
    body = {key: value for key, value in item.items() if key != "unit_id"}
    return canonical_sha256(body)[:24]


# ---- Workload: T4-v2 feasibility grid ----------------------------------------------

#: Repository-relative files the container needs beyond ``src`` and ``scripts``.  Kept
#: explicit: ``diagnostics/`` is 875 MB and must never be copied wholesale into an image.
T4_V2_INPUTS = (
    "diagnostics/t4_support_stage_audit_v1.json",
)

#: Fields excluded from the equivalence comparison because they measure the machine, not
#: the chemistry.  Every other field is compared.
T4_V2_TIMING_FIELDS = ("elapsed_seconds", "wall_seconds", "seconds_per_1k_proposals")


def _t4_gate():
    """Import the production gate/law lazily so this module stays import-cheap."""

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import t4_v2_feasibility_gate as gate

    return gate


def t4_v2_plan(
    *,
    cells: tuple[str, ...] | list[str] = (),
    budget: int = 6000,
    seed: int = 1,
    delta: float = 0.6,
    arms: tuple[str, ...] | list[str] = (),
    ablate: bool = True,
    attribution: bool = True,
    chemistry_filter: bool = True,
    prune_pathological_parents: bool = True,
    granularity: str = "arm",
    **_ignored,
) -> list[dict]:
    """The work list for the four-arm x N-source grid.

    ``granularity='arm'`` emits one unit per (cell, arm), per (cell, frozen component) and
    per (cell, attribution).  ``granularity='cell'`` emits one unit per cell that calls
    ``run_cell`` verbatim -- slower to fan out, but it is the production function, so it
    is the reference the arm decomposition is proved against.
    """

    gate = _t4_gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    selected = tuple(cells) if cells else tuple(gate.FAILED + gate.CONTROLS + gate.SANITY)
    unknown = [cell for cell in selected if cell not in sources]
    if unknown:
        raise ValueError(f"unknown cells: {unknown}")
    requested_arms = tuple(arms) if arms else tuple(gate.ARMS)
    bad = [name for name in requested_arms if name not in gate.ARMS]
    if bad:
        raise ValueError(f"unknown arms: {bad}")
    if granularity not in ("arm", "cell"):
        raise ValueError("granularity must be 'arm' or 'cell'")

    shared = {
        "budget": int(budget),
        "seed": int(seed),
        "delta": float(delta),
        "chemistry_filter": bool(chemistry_filter),
        "prune_pathological_parents": bool(prune_pathological_parents),
    }
    items: list[dict] = []
    for cell in selected:
        if granularity == "cell":
            items.append(
                {
                    "workload": "t4_v2_feasibility",
                    "kind": "cell",
                    "cell": cell,
                    "arms": list(requested_arms),
                    "ablate": bool(ablate),
                    "attribution": bool(attribution),
                    "params": dict(shared),
                }
            )
            continue
        # ARMS order, not the caller's order: ``run_cell`` iterates ``ARMS.items()`` and
        # takes ``slot_preflight`` from the first INCLUDED arm, so the reducer has to see
        # the same order to reassemble the identical payload.
        for name in gate.ARMS:
            if name not in requested_arms:
                continue
            items.append(
                {
                    "workload": "t4_v2_feasibility",
                    "kind": "arm",
                    "cell": cell,
                    "arm": name,
                    "params": dict(shared),
                }
            )
        if ablate:
            for component in gate.FREEZES:
                items.append(
                    {
                        "workload": "t4_v2_feasibility",
                        "kind": "freeze",
                        "cell": cell,
                        "component": component,
                        "params": dict(shared),
                    }
                )
        if attribution:
            items.append(
                {
                    "workload": "t4_v2_feasibility",
                    "kind": "attribution",
                    "cell": cell,
                    "params": {"delta": shared["delta"]},
                }
            )
    for item in items:
        item["unit_id"] = _unit_id(item)
    return items


def t4_v2_execute(item: dict) -> dict:
    """Run ONE unit.  Pure in ``item``; no shared state with any other unit."""

    evidence = assert_zero_oracle()
    gate = _t4_gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    cell = item["cell"]
    smiles = sources[cell]
    params = item["params"]
    started = time.time()

    if item["kind"] == "arm":
        configuration = gate.ARMS[item["arm"]]
        result = gate.law.run_search(
            smiles,
            params["delta"],
            budget=params["budget"],
            seed=params["seed"],
            chemistry_filter=params["chemistry_filter"],
            prune_pathological_parents=params["prune_pathological_parents"],
            **configuration,
        )
        payload = {
            "summary": gate.summarise_arm(result),
            "slot_preflight": result["slot_preflight"],
        }
    elif item["kind"] == "freeze":
        result = gate.law.run_search(
            smiles,
            params["delta"],
            budget=params["budget"],
            seed=params["seed"],
            frozen=(item["component"],),
            chemistry_filter=params["chemistry_filter"],
            prune_pathological_parents=params["prune_pathological_parents"],
        )
        payload = {"funnel": gate.summarise_arm(result)["funnel"]}
    elif item["kind"] == "attribution":
        payload = {
            "attribution": gate.law.attribute_failure(
                smiles, params["delta"], gate.v1_endpoint_sample(audit, cell)
            )
        }
    elif item["kind"] == "cell":
        payload = {
            "run_cell": gate.run_cell(
                cell,
                smiles,
                audit,
                budget=params["budget"],
                seed=params["seed"],
                ablate=item["ablate"],
                delta=params["delta"],
                arms=tuple(item["arms"]),
                chemistry_filter=params["chemistry_filter"],
                prune_pathological_parents=params["prune_pathological_parents"],
                attribution=item["attribution"],
            )
        }
    else:
        raise ValueError(f"unknown unit kind: {item['kind']}")

    return {
        "unit_id": item["unit_id"],
        "workload": item["workload"],
        "kind": item["kind"],
        "cell": cell,
        "arm": item.get("arm"),
        "component": item.get("component"),
        "payload": payload,
        "zero_oracle_evidence": evidence,
        "wall_seconds": round(time.time() - started, 3),
        "hostname": os.uname().nodename,
    }


def t4_v2_reduce(plan: list[dict], rows: list[dict]) -> dict:
    """Rebuild the ``run`` stage payload of ``scripts/t4_v2_feasibility_gate.py``."""

    gate = _t4_gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    by_id = {row["unit_id"]: row for row in rows}
    missing = [item["unit_id"] for item in plan if item["unit_id"] not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} units missing from the result set: {missing[:5]}")

    first = plan[0]["params"]
    merged: dict = {
        "schema_version": gate.law.SCHEMA_VERSION,
        "budget": first.get("budget"),
        "seed": first.get("seed"),
        "delta": first.get("delta"),
        "chemistry_filter": first.get("chemistry_filter"),
        "prune_pathological_parents": first.get("prune_pathological_parents"),
        "cells": {},
    }
    for item in plan:
        if item["kind"] == "cell":
            merged["cells"][item["cell"]] = by_id[item["unit_id"]]["payload"]["run_cell"]
            merged["budget"] = item["params"]["budget"]
            merged["seed"] = item["params"]["seed"]
            merged["delta"] = item["params"]["delta"]
            merged["chemistry_filter"] = item["params"]["chemistry_filter"]
            merged["prune_pathological_parents"] = item["params"][
                "prune_pathological_parents"
            ]
            continue
        cell = item["cell"]
        payload = merged["cells"].get(cell)
        if payload is None:
            payload = {
                "cell": cell,
                "source_smiles": sources[cell],
                "delta": item["params"]["delta"],
                "v1_reference_delta": 0.6,
                "observed_status": audit["cells"][cell]["observed_status"],
                "source": audit["cells"][cell]["source"],
                "v1": gate.v1_funnel(audit, cell),
                "arms": {},
                "h_component_freeze": {},
            }
            merged["cells"][cell] = payload
        row = by_id[item["unit_id"]]
        if item["kind"] == "arm":
            payload["arms"][item["arm"]] = row["payload"]["summary"]
            payload.setdefault("slot_preflight", row["payload"]["slot_preflight"])
        elif item["kind"] == "freeze":
            payload["h_component_freeze"][item["component"]] = row["payload"]["funnel"]
        elif item["kind"] == "attribution":
            payload["attribution"] = row["payload"]["attribution"]
    # ``run_cell`` inserts ``attribution`` before the arm loop populates
    # ``slot_preflight``; key ORDER is irrelevant to the identity used here (canonical
    # sorted-key JSON), and is reproduced only so a human diff of the two artifacts is
    # quiet.
    for cell, payload in merged["cells"].items():
        if "attribution" in payload and "slot_preflight" in payload:
            attribution = payload.pop("attribution")
            preflight = payload.pop("slot_preflight")
            payload["attribution"] = attribution
            payload["slot_preflight"] = preflight
    return merged


def t4_v2_strip_timing(payload: object) -> object:
    """Recursively drop the machine-dependent fields before comparing two runs."""

    if isinstance(payload, dict):
        return {
            key: t4_v2_strip_timing(value)
            for key, value in payload.items()
            if key not in T4_V2_TIMING_FIELDS
        }
    if isinstance(payload, list):
        return [t4_v2_strip_timing(value) for value in payload]
    return payload


# ---- Workload: chemistry-kernel parity ----------------------------------------------

#: Every quantity the T4 gate thresholds on, computed on real corpus molecules.  Two
#: environments that agree here will agree on the gate; two that disagree here cannot be
#: compared field-by-field, and the disagreement localises to the kernel rather than to
#: the harness.
def chemistry_parity_plan(*, cells: tuple[str, ...] | list[str] = (), **_ignored) -> list[dict]:
    gate = _t4_gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    selected = tuple(cells) if cells else tuple(sorted(sources))
    items = [
        {"workload": "chemistry_kernel_parity", "kind": "cell_molecules", "cell": cell}
        for cell in selected
    ]
    for item in items:
        item["unit_id"] = _unit_id(item)
    return items


def chemistry_parity_execute(item: dict) -> dict:
    """Score one cell's molecules with the exact primitives the gate thresholds on."""

    evidence = assert_zero_oracle()
    gate = _t4_gate()
    import t4_v2_feasibility_proposal as law
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    cell = item["cell"]
    source_smiles = sources[cell]
    started = time.time()

    candidates = [source_smiles]
    for row in gate.v1_endpoint_sample(audit, cell):
        if row.get("smiles"):
            candidates.append(row["smiles"])
    # Deterministic, program-free and RNG-fixed: the same molecule list on both sides.
    for row in law.removable_arms(source_smiles, 0.6)[:40]:
        if row.get("smiles"):
            candidates.append(row["smiles"])
    ordered = sorted(dict.fromkeys(candidates))

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    reference = Chem.MolFromSmiles(source_smiles)
    reference_fingerprint = generator.GetFingerprint(reference)

    rows = []
    for smiles in ordered:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            rows.append({"input": smiles, "parses": False})
            continue
        rows.append(
            {
                "input": smiles,
                "parses": True,
                "canonical": Chem.MolToSmiles(molecule),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "rings": molecule.GetRingInfo().NumRings(),
                "aromatic_atoms": sum(
                    1 for atom in molecule.GetAtoms() if atom.GetIsAromatic()
                ),
                "formal_charge": Chem.GetFormalCharge(molecule),
                "qed": round(float(QED.qed(molecule)), 9),
                "sa": round(float(law.sascorer.calculateScore(molecule)), 9),
                "tanimoto_to_source": round(
                    float(
                        DataStructs.TanimotoSimilarity(
                            reference_fingerprint, generator.GetFingerprint(molecule)
                        )
                    ),
                    9,
                ),
            }
        )
    return {
        "unit_id": item["unit_id"],
        "workload": item["workload"],
        "kind": item["kind"],
        "cell": cell,
        "arm": None,
        "component": None,
        "payload": {"molecules": rows, "molecule_count": len(rows)},
        "zero_oracle_evidence": evidence,
        "wall_seconds": round(time.time() - started, 3),
        "hostname": os.uname().nodename,
    }


def chemistry_parity_reduce(plan: list[dict], rows: list[dict]) -> dict:
    by_id = {row["unit_id"]: row for row in rows}
    cells: dict[str, dict] = {}
    for item in plan:
        row = by_id.get(item["unit_id"])
        if row is None:
            raise RuntimeError(f"missing unit {item['unit_id']}")
        cells[item["cell"]] = {
            "molecules": row["payload"]["molecules"],
            "molecule_count": row["payload"]["molecule_count"],
            "cell_sha256": canonical_sha256(row["payload"]["molecules"]),
        }
    return {
        "schema_version": "chemistry_kernel_parity_v1",
        "cells": cells,
        "molecule_total": sum(row["molecule_count"] for row in cells.values()),
        "parity_sha256": canonical_sha256(
            {cell: row["cell_sha256"] for cell, row in cells.items()}
        ),
    }


def identity_strip(payload: object) -> object:
    """This workload has no machine-dependent fields; every value is compared."""

    return payload


# ---- Workload: T4-v2 trajectory trace -----------------------------------------------

#: The per-endpoint scalars the search branches on.  Recorded in EVALUATION ORDER, which
#: is the search's own cache insertion order, so the first index at which two runs differ
#: is the first decision they took differently.
TRACE_FIELDS = (
    "smiles",
    "parses",
    "heavy",
    "similarity",
    "qed",
    "sa",
    "sim_ok",
    "qed_ok",
    "sa_ok",
    "med_chem_ok",
    "benchmark_eligible",
    "stage",
    "eligible",
    "mode",
)


def trajectory_plan(
    *,
    cells: tuple[str, ...] | list[str] = (),
    arms: tuple[str, ...] | list[str] = (),
    budget: int = 600,
    seed: int = 1,
    delta: float = 0.6,
    **_ignored,
) -> list[dict]:
    gate = _t4_gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    selected = tuple(cells) if cells else tuple(sorted(sources))
    requested = tuple(arms) if arms else tuple(gate.ARMS)
    items = [
        {
            "workload": "t4_v2_trajectory_trace",
            "kind": "trace",
            "cell": cell,
            "arm": name,
            "params": {"budget": int(budget), "seed": int(seed), "delta": float(delta)},
        }
        for cell in selected
        for name in gate.ARMS
        if name in requested
    ]
    for item in items:
        item["unit_id"] = _unit_id(item)
    return items


def trajectory_execute(item: dict) -> dict:
    """Replay one arm and record what it saw, in the order it saw it."""

    evidence = assert_zero_oracle()
    gate = _t4_gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    params = item["params"]
    started = time.time()
    result = gate.law.run_search(
        sources[item["cell"]],
        params["delta"],
        budget=params["budget"],
        seed=params["seed"],
        **gate.ARMS[item["arm"]],
    )

    def scalar(value: object) -> object:
        return round(value, 9) if isinstance(value, float) else value

    trace = [
        {field: scalar(row.get(field)) for field in TRACE_FIELDS}
        for row in result["rows"]
    ]
    return {
        "unit_id": item["unit_id"],
        "workload": item["workload"],
        "kind": item["kind"],
        "cell": item["cell"],
        "arm": item["arm"],
        "component": None,
        "payload": {
            "trace": trace,
            "evaluated": len(trace),
            "steps": result["steps"],
            "slot_preflight": result["slot_preflight"],
        },
        "zero_oracle_evidence": evidence,
        "wall_seconds": round(time.time() - started, 3),
        "hostname": os.uname().nodename,
    }


def trajectory_reduce(plan: list[dict], rows: list[dict]) -> dict:
    by_id = {row["unit_id"]: row for row in rows}
    arms: dict[str, dict] = {}
    for item in plan:
        row = by_id.get(item["unit_id"])
        if row is None:
            raise RuntimeError(f"missing unit {item['unit_id']}")
        arms.setdefault(item["cell"], {})[item["arm"]] = row["payload"]
    return {
        "schema_version": "t4_v2_trajectory_trace_v1",
        "cells": arms,
        "trace_sha256": canonical_sha256(arms),
    }


WORKLOADS = {
    "t4_v2_feasibility": {
        "plan": t4_v2_plan,
        "execute": t4_v2_execute,
        "reduce": t4_v2_reduce,
        "strip_timing": t4_v2_strip_timing,
        "inputs": T4_V2_INPUTS,
        "description": (
            "T4-v2 feasibility-conditioned proposal grid: four arms, four h-component "
            "freezes and one bottleneck attribution per source. Zero oracle calls."
        ),
    },
    "t4_v2_trajectory_trace": {
        "plan": trajectory_plan,
        "execute": trajectory_execute,
        "reduce": trajectory_reduce,
        "strip_timing": identity_strip,
        "inputs": T4_V2_INPUTS,
        "description": (
            "Replay a T4-v2 arm and record every evaluated endpoint in evaluation "
            "order, so two environments can be diffed to the first divergent decision."
        ),
    },
    "chemistry_kernel_parity": {
        "plan": chemistry_parity_plan,
        "execute": chemistry_parity_execute,
        "reduce": chemistry_parity_reduce,
        "strip_timing": identity_strip,
        "inputs": T4_V2_INPUTS,
        "description": (
            "Score real corpus molecules with the exact primitives the T4 gate "
            "thresholds on (canonical SMILES, ring perception, QED, SA, Tanimoto), so "
            "two environments can be compared without running a stochastic search."
        ),
    },
}


def workload(name: str) -> dict:
    try:
        return WORKLOADS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown workload {name!r}; registered: {sorted(WORKLOADS)}"
        ) from error


def execute_item(item: dict) -> dict:
    """Dispatch one unit to its workload.  The single entry point both sides call."""

    return workload(item["workload"])["execute"](item)
