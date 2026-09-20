"""Resolve PMO task names against PyTDC's oracle registry WITHOUT building an oracle.

`tdc.Oracle(name=...)` resolves its argument through `fuzzy_search(name, oracle_names)`,
which returns the name unchanged when it is an exact (lowercased) member of the registry
and otherwise falls back to a Levenshtein match at threshold 0.8.  So exact registry
membership is equivalent to "Oracle() would resolve to this name", and can be checked by
reading `tdc/metadata.py` alone -- a pure data module.  Constructing an Oracle would
download predictor pickles (drd2 / gsk3b / jnk3) and is never done here.

The fuzzy fallback is the hazard this tool exists to expose: a near-miss name does not
raise, it silently resolves to a DIFFERENT oracle.  `sitagliptin_mpo_prev` and
`zaleplon_mpo_prev` sit one token away from two PMO tasks.

IMPORTANT: PyTDC does NOT define the PMO suite.  It ships the oracles; the 23-task
membership comes from the PMO benchmark (Gao et al., "Sample Efficiency Matters").  This
tool therefore verifies RESOLVABILITY, and takes suite membership from the published
transcriptions under `docs/`.  It also ships no machine-readable oracle direction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IVG_TABLE = ROOT / "docs/invirtuogen_pmo_targets.json"
GENMOL_TABLE = ROOT / "docs/genmol_pmo_targets.json"


def pmo_suite_23() -> tuple[list[str], dict]:
    """The 23 PMO task names, taken from two independent published transcriptions.

    Both are read and required to agree exactly; a disagreement means one of the
    transcriptions drifted and the suite definition is not trustworthy.
    """

    ivg = json.loads(IVG_TABLE.read_text())
    genmol = json.loads(GENMOL_TABLE.read_text())
    a, b = set(ivg["targets"]), set(genmol["targets"])
    if a != b:
        raise ValueError(f"published PMO task lists disagree: {sorted(a ^ b)}")
    if len(a) != 23:
        raise ValueError(f"expected 23 PMO tasks, found {len(a)}")
    provenance = {
        "note": (
            "PyTDC does not define the PMO suite; it ships the oracles. Suite membership "
            "is taken from two independent published transcriptions that agree exactly."
        ),
        "sources": [
            {"path": str(IVG_TABLE.relative_to(ROOT)), "source": ivg["source"],
             "sha256": hashlib.sha256(IVG_TABLE.read_bytes()).hexdigest()},
            {"path": str(GENMOL_TABLE.relative_to(ROOT)), "source": genmol["source"],
             "sha256": hashlib.sha256(GENMOL_TABLE.read_bytes()).hexdigest()},
        ],
        "agreement": "exact, 23/23",
    }
    return sorted(a), provenance


def load_registry(wheel: Path | None, tdc_root: Path | None) -> tuple[object, dict]:
    """Load `tdc.metadata` as a standalone module. No package import, no network."""

    provenance: dict = {}
    if wheel is not None:
        provenance = {
            "kind": "wheel",
            "path": str(wheel),
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
        extract = Path(tempfile.mkdtemp(prefix="pytdc_registry_"))
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(extract)
        metadata = extract / "tdc" / "metadata.py"
    elif tdc_root is not None:
        metadata = tdc_root / "metadata.py"
        provenance = {"kind": "directory", "path": str(tdc_root)}
    else:
        spec = importlib.util.find_spec("tdc")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit(
                "no PyTDC found: pass --wheel <pytdc-*.whl> or --tdc-root <dir>/tdc"
            )
        metadata = Path(next(iter(spec.submodule_search_locations))) / "metadata.py"
        provenance = {"kind": "installed", "path": str(metadata.parent)}
    if not metadata.is_file():
        raise SystemExit(f"tdc/metadata.py not found at {metadata}")
    provenance["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()

    # metadata.py imports pkg_resources at module scope but never uses it for the
    # registry lists; stub it so the load needs no setuptools and no network.
    sys.modules.setdefault("pkg_resources", types.ModuleType("pkg_resources"))
    spec = importlib.util.spec_from_file_location("_tdc_metadata_readonly", metadata)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, provenance


def resolve(names: list[str], registry) -> dict:
    """Classify each name exactly as `fuzzy_search` would, without calling it."""

    oracle_names = list(registry.oracle_names)
    download = set(registry.download_oracle_names)
    trivial = set(registry.trivial_oracle_names)
    rows = {}
    for name in names:
        lowered = name.lower()
        exact = lowered in oracle_names
        rows[name] = {
            "resolves": exact,
            "resolution": "exact" if exact else "NOT_IN_REGISTRY_would_fuzzy_match",
            # PMO oracles are all maximize on [0, 1]; TDC ships no direction field,
            # so the direction is recorded from the PMO/mol_opt convention and is
            # independently enforced by pmo_top_ten_auc's [0, 1] validation.
            "direction": "maximize",
            "range": [0.0, 1.0],
            "backing": (
                "downloaded_predictor" if lowered in download
                else "analytic_guacamol_or_rdkit" if lowered in trivial
                else "other"
            ),
        }
    near_miss = sorted(n for n in oracle_names if n.endswith("_prev"))
    return {
        "registry_size": len(oracle_names),
        "tasks": rows,
        "all_resolve": all(r["resolves"] for r in rows.values()),
        "unresolved": sorted(n for n, r in rows.items() if not r["resolves"]),
        "direction_source": (
            "PMO/mol_opt convention (all 23 oracles are maximize on [0,1]); PyTDC ships "
            "NO machine-readable direction field. Independently enforced by "
            "compose_v4.control.program_task.pmo_top_ten_auc, which rejects any reward "
            "outside [0,1]."
        ),
        "fuzzy_fallback_hazard": {
            "threshold": 0.8,
            "behaviour": "a non-exact name resolves to the closest registry entry, silently",
            "near_miss_entries_present": near_miss,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, default=None)
    parser.add_argument("--tdc-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    names, suite_provenance = pmo_suite_23()
    registry, registry_provenance = load_registry(args.wheel, args.tdc_root)
    report = {
        "schema_version": "pmo_task_registry_verification_v1",
        "oracle_constructed": False,
        "network_access": False,
        "suite": {"n_tasks": len(names), "tasks": names, "provenance": suite_provenance},
        "registry_provenance": registry_provenance,
        "verification": resolve(names, registry),
        # Sealed fail-closed like every other prepared artifact: this is a
        # read-only verification report and authorizes nothing.
        "scored_launch_authorized": False,
        "modal_launch_authorized": False,
        "oracle_calls_authorized": 0,
        "status": "VERIFICATION_ONLY_FAIL_CLOSED",
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
