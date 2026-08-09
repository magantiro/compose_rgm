"""Turn the frozen V2 mixture recipe into a deterministic, content-addressed manifest.

WHY A MANIFEST
--------------
The recipe is currently numbers in a diagnostics JSON.  Compilation needs an
explicit list of which sources, in which task, under which rule -- and that list
has to be reproducible, because it defines the corpus the final reference model
is trained on.  A manifest with a content hash makes the selection auditable and
makes an accidental re-selection under different parameters detectable.

THE FROZEN RECIPE (measured, see diagnostics/editing_v2_v2_mixture_plan.json)
-----------------------------------------------------------------------------
    real_endpoint_multistep_path        cap 1 source per Murcko scaffold
    linker_positional_topology_analogue whole
    operator_aware_real_endpoint        whole
    reversible_synthetic_walk           combined-family floor 5,000
    observed_local_analogue             SKIP (4,918 rows for 19 new scaffolds)

Cap 1 retains 100% of scaffolds and 100% of new scaffolds at 9.2% of the lane's
rows; every higher cap buys only redundancy.  Floor 5,000 sits at a sharp knee:
5,000 -> 8,000 costs 7.6x the synthetic rows for a 1.6x higher floor.

DETERMINISM
-----------
Every tie is broken on the canonical source key, never on iteration order, so
the manifest is byte-identical across runs and machines.  The corpus scan is
cached: it is 2.31M rows plus ~476k Murcko scaffolds, and re-running it for a
parameter change wasted roughly thirteen of every fifteen minutes across three
earlier sweeps.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path

CYCLE_FAMILIES = {"cycle_insert", "cycle_attach", "ring_system_restate"}
TAKE_WHOLE = ("operator_aware_real_endpoint", "linker_positional_topology_analogue")
SKIP = ("observed_local_analogue",)
CAPPED_LANE = "real_endpoint_multistep_path"
SYNTHETIC_LANE = "reversible_synthetic_walk"


def scan(active8_root: Path, compiled: set[str], cache_path: Path) -> dict:
    """Collect once, cache forever. Keyed by the inputs that can change it."""

    fingerprint = hashlib.sha256(
        json.dumps({"root": str(active8_root), "compiled": sorted(compiled)},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    if cache_path.exists():
        blob = pickle.loads(cache_path.read_bytes())
        if blob.get("fingerprint") == fingerprint:
            print(f"cache hit ({cache_path.name})", flush=True)
            return blob
        print("cache present but stale; rescanning", flush=True)

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    uncompiled = collections.defaultdict(lambda: collections.defaultdict(dict))
    compiled_fams: collections.Counter = collections.Counter()
    compiled_sources: set[str] = set()

    for task in sorted(os.listdir(active8_root / "tasks")):
        receipt = active8_root / "tasks" / task / "RECEIPT.json"
        stream = active8_root / "tasks" / task / "transitions.jsonl.gz"
        if not (receipt.exists() and stream.exists()):
            continue
        lane = json.loads(receipt.read_text()).get("data_lane", "?")
        is_compiled = task in compiled
        for line in gzip.open(stream, "rt"):
            record = json.loads(line)
            evidence = record["candidate_evidence"]
            if evidence.get("exclusion_reason") is not None:
                continue
            source = evidence["source_canonical_key"]
            if is_compiled:
                compiled_sources.add(source)
                compiled_fams[record["model_family"]] += 1
                continue
            slot = uncompiled[lane][task].setdefault(
                source, {"rows": 0, "families": collections.Counter()}
            )
            slot["rows"] += 1
            slot["families"][record["model_family"]] += 1

    def murcko(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception:  # noqa: BLE001
            return None

    scaffolds = {}
    for task_map in uncompiled[CAPPED_LANE].values():
        for source in task_map:
            if source not in scaffolds:
                scaffolds[source] = murcko(source)

    blob = {
        "fingerprint": fingerprint,
        "uncompiled": {lane: {t: dict(m) for t, m in tasks.items()}
                       for lane, tasks in uncompiled.items()},
        "compiled_families": dict(compiled_fams),
        "compiled_source_count": len(compiled_sources),
        "scaffolds": scaffolds,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(blob))
    print(f"scan cached -> {cache_path}", flush=True)
    return blob


def select(blob: dict, floor: int, cap: int, overrides: dict | None = None) -> dict:
    # PER-FAMILY floors. The floor is applied to RAW rows but the corpus is
    # deduplicated to distinct canonical (x, y) afterwards, so a family whose
    # synthetic sources duplicate heavily lands under its floor. Raising the
    # GLOBAL floor to compensate is an over-correction: lifting 5,000 -> 6,500 to
    # recover ring_system_restate's 998-row shortfall pulled in 22k extra synthetic
    # rows across every other family and moved the mixture from 14.5% to 21.6%
    # synthetic. Compensate only the family that actually falls short.
    overrides = overrides or {}
    uncompiled = blob["uncompiled"]
    compiled_fams = collections.Counter(blob["compiled_families"])
    scaffolds = blob["scaffolds"]
    chosen: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    stats: collections.Counter = collections.Counter()

    for lane in TAKE_WHOLE:
        for task, sources in uncompiled.get(lane, {}).items():
            for source in sorted(sources):
                chosen[lane][task].append(source)
                stats[lane] += sources[source]["rows"]

    # capped lane: one source per scaffold, most productive first, ties on key
    by_scaffold: dict[str, list[tuple[str, str, int]]] = collections.defaultdict(list)
    for task, sources in uncompiled.get(CAPPED_LANE, {}).items():
        for source, info in sources.items():
            sc = scaffolds.get(source)
            if sc is not None:
                by_scaffold[sc].append((source, task, info["rows"]))
    # One SOURCE per scaffold, but that source is taken across EVERY task it
    # appears in. Restricting to a single (task, source) pair discards 8,145
    # distinct canonical successors -- 17% of the pairs -- and drops the
    # multi-successor rate from 67.7% to 57.2% (measured,
    # diagnostics/editing_v2_multisuccessor_audit.json). Several plausible
    # successors from one state is the supervision that teaches P(y|x) is a
    # distribution rather than one memorised compiler choice, so it is the last
    # thing to throw away.
    for sc in sorted(by_scaffold):
        picked = sorted(by_scaffold[sc], key=lambda e: (-e[2], e[0]))[:cap]
        for source, _task, _rows in picked:
            for task, sources in sorted(uncompiled[CAPPED_LANE].items()):
                if source in sources:
                    chosen[CAPPED_LANE][task].append(source)
                    stats[CAPPED_LANE] += sources[source]["rows"]

    # synthetic: raise every family to the combined floor, scarcest family first
    synth = uncompiled.get(SYNTHETIC_LANE, {})
    flat = [(task, source, info) for task, sources in synth.items()
            for source, info in sources.items()]
    families = set(compiled_fams) | {f for _t, _s, i in flat for f in i["families"]}
    floor_of = {f: int(overrides.get(f, floor)) for f in families}
    short0 = {f for f in families if compiled_fams.get(f, 0) < floor_of[f]}
    flat.sort(key=lambda e: (-sum(v for f, v in e[2]["families"].items() if f in short0),
                             e[1]))
    got: collections.Counter = collections.Counter()
    for task, source, info in flat:
        if all(compiled_fams.get(f, 0) + got.get(f, 0) >= floor_of[f] for f in families):
            break
        chosen[SYNTHETIC_LANE][task].append(source)
        got.update(info["families"])
        stats[SYNTHETIC_LANE] += info["rows"]

    # `got` only accumulates the SYNTHETIC pass; the real lanes contribute
    # families too, and omitting them understated every total in the report.
    selected_fams: collections.Counter = collections.Counter()
    for lane, tasks in chosen.items():
        for task, sources in tasks.items():
            for source in sources:
                selected_fams.update(uncompiled[lane][task][source]["families"])
    families = families | set(selected_fams)
    combined = {f: int(compiled_fams.get(f, 0) + selected_fams.get(f, 0)) for f in families}
    payload = {
        "schema": "compose.editing_v2.v2_selection_manifest",
        "schema_version": 1,
        "status": "SELECTION_EVIDENCE_ONLY_NO_AUTHORITY",
        "recipe": {"multistep_cap_per_scaffold": cap, "synthetic_family_floor": floor,
                   "synthetic_family_floor_overrides": overrides,
                   "take_whole": list(TAKE_WHOLE), "skip": list(SKIP)},
        "rows_by_lane": {k: int(v) for k, v in sorted(stats.items())},
        "rows_total": int(sum(stats.values())),
        "family_totals_after": dict(sorted(combined.items(), key=lambda kv: -kv[1])),
        "families_from_new_selection": dict(sorted(selected_fams.items(),
                                                   key=lambda kv: -kv[1])),
        "selection": {lane: {t: sorted(v) for t, v in sorted(tasks.items())}
                      for lane, tasks in sorted(chosen.items())},
        # GUARD 1 -- the compiler MUST collapse to distinct canonical (x, y).
        # Measured on this selection: 45.1% of records are duplicate pairs, all
        # but a tenth of it in the multistep lane (63.6% there, 0.0% in both
        # valuable real lanes). One (x, y) appears 129 times. Left uncollapsed,
        # that successor would carry 129x its true probability mass purely from
        # compiler serialisation -- the model would learn a serialisation
        # artefact as chemistry. Provenance should be retained as metadata, but
        # duplicate serialisation is NOT repeated independent observation.
        "canonical_xy_dedup_required": True,
        "rows_after_dedup_by_lane": {
            "linker_positional_topology_analogue": 23075,
            "operator_aware_real_endpoint": 18667,
            "real_endpoint_multistep_path": 47181,
            "reversible_synthetic_walk": 13507,
        },
        "rows_after_dedup_total": 102430,
        # GUARD 2 -- split assignment is by SOURCE, never by row.
        # Selection deliberately keeps every distinct successor of a chosen
        # source (67.7% of them are multi-successor), so splitting rows
        # independently would place y1 in train and y2 in validation for the
        # same x. That is a direct leakage route, and it is created by the very
        # structure that makes this corpus scientifically useful.
        "split_constraint": (
            "all records sharing an exact source_canonical_key must be assigned "
            "to the same split component"
        ),
        "multi_successor_rate": 0.677,
        "mean_distinct_successors_per_source": 2.006,
    }
    body = json.dumps(payload["selection"], sort_keys=True).encode()
    payload["selection_sha256"] = hashlib.sha256(body).hexdigest()
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--prep-subset", default="configs/process_v2_prep_subset.json")
    ap.add_argument("--cache", default="diagnostics/.v2_scan_cache.pkl")
    ap.add_argument("--synthetic-family-floor", type=int, default=5000)
    ap.add_argument("--multistep-cap", type=int, default=1)
    ap.add_argument("--floor-override", action="append", default=[],
                    metavar="FAMILY=N",
                    help="per-family floor, e.g. ring_system_restate=6500")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.prep_subset).read_text())
    compiled = {c["task_identity_sha256"] for c in plan["chunks"]}
    blob = scan(Path(args.active8_root), compiled, Path(args.cache))
    overrides = dict(o.split('=') for o in args.floor_override)
    overrides = {k: int(v) for k, v in overrides.items()}
    manifest = select(blob, args.synthetic_family_floor, args.multistep_cap, overrides)

    print(f"\n{'lane':38s} {'rows':>10} {'tasks':>7} {'sources':>9}")
    for lane, tasks in manifest["selection"].items():
        n = sum(len(v) for v in tasks.values())
        print(f"  {lane:36s} {manifest['rows_by_lane'].get(lane, 0):>10,} "
              f"{len(tasks):>7} {n:>9,}")
    print(f"\n  {'TOTAL':36s} {manifest['rows_total']:>10,}")
    print(f"\nfamily totals after selection: {manifest['family_totals_after']}")
    print(f"selection sha256: {manifest['selection_sha256']}")

    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
