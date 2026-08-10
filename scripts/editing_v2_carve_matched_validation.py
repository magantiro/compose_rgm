#!/usr/bin/env python
"""Carve a source-disjoint, support-stratified validation reserve from training.

WHY THE EXISTING PANEL COULD NOT SELECT CHECKPOINTS
---------------------------------------------------
The frozen panel is the whole ``validation`` role: 16 externally mined shards
of congeneric series. MEASURED against it, two different questions were being
collapsed into one number -- "generalize to a new source on a well-represented
scaffold" and "extrapolate to a scaffold absent from training" -- and its
support composition is an accident of how those shards were mined. Identity
NLL is 1.94 on scaffolds with 25+ training sources against 2.84 at zero
support, and 41.5% of its entries are at zero support, so the accident moves
the number by roughly 0.9 nats.

WHY THIS IS NOT SCAFFOLD-DISJOINT
---------------------------------
Making the primary validation globally scaffold-disjoint sounds maximally
rigorous and is the wrong instrument: it forces train scaffold support to 0 by
construction, which builds a NEW-SCAFFOLD EXTRAPOLATION benchmark and calls it
validation. The reserve here is exact-source disjoint and stratified by
post-split support band, so the extrapolation regime is one readable stratum
rather than the whole panel.

LEAKAGE PROTECTION IS CLUSTER-LEVEL, NOT SCAFFOLD-LEVEL
-------------------------------------------------------
Exact-source disjointness is too weak -- a trivially perturbed near-duplicate
would sit on both sides. Whole-Murcko grouping is too strong, for the reason
above. So very close analogues are clustered (ECFP4 Tanimoto >= 0.95) and whole
clusters are assigned to one side.

Blocking is exhaustive, not sampled. MEASURED: 49,623 molecular-formula blocks
over 106,759 sources, largest 44, 219,034 within-block pairs. Formula and
Murcko scaffold are both used as blocking keys and unioned, since a
near-duplicate shares at least one of them.

THREE VIEWS, ONE RESERVE
------------------------
Removing a near-duplicate cluster barely perturbs scaffold support (2.7 sources
per scaffold on average), so one selection pass stratified by lane, family and
capability cell lands a population-matched support distribution and, because
55% of training scaffolds are singletons, populates the zero-support stratum on
its own. The views are read off the strata:

    matched            support bands reweighted to the TRAINING population.
                       This is the checkpoint-selection number.
    unsupported_scaffold  support == 0: no POST-SPLIT training source shares
                       the scaffold. A property of this split under this
                       scaffold definition, not a claim of chemical novelty.
    supported_scaffold support >= 25. Editing new molecules in familiar series.

The 16-shard panel is retained as an externally mined cross-cohort test and is
deliberately NOT a selection input. final_test is never read here.

NOT A CAUSAL CLAIM
------------------
Support is not asserted to cause the gap. The relationship is not smooth --
bands 0 and 1-4 are indistinguishable and the benefit is concentrated at 25+ --
which is consistent with a latent "typical chemistry" axis. The split only has
to stop over- or under-representing that axis by accident.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SUPPORT_BANDS = ((0, 0), (1, 4), (5, 24), (25, 10**9))
NEAR_DUPLICATE_TANIMOTO = 0.95
#: Below this a band mean is noise and must not carry a selection weight.
MINIMUM_BAND_ENTRIES = 100

#: The existing 16-shard panel, ring-scaffold entries, MEASURED at pilot step
#: 1,750 (diagnostics/editing_v2_r_theta_series_depth.json). Carried here only
#: to show what the reweighting does; nothing selects on it.
EXISTING_PANEL_ENTRIES = {"0": 5863, "1-4": 1891, "5-24": 2988, "25+": 3028}
EXISTING_PANEL_IDENTITY_NLL = {"0": 2.840, "1-4": 2.929, "5-24": 2.690, "25+": 1.935}
EXISTING_PANEL_SHARE = {
    band: round(count / sum(EXISTING_PANEL_ENTRIES.values()), 4)
    for band, count in EXISTING_PANEL_ENTRIES.items()
}


def band_label(value: int) -> str:
    for low, high in SUPPORT_BANDS:
        if low <= value <= high:
            return f"{low}" if low == high else (
                f"{low}-{high}" if high < 10**9 else f"{low}+")
    raise ValueError(f"unbandable support {value}")


class Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[a] = b


def read_library(path: Path) -> list[dict]:
    """One record per DISTINCT entry id.

    MEASURED: the consolidated train library holds 151,082 lines but 151,078
    distinct ids -- 4 ids appear twice, byte-identical in every field including
    the fiber. The validated loader keeps one of each, so counting lines here
    would make this artifact claim 15,031 reserve rows against a packed store
    holding 15,029, and the identity chain would fail downstream on a
    discrepancy that has nothing to do with the split. Exact repeats are
    collapsed; a genuine id collision is refused rather than silently resolved.
    """

    out: list[dict] = []
    seen: dict[str, str] = {}
    for line in gzip.open(path, "rt"):
        entry = json.loads(line)
        fiber = entry.get("teacher_successor_fiber") or {}
        source = str(fiber.get("source_key", ""))
        if not source:
            continue
        entry_id = str(entry["p50_entry_sha256"])
        fingerprint = json.dumps(entry, sort_keys=True)
        if entry_id in seen:
            if seen[entry_id] != fingerprint:
                raise SystemExit(
                    f"conflicting records share entry id {entry_id}; refusing to "
                    "pick one, since the split would silently depend on the choice")
            continue
        seen[entry_id] = fingerprint
        out.append({
            "entry_id": entry_id,
            "source": source,
            "family": str(entry["model_family"]),
            "cell": str(entry["capability_cell_id"]),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-library", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-entries", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()

    entries = read_library(args.train_library)
    with gzip.open(args.provenance, "rt") as handle:
        provenance = json.load(handle)
    synthetic_lane = provenance["synthetic_lane"]
    lane_by_entry = provenance["lane_by_entry_id"]
    for entry in entries:
        lane = lane_by_entry.get(entry["entry_id"])
        entry["lane"] = ("unknown" if lane is None else
                         ("synthetic" if lane == synthetic_lane else "real"))
    sources = sorted({e["source"] for e in entries})
    print(f"[{time.perf_counter()-started:6.1f}s] {len(entries):,} entries, "
          f"{len(sources):,} sources", flush=True)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint: dict[str, object] = {}
    scaffold: dict[str, str] = {}
    formula: dict[str, str] = {}
    unparseable = []
    for source in sources:
        mol = Chem.MolFromSmiles(source)
        if mol is None:
            unparseable.append(source)
            continue
        fingerprint[source] = generator.GetFingerprint(mol)
        scaffold[source] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        formula[source] = CalcMolFormula(mol)
    print(f"[{time.perf_counter()-started:6.1f}s] fingerprinted "
          f"{len(fingerprint):,}; unparseable {len(unparseable)}", flush=True)

    # Cluster very close analogues. Both blocking keys are used and unioned: a
    # near-duplicate shares a formula or a scaffold (usually both), and every
    # block is compared exhaustively rather than sampled.
    union = Union()
    for source in sources:
        union.find(source)
    compared = 0
    for keying in (formula, scaffold):
        blocks: dict[str, list[str]] = collections.defaultdict(list)
        for source, key in keying.items():
            blocks[key].append(source)
        for members in blocks.values():
            if len(members) < 2:
                continue
            for index, left in enumerate(members):
                rest = members[index + 1:]
                if not rest:
                    continue
                scores = DataStructs.BulkTanimotoSimilarity(
                    fingerprint[left], [fingerprint[r] for r in rest])
                compared += len(rest)
                for right, score in zip(rest, scores):
                    if score >= NEAR_DUPLICATE_TANIMOTO:
                        union.union(left, right)
    cluster_of = {s: union.find(s) for s in sources}
    clusters = collections.defaultdict(list)
    for source, root in cluster_of.items():
        clusters[root].append(source)
    multi = sum(1 for v in clusters.values() if len(v) > 1)
    print(f"[{time.perf_counter()-started:6.1f}s] {compared:,} pairs compared; "
          f"{len(clusters):,} clusters, {multi:,} with >1 source, "
          f"largest {max(len(v) for v in clusters.values())}", flush=True)

    entries_by_cluster: dict[str, list[dict]] = collections.defaultdict(list)
    for entry in entries:
        if entry["source"] in cluster_of:
            entries_by_cluster[cluster_of[entry["source"]]].append(entry)

    # Stratified selection of WHOLE clusters, proportional within each
    # (lane, family, cell) stratum so the reserve inherits the training mix.
    strata: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)
    for root, group in entries_by_cluster.items():
        first = group[0]
        strata[(first["lane"], first["family"], first["cell"])].append(root)
    fraction = args.target_entries / max(len(entries), 1)
    rng = random.Random(args.seed)
    reserved_clusters: set[str] = set()
    for key in sorted(strata):
        roots = sorted(strata[key])
        rng.shuffle(roots)
        want = fraction * sum(len(entries_by_cluster[r]) for r in roots)
        taken = 0
        for root in roots:
            if taken >= want:
                break
            # Never empty a stratum on the training side.
            if len(reserved_clusters & set(roots)) + 1 >= len(roots):
                break
            reserved_clusters.add(root)
            taken += len(entries_by_cluster[root])

    reserve = [e for r in reserved_clusters for e in entries_by_cluster[r]]
    remaining = [e for r, g in entries_by_cluster.items()
                 if r not in reserved_clusters for e in g]
    print(f"[{time.perf_counter()-started:6.1f}s] reserve {len(reserve):,} "
          f"entries / {len({e['source'] for e in reserve}):,} sources; "
          f"train keeps {len(remaining):,}", flush=True)

    # POST-split support: scaffolds counted over what training actually retains.
    retained_support = collections.Counter()
    for source in {e["source"] for e in remaining}:
        if source in scaffold:
            retained_support[scaffold[source]] += 1
    for entry in reserve:
        entry["train_support"] = int(
            retained_support.get(scaffold.get(entry["source"], None), 0))
        entry["band"] = band_label(entry["train_support"])

    # The training population's own support profile supplies matched weights.
    train_band = collections.Counter()
    for entry in remaining:
        support = retained_support.get(scaffold.get(entry["source"], None), 0)
        # A training entry does not support itself.
        train_band[band_label(max(support - 1, 0))] += 1
    total_train = sum(train_band.values())
    reserve_band = collections.Counter(e["band"] for e in reserve)

    print(f"\n{'band':>8} {'reserve n':>10} {'reserve %':>10} "
          f"{'train %':>9} {'weight':>8}")
    weights, unusable = {}, []
    for low, high in SUPPORT_BANDS:
        label = band_label(low)
        count = reserve_band.get(label, 0)
        share = train_band.get(label, 0) / max(total_train, 1)
        if count < MINIMUM_BAND_ENTRIES:
            unusable.append(label)
            print(f"{label:>8} {count:10,} {count/max(len(reserve),1):9.1%} "
                  f"{share:8.1%}   (below MINIMUM_BAND_ENTRIES, no weight)")
            continue
        weights[label] = share
        print(f"{label:>8} {count:10,} {count/max(len(reserve),1):9.1%} "
              f"{share:8.1%} {share:8.3f}")
    scale = sum(weights.values())
    weights = {k: v / scale for k, v in weights.items()} if scale else {}

    # What the reweighting does to a number we already have, so the fresh run's
    # matched value is not misread as a regression against the old raw mean.
    existing_total = sum(EXISTING_PANEL_ENTRIES.values())
    existing_raw = sum(EXISTING_PANEL_ENTRIES[b] * EXISTING_PANEL_IDENTITY_NLL[b]
                       for b in EXISTING_PANEL_ENTRIES) / existing_total
    existing_matched = sum(
        (train_band.get(b, 0) / max(total_train, 1)) * EXISTING_PANEL_IDENTITY_NLL[b]
        for b in EXISTING_PANEL_ENTRIES)
    print(f"\nexisting 16-shard panel: raw {existing_raw:.3f} -> under matched "
          f"weights {existing_matched:.3f} ({existing_matched - existing_raw:+.3f})")

    # Gates. Each is a property the reserve must have to carry its claim.
    reserve_sources = {e["source"] for e in reserve}
    train_sources = {e["source"] for e in remaining}
    reserve_roots = {cluster_of[s] for s in reserve_sources}
    train_roots = {cluster_of[s] for s in train_sources}
    gates = {
        "sources_disjoint": not (reserve_sources & train_sources),
        "near_duplicate_clusters_disjoint": not (reserve_roots & train_roots),
        "every_family_present": (
            {e["family"] for e in reserve} == {e["family"] for e in entries}),
        "every_lane_present": (
            {e["lane"] for e in reserve} == {e["lane"] for e in entries}),
        "unsupported_scaffold_stratum_usable":
            reserve_band.get("0", 0) >= MINIMUM_BAND_ENTRIES,
        "supported_scaffold_stratum_usable":
            reserve_band.get("25+", 0) >= MINIMUM_BAND_ENTRIES,
        "training_retains_all_families": (
            {e["family"] for e in remaining} == {e["family"] for e in entries}),
        "no_unparseable_sources": not unparseable,
    }
    print()
    for name, passed in sorted(gates.items()):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    reserve_ids = sorted(e["entry_id"] for e in reserve)
    train_ids = sorted(e["entry_id"] for e in remaining)
    digest = hashlib.sha256(
        ("\n".join(reserve_ids) + "|" + "\n".join(train_ids)).encode()
    ).hexdigest()[:12]
    artifact = {
        "schema": "compose.editing_v2.matched_validation_reserve",
        "status": "PROPOSED_NOT_FROZEN" if not all(gates.values()) else "FROZEN",
        "seed": args.seed,
        "split_digest": digest,
        "near_duplicate_tanimoto": NEAR_DUPLICATE_TANIMOTO,
        "pairs_compared": compared,
        "minimum_band_entries": MINIMUM_BAND_ENTRIES,
        "source_library_entries": len(entries),
        "reserve_entries": len(reserve),
        "reserve_sources": len(reserve_sources),
        "training_entries_retained": len(remaining),
        "reserve_band_counts": dict(reserve_band),
        "training_band_share": {k: v / max(total_train, 1)
                                for k, v in train_band.items()},
        "matched_selection_weights": weights,
        "bands_without_weight": unusable,
        "views": {
            "matched": "all bands, reweighted by matched_selection_weights; "
                       "the checkpoint-selection number",
            "unsupported_scaffold": "band 0 only -- no POST-SPLIT training source shares this Murcko scaffold. A property of this split under this scaffold definition, NOT a claim that the scaffold is chemically novel",
            "supported_scaffold": "band 25+ only; new molecules in familiar series",
            "external_cohort": "the existing 16-shard validation panel, retained "
                               "as a cross-cohort test and NOT a selection input",
        },
        "gates": gates,
        "scaffold_definition": (
            "RDKit MurckoScaffoldSmiles on the source molecule. Acyclic molecules "
            "yield the empty scaffold and are treated as one explicit class, not "
            "as missing -- an earlier pass let the falsy empty string collapse "
            "them into 'no scaffold', which put 361 acyclic entries at the bottom "
            "of a series-depth curve they did not belong on."),
        "band_composition_vs_existing_panel": {
            "existing_16_shard_panel_share": EXISTING_PANEL_SHARE,
            "existing_16_shard_panel_identity_nll": EXISTING_PANEL_IDENTITY_NLL,
            "training_population_share": {
                k: round(train_band.get(k, 0) / max(total_train, 1), 4)
                for k in EXISTING_PANEL_SHARE},
            "reserve_share": {
                k: round(reserve_band.get(k, 0) / max(len(reserve), 1), 4)
                for k in EXISTING_PANEL_SHARE},
            "existing_panel_raw_mean_identity_nll": round(existing_raw, 4),
            "existing_panel_under_matched_weights": round(existing_matched, 4),
            "finding": (
                "The existing panel is BIMODAL against the training population: "
                "it over-represents band 0 (42.6% vs 19.3%) and band 25+ (22.0% "
                "vs 10.4%) while under-representing band 1-4 (13.7% vs 43.4%), "
                "which is the hardest band at 2.93."),
            "matching_does_not_flatter_the_number": (
                f"Applying training weights to the existing panel's own per-band "
                f"means moves it {existing_raw:.2f} -> {existing_matched:.2f}, i.e. "
                f"{existing_matched - existing_raw:+.2f}. Matching buys an "
                "interpretable number, not a better one. An earlier note claimed "
                "2.62 -> 1.94; that conflated the matched view with the "
                "supported-scaffold view, since 1.94 is band 25+ alone."),
            "caveat": (
                "These are the OLD externally mined cohort's per-band means under "
                "new weights. The reserve is drawn from the training population, "
                "so its absolute values should be better; the reweighting "
                "DIRECTION is the claim here, not the magnitude."),
        },
        "not_a_causal_claim": (
            "Support is not asserted to cause the gap. Bands 0 and 1-4 are "
            "indistinguishable and the benefit concentrates at 25+, consistent "
            "with a latent typical-chemistry axis. The split only has to stop "
            "representing that axis by accident."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    ids_path = args.out.with_name(args.out.stem + "_ids.json.gz")
    with gzip.open(ids_path, "wt") as handle:
        json.dump({"reserve_entry_ids": reserve_ids,
                   "training_entry_ids": train_ids,
                   "reserve_band_by_entry_id": {e["entry_id"]: e["band"]
                                                for e in reserve},
                   # The sampling law excludes by SOURCE canonical key, not by
                   # entry id, and it reads Active8 rather than this library --
                   # so the keys are emitted here rather than re-derived
                   # downstream, where a second scaffolding pass could disagree.
                   "reserve_source_keys": sorted(reserve_sources),
                   "training_source_keys": sorted(train_sources)}, handle)
    print(f"\nwrote {args.out} and {ids_path.name}  digest {digest}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
