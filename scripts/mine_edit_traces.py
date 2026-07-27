#!/usr/bin/env python3
"""Shardable, global-grouping edit-trace miner for the B-edit training pool.

The B-edit training mixture is three layers:
  1. CORRUPTION  -- a real molecule corrupted a few legal mark-family edits (both directions). This layer
     is per-molecule and is regenerated IN MEMORY at train time from the corpus + seed
     (``train_tracelet_cnof_gate.py`` :2073), so mining only CHARACTERIZES it (measures the realized
     family histogram / ring-change fraction / source-target Tanimoto) and locks the recipe params.
  2. MMP one-cut -- pairs sharing a constant core differing by one acyclic single-bond cut. Mined offline,
     compiled + verified, serialized to a re-consumable pool JSONL (``--analogue-trace-pool``).
  3. SCAFFOLD-NN -- longer-range pairs sharing a Bemis-Murcko scaffold, k-NN by MCS distance. Mined as
     CANDIDATES; the one-cut-compilable subset joins the pool, the rest are recorded as A2.3-deferred.

Sharding is a genuine map -> global-group -> compile -> global-reduce, NOT independent per-shard mining:

  MAP (shard-local, per-molecule, embarrassingly parallel):
     standardize/validate; emit MMP (core_key, smiles); emit (murcko_scaffold, smiles). No cross-molecule
     state -- a Modal container maps one shard-slice of the TRAIN partition and writes an emission file.
  GLOBAL GROUP (reduce, cross-shard):
     a core_key or a scaffold spans shards, so grouping MUST see every shard's emissions at once. Merge
     all emission files -> global core_index -> within-core MMP pairs; global scaffold_index -> k-NN
     candidate pairs. The per-transformation-signature cap and reverse-pair dedup are GLOBAL counts.
  COMPILE (per-pair, shardable):
     one-cut compile + executor replay -> directed pool records (forward + reverse).
  GLOBAL REDUCE (dedup + cap):
     dedup directed (source_smiles, target_key) across layers; per-source / per-core / per-family caps.

Determinism: the corpus -> TRAIN partition split is the SAME shared ``load_organic_corpus_split`` (broad
scope) the trainer uses (so mining never leaks val/test); the shard rule is a fixed stride over train;
corruption uses a seeded rng. Nothing is hardcoded to 500k -- corpus id/path, split sizes, shard count,
caps, and the output location are all config fields.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse the committed mining primitives

from build_analogue_trace_pool import (  # noqa: E402
    _compile_verify,
    _pool_record,
    iter_one_cut_transformations,
)
from build_edit_data_manifest import _hash_sources  # noqa: E402
from build_murcko_analogue_graph import _fingerprint, _nearest_neighbours  # noqa: E402

from compose_v4.chem.molecular_graph import (  # noqa: E402
    CNOF_VOCABULARY,
    ORGANIC_VOCABULARY,
    MolecularGraphError,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.data.organic_corpus import (  # noqa: E402
    BROAD_ORGANIC_NEUTRAL_V1,
    BROAD_ORGANIC_V1,
    load_organic_corpus_split,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.source_corruption import make_edit_pair  # noqa: E402
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target  # noqa: E402
from compose_v4.rewrite.typed_ring_catalog import (  # noqa: E402
    build_typed_ring_catalog,
    ring_catalog_fingerprint,
)

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent


# ---- config (corpus-agnostic: every corpus/size/cap/output is a field) --------------------------------
@dataclass
class MiningConfig:
    # corpus + the EXACT train/val/test split the trainer will use (mine TRAIN only -> no leakage)
    corpus_id: str = "guacamol_subset_500000_seed0"
    corpus_path: str = "data/guacamol_subset_500000_seed0.smiles"
    train_size: int = 480_000
    validation_size: int = 10_000
    test_size: int = 10_000
    max_atoms: int = 48
    split_seed: int = 20260714
    scope_name: str = "broad_organic_v1"  # LOCKED B-edit scope; neutral/CNOF are ablations only
    scan_all: bool = True
    split_workers: int = 0  # >1 parallelizes the one-time full-corpus canonicalization scan
    # sharding: a fixed stride over the canonical-sorted train list. shard_index is None for the full run.
    n_shards: int = 1
    shard_index: int | None = None
    shard_rule: str = "stride"  # "stride" (representative) | "contiguous"
    # MMP one-cut layer
    max_variable_atoms: int = 8
    max_pairs_per_core: int = 12
    # scaffold-NN layer
    scaffold_k: int = 8
    scaffold_max_per_scaffold: int = 24
    scaffold_max_per_transformation: int = 40
    scaffold_mcs_candidates: int = 16
    scaffold_mcs_timeout: int = 2
    scaffold_max_group_size: int = 200
    # global caps
    max_pairs_per_source: int = 8
    # corruption characterization (recipe is regenerated at train time; here we MEASURE it)
    corruption_depth_max: int = 5
    corruption_couplings_per_target: int = 1
    corruption_sample_size: int = 2_000
    corruption_seed: int = 7
    organic_vocabulary: bool = True
    # output
    out_dir: str = "diagnostics/composition/mining"
    corpus_sha256: str | None = None  # pinned at mining time (computed where the corpus lives)


# ---- deterministic corpus -> train partition -> shard -------------------------------------------------
_SCOPES = {"broad_organic_v1": BROAD_ORGANIC_V1, "broad_organic_neutral_v1": BROAD_ORGANIC_NEUTRAL_V1}


def resolve_scope(config: MiningConfig):
    if config.scope_name not in _SCOPES:
        raise ValueError(f"unknown corpus scope {config.scope_name!r}; known {sorted(_SCOPES)}")
    scope = _SCOPES[config.scope_name]
    if config.max_atoms < scope.max_atoms:
        raise ValueError(
            f"mining slots max_atoms={config.max_atoms} < scope max_atoms={scope.max_atoms}; "
            "a scope-eligible molecule would not fit the mining representation")
    return scope


def train_partition(config: MiningConfig):
    """The broad-organic TRAIN split, via the SHARED loader the trainer uses -- guarantees no val/test
    leakage and the same scope everywhere. Returns the full split (train + census + scope descriptor)."""
    return load_organic_corpus_split(
        REPO / config.corpus_path,
        scope=resolve_scope(config),
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=config.split_seed,
        scan_all=config.scan_all,
        workers=config.split_workers,
    )


def shard_of(smiles: tuple[str, ...], config: MiningConfig) -> list[str]:
    """Deterministic shard of the canonical-ordered train list. stride = representative sample."""
    ordered = sorted(smiles)
    if config.shard_index is None or config.n_shards <= 1:
        return ordered
    if config.shard_rule == "contiguous":
        per = (len(ordered) + config.n_shards - 1) // config.n_shards
        return ordered[config.shard_index * per:(config.shard_index + 1) * per]
    return [s for i, s in enumerate(ordered) if i % config.n_shards == config.shard_index]


# ---- Phase 1: MAP (shard-local, per-molecule) ---------------------------------------------------------
@dataclass
class ShardEmission:
    """Everything a shard emits for the GLOBAL group step. JSON-serializable so Modal map containers can
    write it to the volume and a single reduce container merges every shard's emissions."""
    corpus_id: str
    shard_index: int | None
    n_shards: int
    supported: int = 0
    skipped: int = 0
    mmp: list[list[str]] = field(default_factory=list)       # [core_smiles, member_smiles]
    scaffold: list[list[str]] = field(default_factory=list)  # [murcko_scaffold, member_smiles]

    def merge(self, other: "ShardEmission") -> None:
        self.supported += other.supported
        self.skipped += other.skipped
        self.mmp.extend(other.mmp)
        self.scaffold.extend(other.scaffold)


def map_shard(smiles: list[str], config: MiningConfig) -> ShardEmission:
    """Per-molecule map: validate, then emit MMP core-keys + the Murcko-scaffold membership. Shard-local
    (no cross-molecule state) so it fans out over Modal containers with no coordination."""
    emission = ShardEmission(config.corpus_id, config.shard_index, config.n_shards)
    for smi in smiles:
        try:
            pad_molecular_graph(smiles_to_molecular_graph(smi), config.max_atoms)
        except (MolecularGraphError, ValueError):
            emission.skipped += 1
            continue
        emission.supported += 1
        for core_smi, _variable in iter_one_cut_transformations(
                smi, max_variable_atoms=config.max_variable_atoms):
            emission.mmp.append([core_smi, smi])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:  # noqa: BLE001
            scaffold = ""
        if scaffold:
            emission.scaffold.append([scaffold, smi])
    return emission


# ---- Phase 2: GLOBAL GROUP (reduce; a core/scaffold key spans shards) ---------------------------------
def group_mmp_pairs(emission: ShardEmission, config: MiningConfig) -> list[tuple[str, str, str]]:
    """Global core_index -> within-core unordered pairs (deduped). Splitting mine_one_cut_pairs so the
    per-molecule fragmentation is the (shardable) map and only the grouping is the (global) reduce."""
    core_index: dict[str, list[str]] = defaultdict(list)
    seen_member: set[tuple[str, str]] = set()
    for core_smi, smi in emission.mmp:
        if (core_smi, smi) in seen_member:
            continue
        seen_member.add((core_smi, smi))
        core_index[core_smi].append(smi)
    pairs, seen = [], set()
    for core_smi, members in core_index.items():
        per_core = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sa, sb = members[i], members[j]
                if sa == sb:
                    continue
                key = frozenset((sa, sb))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((sa, sb, core_smi))
                per_core += 1
                if per_core >= config.max_pairs_per_core:
                    break
            if per_core >= config.max_pairs_per_core:
                break
    return pairs


def group_scaffold_pairs(emission: ShardEmission, config: MiningConfig) -> list[dict]:
    """Global scaffold_index -> k-NN candidate pairs (MCS distance, Morgan pre-ranked), with the
    per-transformation-signature cap enforced as a GLOBAL count. Candidates only -- compilation decides
    which are one-cut-realizable vs A2.3-deferred."""
    groups: dict[str, list[str]] = defaultdict(list)
    for scaffold, smi in emission.scaffold:
        groups[scaffold].append(smi)
    multi = {s: sorted(set(m)) for s, m in groups.items() if len(set(m)) >= 2}
    fps: dict[str, object] = {}
    for members in multi.values():
        for smi in members:
            if smi not in fps:
                fps[smi] = _fingerprint(Chem.MolFromSmiles(smi))
    directed, transformation_counts, seen = [], Counter(), set()
    for scaffold in sorted(multi):
        members = multi[scaffold][:config.scaffold_max_group_size]
        scaffold_count = 0
        for source in members:
            if scaffold_count >= config.scaffold_max_per_scaffold:
                break
            for _distance, target, edge in _nearest_neighbours(
                    source, members, fps, k=config.scaffold_k,
                    mcs_candidates=config.scaffold_mcs_candidates,
                    timeout=config.scaffold_mcs_timeout):
                if scaffold_count >= config.scaffold_max_per_scaffold:
                    break
                signature = edge["transformation_signature"]
                if transformation_counts[signature] >= config.scaffold_max_per_transformation:
                    continue
                pair_key = frozenset((source, target))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                transformation_counts[signature] += 1
                scaffold_count += 1
                directed.append({"source": source, "target": target, "scaffold": scaffold, **edge})
    return directed


# ---- Phase 3: COMPILE (per-pair) ----------------------------------------------------------------------
def _compile_directed(sa: str, sb: str, index: int, layer: str, extra: dict,
                      config: MiningConfig) -> list[dict]:
    """One-cut compile + replay both directions -> directed pool records (re-consumable schema)."""
    records = []
    for direction, (src, tgt) in (("forward", (sa, sb)), ("reverse", (sb, sa))):
        result = _compile_verify(src, tgt, n_slots=config.max_atoms,
                                 max_variable_atoms=config.max_variable_atoms)
        if result["outcome"] != "ok":
            continue
        record = _pool_record(index, direction, src, result, config.max_atoms)
        record["layer"] = layer
        record["metadata"] = {**record.get("metadata", {}), "layer": layer, **extra}
        records.append(record)
    return records


def compile_mmp(pairs: list[tuple[str, str, str]], config: MiningConfig) -> list[dict]:
    records = []
    for index, (sa, sb, core) in enumerate(pairs):
        records.extend(_compile_directed(sa, sb, index, "mmp_one_cut", {"core_smiles": core}, config))
    return records


def compile_scaffold(candidates: list[dict], config: MiningConfig) -> tuple[list[dict], int]:
    """Attempt one-cut compilation on scaffold-NN candidates. The compilable subset joins the pool; the
    remainder are longer-range (multi-cut / ring-open) A2.3-deferred candidates (general compiler TODO)."""
    records, deferred = [], 0
    for index, cand in enumerate(candidates):
        got = _compile_directed(cand["source"], cand["target"], index, "scaffold_nn",
                                {"scaffold": cand["scaffold"],
                                 "mcs_distance": cand.get("mcs_distance")}, config)
        if got:
            records.extend(got)
        else:
            deferred += 1
    return records, deferred


# ---- Phase 4: GLOBAL REDUCE (dedup + cap) -------------------------------------------------------------
def dedup_and_cap(records: list[dict], config: MiningConfig) -> tuple[list[dict], dict]:
    """Dedup directed (source_smiles, target_key) across layers; enforce the per-source cap; balance
    reverse pairs (they are separate directed records). Returns (final_pool, cap_report)."""
    seen: set[tuple[str, str]] = set()
    per_source: Counter = Counter()
    final, dropped_dup, dropped_cap = [], 0, 0
    family_counts: Counter = Counter()
    direction_counts: Counter = Counter()
    layer_counts: Counter = Counter()
    for record in records:
        key = (record["source_smiles"], record["target_key"])
        if key in seen:
            dropped_dup += 1
            continue
        if per_source[record["source_smiles"]] >= config.max_pairs_per_source:
            dropped_cap += 1
            continue
        seen.add(key)
        per_source[record["source_smiles"]] += 1
        final.append(record)
        for fam, n in record["operator_histogram"].items():
            family_counts[fam] += n
        direction_counts[record["direction"]] += 1
        layer_counts[record["layer"]] += 1
    report = {
        "records_in": len(records), "records_out": len(final),
        "dropped_duplicate": dropped_dup, "dropped_per_source_cap": dropped_cap,
        "family_histogram": dict(family_counts),
        "direction_balance": dict(direction_counts),
        "layer_counts": dict(layer_counts),
    }
    return final, report


# ---- corruption layer: characterize a sample (records are regenerated at train time) ------------------
def _build_catalog(config: MiningConfig):
    """A typed ring catalog from a few representative rings -- the same construction the trainer uses; it
    enables the clean ring-opening (``ring_system_delete``) corruption family."""
    system = de_novo_rewrite_system()

    def trace(smi: str):
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), config.max_atoms)
        src = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=config.max_atoms)
        return compile_carbon_tree_to_target(src, target, use_bond_reroute=True, align_source=True)

    seeds = ("c1ccccc1", "c1ccncc1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1")
    traces = []
    for smi in seeds:
        try:
            traces.append(trace(smi))
        except Exception:  # noqa: BLE001
            continue
    return build_typed_ring_catalog(tuple(traces)) if traces else None, system


def characterize_corruption(smiles: tuple[str, ...], config: MiningConfig) -> dict:
    """Measure the realized corruption mix on a sample: family histogram, both-direction yield,
    ring-change fraction, source<->target Tanimoto. The recipe (not the records) is locked here."""
    sample = list(smiles)[: config.corruption_sample_size]
    catalog, system = _build_catalog(config)
    vocab = ORGANIC_VOCABULARY if config.organic_vocabulary else CNOF_VOCABULARY
    rng = np.random.default_rng(config.corruption_seed)
    families: Counter = Counter()
    depths, tanimotos = [], []
    trim_yield = grow_yield = attempted = ring_change = 0
    for smi in sample:
        try:
            target = pad_molecular_graph(smiles_to_molecular_graph(smi), config.max_atoms)
        except (MolecularGraphError, ValueError):
            continue
        attempted += 1
        depth = int(rng.integers(1, config.corruption_depth_max + 1))
        trim, grow = make_edit_pair(target, depth, system=system, rng=rng, catalog=catalog,
                                    vocabulary=vocab)
        if trim is not None:
            trim_yield += 1
            for step in trim.steps:
                families[step.rule_name] += 1
            depths.append(len(trim.steps))
            if any(s.rule_name.startswith("ring_system") for s in trim.steps):
                ring_change += 1
            # drift = Tanimoto(real source, CORRUPTED endpoint) -- how far the corruption walked. The
            # trained model must recognize the corrupted source as a variant of the lead (target ~0.4-0.9).
            real_mol = Chem.MolFromSmiles(smi)
            corrupted_smi = molecular_graph_to_smiles(trim.target)
            corrupted_mol = Chem.MolFromSmiles(corrupted_smi) if corrupted_smi else None
            if real_mol is not None and corrupted_mol is not None:
                tanimotos.append(DataStructs.TanimotoSimilarity(
                    _fingerprint(real_mol), _fingerprint(corrupted_mol)))
        if grow is not None:
            grow_yield += 1
    total = sum(families.values()) or 1
    return {
        "sample_attempted": attempted,
        "trim_yield": trim_yield, "grow_yield": grow_yield,
        "family_histogram": dict(families),
        "family_fraction": {k: round(v / total, 4) for k, v in families.items()},
        "ring_change_fraction": round(ring_change / max(1, trim_yield), 4),
        "depth_mean": round(float(np.mean(depths)), 3) if depths else None,
        "source_target_tanimoto": {
            "mean": round(float(np.mean(tanimotos)), 4) if tanimotos else None,
            "p10": round(float(np.percentile(tanimotos, 10)), 4) if tanimotos else None,
            "p90": round(float(np.percentile(tanimotos, 90)), 4) if tanimotos else None,
        },
        "ring_catalog_fingerprint": ring_catalog_fingerprint(catalog) if catalog else None,
        "depth_max": config.corruption_depth_max,
        "couplings_per_target": config.corruption_couplings_per_target,
        "vocabulary": "organic" if config.organic_vocabulary else "cnof",
    }


# ---- orchestration ------------------------------------------------------------------------------------
def run_pipeline(config: MiningConfig) -> dict:
    """In-process map -> group -> compile -> global-reduce (for the local fixture AND a single Modal
    shard). For a true multi-container run, map_shard writes an emission per container and this reduce
    reads them all -- the group/compile/reduce below are identical. Phase timing is recorded so a single
    validation shard projects the full-run cost (one-time scan + per-shard map/compile x n_shards)."""
    t0 = time.time()
    split = train_partition(config)  # ONE-TIME broad-organic scan + deterministic split (+ census)
    train = split.train
    t_scan = time.time() - t0
    shard = shard_of(train, config)
    print(f"[mine] scan+split done: eligible-train={len(train)} shard={len(shard)} ({t_scan:.1f}s)",
          flush=True)

    t0 = time.time()
    emission = map_shard(shard, config)
    mmp_pairs = group_mmp_pairs(emission, config)
    # The scaffold-NN layer's rdFMCS k-NN is the slow phase and is ~mostly A2.3-deferred; skip it when
    # scaffold_k<=0 (validation shards) -- the MMP + corruption layers carry the pool.
    scaffold_cands = group_scaffold_pairs(emission, config) if config.scaffold_k > 0 else []
    print(f"[mine] map+group done: mmp_pairs={len(mmp_pairs)} scaffold_cands={len(scaffold_cands)}",
          flush=True)
    mmp_records = compile_mmp(mmp_pairs, config)
    scaffold_records, scaffold_deferred = compile_scaffold(scaffold_cands, config)
    pool, cap_report = dedup_and_cap(mmp_records + scaffold_records, config)
    t_shard = time.time() - t0
    print(f"[mine] compile+reduce done: pool={len(pool)} "
          f"(mmp={len(mmp_records)} scaffold={len(scaffold_records)}) ({t_shard:.1f}s)", flush=True)

    t0 = time.time()
    corruption = characterize_corruption(tuple(shard), config)
    t_corruption = time.time() - t0
    print(f"[mine] corruption characterization done ({t_corruption:.1f}s)", flush=True)

    return {
        "config": asdict(config),
        "provenance": _provenance(config),
        "corpus": {
            "scope": split.scope,
            "train_partition_size": len(train), "shard_size": len(shard),
            "supported": emission.supported, "skipped": emission.skipped,
            "census": split.census,
        },
        "layers": {
            "corruption": corruption,
            "mmp_one_cut": {"pairs_grouped": len(mmp_pairs), "records_compiled": len(mmp_records)},
            "scaffold_nn": {"candidates_grouped": len(scaffold_cands),
                            "records_compiled": len(scaffold_records),
                            "a2_3_deferred": scaffold_deferred},
        },
        "timing_seconds": {
            "corpus_scan_split_one_time": round(t_scan, 2),
            "shard_map_group_compile": round(t_shard, 2),
            "corruption_characterization": round(t_corruption, 2),
        },
        "global_reduce": cap_report,
        "pool": pool,
    }


def mine_mmp_pairs(config: MiningConfig) -> dict:
    """Phase A of the multi-container fan-out: scan+split, map the FULL train partition, GLOBAL-group the
    MMP one-cut pairs, and characterize corruption -- but do NOT compile. Returns the pairs (JSON-friendly
    ``[sa, sb, core]``) for parallel compilation + census + corruption + provenance. Scaffold-NN is skipped
    (scaffold_k<=0). The compile of the (very many) pairs is fanned out over containers via ``compile_mmp``,
    then merged with ``dedup_and_cap`` -- see modal_apps/mine_edit_traces_app.py."""
    t0 = time.time()
    split = train_partition(config)
    train = split.train
    t_scan = time.time() - t0
    shard = shard_of(train, config)
    print(f"[mine] scan+split done: eligible-train={len(train)} shard={len(shard)} ({t_scan:.1f}s)",
          flush=True)

    t0 = time.time()
    emission = map_shard(shard, config)
    mmp_pairs = group_mmp_pairs(emission, config)
    t_map = time.time() - t0
    print(f"[mine] map+group done: mmp_pairs={len(mmp_pairs)} ({t_map:.1f}s)", flush=True)

    t0 = time.time()
    corruption = characterize_corruption(tuple(shard), config)
    t_corruption = time.time() - t0
    print(f"[mine] corruption characterization done ({t_corruption:.1f}s)", flush=True)

    return {
        "config": asdict(config),
        "provenance": _provenance(config),
        "corpus": {"scope": split.scope, "train_partition_size": len(train), "shard_size": len(shard),
                   "supported": emission.supported, "skipped": emission.skipped, "census": split.census},
        "corruption": corruption,
        "pairs": [[sa, sb, core] for (sa, sb, core) in mmp_pairs],
        "timing_seconds": {"corpus_scan_split_one_time": round(t_scan, 2),
                           "map_group": round(t_map, 2),
                           "corruption_characterization": round(t_corruption, 2)},
    }


def _provenance(config: MiningConfig) -> dict:
    import subprocess
    try:  # the Modal container has no git binary; the caller (mine_shard) overrides with the launch commit
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, OSError):
        commit = ""
    return {
        "commit": commit,
        "corpus_id": config.corpus_id,
        "corpus_sha256": config.corpus_sha256,
        "corpus_scope": config.scope_name,
        "corpus_scope_hash": resolve_scope(config).scope_hash(),
        "standardization_hash": _hash_sources(
            ["src/compose_v4/chem/molecular_graph.py", "src/compose_v4/chem/state.py"]),
        "operator_registry": _hash_sources(
            ["src/compose_v4/rewrite/operators.py", "src/compose_v4/rewrite/kernel.py",
             "src/compose_v4/rewrite/factorized_fiber.py"]),
        "compiler_mmp": _hash_sources(["scripts/build_analogue_trace_pool.py"]),
        "compiler_corruption": _hash_sources(["src/compose_v4/rewrite/source_corruption.py"]),
        "corpus_scope_module": _hash_sources(["src/compose_v4/data/organic_corpus.py"]),
        "miner": _hash_sources(["scripts/mine_edit_traces.py"]),
    }


def _write_outputs(summary: dict, config: MiningConfig) -> tuple[Path, Path]:
    out_dir = REPO / config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "full" if config.shard_index is None else f"shard{config.shard_index:04d}"
    pool_path = out_dir / f"edit_pool_{tag}.jsonl"
    with pool_path.open("w") as handle:
        for record in summary.pop("pool"):
            handle.write(json.dumps(record) + "\n")
    summary_path = out_dir / f"mining_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return pool_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", default=MiningConfig.corpus_id)
    parser.add_argument("--corpus-path", default=MiningConfig.corpus_path)
    parser.add_argument("--corpus-sha256", default=None)
    parser.add_argument("--train-size", type=int, default=MiningConfig.train_size)
    parser.add_argument("--validation-size", type=int, default=MiningConfig.validation_size)
    parser.add_argument("--test-size", type=int, default=MiningConfig.test_size)
    parser.add_argument("--max-atoms", type=int, default=MiningConfig.max_atoms)
    parser.add_argument("--n-shards", type=int, default=MiningConfig.n_shards)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--corruption-sample-size", type=int, default=MiningConfig.corruption_sample_size)
    parser.add_argument("--out-dir", default=MiningConfig.out_dir)
    parser.add_argument("--cnof", action="store_true", help="CNOF vocab (default: organic)")
    args = parser.parse_args()

    config = MiningConfig(
        corpus_id=args.corpus_id, corpus_path=args.corpus_path, corpus_sha256=args.corpus_sha256,
        train_size=args.train_size, validation_size=args.validation_size, test_size=args.test_size,
        max_atoms=args.max_atoms, n_shards=args.n_shards, shard_index=args.shard_index,
        corruption_sample_size=args.corruption_sample_size, out_dir=args.out_dir,
        organic_vocabulary=not args.cnof,
    )
    summary = run_pipeline(config)
    pool = summary["pool"]
    print(json.dumps({"corpus": summary["corpus"], "layers": summary["layers"],
                      "global_reduce": summary["global_reduce"]}, indent=2))
    pool_path, summary_path = _write_outputs({**summary, "pool": pool}, config)
    print(f"\npool  -> {pool_path.relative_to(REPO)} ({len(pool)} records)")
    print(f"summary -> {summary_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
