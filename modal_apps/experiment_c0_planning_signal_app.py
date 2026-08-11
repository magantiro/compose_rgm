"""Experiment C0: is there a planning problem here at all?

THE QUESTION
------------
Before training a value network h_phi, establish that future value would change
decisions that matter.  If the action with the best estimated FUTURE value is
almost always the action with the best IMMEDIATE value, then greedy already is
the planner on this task, and amortising an expensive planner into h_phi buys
nothing.  This probe answers that for ~a tenth of the cost of the full six-arm
pilot, and gates it.

WHAT IS MEASURED, AND WHY IT IS MEASURED THIS WAY
-------------------------------------------------
At each decision state along a greedy trajectory, every mask-legal successor is
scored for immediate DRD2 (exact, no estimation), and a subset is scored for
future value by Monte-Carlo rollout under the frozen reference law.

Global rank correlation is deliberately NOT the deciding statistic.  With ~600
successors, immediate and future value can agree across hundreds of obviously
bad edits while disagreeing on the handful that matter.  The decisive statistics
are about the top action:

    top-1 disagreement    does MC value pick a different edit than greedy?
    greedy future regret  h(MC-best) - h(greedy-best)
    sacrifice-to-win      is the MC-best edit WORSE immediately and BETTER
                          after the remaining horizon?

The last one names the phenomenon directly: the best route sometimes requires
an edit that does not look best right now.

WINNER'S CURSE
--------------
h is estimated from few rollouts, so argmax over K noisy estimates is biased
upward -- it would inflate both disagreement and regret, i.e. bias the result
toward "planning helps", which is the dangerous direction.  So selection and
evaluation use INDEPENDENT rollout samples: MC-best is chosen on one sample and
the reported regret is computed on a fresh one.  Immediate value needs no such
care; it is exact for every legal successor.

WHAT THIS PROBE IS NOT
----------------------
Planning is scored in MARGIN units -- logit P(active), the SVM margin.  The
margin is strictly monotone with the benchmark probability, so a greedy
controller makes identical choices under either; what it adds is resolution,
since P(active) spans 0.00003 to 0.048 across the benchmark's own sources and a
few edits through that region move it by less than rollout noise.

Monotonicity does NOT survive taking expectations: E[max margin] and
E[max P(active)] are different objectives and can rank candidates differently.
So this is a MECHANISM-DETECTION probe -- "is there evidence that future
reachability can change decisions?" -- and nothing here is a benchmark result.
Demonstrating the real thing on benchmark outcomes (P(active), success at 0.5,
oracle cost) is Experiment C's job.

DESIGN POINTS THAT DECIDE WHETHER THE ANSWER MEANS ANYTHING
-----------------------------------------------------------
Sources are drawn at RANDOM from the eligible pool at a fixed seed.  Taking the
head of a sorted SMILES list clusters by leading element and returns a
chemically unrepresentative sample.

The candidate pool is three strata -- top margin, top reference-law probability,
and random reference-law draws.  Comparing greedy against a handful of purely
random draws out of ~500 legal successors made MC-best = greedy-best almost by
construction, since greedy's margin is the maximum over ALL successors and a
random draw's short rollout essentially never overtakes it.  The strata give
both obvious high-reward edits and plausible "sacrifice now, win later" edits a
chance to be examined.  Only ~8 of ~500 successors are evaluated, so measured
discordance still UNDERSTATES the real signal.

The kernel is NOT given a visited-set.  Masking revisited states would redefine
R_theta, and the reference law has to stay the law being studied; revisits are
recorded and dropped from the statistics instead.

CPU ONLY.  Executor and oracle work; no GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_file(
        ROOT / "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz",
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz"),
        copy=True)
    .add_local_file(
        ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json",
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"),
        copy=True)
)
app = modal.App("compose-v4-experiment-c0-planning-signal")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
ORACLE_DIR = REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"

# ---- preregistered task definition ---------------------------------------
#: Source molecules must start below this DRD2 score.
SOURCE_CEILING = 0.05
#: A molecule counts as solved at or above this DRD2 score.
TARGET_FLOOR = 0.5
#: Hard similarity constraint to the SOURCE, applied to every arm alike.
SIMILARITY_FLOOR = 0.4
#: Productive edits available.
BUDGET = 6
#: Steps of rollout beyond a candidate when estimating its future value.  A
#: six-edit task probed with two-step planning is too easy to falsely call
#: "no planning signal", so the horizon is half the budget.
LOOKAHEAD_DEPTH = 3
#: Heavy-atom band for sources, matching the benchmark's own DRD2 source set
#: (median 24, range 13-41).  Outcome-independent: it looks at molecule size,
#: never at reachability or reward.  Below this band a Tanimoto >= 0.4 mask
#: leaves only a handful of legal edits.
MIN_HEAVY_ATOMS = 13
MAX_HEAVY_ATOMS = 41

# ---- candidate pool ------------------------------------------------------
# Three strata, unioned and deduplicated.  Greedy's own pick is always present
# (it heads the top-margin stratum), so the regret comparison is never against
# an action greedy would not have taken.
#: Best immediate margin -- where a "sacrifice among plausible actions" lives.
CANDIDATES_TOP_MARGIN = 4
#: Highest reference-law probability -- what R_theta itself considers likely.
CANDIDATES_TOP_REFERENCE = 2
#: Drawn from the reference law -- the only stratum that can reach the tail.
CANDIDATES_RANDOM = 2

#: Rollouts per candidate for SELECTING the MC-best action.
ROLLOUTS_SELECT = 2
#: Fresh, independent rollouts for SCORING the selected action (winner's curse).
ROLLOUTS_EVAL = 3
#: Kernel time point, matching the training and partition conventions.
TIME_POINT = 0.5


@app.function(
    image=image, cpu=8.0, memory=64 * 1024, timeout=60 * 60,
    max_containers=1, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def select_sources(wanted: int = 12, scan_limit: int = 12000,
                   selection_seed: int = 20260811) -> dict[str, Any]:
    """Choose probe sources on OUTCOME-INDEPENDENT criteria only.

    Representable by COMPOSE, genuinely held out, below the DRD2 source ceiling,
    not already solved, and within the benchmark's own heavy-atom band.  Then a
    random fixed-seed draw.

    Deliberately NO rollout-based reachability screen: removing sources that
    look hard under exploratory rollouts would discard exactly the instances
    where planning is supposed to pay, and "unsolvable in six edits" is a
    scientifically meaningful outcome rather than grounds for exclusion.
    """

    import gzip
    import sys

    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.corpus_training_library import load_corpus_training_library
    from compose_v4.drd2_oracle import DRD2Oracle

    started = time.perf_counter()
    artifact_volume.reload()
    inputs = Path(RUN_ROOT) / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())
    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())

    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False)
    with gzip.open(
            inputs / "editing_v2_matched_validation_reserve_ids.json.gz", "rt") as handle:
        reserve = json.load(handle)
    reserve_ids = set(reserve["reserve_entry_ids"])
    print(f"[{time.perf_counter()-started:6.1f}s] library loaded", flush=True)

    # One representative entry per reserve SOURCE.
    representative: dict[str, str] = {}
    for entry in library.entries:
        if entry.entry_id not in reserve_ids:
            continue
        key = entry.teacher_fiber.state_support.source_key
        representative.setdefault(key, entry.entry_id)
    candidates = sorted(representative)[:scan_limit]
    print(f"  {len(representative):,} reserve sources; scanning {len(candidates):,}",
          flush=True)

    oracle = DRD2Oracle.from_manifest(ORACLE_DIR / "drd2_oracle_manifest.json")
    scores = oracle.score_many(candidates)

    # The probe containers need exactly one thing from the library: the slot
    # count its own padding assigned to this source.  Resolving it here means
    # they never load the corpus at all, which is most of their memory and most
    # of their startup.
    # Heavy atoms come from RDKit, not from counting slots: the state tensor is
    # padded and its NULL slots are not distinguishable by sign, so
    # `(atom_types >= 0).sum()` returns the slot count for every molecule alike.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def heavy_atoms(smiles: str) -> int:
        mol = Chem.MolFromSmiles(smiles)
        return -1 if mol is None else int(mol.GetNumHeavyAtoms())

    sizes = np.array([heavy_atoms(k) for k in candidates])
    below_ceiling = scores < SOURCE_CEILING
    in_band = (sizes >= MIN_HEAVY_ATOMS) & (sizes <= MAX_HEAVY_ATOMS)
    already_solved = int((scores >= TARGET_FLOOR).sum())
    keep = np.flatnonzero(below_ceiling & in_band)
    print(f"  DRD2 < {SOURCE_CEILING}: {int(below_ceiling.sum()):,}; "
          f"already solved: {already_solved}; "
          f"in {MIN_HEAVY_ATOMS}-{MAX_HEAVY_ATOMS} heavy atoms: "
          f"{int(in_band.sum()):,}; eligible on both: {len(keep):,}", flush=True)
    if len(keep) == 0:
        raise RuntimeError("no eligible sources after the preregistered filters")

    # Random, fixed-seed draw.  Taking the head of a sorted SMILES list is not a
    # neutral sample: SMILES order tracks leading element, so it returns an
    # alphabetically clustered and chemically unrepresentative set.
    order = np.random.default_rng(selection_seed).permutation(len(keep))
    drawn = [int(keep[i]) for i in order[:wanted]]

    chosen = []
    for position in drawn:
        key = candidates[position]
        entry_id = representative[key]
        state, _f, _e = library.inputs_for([entry_id])
        chosen.append({
            "source": key,
            "entry_id": entry_id,
            "drd2": float(scores[position]),
            "slots": int(state[0].atom_types.shape[0]),
            "heavy_atoms": int(sizes[position]),
        })
    eligible_total = len(keep)
    print(f"  drew {len(chosen)} sources at seed {selection_seed}; "
          f"heavy atoms {min(c['heavy_atoms'] for c in chosen)}-"
          f"{max(c['heavy_atoms'] for c in chosen)}", flush=True)
    payload = {
        "schema": "compose.editing_v2.c0_source_selection",
        "criteria": {
            "source_ceiling": SOURCE_CEILING, "target_floor": TARGET_FLOOR,
            "min_heavy_atoms": MIN_HEAVY_ATOMS, "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "outcome_independent": True,
            "selection_seed": selection_seed,
            "note": "no rollout-based reachability screen; unsolvable sources are "
                    "reported, not excluded. The heavy-atom band matches the "
                    "benchmark's own DRD2 source set and looks only at molecule "
                    "size. Sources are drawn at random from the eligible pool: "
                    "taking the head of a sorted SMILES list clusters by leading "
                    "element and is not a neutral sample.",
        },
        "reserve_sources": len(representative),
        "scanned": len(candidates),
        "eligible": eligible_total,
        "below_ceiling": int(below_ceiling.sum()),
        "in_heavy_atom_band": int(in_band.sum()),
        "already_solved": already_solved,
        "selected": chosen,
    }
    out = Path(RUN_ROOT) / "c0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    return payload


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
    max_containers=20, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def probe_source(task: dict[str, Any]) -> dict[str, Any]:
    """Greedy trajectory from one source, with an MC value probe at each state."""

    import numpy as np
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
        canonical_successor_result,
        enumerate_factorized_marked_law,
    )
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.drd2_oracle import DRD2Oracle, tanimoto_to

    started = time.perf_counter()
    source_key = task["source"]
    rng = np.random.default_rng(task["seed"])
    artifact_volume.reload()
    inputs = Path(RUN_ROOT) / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())

    runtime_source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    state_bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        runtime_source, materialized_state=state_bundle)
    model = runtime.model
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()

    # The slot count is INHERITED from the library's own padding (resolved during
    # selection) rather than chosen here, so every state the kernel sees carries
    # the same insertion headroom it was trained with.  Rebuilding the start
    # state from its canonical key reproduces the library's state exactly while
    # letting this container skip loading the corpus.
    slots = int(task["slots"])
    start_state = pad_molecular_graph(smiles_to_molecular_graph(source_key), slots)
    oracle = DRD2Oracle.from_manifest(ORACLE_DIR / "drd2_oracle_manifest.json")
    print(f"[{time.perf_counter()-started:6.1f}s] ready: {source_key} "
          f"({task['heavy_atoms']} heavy atoms in {slots} slots, "
          f"step {checkpoint['selected_step']:,})", flush=True)

    enumeration_cache: dict[str, list[tuple[str, float]]] = {}
    kernel_calls = 0

    def successors(state, key: str) -> list[tuple[str, float]]:
        """Mask-legal successors of `state` with reference-law probabilities."""

        nonlocal kernel_calls
        if key in enumeration_cache:
            return enumeration_cache[key]
        kernel_calls += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        rows: list[tuple[str, float]] = []
        for successor in result.batch.successors:
            # The similarity constraint is to the SOURCE, not to the current
            # state, and binds every arm identically.
            if tanimoto_to(source_key, successor.key) >= SIMILARITY_FLOOR:
                rows.append((successor.key, float(successor.probability)))
        enumeration_cache[key] = rows
        return rows

    def rebuild(key: str):
        try:
            return pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001 - an unrepresentable successor is skipped
            return None

    def sample_index(probabilities: np.ndarray) -> int:
        total = probabilities.sum()
        if total <= 0:
            return int(rng.integers(len(probabilities)))
        return int(rng.choice(len(probabilities), p=probabilities / total))

    # ---- cheap rollout stepping -----------------------------------------
    # A rollout needs ONE sampled successor, not the whole canonical partition.
    # Profiling one enumeration: 59% of the cost is applying and canonicalising
    # every mark (25,581 molecular_graph_to_smiles calls for a single state),
    # and only 41% is the marked law the probabilities come from.
    #
    # P(canonical successor y) is exactly the sum of the mark probabilities that
    # reach y, so drawing a mark in proportion to its probability and
    # canonicalising ONLY that mark samples the canonical law.  Verified
    # deterministically rather than by Monte Carlo: identical support, virtual
    # mass equal to 0.0e+00, probabilities agreeing to 1.6e-8 -- summation-order
    # associativity on float32 model outputs, far below anything that moves a
    # sample.
    #
    # Self-loops and mask failures are handled by rejection, which reproduces the
    # masked-and-renormalised law exactly.  On giving up, the exact path is used
    # instead; a mixture of two samplers of the SAME law is still that law, so
    # the fallback introduces no bias.
    rewrite_system = _default_rewrite_system(model)
    marked_law_cache: dict[str, Any] = {}
    mark_resolution: dict[tuple[str, int], str | None] = {}
    REJECTION_TRIES = 12

    def marked_law(state, key: str):
        nonlocal kernel_calls
        if key not in marked_law_cache:
            kernel_calls += 1
            with torch.no_grad():
                marked_law_cache[key] = enumerate_factorized_marked_law(
                    model, state, float(TIME_POINT))
        return marked_law_cache[key]

    def resolve_mark(state, key: str, law, index: int) -> str | None:
        """Canonical successor of one mark; None if it is a self-loop."""

        cached = mark_resolution.get((key, index))
        if cached is not None or (key, index) in mark_resolution:
            return cached
        mark = law.marks[index]
        successor = rewrite_system.apply(state, mark.executor_rule_name, mark.action)
        resolved = canonical_state_key(successor)
        if resolved == law.source_key:
            resolved = None
        mark_resolution[(key, index)] = resolved
        return resolved

    def sample_successor(state, key: str) -> str | None:
        if key in enumeration_cache:
            # Already enumerated exactly -- reuse it rather than re-deriving.
            rows = enumeration_cache[key]
            if not rows:
                return None
            keys = [r[0] for r in rows]
            weights = np.array([r[1] for r in rows], dtype=np.float64)
            return keys[sample_index(weights)]

        law = marked_law(state, key)
        if not law.marks:
            return None
        weights = np.array([float(m.probability) for m in law.marks], dtype=np.float64)
        for _ in range(REJECTION_TRIES):
            index = sample_index(weights)
            resolved = resolve_mark(state, key, law, index)
            if resolved is None:
                continue
            if tanimoto_to(source_key, resolved) >= SIMILARITY_FLOOR:
                return resolved
        # Rejection kept failing: fall back to the exact enumeration.
        rows = successors(state, key)
        if not rows:
            return None
        keys = [r[0] for r in rows]
        exact_weights = np.array([r[1] for r in rows], dtype=np.float64)
        return keys[sample_index(exact_weights)]

    margin_cache: dict[str, float] = {}

    def margin(key: str) -> float:
        if key not in margin_cache:
            margin_cache[key] = float(oracle.margin_many([key])[0])
        return margin_cache[key]

    def rollout_prefix(key: str, depth: int) -> np.ndarray:
        """Running best MARGIN over 0..depth steps of a masked-reference rollout.

        Returns the whole prefix, not just the endpoint, because a depth-3
        rollout CONTAINS its depth-1 and depth-2 prefixes: reading the running
        maximum at each step yields all three horizons from one set of rollouts,
        at no extra kernel calls.  That turns "depth 3 is enough" into a
        measurable trend across depths instead of an assertion.

        The task is to REACH an active molecule within the budget, not to be
        active exactly at the horizon, so a path is worth its best molecule
        rather than its endpoint.

        The margin, not the probability, is what a rollout accumulates: P(active)
        spans 0.00003 to 0.048 across the benchmark's own sources, so a few edits
        through that region move it by amounts that vanish against rollout noise
        and every candidate ties.  The margin is the same ordering with usable
        resolution.
        """

        best = margin(key)
        prefix = [best]
        current_key = key
        for _ in range(depth):
            state = rebuild(current_key)
            nxt = None if state is None else sample_successor(state, current_key)
            if nxt is None:
                break
            current_key = nxt
            best = max(best, margin(current_key))
            prefix.append(best)
        # A rollout that dies early cannot improve further, so its later
        # horizons carry the best it did reach -- not a missing value, which
        # would silently drop that candidate from the deeper comparisons.
        while len(prefix) < depth + 1:
            prefix.append(best)
        return np.asarray(prefix, dtype=np.float64)

    def estimate_value(key: str, depth: int, rollouts: int) -> np.ndarray:
        """Mean prefix-value vector over `rollouts` independent rollouts."""

        if depth <= 0:
            return np.full(1, margin(key), dtype=np.float64)
        return np.mean([rollout_prefix(key, depth) for _ in range(rollouts)], axis=0)

    decisions: list[dict[str, Any]] = []
    current_key = source_key
    current_state = start_state
    trajectory = [source_key]
    seen_states: set[str] = set()
    solved_at = None

    for step in range(BUDGET):
        remaining = BUDGET - step
        rows = successors(current_state, current_key)
        if not rows:
            print(f"  step {step}: no mask-legal successors; stopping", flush=True)
            break

        keys = [r[0] for r in rows]
        reference = np.array([r[1] for r in rows], dtype=np.float64)
        # Exact for every legal successor -- no estimation anywhere in `immediate`.
        immediate = oracle.margin_many(keys)
        for key, value in zip(keys, immediate):
            margin_cache.setdefault(key, float(value))
        greedy_index = int(np.argmax(immediate))

        # Candidate pool, three strata, unioned and deduplicated:
        #   top margin     -- where a sacrifice AMONG PLAUSIBLE ACTIONS lives
        #   top reference  -- what R_theta itself considers likely
        #   random draws   -- the only stratum that reaches the tail
        # Greedy's pick heads the first stratum, so it is always evaluated.
        # Comparing greedy against a handful of purely random draws out of ~500
        # made MC-best = greedy-best almost by construction: greedy's margin is
        # the maximum over ALL successors, which a random draw's short rollout
        # essentially never overtakes.
        # Greedy's own index is seeded FIRST rather than assumed to head the
        # top-margin stratum. Margins tie exactly whenever two successors fold to
        # the same fingerprint -- symmetric edits do this routinely -- and
        # np.argsort is not stable, so with enough ties at the maximum the
        # argmax index can fall outside the top-K and the regret comparison would
        # have no greedy value to compare against.
        picked: list[int] = [greedy_index]
        seen_candidates = {greedy_index}
        for stratum, count in ((-immediate, CANDIDATES_TOP_MARGIN),
                               (-reference, CANDIDATES_TOP_REFERENCE)):
            for index in np.argsort(stratum)[:count]:
                if int(index) not in seen_candidates:
                    picked.append(int(index))
                    seen_candidates.add(int(index))
        rest = [i for i in range(len(keys)) if i not in seen_candidates]
        if rest and CANDIDATES_RANDOM > 0:
            weights = reference[rest]
            take = min(CANDIDATES_RANDOM, len(rest))
            if weights.sum() > 0:
                drawn = rng.choice(rest, size=take, replace=False,
                                   p=weights / weights.sum())
            else:
                drawn = rng.choice(rest, size=take, replace=False)
            picked.extend(int(i) for i in drawn)
        candidate_indices = picked

        depth = min(LOOKAHEAD_DEPTH, remaining - 1)
        selection = {i: estimate_value(keys[i], depth, ROLLOUTS_SELECT)
                     for i in candidate_indices}

        # One MC choice per horizon, all read off the same rollouts.
        mc_by_depth = {d: max(selection, key=lambda i: selection[i][d])
                       for d in range(1, depth + 1)}
        mc_index = mc_by_depth.get(depth, greedy_index)

        # Fresh, independent rollouts for the reported comparison.  Every
        # candidate any horizon selected is re-evaluated once; a single fresh
        # estimate serves all depths because it too returns the whole prefix.
        needs_fresh = {greedy_index} | set(mc_by_depth.values())
        fresh = {i: estimate_value(keys[i], depth, ROLLOUTS_EVAL)
                 for i in needs_fresh}

        by_depth = {}
        for d in range(1, depth + 1):
            chosen = mc_by_depth[d]
            regret = float(fresh[chosen][d] - fresh[greedy_index][d])
            by_depth[str(d)] = {
                "mc_key": keys[chosen],
                "top1_disagreement": bool(chosen != greedy_index),
                "mc_immediate_margin": float(immediate[chosen]),
                "fresh_regret": regret,
                "sacrifice_to_win": bool(
                    chosen != greedy_index
                    and immediate[chosen] < immediate[greedy_index]
                    and regret > 0),
            }

        fresh_mc = float(fresh[mc_index][depth])
        fresh_greedy = float(fresh[greedy_index][depth])

        # A revisited state is recorded but excluded from the statistics. The
        # kernel is NOT given a visited-set: masking revisits would redefine
        # R_theta, and the reference law has to stay the law being studied.
        duplicate = current_key in seen_states
        seen_states.add(current_key)

        greedy_probability = float(oracle.score(keys[greedy_index]))
        decisions.append({
            "step": step,
            "remaining_budget": remaining,
            "lookahead_steps": depth,
            "state": current_key,
            "duplicate_state": duplicate,
            "legal_successors": len(keys),
            "candidates_evaluated": len(candidate_indices),
            "greedy_key": keys[greedy_index],
            "mc_key": keys[mc_index],
            "greedy_immediate_margin": float(immediate[greedy_index]),
            "mc_immediate_margin": float(immediate[mc_index]),
            "greedy_probability": greedy_probability,
            "mc_probability": float(oracle.score(keys[mc_index])),
            "greedy_value_selection": float(selection[greedy_index][depth]),
            "mc_value_selection": float(selection[mc_index][depth]),
            "greedy_value_fresh": fresh_greedy,
            "mc_value_fresh": fresh_mc,
            "by_depth": by_depth,
            "top1_disagreement": bool(mc_index != greedy_index),
            "fresh_regret": float(fresh_mc - fresh_greedy),
            "immediate_sacrifice": float(immediate[greedy_index]
                                         - immediate[mc_index]),
            "sacrifice_to_win": bool(
                mc_index != greedy_index
                and immediate[mc_index] < immediate[greedy_index]
                and fresh_mc > fresh_greedy),
            "best_immediate_margin": float(immediate.max()),
        })
        print(f"  step {step}: {len(keys):4d} legal  {len(candidate_indices)} cand  "
              f"margin*={immediate.max():+.3f} P={greedy_probability:.4f}  "
              f"disagree={mc_index != greedy_index}  "
              f"regret={fresh_mc - fresh_greedy:+.4f}"
              f"{'  [dup]' if duplicate else ''}  "
              f"({time.perf_counter()-started:.0f}s, {kernel_calls} kernel calls)",
              flush=True)

        # Advance greedily: the probe asks what a planner would have done at the
        # states the greedy comparator actually reaches.
        current_key = keys[greedy_index]
        nxt = rebuild(current_key)
        if nxt is None:
            break
        current_state = nxt
        trajectory.append(current_key)
        # Success is the BENCHMARK quantity -- P(active) against 0.5 -- not the
        # margin used for planning. Comparing a margin to 0.5 would be comparing
        # a log-odds to a probability.
        if greedy_probability >= TARGET_FLOOR and solved_at is None:
            solved_at = step + 1
            break

    payload = {
        "schema": "compose.editing_v2.c0_probe",
        "source": source_key,
        "entry_id": task["entry_id"],
        "source_drd2": task["drd2"],
        "seed": task["seed"],
        "selected_step": int(checkpoint["selected_step"]),
        "config": {
            "budget": BUDGET, "similarity_floor": SIMILARITY_FLOOR,
            "target_floor": TARGET_FLOOR, "lookahead_depth": LOOKAHEAD_DEPTH,
            "candidates_top_margin": CANDIDATES_TOP_MARGIN,
            "candidates_top_reference": CANDIDATES_TOP_REFERENCE,
            "candidates_random": CANDIDATES_RANDOM,
            "rollouts_select": ROLLOUTS_SELECT,
            "rollouts_eval": ROLLOUTS_EVAL, "time_point": TIME_POINT,
            "planning_signal": "logit P(active) (SVM margin); monotone with the "
                               "benchmark probability so greedy is unchanged, but "
                               "with resolution in the low-activity regime",
            "outcome_signal": "P(active) against 0.5, the benchmark quantity",
        },
        "greedy_trajectory": trajectory,
        "greedy_solved_at": solved_at,
        "decisions": decisions,
        "kernel_calls": kernel_calls,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / "c0" / "probes"
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in source_key)[:80]
    (out / f"{task['index']:03d}_{safe}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    print(f"[{time.perf_counter()-started:6.1f}s] done: {len(decisions)} decisions, "
          f"{kernel_calls} kernel calls", flush=True)
    return payload


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-source probes into the reported statistics.

    A module-level function rather than inline in the entrypoint so it can be
    exercised without Modal.  This is the one stage that runs only after every
    container has been paid for, so a crash here is the most expensive kind.
    """

    # A revisited state contributes nothing new and would double-count whatever
    # it contributed the first time; the kernel was left untouched, so revisits
    # are dropped HERE rather than prevented there.
    everything = [d for r in results for d in r["decisions"]]
    duplicates = sum(d["duplicate_state"] for d in everything)
    decisions = [d for d in everything
                 if d["lookahead_steps"] >= 1 and not d["duplicate_state"]]
    print(f"\n{len(decisions)} decision states with a non-trivial horizon "
          f"(from {len(results)} sources; {duplicates} revisited states dropped, "
          f"{len(everything)} total)")
    if not decisions:
        return {"schema": "compose.editing_v2.c0_summary",
                "sources": len(results), "decision_states": 0,
                "verdict": "NO DATA -- no decision states with a horizon"}

    disagreements = sum(d["top1_disagreement"] for d in decisions)
    sacrifices = sum(d["sacrifice_to_win"] for d in decisions)
    regrets = [d["fresh_regret"] for d in decisions]
    positive = sum(1 for r in regrets if r > 0)
    mean_regret = sum(regrets) / len(regrets)
    solved = sum(1 for r in results if r["greedy_solved_at"] is not None)

    print(f"\n  top-1 disagreement   {disagreements:4d} / {len(decisions)} "
          f"({disagreements/len(decisions):6.1%})")
    print(f"  sacrifice-to-win     {sacrifices:4d} / {len(decisions)} "
          f"({sacrifices/len(decisions):6.1%})")
    print(f"  fresh regret > 0     {positive:4d} / {len(decisions)} "
          f"({positive/len(decisions):6.1%})")
    print(f"  mean fresh regret    {mean_regret:+.5f}  (margin units)")
    print(f"  greedy solved        {solved} / {len(results)} sources "
          f"(benchmark P >= {TARGET_FLOOR})")
    print(f"  total kernel calls   {sum(r['kernel_calls'] for r in results):,}")

    # ---- the depth ladder ------------------------------------------------
    # Read off the same rollouts, so this costs nothing extra. The convincing
    # pattern is monotone: depth 1 close to greedy, later depths diverging. A
    # single depth in isolation cannot distinguish "planning helps" from "this
    # particular horizon happened to look good".
    print(f"\n  depth  disagree   sacrifice-to-win   mean fresh regret   "
          f"sacrificial win rate")
    ladder = {}
    for d in range(1, LOOKAHEAD_DEPTH + 1):
        at_depth = [row["by_depth"][str(d)] for row in decisions
                    if str(d) in row.get("by_depth", {})]
        if not at_depth:
            continue
        d_disagree = sum(r["top1_disagreement"] for r in at_depth)
        d_sacrifice = sum(r["sacrifice_to_win"] for r in at_depth)
        d_regret = sum(r["fresh_regret"] for r in at_depth) / len(at_depth)
        d_sacrificial = [r for r in at_depth if r["top1_disagreement"]]
        d_won = sum(1 for r in d_sacrificial if r["fresh_regret"] > 0)
        rate = (d_won / len(d_sacrificial)) if d_sacrificial else None
        ladder[str(d)] = {
            "decision_states": len(at_depth),
            "top1_disagreement": d_disagree,
            "sacrifice_to_win": d_sacrifice,
            "mean_fresh_regret": d_regret,
            "sacrificial_actions": len(d_sacrificial),
            "sacrificial_won": d_won,
            "sacrificial_win_rate": rate,
        }
        print(f"  {d:5d}  {d_disagree:4d}/{len(at_depth):-4d}  "
              f"{d_sacrifice:11d}/{len(at_depth):-4d}   {d_regret:+15.5f}   "
              f"{'n/a' if rate is None else f'{rate:.1%} of {len(d_sacrificial)}'}")

    # PRIMARY EVIDENCE.  Among decisions where the MC action differs from greedy
    # AND is immediately worse, "no planning signal" predicts the fresh regret is
    # positive about half the time -- the evaluation sample is independent of the
    # one that selected the action, so under the null its sign is a coin flip.
    # A rate near 50% is noise however large the disagreement count is; a rate
    # well above 50% is the phenomenon.
    sacrificial = [d for d in decisions
                   if d["top1_disagreement"]
                   and d["mc_immediate_margin"] < d["greedy_immediate_margin"]]
    won = sum(1 for d in sacrificial if d["fresh_regret"] > 0)
    if sacrificial:
        share = won / len(sacrificial)
        # Binomial standard error on the rate, so a near-chance result is not
        # read as a trend. With this many observations anything inside roughly
        # +-2 SE of 50% is not distinguishable from the null.
        standard_error = (0.25 / len(sacrificial)) ** 0.5
        print(f"\n  immediately-worse MC actions: {len(sacrificial)}")
        print(f"    better after {LOOKAHEAD_DEPTH} edits: {won} ({share:.1%})"
              f"  [null 50%, SE {standard_error:.1%}, "
              f"{abs(share-0.5)/standard_error:.1f} SE from chance]")
        mean_sacrifice = (sum(d["immediate_sacrifice"] for d in sacrificial)
                          / len(sacrificial))
        print(f"    mean immediate margin given up: {mean_sacrifice:+.5f}")
    else:
        print("\n  no immediately-worse MC actions were chosen at all")

    if not sacrificial or disagreements == 0:
        verdict = ("STOP -- future value never changed a decision; "
                   "greedy is already the planner on this task")
    elif won / len(sacrificial) > 0.5 and mean_regret > 0:
        verdict = (f"GO -- sacrificial actions pay off {won/len(sacrificial):.0%} "
                   f"of the time against a ~50% null, with positive mean regret")
    else:
        verdict = ("STOP / INCONCLUSIVE -- sacrificial actions pay off at about "
                   "chance; no evidence future value changes decisions for the better")
    print(f"\n  {verdict}")

    summary = {
        "schema": "compose.editing_v2.c0_summary",
        "status": "MECHANISM_DETECTION_PROBE_NOT_A_BENCHMARK_RESULT",
        "reading": (
            "Regret is in MARGIN units (logit P(active)). The margin is monotone "
            "with the benchmark probability, so greedy's ranking is identical "
            "under either -- but monotonicity does NOT survive taking "
            "expectations, so E[max margin] and E[max P] are different "
            "objectives. This probe therefore answers only 'is there evidence "
            "that future reachability changes decisions'. Benchmark outcomes -- "
            "P(active), success at 0.5, oracle cost -- are Experiment C's job."),
        "sources": len(results),
        "decision_states": len(decisions),
        "decision_states_total": len(everything),
        "duplicate_states_dropped": duplicates,
        "top1_disagreement": disagreements,
        "sacrifice_to_win": sacrifices,
        "fresh_regret_positive": positive,
        "mean_fresh_regret": mean_regret,
        "sacrificial_actions": len(sacrificial),
        "sacrificial_won": won,
        "sacrificial_win_rate": (won / len(sacrificial)) if sacrificial else None,
        "null_win_rate": 0.5,
        "by_depth": ladder,
        "greedy_solved_sources": solved,
        "kernel_calls": sum(r["kernel_calls"] for r in results),
        "verdict": verdict,
    }
    return summary


@app.local_entrypoint()
def main(sources: int = 12, selection_seed: int = 20260811,
         label: str = "preregistered") -> None:
    """`selection_seed` exists so plumbing can be validated on THROWAWAY sources.

    A dry run at a different seed draws a disjoint sample, so nothing observed
    while checking that the code executes can touch the sources the reported run
    uses.
    """

    selection = select_sources.remote(wanted=sources, selection_seed=selection_seed)
    chosen = selection["selected"]
    print(f"\n[{label}] {len(chosen)} sources at seed {selection_seed}, from "
          f"{selection['eligible']:,} eligible "
          f"({selection['reserve_sources']:,} reserve sources scanned)")
    if not chosen:
        raise SystemExit("no eligible sources; nothing to probe")

    tasks = [{**row, "index": i, "seed": selection_seed + i}
             for i, row in enumerate(chosen)]
    results = [r for r in probe_source.map(tasks) if r]

    summary = summarise(results)
    destination = Path(
        f"diagnostics/editing_v2_experiment_c0_planning_signal_{label}.json")
    destination.write_text(
        json.dumps({"summary": summary, "per_source": results}, indent=2) + "\n")
    print(f"  wrote {destination}")
