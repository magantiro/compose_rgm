#!/usr/bin/env python3
"""What the validity-closed legal-event fiber buys, measured against blind draw + reject.

COMPOSE claims its proposal machinery draws each step from the set of rewrites that
are executable *from the current state*, so every committed intermediate is a valid
molecule by construction.  This diagnostic quantifies that claim against the obvious
alternative on the same program space.

    ARM F  fiber-conditioned  -- the shipped shallow-lane sampler, called through the
           production `t4_fiber_campaign.expand`.  Nothing here is transcribed: the
           real `synthesize_dynamic_program`, the real executor and the real
           `Fiber.check` decide every outcome.  This harness only WRAPS them to see
           the funnel interior, which `expand` does not expose in its return value.

    ARM U  unconditioned + execute/reject -- the SAME program space (same 13 module
           families, same coordinate ranges, same parameter bounds) with the
           state-dependent executability predicates removed at the draw site.  The
           drawn coordinates then go through the SAME production executor, which
           refuses what is illegal.

The line between "the space" and "the conditioning" is read straight off the
production enumerators, which are all written in the shape

    enumerate_X(state) = tuple(action for <coordinates> in <DECLARED RANGE>
                               if <STATE-DEPENDENT ADMISSION PREDICATE>)

so the rule adopted here is: **everything in the `for` clause is the space; every
`if` that consults graph structure to decide executability is the conditioning.**
Arm U keeps the `for`, drops the `if`, and lets the executor refuse.

Three conditioning points could NOT be separated from the space definition and are
therefore RETAINED in arm U (see `arm_definitions.retained` in the artifact).  Arm U
is consequently only *partially* unconditioned, which makes every gap reported here a
LOWER BOUND on what full fiber conditioning is worth.

Zero oracle calls, zero docking, zero Modal.  Run under the pinned kernel:

    PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      /Users/rmaganti/compose_region_pinned_env/bin/python \
      scripts/t4_validity_fiber_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control import current_state_edits as cse
from compose_v4.control import dynamic_program_synthesis as dps
from compose_v4.experiments import t4_fiber_campaign as t4fc
from compose_v4.experiments.whole_ring_plan import execute_program, fresh_slot
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.operators import (
    MICRO_BOND_CLASSES,
    AtomDelete,
    AtomInsert,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA_VERSION = "t4_validity_fiber_diagnostic_v1"

DELTA = 0.6
SUPPORT = "compose_valid"
SLOTS = 48                      # t4_fiber_campaign.py:257 pads every parent to 48 slots
HORIZON = 3                     # expand(..., horizon=3) -> max_modules=3
DEFAULT_DRAWS = 256

#: Arm U's reject budget for the region draw.  Production's candidate list length is a
#: function of the graph's bridge structure, so reusing it would leak the very
#: conditioning arm U removes.  A fixed constant is used instead.  Raising it can only
#: HELP arm U, so this choice is conservative with respect to the reported direction.
ARM_U_REGION_CANDIDATES = 64


# ----------------------------------------------------------------------------------
# Conditioning-point registry.  Emitted verbatim into the artifact so the arm
# definition is auditable without reading this file.
# ----------------------------------------------------------------------------------

CONDITIONING_REMOVED = [
    {
        "site": "current_state_edits.ENUMERATORS['atom_restate_semantic']",
        "production": "rewrite/operators.py:600-629 enumerate_semantic_atom_restates",
        "declared_range": "real_atom_slots x range(len(ORGANIC_VOCABULARY))",
        "predicate_removed": (
            "resolve_semantic_atom_restate_action(...).admitted and "
            "canonical_state_key(successor) != source_key  (operators.py:622-627)"
        ),
        "why_it_is_executability_not_space": (
            "The for-clause enumerates every (vertex, target class) pair; the if-clause "
            "asks the semantic resolver whether THIS state admits that pair. The pair "
            "is well-typed regardless of the state."
        ),
        "families_affected": ["heteroatom_substitute"],
    },
    {
        "site": "current_state_edits.ENUMERATORS['cycle_open']",
        "production": "rewrite/operators.py:259-268 enumerate_cycle_open_edges",
        "declared_range": "combinations(real_atom_slots, 2)",
        "predicate_removed": (
            "int(mg.bonds[a, b]) != 0 and is_valid_cycle_open_edge(mg, action)  "
            "(operators.py:266-267)"
        ),
        "why_it_is_executability_not_space": (
            "CycleOpenEdge(a, b) is well-typed for any ordered pair of real slots. "
            "Whether a bond exists there, and whether removing it keeps the molecule "
            "connected, are properties of the current graph."
        ),
        "families_affected": ["cycle_open"],
    },
    {
        "site": "current_state_edits.ENUMERATORS['cycle_close']",
        "production": "rewrite/operators.py:324-346 enumerate_cycle_close_edges",
        "declared_range": "combinations(real_atom_slots, 2) x MICRO_BOND_CLASSES",
        "predicate_removed": "resolve_cycle_close_edge(...).admitted (operators.py:337-346)",
        "why_it_is_executability_not_space": (
            "The bond order is drawn from the declared MICRO_BOND_CLASSES tuple; the "
            "resolver decides whether the current valences admit it."
        ),
        "families_affected": ["cycle_close"],
    },
    {
        "site": "current_state_edits.ENUMERATORS['bond_reroute']",
        "production": (
            "rewrite/factorized_fiber.py:307-346 pendant_graft_candidates / "
            "enumerate_pendant_graft_actions"
        ),
        "declared_range": (
            "ordered triples (moved, removed_neighbor, target) of distinct real slots, "
            "assembled into the restricted BondReroute(a=moved, b=removed_neighbor, "
            "u=moved, v=target) form the production enumerator emits"
        ),
        "predicate_removed": (
            "single-bond bridge test, _induced_is_tree(pendant), MAX_H_COUNT room on "
            "removed_neighbor, bonds[moved,target]==0 and implicit_h[target]>=1 "
            "(factorized_fiber.py:321-335)"
        ),
        "why_it_is_executability_not_space": (
            "The action carries three atom slots. Bridge-ness, tree-ness and hydrogen "
            "availability are all facts about the current graph, checked to decide "
            "whether the executor would accept the triple."
        ),
        "families_affected": ["bond_reroute"],
    },
    {
        "site": "dynamic_program_synthesis._grow_actions",
        "production": "control/dynamic_program_synthesis.py:119-133",
        "declared_range": "real_atom_slots (anchor) x declared element tuple x length",
        "predicate_removed": (
            "int(current.implicit_h_counts[i]) >= 1 restricting the anchor to "
            "hydrogen-bearing atoms (dynamic_program_synthesis.py:124)"
        ),
        "why_it_is_executability_not_space": (
            "The element tuple and the length bound ARE the space and are kept. Free "
            "valence on the chosen anchor is exactly the condition under which the "
            "executor will accept the AtomInsert."
        ),
        "families_affected": ["segment_grow", "functionalize", "segment_replace"],
    },
    {
        "site": "dynamic_program_synthesis._terminal_shrink",
        "production": "control/dynamic_program_synthesis.py:198-227",
        "declared_range": "real_atom_slots",
        "predicate_removed": (
            "int(np.count_nonzero(current.bonds[i])) == 1 restricting deletion to "
            "degree-one atoms (dynamic_program_synthesis.py:203)"
        ),
        "why_it_is_executability_not_space": (
            "AtomDelete(slot) is well-typed for any real slot. Degree one is the "
            "condition under which deleting it leaves the molecule connected."
        ),
        "families_affected": ["segment_shrink"],
    },
    {
        "site": "dynamic_program_synthesis._pendant_fragments",
        "production": "control/dynamic_program_synthesis.py:242-258",
        "declared_range": (
            "an atom subset of size 1..MAX_SEGMENT_LENGTH drawn uniformly from the real "
            "slots, plus an anchor drawn uniformly from its complement -- the same "
            "(fragment, anchor) region object the production law weights"
        ),
        "predicate_removed": (
            "the bridge test `if b in left: continue` and the size/properness bound "
            "`1 <= len(fragment) <= maximum and len(fragment) < len(real)` "
            "(dynamic_program_synthesis.py:251,255)"
        ),
        "why_it_is_executability_not_space": (
            "The region is a set of atoms plus a retained anchor. Bridge-separation is "
            "the condition under which the leaf-deletion schedule in "
            "_delete_pendant_fragment can excise it without disconnecting the rest; the "
            "size bound is kept as the declared bound and only the bridge property is "
            "dropped."
        ),
        "families_affected": ["substituent_delete", "segment_replace"],
    },
]

CONDITIONING_RETAINED = [
    {
        "site": "current_state_edits.ENUMERATORS['ring_system_restate']",
        "production": "rewrite/tracelet_fiber.py:353-387 _ring_system_restate_candidates",
        "verdict": "INSEPARABLE_FROM_SPACE_DEFINITION",
        "reason": (
            "RingSystemRestate carries an arbitrary-length tuple of BondOrderChange. "
            "There is no bounded coordinate range to draw from: the candidate set is "
            "CONSTRUCTED by perceiving bridges, taking the ring-only connected "
            "components, and enumerating maximum-cardinality matchings to build "
            "coherent Kekule alternations (tracelet_fiber.py:362-386). The matching "
            "computation does not FILTER a declared space, it MANUFACTURES the "
            "parameter. An 'unconditioned' draw over all subsets of all real bond "
            "pairs x orders is a combinatorially different and astronomically larger "
            "space, which would violate the same-space requirement."
        ),
        "arm_u_behaviour": "family declines the state (raises ValueError), as production does when the fiber is empty",
    },
    {
        "site": "control/ring_program.py:188-216 construction_branches",
        "production": "reached from dynamic_program_synthesis._ring_module:329-378",
        "verdict": "RETAINED_STRUCTURAL",
        "reason": (
            "For topology='fused' the anchor domain is _ring_edges(graph, locus) -- a "
            "ring edge only exists relative to the current graph's cycle structure, so "
            "the anchor is not a state-independent coordinate. The element quota "
            "(_quota_possible/residual) is separable but was not removed in isolation, "
            "because doing so without also unconditioning the anchor would produce a "
            "half-arm whose scope is harder to state than to keep conditioned."
        ),
        "arm_u_behaviour": "identical to arm F",
        "families_affected": ["append_ring", "fuse_ring"],
    },
    {
        "site": "dynamic_program_synthesis.compile_generic_module inline predicates",
        "production": "control/dynamic_program_synthesis.py (carbonyl anchor filter; capacity clamps)",
        "verdict": "RETAINED_INLINE",
        "reason": (
            "The carbonyl_insert anchor filter (atom_types==C and implicit_h_counts>=2) "
            "and the segment_grow / segment_replace capacity clamps "
            "min(MAX_SEGMENT_LENGTH, 40 - n_real_atoms) are written inline inside "
            "compile_generic_module rather than in a helper, so removing them would "
            "require transcribing ~200 lines of the production module. A transcribed "
            "arm F cannot fail usefully and a transcribed arm U invites drift, so they "
            "are left conditioned and declared here."
        ),
        "arm_u_behaviour": "identical to arm F",
        "families_affected": ["carbonyl_insert", "segment_grow", "segment_replace"],
    },
    {
        "site": "dynamic_program_synthesis.synthesize_dynamic_program family loop",
        "production": "control/dynamic_program_synthesis.py:621-636",
        "verdict": "RETAINED_BY_DESIGN",
        "reason": (
            "The `for family in _weighted_module_order(...): try: ... except ValueError: "
            "continue` loop IS production's own reject step. It is arm U's execute/reject "
            "mechanism and is deliberately identical in both arms; removing it would "
            "compare two different algorithms rather than two different draw laws."
        ),
        "arm_u_behaviour": "identical to arm F",
    },
]


# ----------------------------------------------------------------------------------
# Lazy coordinate space -- `current_state_program` only needs len() and [i].
# ----------------------------------------------------------------------------------


class _CoordinateSpace:
    """A uniform draw over a declared coordinate range, materialised on demand.

    `current_state_program` (current_state_edits.py:43-52) calls `len(actions)` and
    `actions[i]` and nothing else, so the unconditioned range never has to be built.
    Keeping the draw uniform over the FULL declared range is what makes arm U's
    proposal law the unconditioned counterpart of production's uniform draw over the
    admitted fiber, rather than a differently shaped sampler.
    """

    __slots__ = ("_build", "_n")

    def __init__(self, n: int, build):
        self._n, self._build = int(n), build

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        return self._build(int(index))


def _real_slots(state) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))


def _u_atom_restate(state):
    real = _real_slots(state)
    classes = len(ORGANIC_VOCABULARY)
    return _CoordinateSpace(
        len(real) * classes,
        lambda i: SemanticAtomRestate(real[i // classes], i % classes),
    )


def _u_cycle_open(state):
    pairs = tuple(combinations(_real_slots(state), 2))
    return _CoordinateSpace(len(pairs), lambda i: CycleOpenEdge(*pairs[i]))


def _u_cycle_close(state):
    pairs = tuple(combinations(_real_slots(state), 2))
    orders = MICRO_BOND_CLASSES
    return _CoordinateSpace(
        len(pairs) * len(orders),
        lambda i: CycleCloseEdge(*pairs[i // len(orders)], orders[i % len(orders)]),
    )


def _u_bond_reroute(state):
    real = _real_slots(state)
    n = len(real)
    if n < 3:
        return ()

    def build(index: int):
        # index -> (moved, removed_neighbor, target) over distinct ordered triples
        target_i = index % (n - 2)
        rest = index // (n - 2)
        removed_i = rest % (n - 1)
        moved_i = rest // (n - 1)
        moved = real[moved_i]
        removed = real[removed_i if removed_i < moved_i else removed_i + 1]
        pool = [v for v in real if v != moved and v != removed]
        target = pool[target_i]
        return BondReroute(a=moved, b=removed, u=moved, v=target)

    return _CoordinateSpace(n * (n - 1) * (n - 2), build)


def _u_ring_system_restate(state):
    raise ValueError(
        "ring_system_restate has no state-independent coordinate range; see "
        "arm_definitions.retained[0]"
    )


ARM_U_ENUMERATORS = {
    "atom_restate_semantic": _u_atom_restate,
    "bond_reroute": _u_bond_reroute,
    "cycle_close": _u_cycle_close,
    "cycle_open": _u_cycle_open,
    "ring_system_restate": _u_ring_system_restate,
}


# ----------------------------------------------------------------------------------
# Production helpers copied with ONE predicate made switchable.
#
# Each carries `conditioned=True`, and `equivalence_report` requires that setting to
# reproduce the production helper byte-for-byte on real states under the same seed.
# Without that check a copied helper cannot fail usefully if production drifts.
# ----------------------------------------------------------------------------------


def _grow_actions_variant(source, rng, *, length, elements, anchor=None, conditioned=True):
    """Copy of dynamic_program_synthesis._grow_actions:119-133, anchor filter switchable."""
    current, actions = source, []
    anchors = [
        int(i)
        for i in np.flatnonzero(is_element(current.atom_types))
        if (not conditioned) or int(current.implicit_h_counts[i]) >= 1  # :124
    ]
    if anchor is not None:
        anchors = [anchor] if anchor in anchors else []
    if not anchors:
        raise ValueError("segment growth has no hydrogen-bearing anchor")
    at = anchors[int(rng.integers(len(anchors)))]
    chosen = []
    for _ in range(length):
        element = elements[int(rng.integers(len(elements)))]
        valences = ALLOWED_VALENCES[element]
        if len(valences) != 1:
            raise ValueError("dynamic growth requires an unambiguous neutral valence")
        action = AtomInsert(
            fresh_slot(current), ELEMENT_TO_IDX[element], 0, valences[0] - 1, ((at, 1),)
        )
        record = encode_action("atom_insert", action)
        current, _ = execute_program(current, [record])
        actions.append(record)
        at = action.slot
        chosen.append(element)
    return actions, at, chosen


def _terminal_shrink_variant(source, rng, *, requested_length, conditioned=True):
    """Copy of dynamic_program_synthesis._terminal_shrink:198-227, degree filter switchable."""
    current, actions, path, anchor = source, [], [], None
    preferred = None
    for _ in range(requested_length):
        real = [int(i) for i in np.flatnonzero(is_element(current.atom_types))]
        terminal = [
            i
            for i in real
            if (not conditioned) or int(np.count_nonzero(current.bonds[i])) == 1  # :203
        ]
        if preferred in terminal:
            candidates = [preferred]
        else:
            candidates = terminal
        if not candidates or len(real) <= 1:
            break
        slot = candidates[int(rng.integers(len(candidates)))]
        neighbors = [int(i) for i in np.flatnonzero(current.bonds[slot])]
        if not neighbors:
            preferred = None
            continue
        next_slot = neighbors[0]
        record = encode_action("atom_delete", AtomDelete(slot))
        try:
            following, _ = execute_program(current, [record])
        except ValueError:
            preferred = None
            continue
        actions.append(record)
        path.append(slot)
        current = following
        anchor = next_slot
        preferred = next_slot
    if not actions:
        raise ValueError("segment shrink found no legal terminal deletion")
    return actions, current, anchor, path


def _pendant_fragments_unconditioned(graph, *, maximum, harness_rng, n_samples):
    """Uniform draw over (atom subset of size 1..maximum, anchor in its complement).

    The production enumerator (dynamic_program_synthesis.py:242-258) restricts this to
    subsets that are bridge-separated pendant fragments. That restriction is dropped;
    _delete_pendant_fragment's leaf-deletion schedule and the executor then refuse what
    cannot actually be excised, which is arm U's reject step.
    """
    real = _real_slots(graph)
    if len(real) < 2:
        return ()
    out, seen = [], set()
    upper = min(maximum, len(real) - 1)
    for _ in range(n_samples):
        size = int(harness_rng.integers(1, upper + 1))
        fragment = tuple(sorted(int(v) for v in harness_rng.choice(real, size=size, replace=False)))
        complement = [v for v in real if v not in fragment]
        if not complement:
            continue
        anchor = int(complement[int(harness_rng.integers(len(complement)))])
        if (fragment, anchor) not in seen:
            seen.add((fragment, anchor))
            out.append((fragment, anchor))
    return tuple(out)


# ----------------------------------------------------------------------------------
# Equivalence self-check: the copies must reproduce production when conditioned.
# ----------------------------------------------------------------------------------


def equivalence_report(states) -> dict:
    """Require the switchable copies to equal the production helpers at conditioned=True."""
    checks, mismatches = 0, []
    for label, state in states:
        # _pendant_fragments is pure -- exact tuple equality.
        want = dps._pendant_fragments(state, maximum=dps.MAX_SEGMENT_LENGTH)
        # conditioned production form is used directly; the unconditioned form is a
        # different law by construction and is not compared here.
        checks += 1
        if want != dps._pendant_fragments(state, maximum=dps.MAX_SEGMENT_LENGTH):
            mismatches.append(f"{label}: _pendant_fragments not deterministic")

        for seed in (11, 12, 13):
            for length, elements in ((1, ("C", "N", "O", "F")), (3, ("C", "N", "O"))):
                try:
                    got = _grow_actions_variant(
                        state,
                        np.random.default_rng(seed),
                        length=length,
                        elements=elements,
                        conditioned=True,
                    )
                except ValueError as error:
                    got = f"ValueError:{error}"
                try:
                    ref = dps._grow_actions(
                        state, np.random.default_rng(seed), length=length, elements=elements
                    )
                except ValueError as error:
                    ref = f"ValueError:{error}"
                checks += 1
                if repr(got) != repr(ref):
                    mismatches.append(f"{label}: _grow_actions seed={seed} len={length}")

            for requested in (1, 4):
                try:
                    got = _terminal_shrink_variant(
                        state,
                        np.random.default_rng(seed),
                        requested_length=requested,
                        conditioned=True,
                    )
                except ValueError as error:
                    got = f"ValueError:{error}"
                try:
                    ref = dps._terminal_shrink(
                        state, np.random.default_rng(seed), requested_length=requested
                    )
                except ValueError as error:
                    ref = f"ValueError:{error}"
                checks += 1
                # compare actions/anchor/path; the intermediate graph is compared by key
                if repr(_shrink_key(got)) != repr(_shrink_key(ref)):
                    mismatches.append(f"{label}: _terminal_shrink seed={seed} n={requested}")
    return {"checks": checks, "mismatches": mismatches, "pass": not mismatches}


def _shrink_key(value):
    if isinstance(value, str):
        return value
    actions, current, anchor, path = value
    return actions, int(current.n_real_atoms), anchor, path


# ----------------------------------------------------------------------------------
# Arm U patch
# ----------------------------------------------------------------------------------


@contextmanager
def arm_u_patch(harness_rng):
    """Install arm U's unconditioned draw sites; restore production on exit."""
    saved = (cse.ENUMERATORS, dps._grow_actions, dps._terminal_shrink, dps._pendant_fragments)
    cse.ENUMERATORS = ARM_U_ENUMERATORS
    dps._grow_actions = (
        lambda source, rng, *, length, elements, anchor=None: _grow_actions_variant(
            source, rng, length=length, elements=elements, anchor=anchor, conditioned=False
        )
    )
    dps._terminal_shrink = lambda source, rng, *, requested_length: _terminal_shrink_variant(
        source, rng, requested_length=requested_length, conditioned=False
    )
    dps._pendant_fragments = lambda graph, *, maximum: _pendant_fragments_unconditioned(
        graph, maximum=maximum, harness_rng=harness_rng, n_samples=ARM_U_REGION_CANDIDATES
    )
    try:
        yield
    finally:
        (
            cse.ENUMERATORS,
            dps._grow_actions,
            dps._terminal_shrink,
            dps._pendant_fragments,
        ) = saved


# ----------------------------------------------------------------------------------
# Instrumentation: wrap, never transcribe.
# ----------------------------------------------------------------------------------


def _chemically_valid(encoded) -> bool:
    try:
        smiles = molecular_graph_to_smiles(decode_state(encoded))
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError):
        return False
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def run_arm(row, arm: str, draws: int, seed: int) -> dict:
    """One arm on one state. Arm F leaves production untouched; arm U installs patches."""
    fiber = t4fc.Fiber(row["smiles"], DELTA, support=SUPPORT)
    rng = np.random.default_rng(seed)
    harness_rng = np.random.default_rng(seed ^ 0x5F17)

    per_draw: list[dict] = []
    families = Counter()
    family_attempts = Counter()
    family_refusals = Counter()
    production_failures = Counter()
    state = {"draw": None}

    real_synth = t4fc.synthesize_dynamic_program
    real_compile = dps.compile_generic_module
    real_check = fiber.check

    def wrapped_compile(*args, **kwargs):
        family = args[2] if len(args) > 2 else kwargs.get("family")
        state["draw"]["attempts"] += 1
        family_attempts[family] += 1
        try:
            out = real_compile(*args, **kwargs)
        except ValueError:
            state["draw"]["module_refusals"] += 1
            family_refusals[family] += 1
            raise
        return out

    def wrapped_synth(source, rng_, **kwargs):
        record = {
            "ok": False,
            "attempts": 0,
            "module_refusals": 0,
            "modules": 0,
            "primitive_edits": 0,
            "committed_states": 0,
            "invalid_states": 0,
            "eligible": 0,
            "families": [],
        }
        state["draw"] = record
        per_draw.append(record)
        result = real_synth(source, rng_, **kwargs)
        _, _, _, trace, metadata = result
        record["ok"] = True
        record["modules"] = int(metadata["completed_module_count"])
        record["primitive_edits"] = int(trace.get("primitive_edits", 0))
        record["families"] = [m["family"] for m in metadata["modules"]]
        families.update(record["families"])
        committed = list(trace["states"])[1:]          # index 0 is the source, not a product
        record["committed_states"] = len(committed)
        record["invalid_states"] = sum(1 for s in committed if not _chemically_valid(s))
        # Production's OWN accounting, as a second independent instrument on the same
        # quantity: `module_failure_counts` also carries post-compile rejections
        # (extract_program failures and the primitive/block work limit) that a wrapper
        # on compile_generic_module cannot see.
        for key in metadata.get("module_failure_counts", {}):
            reason = (
                "work_limit" if key.endswith(":work_limit")
                else "extract" if ":extract:" in key
                else "compile"
            )
            production_failures[reason] += metadata["module_failure_counts"][key]
        return result

    def wrapped_check(smiles):
        gate = real_check(smiles)
        if gate is not None and state["draw"] is not None:
            state["draw"]["eligible"] += 1
        return gate

    fiber.check = wrapped_check
    t4fc.synthesize_dynamic_program = wrapped_synth
    dps.compile_generic_module = wrapped_compile
    patch = arm_u_patch(harness_rng) if arm == "U" else _nullcontext()
    try:
        with patch:
            start = time.perf_counter()
            endpoints = t4fc.expand(
                row["smiles"], 0.0, fiber, rng, draws=draws, proposal_lane="shallow"
            )
            wall = time.perf_counter() - start
    finally:
        t4fc.synthesize_dynamic_program = real_synth
        dps.compile_generic_module = real_compile

    executed = [d for d in per_draw if d["ok"]]
    committed_total = sum(d["committed_states"] for d in executed)
    invalid_total = sum(d["invalid_states"] for d in executed)
    attempts_total = sum(d["attempts"] for d in per_draw)
    refusals_total = sum(d["module_refusals"] for d in per_draw)
    return {
        "arm": arm,
        "draws": len(per_draw),
        "executable_fraction": len(executed) / len(per_draw) if per_draw else 0.0,
        "valid_intermediate_fraction": (
            (committed_total - invalid_total) / committed_total if committed_total else None
        ),
        "committed_intermediate_states": committed_total,
        "invalid_intermediate_states": invalid_total,
        "unique_eligible_endpoints": len(endpoints),
        "wasted_proposals": len(per_draw) - len(executed),
        "draws_with_zero_eligible_endpoints": sum(1 for d in per_draw if d["eligible"] == 0),
        "module_compile_attempts": attempts_total,
        "module_compile_refusals": refusals_total,
        "module_refusal_rate": refusals_total / attempts_total if attempts_total else None,
        "attempts_per_accepted_module": (
            attempts_total / sum(d["modules"] for d in executed)
            if sum(d["modules"] for d in executed)
            else None
        ),
        "mean_modules_per_executed_program": (
            sum(d["modules"] for d in executed) / len(executed) if executed else None
        ),
        "mean_primitive_edits_per_executed_program": (
            sum(d["primitive_edits"] for d in executed) / len(executed) if executed else None
        ),
        "realized_family_mix": dict(families.most_common()),
        "family_compile_attempts": dict(family_attempts.most_common()),
        "family_compile_refusals": dict(family_refusals.most_common()),
        "production_reported_failures": dict(production_failures),
        "wall_seconds": round(wall, 3),
        "seed": seed,
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ----------------------------------------------------------------------------------


def zero_discrimination(per_state) -> dict:
    """What each reported zero does and does not discriminate.

    A zero is only evidence of absence if the arm had the power to see a nonzero.
    For every cell where one arm returned zero eligible endpoints, the OTHER arm's
    realized per-draw yield gives the expected count, and exp(-expected) is the
    approximate probability of observing zero at that rate. The approximation is
    Poisson on a DEDUPLICATED count (distinct endpoints), so it slightly overstates
    the expected number of independent successes and is therefore conservative
    against calling a zero decisive.
    """
    rows, undiscriminating = [], []
    for cell in per_state:
        draws = cell["arm_F"]["draws"]
        f, u = (
            cell["arm_F"]["unique_eligible_endpoints"],
            cell["arm_U"]["unique_eligible_endpoints"],
        )
        for zero_arm, other in (("U", f), ("F", u)):
            mine = u if zero_arm == "U" else f
            if mine != 0:
                continue
            rate = other / draws
            expected = rate * draws
            row = {
                "idx": cell["idx"],
                "target": cell["target"],
                "zero_arm": zero_arm,
                "other_arm_eligible": other,
                "draws": draws,
                "other_arm_rate_per_draw": round(rate, 5),
                "expected_at_other_arm_rate": round(expected, 2),
                "approx_p_zero_at_that_rate": round(float(np.exp(-expected)), 6),
                "verdict": (
                    "BOTH_ZERO_UNDISCRIMINATING" if other == 0
                    else "DISCRIMINATING" if expected >= 5
                    else "WEAK"
                ),
            }
            rows.append(row)
            if other == 0:
                undiscriminating.append(cell["idx"])
    return {
        "note": (
            "Cells where BOTH arms return zero carry no information about the arms: "
            "the fiber gate admits nothing reachable there at delta=0.6 under either "
            "draw law, which is a property of the cell, not of the conditioning."
        ),
        "cells_where_both_arms_zero": sorted(set(undiscriminating)),
        "per_state": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--limit", type=int, default=0, help="first N seeds (debug only)")
    parser.add_argument("--idx", type=int, nargs="*", default=None, help="only these seed idx (debug only)")
    parser.add_argument(
        "--out",
        default="diagnostics/t4_validity_fiber_diagnostic_v1.json",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="recompute the derived discrimination block from an existing artifact, "
             "without re-measuring (the measured rows are untouched)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.reanalyze:
        target = repo / args.out
        payload = json.loads(target.read_text())
        payload["zero_discrimination"] = zero_discrimination(payload["per_state"])
        target.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"reanalyzed {target}")
        for row in payload["zero_discrimination"]["per_state"]:
            print("  ", row)
        return 0
    seeds = json.loads((repo / "docs" / "GENMOL_T4_SEEDS.json").read_text())
    if args.idx:
        seeds = [r for r in seeds if int(r["idx"]) in set(args.idx)]
    if args.limit:
        seeds = seeds[: args.limit]

    padded = [
        (f"idx{row['idx']}_{row['target']}", pad_molecular_graph(
            smiles_to_molecular_graph(row["smiles"]), SLOTS))
        for row in seeds[:4]
    ]
    equivalence = equivalence_report(padded)
    if not equivalence["pass"]:
        print("EQUIVALENCE FAILED:", equivalence["mismatches"], file=sys.stderr)
        return 2
    print(f"equivalence self-check: {equivalence['checks']} checks, 0 mismatches", flush=True)

    import rdkit
    import scipy

    measured_under = {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "interpreter": sys.executable,
    }
    print("measured_under:", measured_under["python"], "rdkit", measured_under["rdkit"], flush=True)

    per_state = []
    for row in seeds:
        seed = args.seed + 1_000_003 * int(row["idx"])
        cell = {
            "idx": int(row["idx"]),
            "target": row["target"],
            "smiles": row["smiles"],
            "heavy_atoms": int(row["heavy"]),
            "delta": DELTA,
            "support": SUPPORT,
            "slots": SLOTS,
        }
        for arm in ("F", "U"):
            started = time.perf_counter()
            cell[f"arm_{arm}"] = run_arm(row, arm, args.draws, seed)
            print(
                f"  idx{row['idx']:>2} {row['target']:<6} arm {arm}: "
                f"exec={cell[f'arm_{arm}']['executable_fraction']:.3f} "
                f"valid={cell[f'arm_{arm}']['valid_intermediate_fraction']} "
                f"eligible={cell[f'arm_{arm}']['unique_eligible_endpoints']:<4} "
                f"wasted={cell[f'arm_{arm}']['wasted_proposals']:<4} "
                f"{time.perf_counter()-started:.1f}s",
                flush=True,
            )
        per_state.append(cell)

    aggregate = {}
    for arm in ("F", "U"):
        rows = [c[f"arm_{arm}"] for c in per_state]
        committed = sum(r["committed_intermediate_states"] for r in rows)
        invalid = sum(r["invalid_intermediate_states"] for r in rows)
        drawn = sum(r["draws"] for r in rows)
        attempts = sum(r["module_compile_attempts"] for r in rows)
        refusals = sum(r["module_compile_refusals"] for r in rows)
        mix, fam_att, fam_ref, prod_fail = Counter(), Counter(), Counter(), Counter()
        for r in rows:
            mix.update(r["realized_family_mix"])
            fam_att.update(r["family_compile_attempts"])
            fam_ref.update(r["family_compile_refusals"])
            prod_fail.update(r["production_reported_failures"])
        aggregate[f"arm_{arm}"] = {
            "draws": drawn,
            "executable_fraction": sum(
                r["executable_fraction"] * r["draws"] for r in rows
            ) / drawn,
            "valid_intermediate_fraction": (committed - invalid) / committed if committed else None,
            "committed_intermediate_states": committed,
            "invalid_intermediate_states": invalid,
            "unique_eligible_endpoints": sum(r["unique_eligible_endpoints"] for r in rows),
            "states_with_any_eligible_endpoint": sum(
                1 for r in rows if r["unique_eligible_endpoints"] > 0
            ),
            "wasted_proposals": sum(r["wasted_proposals"] for r in rows),
            "module_compile_attempts": attempts,
            "module_compile_refusals": refusals,
            "module_refusal_rate": refusals / attempts if attempts else None,
            "realized_family_mix": dict(mix.most_common()),
            "family_compile_attempts": dict(fam_att.most_common()),
            "family_compile_refusals": dict(fam_ref.most_common()),
            "family_refusal_rate": {
                f: round(fam_ref.get(f, 0) / fam_att[f], 4) for f in sorted(fam_att) if fam_att[f]
            },
            "production_reported_failures": dict(prod_fail),
            "wall_seconds": round(sum(r["wall_seconds"] for r in rows), 3),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_under": measured_under,
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_jobs": 0,
        "configuration": {
            "draws_per_state_per_arm": args.draws,
            "delta": DELTA,
            "support": SUPPORT,
            "slots": SLOTS,
            "horizon": HORIZON,
            "region_law": None,
            "completion_law": None,
            "base_seed": args.seed,
            "seed_rule": "base_seed + 1_000_003 * idx, identical across arms",
            "arm_u_region_candidates": ARM_U_REGION_CANDIDATES,
            "execution": "serial, single process (a loaded machine corrupts wall_seconds)",
        },
        "arm_definitions": {
            "rule": (
                "The production enumerators are written as "
                "`tuple(action for <coords> in <DECLARED RANGE> if <ADMISSION PREDICATE>)`. "
                "Everything in the for-clause is the SPACE; every if-clause that consults "
                "graph structure to decide executability is the CONDITIONING. Arm U keeps "
                "the for, drops the if, and lets the production executor refuse."
            ),
            "arm_F": {
                "what": "shipped shallow-lane sampler via t4_fiber_campaign.expand",
                "transcribed": False,
                "note": (
                    "expand, synthesize_dynamic_program, the executor and Fiber.check are "
                    "the real production objects; the harness only wraps them to observe "
                    "the funnel, which expand's return value does not expose."
                ),
            },
            "arm_U": {
                "what": "same program space, admission predicates removed at the draw site",
                "removed": CONDITIONING_REMOVED,
                "families_unconditioned": sorted(
                    {f for c in CONDITIONING_REMOVED for f in c["families_affected"]}
                ),
            },
            "retained": CONDITIONING_RETAINED,
            "consequence": (
                "Three conditioning points are retained (one inseparable, two for scope), "
                "so arm U is only partially unconditioned and every gap reported here is a "
                "LOWER BOUND on what full fiber conditioning is worth."
            ),
        },
        "equivalence_self_check": {
            **equivalence,
            "what_it_proves": (
                "The switchable copies of _grow_actions and _terminal_shrink reproduce the "
                "production helpers exactly at conditioned=True under identical seeds, so a "
                "copy that drifts from production fails this check instead of silently "
                "measuring a different sampler."
            ),
        },
        "per_state": per_state,
        "aggregate": aggregate,
    }

    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)
    for arm in ("F", "U"):
        a = aggregate[f"arm_{arm}"]
        print(
            f"ARM {arm}: exec={a['executable_fraction']:.4f} "
            f"valid_int={a['valid_intermediate_fraction']} "
            f"eligible={a['unique_eligible_endpoints']} "
            f"wasted={a['wasted_proposals']} "
            f"refusal_rate={a['module_refusal_rate']:.4f} "
            f"wall={a['wall_seconds']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
