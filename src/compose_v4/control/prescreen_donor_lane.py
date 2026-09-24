"""Transplant a coherent region from one scored molecule into another.

WHY THIS EXISTS.  The live recombiner, `ProgramOptimizer._recombine`, decomposes a
donor's edit PROGRAM into dependency-closed branches.  A molecule that arrived from the
ZINC prescreen carries only `{endpoint, source_id, state}` -- no program, no assignment,
no construction history -- so it is structurally invisible to that path however good a
molecule it is.  `compile_transplant` takes two molecular STATES plus cuts, so it does
not care where either came from.  That is the whole gap this lane closes.

MEASURED, zero oracle, 240 fixed (source, donor, cut, cut) tuples over four tasks:
  compiled 129/240 (53.8%); of the 61 inside the ceiling, source retention median 0.963,
  donor incorporation median 9 heavy atoms, and 61/61 connected, distinct from BOTH
  parents, and replaying exactly.

THE CEILING IS NOT ARBITRARY.  Transplants of 24-32 primitives exist (20.2% of compiled)
but verifying them shows median retention 0.036 and median donor contribution of ONE
atom: long transplants demolish the source and rebuild it, which is valid and useless.
The 23-primitive limit is what separates region transplantation from reconstruction, so
it is enforced here rather than left to whoever calls this.

SEARCH BUDGET IS DEFAULT ON PURPOSE.  Sweeping `PathConfig` from 1s/128 expansions to
10s/512 left the compile rate at exactly 53.8% and tripled wall time; the unresolved
mappings are real incompatibilities, not starvation.

INFORMATION BOUNDARY.  Donor and cut selection see the state, the archive's own charged
oracle scores and executability.  No target structure, no target similarity, no
task-specific motif.  This module never imports a reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.state import is_element
from compose_v4.control.donor_program import compile_transplant, pendant_cuts

#: A transplant longer than this stops being a region swap.  See the module docstring.
MAX_TRANSPLANT_PRIMITIVES = 23
#: Below this, the "transplant" released most of the source and rebuilt it.
MIN_SOURCE_RETENTION = 0.5
#: A donor that contributes fewer atoms than this did not really donate a region.
MIN_DONOR_ATOMS = 3
#: The executor refuses endpoints above this; a cut pair that cannot fit is not worth
#: compiling, so the ranking filters on it rather than discovering it as a refusal.
REPRESENTABLE_HEAVY_ATOMS = 40


@dataclass(frozen=True)
class TransplantProposal:
    endpoint: str
    primitives: int
    retention: float
    removed_atoms: int
    added_atoms: int
    donor_endpoint: str
    actions: tuple
    states: tuple


def _real_atoms(graph) -> int:
    return int(np.count_nonzero(is_element(graph.atom_types)))


def candidate_cut_pairs(source, donor, *, limit: int):
    """Oriented single-bond bridge cuts on both sides, cheapest predicted work first.

    The ordering is a HEURISTIC over a lower bound: the compiler's realized primitive
    count is what actually decides admission, and it is frequently larger than
    `removed + added`.  Ranking on the estimate only decides what to try first; nothing
    is accepted on it.
    """
    source_cuts, donor_cuts = pendant_cuts(source), pendant_cuts(donor)
    n_source, n_donor = _real_atoms(source), _real_atoms(donor)
    ranked = []
    for cut in source_cuts:
        removed = len(cut.component)
        if not 0 < removed < n_source:
            continue
        for other in donor_cuts:
            added = len(other.component)
            if not 0 < added < n_donor:
                continue
            # A donor piece the same size as what we removed usually reproduces the
            # source exactly and burns the attempt on a self-proposal.
            if added == removed and added <= 2:
                continue
            # The endpoint must stay representable.  Ranking purely by donor size
            # without this made `unsupported_size` refuse 90 of 90 attempts on every
            # large source, while ranking purely by smallest work made
            # `donor_contribution_trivial` refuse 40-89 of 90.  Both extremes are
            # measured; the admissible band is what is left between them.
            if n_source - removed + added > REPRESENTABLE_HEAVY_ATOMS:
                continue
            # `removed + added` is a lower bound on the realized primitive count, so a
            # pair already over the ceiling on the bound cannot come in under it.
            # Without this the largest-donor ranking spent 82 of 90 attempts on
            # `over_primitive_ceiling` and 65-90 on `search_unresolved`.
            if removed + added > MAX_TRANSPLANT_PRIMITIVES:
                continue
            ranked.append((added, removed, cut, other))
    # Largest donor region first among the pairs that still fit.
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [(cut, other) for _work, _ret, cut, other in ranked[:limit]]


def propose_transplants(source, donors, rng, *, attempts: int, pairs_per_donor: int = 3):
    """Compile region transplants from `source` using `donors` as structural material.

    `donors` are ordinary molecular states.  They need no program history, which is the
    point.  Returns only transplants that are inside the primitive ceiling, retentive,
    and carry a real donor region; everything else is reported through `rejections` so a
    caller can see WHY the lane was quiet rather than guessing.
    """
    accepted, rejections, tried = [], {}, 0
    order = rng.permutation(len(donors)) if len(donors) else []
    for index in order:
        if tried >= attempts:
            break
        donor_state, donor_smiles = donors[int(index)]
        try:
            pairs = candidate_cut_pairs(source, donor_state, limit=pairs_per_donor)
        except (ValueError, RuntimeError) as error:
            rejections[f"cuts:{error!s}"] = rejections.get(f"cuts:{error!s}", 0) + 1
            continue
        for cut, other in pairs:
            if tried >= attempts:
                break
            tried += 1
            try:
                result = compile_transplant(source, donor_state, cut, other)
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                key = f"raised:{type(error).__name__}"
                rejections[key] = rejections.get(key, 0) + 1
                continue
            status = str(result.get("status", "unknown"))
            if status != "compiled":
                rejections[status] = rejections.get(status, 0) + 1
                continue
            primitives = int(result.get("primitive_steps", 10**6))
            retention = 1.0 - float(result.get("released_fraction") or 0.0)
            added = int(result.get("added_atoms") or 0)
            endpoint = result.get("smiles")
            if primitives > MAX_TRANSPLANT_PRIMITIVES:
                rejections["over_primitive_ceiling"] = (
                    rejections.get("over_primitive_ceiling", 0) + 1)
                continue
            if retention < MIN_SOURCE_RETENTION:
                rejections["destroys_source"] = rejections.get("destroys_source", 0) + 1
                continue
            if added < MIN_DONOR_ATOMS:
                rejections["donor_contribution_trivial"] = (
                    rejections.get("donor_contribution_trivial", 0) + 1)
                continue
            if not endpoint:
                rejections["no_endpoint"] = rejections.get("no_endpoint", 0) + 1
                continue
            accepted.append(TransplantProposal(
                endpoint=endpoint, primitives=primitives, retention=retention,
                removed_atoms=int(result.get("removed_atoms") or 0), added_atoms=added,
                donor_endpoint=donor_smiles,
                actions=tuple(result.get("actions") or ()),
                states=tuple(result.get("states") or ()),
            ))
    return accepted, {"attempted": tried, "rejections": rejections}


def transplant_program(source, donors, rng, *, attempts: int, max_primitives: int = 32,
                       max_blocks: int = 8, exclude: set[str] | None = None):
    """A transplant as an `(source, EditProgram, assignment, metadata)` tuple.

    The controller's recombination contract is a PROGRAM, not an action list, so the
    accepted transplant is replayed through the production executor and extracted the
    same way every generic module is.  That also re-proves the actions execute on this
    exact source rather than trusting the compiler's own replay.
    """
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.control.dynamic_program_synthesis import _stage
    from compose_v4.control.edit_program import extract_program
    from compose_v4.experiments.whole_ring_plan import execute_program

    accepted, telemetry = propose_transplants(source, donors, rng, attempts=attempts)
    if not accepted:
        raise ValueError(f"no admissible transplant: {telemetry}")
    # Largest donated region first; ties by fewest primitives.
    accepted.sort(key=lambda p: (-p.added_atoms, p.primitives))
    # MEASURED: this ranking is deterministic, so a recurring parent against a slowly
    # changing archive returns the SAME transplant every round -- 51.3% of donor attempts
    # were duplicates against 5.7% for every other proposal. `exclude` carries the
    # endpoints this source has already produced, so the lane walks down its own ranking
    # instead of re-offering its top pick forever. The chemistry is untouched: the
    # candidate set, the ordering and the ceiling are exactly as before.
    seen = exclude if exclude is not None else set()
    duplicates = 0
    for proposal in accepted:
        try:
            product, receipt = execute_program(source, list(proposal.actions))
        except (ValueError, RuntimeError, KeyError, IndexError):
            continue
        endpoint = molecular_graph_to_smiles(product)
        if endpoint is not None and endpoint in seen:
            duplicates += 1
            continue
        stage = _stage("prescreen_transplant", receipt, {
            "donor_endpoint": proposal.donor_endpoint,
            "added_atoms": proposal.added_atoms,
            "removed_atoms": proposal.removed_atoms,
            "retention": round(proposal.retention, 4),
            "transplant_primitives": proposal.primitives,
        })
        try:
            program, assignment = extract_program(source, [stage])
        except ValueError:
            continue
        if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
            continue
        if endpoint is not None:
            seen.add(endpoint)
        return source, program, assignment, {
            "prescreen_transplant": {
                "donor_endpoint": proposal.donor_endpoint,
                "added_atoms": proposal.added_atoms,
                "removed_atoms": proposal.removed_atoms,
                "retention": proposal.retention,
                "primitives": proposal.primitives,
                "endpoint": endpoint,
                "skipped_as_duplicate": duplicates,
                "telemetry": telemetry,
            }
        }
    raise ValueError(
        f"transplants compiled but none were new: {duplicates} duplicates, {telemetry}")
