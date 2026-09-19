"""Deep constructive composition: a high-volume proposal law over unchanged pieces.

Bounded T4 search proposes a median of 5.5 primitives while the docking value from a
benchmark root sits at 15 or more. Two measured facts set the design:

* `synthesize_dynamic_program` caps at three generic modules and orders its thirteen
  families by uniform weight, only six of which create anything, so programs grow by
  scattering edits across the original molecule and the similarity gate refuses them.
* Composing deeper does reach the size band, but the endpoint gate closes: roughly one
  proposal in two hundred is eligible at 15 or more primitives.

The second fact is a cost, not a wall. A proposal costs no oracle call, and eligibility
is an exact local computation, so the docking budget is only ever spent on survivors.
This module therefore generates in volume and filters for free, rather than trying to
make every draw count.

Nothing here modifies v0. `synthesize_dynamic_program` keeps its three-module cap and
its lane is untouched; this is a separate law built from the same production pieces,
inside the declared 32-primitive/8-block support.
"""

from __future__ import annotations

from collections import Counter
from time import perf_counter

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    _weighted_module_order,  # the production family-order law, reused rather than copied
    compile_generic_module,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph

# The declared per-program support. Read from the contract; never widened here.
MAX_PRIMITIVES = 32
MAX_BLOCKS = 8

# Families whose median effect creates at least as many atoms as it disturbs, measured
# over single-family records and restricted to v0's own thirteen. A fallback only: the
# caller should pass the set derived from the corpus it is actually working with.
DEFAULT_BUILDERS = ("append_ring", "fuse_ring", "functionalize", "segment_grow")


def family_order(rng, builders, *, constructive: bool) -> list[str]:
    """The production family order, optionally with builders moved to the front.

    This changes which family is *tried first*, never which families exist, so the
    supported action set is identical to v0's.
    """
    order = _weighted_module_order(rng, near_capacity=False)
    if not constructive:
        return order
    preferred = set(builders)
    return [f for f in order if f in preferred] + [f for f in order if f not in preferred]


def compose(source, rng, modules: int, *, constructive: bool = True, builders=None):
    """Compose up to `modules` generic modules into one program.

    Returns the executed trace and the families used. Raises `ValueError` when no
    module executes at this source, which is a legitimate outcome and not an error
    condition for the caller to hide.
    """
    if not isinstance(modules, int) or not 1 <= modules <= MAX_BLOCKS:
        raise ValueError(f"composition depth must fit the declared {MAX_BLOCKS}-block support")
    builders = DEFAULT_BUILDERS if builders is None else tuple(builders)
    current, stages = source, []
    for _ in range(modules):
        accepted = None
        for family in family_order(rng, builders, constructive=constructive):
            try:
                product, stage = compile_generic_module(current, rng, family)
                program, _ = extract_program(source, [*stages, stage])
            except ValueError:
                continue
            if len(program.marks) > MAX_PRIMITIVES or len(program.blocks) > MAX_BLOCKS:
                continue
            accepted = (product, stage)
            break
        if accepted is None:
            break
        current, stage = accepted
        stages.append(stage)
    if not stages:
        raise ValueError("no generic module executed at this source")
    program, binding = extract_program(source, stages)
    _, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        binding,
        max_primitives=MAX_PRIMITIVES,
        max_blocks=MAX_BLOCKS,
    )
    return trace, tuple(stage["name"] for stage in stages)


def prospect(
    source,
    eligibility,
    rng,
    *,
    attempts: int,
    minimum_primitives: int = 15,
    depths=(5, 6, 7, 8),
    constructive: bool = True,
    builders=None,
    wall_seconds: float | None = None,
    progress=None,
):
    """Generate in volume, keep only eligible programs at or above a size floor.

    Zero oracle calls: eligibility is the frozen endpoint gate, computed locally. The
    returned pool is deduplicated by canonical endpoint, because two programs reaching
    one molecule are one docking candidate.

    `attempts` is the budget that binds. `wall_seconds`, when given, is a disclosed
    stopping budget and the realized attempt count is always reported.
    """
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("prospecting needs a positive attempt budget")
    if not depths or any(not 1 <= d <= MAX_BLOCKS for d in depths):
        raise ValueError(f"every depth must fit the declared {MAX_BLOCKS}-block support")
    began = perf_counter()
    kept: dict[str, dict] = {}
    sizes, rejections, families = Counter(), Counter(), Counter()
    realized = failed = below_floor = 0
    for index in range(attempts):
        if wall_seconds is not None and perf_counter() - began >= wall_seconds:
            break
        realized += 1
        if progress is not None:
            # Report on attempts, not only on successes: a run that finds nothing must
            # still be distinguishable from a hung one.
            progress({"attempts": realized, "eligible_endpoints": len(kept)})
        depth = depths[int(rng.integers(len(depths)))]
        try:
            trace, used = compose(source, rng, depth, constructive=constructive, builders=builders)
        except (ValueError, RuntimeError):
            failed += 1
            continue
        primitives = len(trace["actions"])
        sizes[primitives] += 1
        if primitives < minimum_primitives:
            below_floor += 1
            continue
        properties = eligibility({"smiles": trace["endpoint"]})
        if properties.get("oracle_eligible") is not True:
            for reason in properties.get("endpoint_exclusion_reasons") or []:
                rejections[reason] += 1
            continue
        families.update(used)
        kept.setdefault(
            trace["endpoint"],
            {
                "endpoint": trace["endpoint"],
                "primitives": primitives,
                "families": list(used),
                "properties": properties,
                "changed_originals": len(trace["actual_changes"]["changed_original_slots"]),
                "created": trace["actual_changes"]["surviving_new_atoms"],
                "trace": trace,
                "found_at_attempt": index,
            },
        )
    return {
        "schema_version": "constructive_prospect_v1",
        "pool": sorted(kept.values(), key=lambda row: (-row["primitives"], row["endpoint"])),
        "attempted": realized,
        "attempt_budget": attempts,
        "build_failures": failed,
        "below_size_floor": below_floor,
        "eligible_endpoints": len(kept),
        "eligible_per_attempt": len(kept) / realized if realized else 0.0,
        "attempts_per_eligible_endpoint": realized / len(kept) if kept else None,
        "minimum_primitives": minimum_primitives,
        "size_histogram": dict(sorted(sizes.items())),
        "rejections": dict(rejections.most_common()),
        "realized_families": dict(sorted(families.items())),
        "elapsed_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
        "pool_id": identity(sorted(kept)),
    }
