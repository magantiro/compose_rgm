"""Why bounded T4 search cannot reach the region where the docking value lives.

The measured chain this module implements, in order:

1. The 77 exact teacher routes decompose into a median of two dependency regions
   of about eight primitives each, and 65 of 77 reuse a handle they created. They
   are a few coordinated transitions, not twenty-one independent decisions.
2. 76 of 77 have an ineligible internal prefix (median 11), so those endpoints are
   not reachable as a chain of individually eligible small programs. A program has
   to span a whole region, which the declared 32-primitive/8-block support allows.
3. From the benchmark root the value lives at 15-22 primitives, and the winning
   shape is constructive: PARP1's best is 22 primitives touching 6 original atoms
   and creating 14.
4. `synthesize_dynamic_program` caps at three modules and orders its thirteen
   families by uniform weight, only five of which create anything, so it lands at
   a median of 5.5 primitives by scattering edits.
5. Composing deeper reaches the size band but the endpoint gate closes: eligibility
   falls off a cliff between 6-9 and 10-14 primitives and is zero beyond it.
6. Yet the region is densely populated historically: 479 eligible >=15-primitive
   constructions were applied directly to the PARP1 root alone.

So the failure is reachability, not capability. Nothing here fits a model, calls an
oracle, or changes any production sampler; `compose_deep` is a new composition law
over the unchanged production pieces, offered for measurement.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

import numpy as np

from compose_v4.control.constructive_composition import (
    DEFAULT_BUILDERS,
    MAX_BLOCKS,
    MAX_PRIMITIVES,
    compose,
)

SCHEMA_VERSION = "t4_construction_scale_v1"

# Descriptive bands. Chosen after inspecting the corpus, so they describe the
# measurement rather than preregistering it; the prospective gate is separate.
SIZE_BANDS = ((1, 2), (3, 5), (6, 9), (10, 14), (15, 22), (23, MAX_PRIMITIVES))


def band_of(primitives: int) -> str:
    for low, high in SIZE_BANDS:
        if low <= primitives <= high:
            return f"{low}-{high}"
    return f"{SIZE_BANDS[-1][1]}+"


# ---- What each generic family does to a molecule ----


def family_profile(records, *, minimum: int = 30) -> dict:
    """Median effect of each family, from records that used exactly that one family.

    Isolating single-family records is what separates a family's own effect from
    whatever it was composed with.
    """
    profile = {}
    for family in sorted({f for record in records for f in record["construction_families"]}):
        group = [r for r in records if r["construction_families"] == [family]]
        if len(group) < minimum:
            continue
        profile[family] = {
            "records": len(group),
            "median_primitives": statistics.median(r["primitive_count"] for r in group),
            "median_changed_originals": statistics.median(r["changed_slot_count"] for r in group),
            "median_created": statistics.median(r["net_created"] for r in group),
            "median_deleted": statistics.median(r["net_deleted"] for r in group),
        }
    return profile


def builder_families(profile: dict, universe=None) -> tuple[str, ...]:
    """Families that build: they create at least as many atoms as they disturb.

    Scattering edits across the original molecule is what breaks the similarity
    gate, so this is the axis that matters for reaching a large eligible program.
    """
    chosen = sorted(
        family
        for family, row in profile.items()
        if row["median_created"] > 0 and row["median_created"] >= row["median_changed_originals"]
    )
    if universe is not None:
        chosen = sorted(set(chosen) & set(universe))
    return tuple(chosen)


# ---- Teacher-route decomposition ----


def route_decomposition(descriptors) -> dict:
    """Region structure of the exact winning routes; the unit a program must span."""
    regions = [d["dependency_regions"] for d in descriptors]
    primitives = [d["primitives"] for d in descriptors]
    per_region = [n for d in descriptors for n in (d.get("region_primitives") or [])]
    prefixes = [d["ineligible_internal_prefixes"] for d in descriptors]
    return {
        "routes": len(descriptors),
        "regions_per_route": dict(sorted(Counter(regions).items())),
        "median_regions": statistics.median(regions),
        "median_primitives": statistics.median(primitives),
        "primitives_per_region": {
            "n": len(per_region),
            "p25": float(np.percentile(per_region, 25)),
            "median": float(np.percentile(per_region, 50)),
            "p75": float(np.percentile(per_region, 75)),
            "p90": float(np.percentile(per_region, 90)),
        },
        "median_created_handles": statistics.median(d["created_handles"] for d in descriptors),
        "median_reused_handles": statistics.median(
            d["reused_created_handles"] for d in descriptors
        ),
        "routes_reusing_a_created_handle": sum(
            1 for d in descriptors if d["reused_created_handles"] > 0
        ),
        "median_ineligible_internal_prefixes": statistics.median(prefixes),
        "routes_with_an_ineligible_interior": sum(1 for p in prefixes if p > 0),
        "interpretation": (
            "a route whose interior is ineligible cannot be walked as a chain of "
            "individually eligible small programs; one program must span the region"
        ),
    }


# ---- Where the value lives ----


def score_by_band(records, cells) -> dict:
    """Score distribution by program size, within the named cells."""
    rows = [r for r in records if r["cell"] in cells]
    result = {}
    for low, high in SIZE_BANDS:
        band = [r for r in rows if low <= r["primitive_count"] <= high]
        if not band:
            continue
        scores = sorted(r["score_mean"] for r in band)
        result[f"{low}-{high}"] = {
            "n": len(band),
            "best": scores[0],
            "p5": float(np.percentile(scores, 5)),
            "median": float(np.percentile(scores, 50)),
        }
    return result


def root_construction_density(records, roots, *, threshold: int = 15) -> dict:
    """Eligible large constructions applied directly to each benchmark root.

    Every corpus record is eligible by construction, so this measures how densely
    the historical search populated the region, and proves it is not empty.
    """
    result = {}
    for cell, root in sorted(roots.items()):
        rows = [r for r in records if r["cell"] == cell and r["input_state_sha256"] == root]
        if not rows:
            continue
        large = [r for r in rows if r["primitive_count"] >= threshold]
        result[cell] = {
            "root_applied": len(rows),
            "at_or_above_threshold": len(large),
            "share": len(large) / len(rows),
            "distinct_endpoints": len({r["endpoint_sha256"] for r in large}),
            "best": min((r["score_mean"] for r in large), default=None),
            "p5": float(np.percentile([r["score_mean"] for r in large], 5)) if large else None,
            "median": float(np.percentile([r["score_mean"] for r in large], 50)) if large else None,
        }
    return result


# ---- The deep constructive composition law ----

# The law itself lives in control/, next to the other proposal code, because it is a
# proposal law rather than a diagnostic. It is re-exported here under the name this
# module's tests and report already use, so there is exactly one implementation.
compose_deep = compose


def eligibility_by_size(
    source, eligibility, rng, *, attempts: int, constructive: bool, builders=None
) -> dict:
    """Eligible rate as a function of realized program size. Zero oracle calls."""
    bands = defaultdict(lambda: {"n": 0, "eligible": 0})
    rejections = Counter()
    for _ in range(attempts):
        depth = int(rng.integers(1, MAX_BLOCKS + 1))
        try:
            trace, _ = compose_deep(
                source, rng, depth, constructive=constructive, builders=builders
            )
        except (ValueError, RuntimeError):
            continue
        properties = eligibility({"smiles": trace["endpoint"]})
        band = bands[band_of(len(trace["actions"]))]
        band["n"] += 1
        if properties.get("oracle_eligible"):
            band["eligible"] += 1
        else:
            for reason in properties.get("endpoint_exclusion_reasons") or []:
                rejections[reason] += 1
    return {
        "bands": {
            key: {**value, "rate": value["eligible"] / value["n"]}
            for key, value in sorted(bands.items())
        },
        "rejections": dict(rejections.most_common()),
        "attempts": attempts,
        "constructive": constructive,
        "builders": list(DEFAULT_BUILDERS if builders is None else builders),
        "new_oracle_calls": 0,
    }


def rule_of_three_upper_bound(trials: int) -> float:
    """95% upper bound on a rate after observing zero successes."""
    if trials <= 0:
        raise ValueError("an upper bound needs at least one trial")
    return 3.0 / trials


def upper_bound_95(successes: int, trials: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound on a rate.

    A rate this small must always be reported with its bound: one success in 249 is
    not zero, and quoting either as a bare point estimate overstates what the sample
    settles. At zero successes this reduces to the rule of three.
    """
    from scipy.stats import beta

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("an upper bound needs 0 <= successes <= trials and trials > 0")
    if successes == trials:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, trials - successes))
