# SA census — model-free half: source applicability

**Status: `SMOKE_HELD_IN`.** Held-out never opened. No kernel, no `R_θ`, no
oracle, no Gate-0. CPU only, nothing installed.

**Artifact:** `diagnostics/constraints_hard_sa_applicability.json`
**Producer:** `scripts/constraints_hard_sa_applicability.py`
**τ rule:** fixed in `SA_CENSUS_PROTOCOL.md` and **committed at `034c294`,
before this measurement existed.**

---

## 0. Headline

**The SA constraint is far more applicable than the scaffold was, and the median
source sits almost exactly on the primary threshold.**

| | Bemis–Murcko scaffold | **SA ≤ 3.0** |
|---|---:|---:|
| eligible fraction, held-in pool | 22.03% | **49.79%** |

But the number that decides this constraint's fate is the second one:

> **At the primary threshold, the median eligible source has only 0.11 SA units
> of slack.**

That is the constraint sitting on the boundary — which is exactly what makes it
*bite*, and exactly what threatens the retention floor. §3.

## 1. Results — held-in pool, n = 96,094

`training_source_keys` only; `reserve_source_keys` is popped from the payload
before scoring, so the held-out reserve is structurally not opened.

**SA distribution:** median **2.8903**, mean 3.1591, p10 2.0987, p90 4.6829,
range [1.0687, 8.5187].

| τ | satisfies `SA(x₀) ≤ τ` | **eligible** (+ frozen size band) | median slack `τ − SA(x₀)` |
|---|---:|---:|---:|
| **3.0 — PRIMARY** | 54.37% | **49.79%** | **0.1097** |
| 3.5 | 70.15% | 64.12% | 0.6097 |
| 4.0 | 80.94% | 73.60% | 1.1097 |
| 4.5 | 88.04% | 79.75% | 1.6097 |

### Held-in developability cohort, n = 30

Median SA 2.6724, mean 3.2015, range [1.7995, 6.3987]. Eligible: 63.3% / 70.0% /
76.7% / 83.3% across the four τ. Directionally consistent with the pool; the
cohort skews slightly more accessible.

## 2. What this establishes, and what it does not

**Establishes:** source applicability. At the primary threshold **roughly half**
the held-in pool is eligible — ~47,800 sources, far more than any panel needs.
The predicate is not applicability-limited the way the scaffold was.

**Does not establish anything about the fiber.** Not violation prevalence, not
retained support, not mask-empty rate, not headroom, not recoverability. All five
gates in `SA_CENSUS_PROTOCOL.md` §5 remain **unmeasured**. This is not the census
verdict; it is the half that runs without the kernel.

## 3. The decisive number, and the risk it creates

**Median slack at τ = 3.0 is 0.1097 SA units.** The median eligible source is
essentially *on* the threshold.

This cuts hard in both directions, and the protocol predicted the direction
before the number existed:

| gate | effect of near-zero slack | reading |
|---|---|---|
| **V3 nonvacuity** (≥ 20 events) | **strongly helped** — almost any complexity-increasing edit crosses τ | the constraint will very likely bite |
| **V4a retained support** (≥ 0.10) | **strongly threatened** — if most successors add complexity, most get masked | the live failure mode |
| **V4b mask-empty** (≤ 0.05) | **threatened** for the same reason | — |
| **headroom inside `F_C`** | **threatened** — improvement may require complexity | — |

> ### The outcome that must be declared in advance
>
> **It is possible that NO τ passes all gates simultaneously.** Tight τ passes
> nonvacuity and fails retention; loose τ passes retention and fails nonvacuity.
> If the whole curve behaves that way, **the constraint dies** — that is the kill
> rule in `SA_CENSUS_PROTOCOL.md` §6, and it is declared here, before the fiber
> census, so it cannot later look like a discovery that happened to justify a
> rescue.
>
> Declaring it now is the point. A gate whose failure mode is only recognized
> after the data arrives is a gate that gets renegotiated.

## 4. The uncertainty this cannot resolve, and why I am not proxying it

Whether 0.11 units of slack is "one edit from violating" or "twenty" depends on
**how much SA moves per legal rewrite** — a property of the successor fiber,
not of the source distribution.

**I did not build a matched-pair proxy for it.** The project rule is explicit:
*"Do not use one-cut matched-pair proxies for load-bearing reachability or
control-geometry conclusions when exact successor fibers are available."* It was
adopted after a proxy said a DRD2 threshold was out of reach in four edits and
the real fiber climbed +4.42 log-odds in three. A per-edit ΔSA estimate is
exactly that kind of load-bearing reachability claim, and the honest move is to
leave it to the fiber census rather than to anticipate it with a cheaper
instrument.

**So: the single highest-value quantity the fiber census returns is the
distribution of ΔSA per legal rewrite.** Everything about whether this constraint
is workable follows from it.

## 5. Applicability cost, reported not hidden

At the primary threshold **~46% of held-in sources are excluded** because they
already violate `SA(x₀) ≤ 3.0`. That is a real scope limitation and it belongs in
the paper:

> The SA-constrained result is scoped to sources that already satisfy the
> constraint — **49.8% of the held-in pool at τ = 3.0** — because the constraint
> is a *maintenance* requirement (do not make the molecule harder to synthesize)
> rather than a *repair* one.

Compare the scaffold disclosure at 22.03%. This is a materially better position,
and it is the same kind of honest scoping either way.

## 6. Reproduction

```bash
python3 scripts/constraints_hard_sa_applicability.py \
    --cohort diagnostics/retarget_calibration_cohort.json \
    --pool   diagnostics/editing_v2_matched_validation_reserve_ids.json.gz \
    --out    diagnostics/constraints_hard_sa_applicability.json
```

RDKit **2025.09.6** locally; production pin **2024.03.5**. The artifact carries
`rdkit_pin_matches_production: false`.

**This matters more here than it did for the scaffold.** CDD pins no RDKit
version, and SA values shift with the fragment-contribution tables, so the τ
boundaries are **not version-portable**. With median slack at 0.11, a small
version-to-version shift in `sascorer` could move a non-trivial number of sources
across the eligibility line. **Any claim-bearing run uses the pinned shared
evaluator, and the fiber census should re-report applicability under the pin as a
cheap consistency check.**
