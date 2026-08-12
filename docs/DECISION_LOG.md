# Decision Log — what is established, and what was refuted

**Read this before re-running an experiment.**

This is not a replay of the wiped task board. Task titles are cheap to lose and
recoverable from git; what is expensive to lose is *which hypotheses were tested
and killed*, because nothing stops someone from testing them again. Six of the
agent's own hypotheses were refuted during this work, several published claims
were corrected, and none of that is legible from a list of commit subjects.

Scope: the `run_v2_01` line, 2026-08-10. Plan of record:
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

---

## Established

**Ordinary canonical-successor likelihood learns the declared reference process.**
Reference-law-weighted NLL falls from **5.4603** at the frozen initialization to
**2.9238** at the selected step 8,500 — 2.54 nats over two epochs.

**Within-family identity NLL improved in all eight families** against R_θ0:

| family | init → final | Δ |
|---|---|---|
| cycle_insert | 4.7147 → 0.9071 | −3.81 |
| atom_insert | 5.1526 → 3.2900 | −1.86 |
| atom_restate | 5.2047 → 3.3868 | −1.82 |
| bond_reroute | 3.6376 → 2.5431 | −1.09 |
| atom_delete | 1.6492 → 1.0052 | −0.64 |
| bond_reorder | 1.5457 → 1.3122 | −0.23 |
| ring_system_restate | 0.7139 → 0.5007 | −0.21 |
| cycle_attach | 2.5427 → 2.5360 | −0.01 |

**Improvement scales with training scaffold support**, over two epochs:
−0.22 / −0.21 / −0.38 / **−0.62** across bands 0 / 1–4 / 5–24 / 25+. Supportable
phrasing: *transfers to unsupported chemistry, benefits increasingly from denser
support*. **Not** supportable: a strict monotone-in-support claim — the two low
bands are tied within 0.015, and at the epoch-1 boundary they inverted.

**No cross-cohort cost.** External 16-shard cohort: 4.3364 → 4.5787 → 4.3817 →
4.3288 across steps 3,000 / 4,500 / 8,502 / 9,000. Flat to slightly improving
while in-population improved 0.20 nats.

**The metric's resolution is ~0.026 nats.** Steps 7,956 and 8,000 are 44
optimizer steps apart and differ by that much on a deterministic evaluation —
comparable to a typical 500-step gain. The pre-run freeze gate predicted 0.024
standard error. **Adjacent evaluations closer than ~0.03 nats are not
interpretable.** Four trend calls made from consecutive points were each
overturned at the next point.

**The initialization is bit-reproducible.** R_θ0 scored identically (0.00e+00
difference) across three fresh containers, including one following an
involuntary preemption.

---

## Refuted — do not re-test without new reason

| Hypothesis | Verdict | Evidence |
|---|---|---|
| The compact successor fiber is the object to optimize | **Wrong target** | Partitions are discarded by the training loop; only teacher fibers are read |
| Larger batches will speed training | **Refuted** | Batch 128 is 3× worse than 32 |
| Vectorized backward pass gives a large speedup | **Refuted** | 2%, not 20× |
| The model memorizes slot permutations | **Refuted** | Scorer is equivariant to 1e-6 |
| The gap is a representation problem | **Refuted** | Provenance correlation ρ = +0.762 |
| Validation is a different chemical cohort | **Refuted at role level** | 85.5% scaffold coverage — *but see the correction below* |
| Held-out error grows with analogue-series depth | **Refuted, opposite direction** | Error *falls* 3.87 → 2.42 with depth, surviving both fixed-zero-support (n=5,863) and fixed-molecule-size (n=6,992) controls |
| The most-oversampled strata overfit first | **Refuted** | cycle_attach, bond_reorder, ring_system_restate sit at the 3.0× oversample ceiling and improved most |
| `atom_restate` suffered capability collapse | **Refuted** | Its *identity* improved 1.82 nats; the joint gate was reading family-head reallocation |
| Cross-cohort performance is degrading | **Refuted** | Third and fourth readings returned to baseline |
| DRD2 is rugged enough that future value beats greedy | **Refuted** | C0: mean fresh regret **negative at every depth**; sacrificial win rate 57.1% at **0.5 SE** from chance. Greedy solved **7/12** sources — outcomes are bimodal (1–5 edits, or never close), leaving planning nothing to buy |
| Historical traces can be reconstructed from the corpus | **Refuted, structurally** | The corpora are samples of `(state, edit)` pairs across capability cells, not paths: **29,928 forked states** in train, 2,380 in a 14,140-entry panel. Strict funnel reached **0** on both partitions |
| The `linker_positional_topology_analogue` lane implies two-cut/topology mining | **Refuted** | It is a **post-hoc router label** for candidates whose operator family is in `{bond_reroute, cycle_insert, cycle_attach, ring_system_restate}`. No two-cut or ring-analogue miner exists — zero hits for `two_cut`/`double_cut`/`n_cuts` across the repo |

---

## Established — when planning matters, and when it does not

Two results that read as contradictory are actually a characterisation.

**DRD2 (C0): planning did not help.** Mean fresh regret negative at every depth,
sacrificial win rate 0.5 SE from chance, greedy solved 7/12 sources. Outcomes
were bimodal — solved in 1–5 edits or never close — so a locally easy landscape
left planning nothing to buy.

**Known-reachable target recovery (C): planning rescued half of greedy's
failures.** 12/24 → 18/24 exact recovery; **6 of the 12 targets greedy missed
were recovered**, none lost; best similarity 0.8589 → 0.9335; cost 5.2×.

Together these characterise *when* remaining-budget information matters, rather
than searching for a benchmark where the method wins. That framing is stronger
than either result alone.

**Do not headline the p-value.** McNemar one-sided p = 0.0156, but the rollout
policy is constructed to be no worse than its greedy base under the value used
for improvement, so `greedy-only = 0` is a structural guarantee, not an
observation. The scientific effect is the 50% rescue rate among greedy failures.

**Two claims the panel cannot support**: any horizon trend (+3 / 0 / +2 at 4/5/6
steps) and any mechanism story — every pair is a delete-insert fragment swap.

---

## The amortisation gap — why a learned value is not a cheap planner

Explicit full-horizon rollout rescues half of greedy's failures (18/24 vs
12/24, six rescues, **zero** losses, 5.2x cost). A learned `h_phi` trained on
14,110 teacher labels from that exact teacher **does not retain the benefit**.

| data | rescue ranking | harmful override | contrastive |
|---|---|---|---|
| 2k | 30.5% ± 7.8% | 21.5% ± 1.2% | 52.3% |
| 5k | 35.2% ± 6.9% | 17.2% ± 1.3% | 59.3% |
| 10k | 42.2% ± 6.0% | 13.8% ± 2.1% | 65.9% |
| all | 44.5% ± 6.9% | 16.0% ± 2.1% | 63.6% |

Baselines — rescue: greedy 0% by construction, random 20.8%,
second-highest-similarity 33.3%. Harmful: greedy 0%, random 27.0%.

`h_phi` learns real signal: every metric beats its baseline and improves
monotonically with data. But **no override threshold yields net benefit** —
−10.2 states at margin 0, −2.2 at 1.5, −0.2 at 4.0, and the sole positive
(+0.5 at margin 6.0) is where it overrides 9 of ~415 states and has stopped
acting.

**The structural cause, which generalises beyond this model.** States where
greedy is already safe (158) outnumber states needing rescue (36) by 4:1, so
**precision dominates recall** and a learned value must be very precise before
overriding pays. Explicit rollout never faced this: policy improvement
guaranteed it could never harm. That guarantee is doing far more work than
"slower but equivalent" suggests.

**Not saturated.** Rescue ranking still climbs at the largest subset, so
data/capacity limitation and genuine unlearnability are both consistent with
this evidence. This experiment cannot separate them, and the writeup should not
imply otherwise.

---

## Structural limit — one-cut MMP cannot produce diverse multi-step panels

Measured while building the fresh evaluation panel, and the reason the
non-locality experiment is on hold rather than running.

Compile outcomes over 972 nominated pairs: 160 `bond_reroute` + 146
`atom_restate` + 17 `bond_reorder` = **323 direct compiles**, against **exactly
323 paths of length 1** in the length distribution. Every direct compile is a
single step; everything of length ≥2 is `delete_insert_fallback`. There is no
middle.

So **every 4–6 step transformation this machinery can produce is a delete-insert
fragment swap by construction.** Consequences, all observed:

- `ring_delta` is **0 across all 91** accepted pairs — `iter_one_cut_transformations`
  skips ring bonds and requires an acyclic variable fragment, so ring systems
  can never change.
- The 86% `REFERENCE_DIP` rate is a **compiler artifact**: delete-then-insert
  removes the fragment before adding its replacement, so it mechanically dips.
  A dip stratum built this way is confounded with path-construction strategy.

The 91 pairs remain valid as a **held-out multi-step recovery cohort** — both
endpoints held out (reserve keys are disjoint from the 96,094 training-source
universe), fresh post-split compilation, replay-verified endpoints, and 99 pairs
rejected against 130,432 supervised transitions. They support exact target
recovery under a declared budget over a *single* transformation class. They do
not support any non-locality claim.

Provenance claim to make: **held-out endpoints plus unseen supervised
transitions.** Not "the model never saw any intermediate" — 38 of 91 have an
intermediate appearing elsewhere as a training source, which is described rather
than excluded.

---

## Corrections to claims that were made and then found wrong

- **Scaffold coverage is 58.5% of panel entries, not 85.5%.** The 85.5% figure counted train-*role* sources across 30 Active8 shards; the model trains on the compiled **library**, 106,759 of 564,316 role sources. This reversed a retraction: chemical novelty *is* substantial from the model's point of view.
- **Support-matching does not flatter the number.** An earlier note claimed 2.62 → 1.94; that conflated the matched view with the supported-scaffold view (1.94 is band 25+ alone). Training weights on the same per-band means give 2.74 — matching moves the number **+0.12**, slightly worse. The re-freeze buys an interpretable number, not a better one.
- **Depth 0 meant acyclic, not shallow.** All 361 depth-0 entries were molecules with an empty Murcko scaffold; `if s:` treated the empty string as no-scaffold. A category error at the exact end of the curve the hypothesis was about.
- **The family floor was never enforced.** Clamping to [floor, cap] then renormalizing divides by the sum, so a family pinned *at* the floor lands under it. The pilot's law claimed 0.05 and delivered 0.0497. Replaced with water-filling.
- **The joint NLL is the wrong quantity for a capability gate.** It moves with family-head reallocation the model is entitled to perform. Across eight families the largest identity movement was 0.062 nats against family movement of 0.317. The gate now reads identity, over two consecutive evaluations.

---

## Defects found in the harness itself

These cost real money or hid real results, and are fixed.

- **No evaluation ran on the final step.** One epoch is 4,251 steps against an interval of 500, so runs ended with their last evaluation 251 steps stale. The final model state was never scored and could never be selected — this hid step 4,000, whose weights are now permanently lost, and step 4,251, which was eligible and better than the then-selected checkpoint.
- **The eval line logged everything except the number that selects.** Panel-native, deployment and minimum probability were printed; reference-law-weighted NLL and the gate verdict were not.
- **Re-collating per batch cost 98.17 s per 32 examples** — 97.2% of step time. Fixed by the packed store; GPU data-wait fell to 0.93%.
- **Three earlier logging defects:** throughput charged eval and checkpoint time, producing a false WARN at every eval; the alarm watched only panel-native and missed a 9.2% deployment regression; the per-family breakdown was computed and discarded.
- **The carve was defined over the wrong set** — 151,078 entries where the loader yields 151,059; 12 of 97 precedence-held-out source keys were live, 2 of them in the reserve.
- **Four byte-identical duplicate entry ids** would have made the artifact claim 15,031 reserve rows against a store holding 15,029.
- **Cost estimates counted work, not billed resources.** C0 was quoted at ~$0.50 from kernel-call arithmetic; Modal bills **memory-time**, which dominated, and the real figure for the config as written was ~$7 — no cheaper than the experiment it was meant to gate. Fixed structurally, by removing the corpus load from the probe containers rather than by re-quoting. Assume this failure mode until a quote is built from allocation × wall-time.
- **`compose_v4/oracles/__init__.py` eagerly imports `joblib`.** Putting a new module in that package makes every consumer pay for the unrelated pan-lung oracle stack, which is absent from the Modal image. Caught by a 2-minute smoke test rather than 40 minutes into a 12-way fan-out; the DRD2 oracle now lives at `compose_v4.drd2_oracle`.

---

## Oracle provenance — DRD2

The classic benchmark SVM ships as a Python-3.6 sklearn pickle. It is opened
**once**, by `scripts/drd2_oracle_extract.py`, and the runtime thereafter
evaluates frozen arrays in numpy — no sklearn import, no unpickling.

Two things had to be reproduced rather than assumed:

- **libsvm's Wu-Lin-Weng coupling**, not just the Platt sigmoid. The exact fixed
  point of that iteration *is* the sigmoid, but libsvm stops at
  `max_error < 0.005/k`. Short-circuiting it left a **1.7e-3** discrepancy —
  enough to move molecules sitting on the 0.5 success threshold, which is
  exactly where success rates are decided.
- **The Platt orientation.** An inverted oracle returns entirely plausible
  probabilities in [0,1] while rewarding the wrong molecules. Both orientations
  are scored against the original estimator and the agreeing one is pinned; the
  rejected one is asserted to *fail* parity, so agreement is evidence rather
  than luck.

Parity: **2.19e-14** on probabilities, **1.35e-13** on decision values, over a
200-molecule panel. Discrimination: source panel max **0.0482** (independently
confirming it is the `<0.05` set the benchmark uses), active panel **300/300**
above 0.5 — a constant scorer would pass parity on inactives alone.

Note the task uses **two different fingerprints**: activity is scored with
count-based FCFP6 (radius 3, `useFeatures=True`, folded by modulo), while the
similarity constraint uses ECFP4 bits (radius 2, 2048). Conflating them would
silently redefine the constrained benchmark.

---

## Identity chain — `run_v2_01`

```
reserve   b580fdef6486     15,031 reserve / 136,028 training / 151,059 total
law       b0cc66f168f1     7 gates; synthetic 21.0%; family floors exact at 5.00%
manifest  e27494a46250     5 gates; drift ≤0.003%; sequence 136,027
store     b232a6fa069f     unchanged — already covered the split exactly
gate      FROZEN 9/9       primary-metric standard error 0.0237 (bound 0.05)
```

**The law never draws 29,600 of 136,028 rows** — whole synthetic strata in
families whose real supply already meets target: `atom_delete|synthetic` and
`atom_insert|synthetic` at exactly 0%, `atom_restate|synthetic` 99.6%,
`bond_reroute|synthetic` 97.3%. The sequence is fixed, so this is **permanent
exclusion, not slow exposure** — no number of epochs reaches them. Those strata
degraded +0.80 to +1.79 nats during training and are reported as diagnostics,
never weighted into selection.

Cost to date ≈ **$2.6** on A10G (preemptible), against a $4 cap.
