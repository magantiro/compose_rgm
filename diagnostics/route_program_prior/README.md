# Offline structural prior from teacher routes — held-out result

**Verdict: do NOT put this prior underneath PMO as a complete-program proposal.**
It does not beat the generic control on complete-program yield on held-out
sources, and its region-ranking advantage is not distinguishable from chance at
this corpus size. One component is worth keeping, for a different reason, and is
scoped at the bottom of this file.

Zero oracle calls, zero docking calls, zero Modal launches. Pinned kernel
throughout: python 3.11.13 / rdkit 2024.3.5 / numpy 1.26.4.

## What was trained

`pi_route(Z | G) = p(K | G) * prod_i p(F_i | G) * p(R | G)` — a parent-conditioned
prior over complete program structures: how many modules, which production
module families, and which bridge-separated region to release. `H`, `alpha` and
`D` are realized by the production synthesizer against the current parent.

Implementation: `src/compose_v4/control/route_program_prior.py`. It implements
the production region-law interface `order(graph, rng)`, so it is accepted
wherever `dynamic_program_synthesis` takes `region_law=`.

Every stored quantity is a ROLE computed on the molecule being proposed for —
relative region size, ring content, anchor environment, cut-bond order — never
an address or an elemental-neighbourhood fingerprint of a teacher molecule.
`assert_payload_is_address_free` enforces this rather than asserting it in
prose, and the test suite mutates a stored table to prove the check bites.

## Corpora, and the exclusion certified

ADMITTED — `diagnostics/ivg_winner_paths/pairs/*.json.gz`: 160 exact compiled
witness routes over 15 T4 lead-optimization source molecules, 3,070 primitive
transitions, 238 dependency-connected components, 213 bound region choices.
These are T4 routes. No PMO oracle value, task identity or selection appears in
them.

ADMITTED — `data/jin/hphi_train_1024.txt` (32 ZINC-derived molecules), used ONLY
to measure the production module families' emission matrix. That measurement
uses no teacher label; it is an instrument reading of the synthesizer.

REFUSED, and the fit script raises if pointed at them:
`diagnostics/pmo_route_distillation/` (184 routes; its own result file records
"All PMO supervision is answer-known, panel-informed or winner-informed
development evidence"), `diagnostics/pmo_teacher_route_gap_v1.json` (routes
indexed by PMO objective — gsk3b, celecoxib_rediscovery, albuterol_similarity),
and the winner-curriculum and winner-recovery directories.

**Finding worth acting on separately:** the existing
`t4_compositional_structural_subgoal_generator` trains on those PMO routes —
`training_trace_counts` is `{pmo_dependency_region: 85, t4_complete: 50}` in
fold 0 and 85 of 191/193 in folds 1 and 2, i.e. 44–63% PMO. Whatever its merits
as a T4 result, it is not admissible as a no-prescreen PMO prior.

## Held-out comparison

`heldout_comparison_v1.json`. 48 draws per parent per arm; module count drawn
once per (parent, draw) and shared by every declared arm, so no arm can win by
declaring shorter programs. 11,232 draws, 4,344 s.

Two populations. **t4_heldout** — the 15 T4 sources, each scored under the prior
fitted WITHOUT any route from that source (leave-source-out). **generic** — 24
held-out ZINC molecules under the train-on-all prior; no generic molecule
appears in any teacher route, so there is nothing to leave out. The generic
population is the one that speaks to PMO, whose parents are initialization-pool
molecules rather than docking leads.

### Complete-program yield — the headline

| arm | T4 held-out | generic |
|---|---|---|
| `production_v1` (shipped) | **1.000** | **1.000** |
| `production_route_law` | 0.994 | 0.997 |
| `declared_uniform` (control) | **0.850** | **0.962** |
| `declared_prior` | **0.749** | **0.929** |
| `declared_prior_families` | 0.831 | 0.971 |
| `declared_prior_region` | 0.786 | 0.936 |

Paired per-parent, `declared_prior` vs `declared_uniform`: T4 **−0.1014**
[bootstrap 95% −0.164, −0.042], sign test +3/−8, p = 0.227. Generic **−0.0330**
[−0.057, −0.006], +5/−17, **p = 0.017**.

Attribution is consistent across both populations: the **region law** is what
costs yield (T4 −0.064, p = 0.0034; generic −0.026, p = 0.0001), while the
**family projection is neutral** (T4 −0.019, p = 1.00; generic +0.010, p = 0.63).

Diversity moves the same way: `declared_prior` produces significantly FEWER
distinct endpoints per parent (generic −2.50, p = 0.0043; T4 −5.07, p = 0.057).

### Teacher recovery

Exact teacher endpoint recovery is **0 out of 11,232 draws** in every arm. No
arm was expected to recover one, and none did.

Teacher REGION recall (did a complete program release inside the bridge region
the teacher released from) is the coordinate the law is answerable for.
`declared_prior` 0.229 vs control 0.191 — but paired, +6/−8, p = 0.79. Not
established.

`region_ranking_v1.json` asks the same question directly, against an ANALYTIC
uniform control (exact hypergeometric — no sampling error on either side), over
15 leave-source-out sources with a median support of 16 regions and 3 teacher
targets:

| | learned | uniform | exact p |
|---|---|---|---|
| recall@1 | 0.267 (4 hits) | 0.211 (3.17 expected) | 0.39 |
| recall@3 | 0.533 (8) | 0.496 (7.45) | 0.48 |
| recall@5 | 0.867 (13) | 0.665 (9.97) | 0.054 |
| recall@10 | 0.933 (14) | 0.897 (13.46) | 0.52 |
| MRR | 0.471 | 0.410 | — |

One cell at p = 0.054 uncorrected across four comparisons. **The ranking
advantage is not established at n = 15 sources.**

### What the prior DOES change, and it is large

Region scale, mean over parents:

| arm | released atoms (mean) | released atoms (max) | primitives | changed fraction |
|---|---|---|---|---|
| `production_v1` | 1.44 | 8.33 | 5.33 | 0.188 |
| `production_route_law` | **3.86** | **21.00** | 8.22 | 0.268 |
| `declared_uniform` | 0.86 | 6.73 | 4.63 | 0.180 |
| `declared_prior` | **3.44** | **19.80** | 8.70 | 0.267 |

On the SHIPPED path the learned law multiplies mean excision scale by 2.7x and
maximum excision by 2.5x — from 8.3 atoms, which is the v1 cap, to 21.0 — at a
yield cost of 0.6 points that the sign test cannot separate from zero
(+0/−3/=12, p = 0.25). The measured T4 witness census says every clean eligible
witness in the five exhausted delta=0.6 cells is a 7–15 heavy-atom EXCISION, a
move class the v1 law cannot express in one draw.

## Reading this honestly

The central lesson this task was scoped around reappears one level down.
Recognition is not generation: the law is (weakly, unestablished) better than
chance at ranking the region a teacher released, and it is significantly WORSE
at producing a complete program. Teacher-like regions are larger, and larger
excisions are refused more often by the executor.

Limits that belong beside every number here:

* The teacher routes are COMPILED witnesses — a search reconstructed a reported
  endpoint — so the corpus rule mix is partly a property of that compiler. Yield
  is unaffected by this (a program compiles or it does not), but no claim that
  the prior learned *chemistry* rather than *route-compiler habit* is supported.
* 15 source molecules. Every leave-source-out number rests on 15 groups.
* The declared-arm control is uniform over the thirteen families, which is
  weaker than the production capacity-aware weighting. The production experiment
  covers that stronger control but varies only the region law.
* Wall seconds are load-contaminated (other jobs share this machine). Module
  compiles attempted and primitive rewrites executed are the load-independent
  counters and are reported per arm.

## Recommendation

Do NOT adopt this prior as the complete-program proposal underneath PMO. On the
evidence here it lowers complete-program yield and structural diversity on
unseen parents, and its one candidate advantage does not survive an exact test.

The one part worth carrying forward is narrower, and it is measured.
`gated_law_reference_v1.json` puts the learned law beside `free_gate_margin_v1`
— the engineered T4 region repair, which consumes the task's declared delta and
the benchmark source as a similarity reference — on the production path, 15
parents x 48 draws, zero oracle:

| arm | complete yield | released atoms (mean) | released atoms (max) |
|---|---|---|---|
| `production_v1` | 1.000 | 1.44 | 8.33 |
| `production_route_law` (learned, leave-source-out) | 0.994 | **3.86** | **21.00** |
| `production_free_gate_v1` (engineered, task-gated) | 0.999 | 3.27 | 15.00 |

Paired per-parent against v1, mean excision scale: learned **+2.424, unanimous
+15/−0**; gated +1.828, +13/−2. Learned against gated: +0.596, +11/−4. Yield
cost is −0.006 and −0.001 respectively.

So the learned structural law **recovers, and slightly exceeds, the task-gated
law's shift in excision scale — on every one of the fifteen parents — without
consuming a similarity reference or a delta.** That matters because PMO declares
neither, so `free_gate_margin_v1` is not expressible there and this law is.

That is a T4 region-repair result about ONE coordinate. It is not evidence for
a PMO complete-program prior, and it does not weaken the verdict above: the same
law, asked to carry a whole declared program, significantly lowers yield. If it
is adopted anywhere it should be as a region-draw law on the production
synthesizer, decided on T4's own rescue evidence, and never as the program
proposal itself.
