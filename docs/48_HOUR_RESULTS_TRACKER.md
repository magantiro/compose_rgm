# COMPOSE 48-hour results tracker

**Window opened:** 2026-07-19 18:08 EDT  
**Rule:** no experiment starts without a named failure, a bounded runtime,
reused artifacts, a stop rule, and a result that changes the next decision.

This is a computational-results sprint.  It can produce qualified generators,
benchmark measurements, conditional proof points, lipid data audits, and lipid
smoke samples.  It cannot produce prospective synthesis, formulation, or
in-vivo results within 48 hours; those remain explicit Paper-2 future work.

## Live tracker

| Block | Deliverable | Runtime authorization | Pass / stop rule | Status |
|---|---|---:|---|---|
| U1 | Quotient-correct unconditional continuation from selected step 2,500 | Two bounded cached arms; no new chemistry compilation | Promote only on the frozen validation gate before rollouts | **Completed, no promotion** — fresh-optimizer arm failed; legacy-transfer arm best 12.7654/53.81% versus incumbent 11.6977/58.08% |
| U2 | 100-sample checkpoint audit | One deterministic sample set per materially improved checkpoint | 100% final and intermediate validity/connectivity; zero molecular self-events; report operator, atom, bond, size, QED, ring, and trajectory diagnostics | **Completed on incumbent** — perfect validity/connectivity/uniqueness/novelty and zero self-events, but 40% small-ring incidence; all new small rings attributed to whole-ring grow |
| U3 | Unconditional MVP decision | One support-versus-logit diagnostic, then bounded cached comparisons | Keep the incumbent unless a matched pilot improves learning and ring topology without degrading validity or other chemistry | **Developmental gate passed on 600 fixed samples** — exact CTMC thinning reduced delete and 3/4-ring intensities without renormalizing other learned rates; the checkpoint is promoted to QED conditioning while publication-scale unconditional confirmation is deferred |
| U4 | Root-fix pilots for remaining unconditional chemistry holes | P1/P2 only on existing paths/support; 250--500 updates and 200 fixed rollouts each | Improve triple/heteroatom and ring-topology marginals without any validity or trajectory regression; do not rebuild the ring catalog unless bounded pilots fail | **Implementation active** — audit complete; empirical base-rate residual and topology-group absolute-rate pilots are being added behind explicit flags |
| C1 | Matched QED protocol and valid-rewrite conditional smoke | Engineering/smoke only before U3; bounded small oracle budget after U3 | Same starts, proposals, oracle calls, validity denominator, seeds, and similarity rule for selection and guidance arms | **500-update pilot complete** — conditioned evaluation cache persisted; selected validation loss improved 38.7554 to 11.6031 and final test family accuracy is 60.50% |
| C2 | QED authorization result | Small pilot first; expand only on positive paired signal | Direct conditioned generation must show target response under matched classifier-free controls without sacrificing all-attempt validity | **Matched rollouts active** — four QED targets (0.3/0.5/0.7/0.9), ten attempts each, plus classifier-free control for both an early checkpoint and the immutable final step-500 checkpoint; first control arm is 10/10 valid and unique |
| L1 | Lipid dataset registry and audit | Starts immediately when files/links and endpoint schema arrive | Provenance, licenses, deduplication, chemistry coverage, graph size, charge, rings, components, assay/formulation separation, and leakage-safe split all recorded | **Verified-source baselines complete; unified matrix in progress** — LNPDB lung (1,975 observations/291 lipids), LUMI (1,920 structures), and LuT (444 in-vivo compounds) are audited; the final matrix is representation by model family over typed task heads, not dataset by model |
| L2 | Lipid kernel/throughput smoke | Small audited slice only; no corpus-scale cache build without a measured pilot | Exact reachability on eligible molecules and tractable measured examples/s at lipid sizes | Queued after L1; independent of final Paper-1 tables |
| L3 | COMPOSE-Lipid smoke samples | Restricted, non-claim-bearing run after U3 and C2 | 100% validity/connectivity, corpus-matched size/charge/head-linker-tail diagnostics, unique samples, complete provenance | Queued |
| P1 | Paper-1 results package | Populate only measured results | Unconditional MVP + QED proof + retained formal/executor evidence; all unfinished cells remain `XXX`, never invented | In progress |
| P2 | Paper-2 computational package | Populate only measured retrospective/smoke results | Dataset card + chemistry/kernel qualification + lipid generator smoke; prospective panels remain preregistered future work | In progress through L1 planning |

## Completed continuation decision and current U3 diagnostic

- The fixed-`1e-4` continuation and the shape-compatible legacy transfer both
  failed the frozen incumbent gate and are retired.
- The incumbent 100-sample replay proves the ring failure is not Graft,
  delete/regrow collapse, invalid execution, or a catalog dominated by small
  rings.  Whole-ring grow selects three/four-member topologies about 20% of the
  time despite only 3.01% empirical catalog mass.
- The exact diagnostic found ordinary states with 113--1,098 legal templates
  and only about 2--5% production small-ring mass, but late decorated states
  with only 2--5 legal templates, all small-ring topologies.  The hierarchical
  family head therefore sometimes selects ring-grow from a degenerate legal
  residue and forces the error; this is not global neural over-preference.
- The first inference-only ablation excludes templates containing 3/4-membered
  cycles and falls back to another valid family when none remain.  Its 4-sample
  gate retained only 5/6-member rings, 100% all-step validity/connectivity,
  zero self-events, and ordinary fused/bridged/aromatic/saturated chemistry.
  The 100-sample run uses the incumbent checkpoint and baseline seed partition.
- The original 3,000-step run used a 3,000-step cosine horizon and had already
  decayed to 4.22e-5 at step 2,500.  A from-scratch control now keeps the
  intended 30,000-step schedule horizon while retaining a hard 3,000-update
  pilot cap.  Its step-250 accuracy is 43.19% during warmup; this is an early
  diagnostic, not a promotion result.  The control reached 52.60% at step 750
  and 52.67% at step 1,000, so it was stopped at the promised gate rather than
  consuming the remaining 2,000 updates.
- A matched superposed-rate arm added each family's executable canonical-match
  log partition before family selection.  This is the normalized marked-rate
  form of superposing rewrite matches: a rare residual ring support no longer
  inherits all ring-family probability merely because common matches are
  absent.  It reuses the exact same paths, evaluation batches, and support
  cache and had the same decision points.  It reached 56.94% at step 750,
  then fell to 55.07% and 54.61% at steps 1,000 and 1,250.  It was stopped:
  rate normalization alone does not recover the pancake teacher signal.
- The retained step-6,250 pancake checkpoint is a viable direct rescue asset:
  all 110 learned tensors match the current shared architecture by name and
  shape; only four subsequently introduced ring-role tensors are absent.  Its
  original decoder/executor can therefore be wrapped with canonical no-op
  rejection and support-aware ring rates without treating the weak quotient
  checkpoint as a prerequisite.
- The exact 2,000-sample pancake audit identifies two separable failures.  It
  underproduces atoms, bonds, and total cycle rank, while 76.07% of sampled
  events are Grafts and 99.31% of ring events occur in the final event decile.
  A conservative adjacent-commutation scheduler now moves a whole-ring event
  earlier only when both orders execute and reach the identical padded state;
  insert/delete remain hard barriers.  Across the first aromatic, saturated,
  fused, bridged, and spiro tests, ring commitment moved from the final step to
  50--86% of the trace with exact endpoints and valid connected intermediates.
- The direct rescue evaluation preserves all 110 learned pancake tensors,
  zeros only the four absent post-hoc ring-role parameters, and uses the
  canonical colored-tree Graft mask to reject molecular self-transitions.  It
  is inference-only and therefore decides whether retraining is necessary
  before any new corpus-scale preprocessing.
- The first 80-sample canonical-mask rescue was stopped after showing that
  deletion plus renormalization is not lossless: validity stayed perfect and
  self-events vanished, but mean size rose to 30.75 atoms and delete events
  nearly disappeared.  The active replacement uses CTMC thinning: a
  symmetry-only Graft advances time as a separately recorded virtual jump but
  leaves the molecule unchanged.  This retains the learned pre-quotient clock
  and conditional mark law instead of reallocating self mass to chemical
  successors.
- A frozen inference-only calibration then multiplied atom-delete intensity by
  `exp(-0.5)` and whole-ring actions containing a 3/4-member cycle by
  `exp(-1.5)`, using exact CTMC thinning.  It never enables an action,
  renormalizes a family, or changes the checkpoint.  On the same 100-source
  seed partition, mean atoms improved from 24.20 to 26.43 against a 26.61
  reference; atom-count TV improved from 0.2945 to 0.2590; bond-order TV from
  0.0898 to 0.0782; cycle-rank TV from 0.3380 to 0.2640; and small-ring
  prevalence fell from 35% to 17%.  Mean cycle rank increased from 2.42 to
  2.56, while fused prevalence changed from 33% to 30%.
- All 100 calibrated endpoints and all 2,987 observed intermediate states were
  valid and connected; uniqueness and novelty were 100%, molecular self-events
  and immediate backtracks were zero, and no trajectory hit one atom or null.
  Individual source-to-final changes ranged from -13 to +15 atoms (29 grew, 14
  were unchanged, and 57 shrank), with mean -1.15 versus the approximately
  -0.97 change required by the source/reference means.  The 100-sample matched
  FCD moved directionally from 20.41 to 19.79 but is not treated as a stable
  FCD estimate.  The same calibration was then advanced to the fixed
  600-sample developmental audit described below; publication-scale FCD is
  deferred because it is not the conditional-activation criterion.
- Three deterministic rollout shards yielded a fixed 600-sample developmental
  audit before the remaining workers were preempted.  The completed samples
  are sufficient for the conditional activation decision and are retained in
  `diagnostics/pancake6250_calibration_eval600_metrics.json`; the unfinished
  2,000-sample run was stopped rather than allowed to hold conditional work
  hostage.  All 600 endpoints and all 18,646 observed states are valid and
  connected; uniqueness is 100%, there are zero molecular self-events, zero
  null/one-atom collapses, and zero event-budget failures.  Mean atoms/bonds
  are 27.10/28.56 versus 26.35/28.70 in the matched reference, with element,
  bond-order, and atom-count total variation 0.054, 0.093, and 0.143.
- The 600-sample audit also records the remaining non-blocking base-model
  weaknesses: cycle rank 2.46 versus 3.35, fused-ring prevalence 28.5% versus
  58.0%, small-ring prevalence 11.3% versus 6.0%, QED 0.439 versus 0.602, and
  SA score 4.49 versus 2.88.  Graft activity is no longer the earlier dominant
  pathology: bond reroutes are 49.5% of events, while insert, delete, retype,
  bond-order, and whole-ring actions all remain active.  This qualifies the
  executable rewrite substrate for direct conditional training but not a
  final unconditional distribution claim.
- No rollout is run for a checkpoint that fails the teacher-space incumbent
  gate, and no further legacy transfer is authorized.

## What is deliberately not blocking progress

- A publication-quality 10,000-sample unconditional table is not required to
  begin the QED proof.
- Perfect GuacaMol fused-ring fidelity is not required to audit lipid data,
  whose ring support will be derived from the actual corpus.
- The sparse unique-state/path-witness compiler is required before another
  large from-scratch or lipid-scale stream, but not for the cached U1 run.
- Paper 2 does not require Paper 1 submission or acceptance.  Our operational
  gate is a credible base editor plus one matched QED proof before expensive
  lipid generation and prospective work.

## Change-authorization card

Before any further scientific or infrastructure change, record:

1. the measured failure and artifact that demonstrates it;
2. whether the change alters semantics, learning, or only representation;
3. pilot size, expected wall time, compute type, and maximum cost;
4. which caches remain reusable and which signature changes invalidate them;
5. the numerical pass criterion and automatic stop criterion; and
6. the downstream decision enabled by either a positive or negative result.
