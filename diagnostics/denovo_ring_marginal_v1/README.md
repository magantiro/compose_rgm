# Pinning a global ring marginal for de-novo generation

**Question.** COMPOSE's ring catalog is faithful, but its REALIZED ring marginal
drifts during rollout because the state-dependent executable support distorts
it before the learned rates ever choose. Both published comparators avoid this
by pinning a global marginal *before* generation rather than controlling rings
*during* it — InVirtuoGen factorizes `p(x) = p(n) p_theta(x | n)` with `p(n)`
fitted empirically on ZINC250k; GenMol draws its mask count from an empirical
training distribution. Does the same move work for the ring skeleton?

    p(G) = p(R) p_theta(G | R)

`R` is a ring-system multiset — for example `((6,), (5, 6))`, a benzene plus a
fused 5-6 bicyclic — drawn once, before generation, from the corpus law.

## The mechanism this is aimed at

`_eligible_grow_host_graph` restricts v1 ring installation to a scaffold of
**acyclic, neutral, single-bonded carbon**. Every committed ring permanently
removes its atoms from that host, so each successive ring decision is made
against a smaller host — and a small host only fits small rings. Nothing
downstream of the rates can repair that: on 3 of 12 audited rollout states
EVERY legal template is a small ring, so the model has no non-small option to
choose. See `diagnostics/denovo_ring_support_host_v1/FINDING.md`.

## Files

| file | what it is |
| --- | --- |
| `plan_prior_v1.json` | `p(R \| heavy-atom bin)`, fitted on all 500,000 GuacaMol training molecules |
| `corpus_ring_census_v1.json` | the reference ring law the arms are scored against, from the same pass |
| `arms_report_v1.json` | the scored arms |
| `arms_table_v1.txt` | the rendered comparison |
| `plan_hazard_collapse_v1.json` | why the plan cannot be realized at `t = 0` |
| `ring_time_sensitivity_v1.json` | how the ring-template law moves with conditioning time |

## Arms

| arm | pinned | realized at |
| --- | --- | --- |
| `A` | nothing (shipped process) | — |
| `B0` | ring-system COUNT | `t = 0` |
| `C0` | full ring-system SIGNATURE | `t = 0` |
| `B1` | ring-system COUNT | the model's own first ring event |
| `C1` | full ring-system SIGNATURE | the model's own first ring event |

`A -> B` attributes earliness plus the count; `B -> C` attributes the size law;
`0 -> 1` attributes the realization point.

**The `B` arms are a declared SUBSTITUTION.** The brief's arm B was
`exact_early_ring`, a TEACHER-TRACE schedule applied at training time. Lineage B
was trained on `sequential`, so that arm cannot be obtained from this checkpoint
by any inference flag — only by a retrain. `B` here is the inference-time
analogue of its intent (commit ring transactions at the earliest state that can
carry them) and is the right ablation for separating earliness from the pinned
marginal, but it is not the same object.

## Discipline

* The latent is drawn **before** generation. No endpoint is ever rejected on how
  its rings came out; a plan the host cannot carry is RECORDED as unrealized,
  never retried until it looks right.
* **No SA or small-ring reward term exists anywhere in this experiment.** The
  diagnosis is that the defect sits upstream of the learned rates.
* Realization runs the production sampler verbatim, through the two restriction
  attributes it already reads, so template choice, placement and the semantic
  electronic decoding are never transcribed.
* Every arm draws its `t = 0` tree as the first use of its RNG, so matched
  trajectory seeds give matched initial states and the arms diverge only
  afterwards.
* Zero oracle calls. Every quantity here is a free RDKit computation.
