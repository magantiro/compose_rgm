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

## Answer

**Yes for the ring-size law, partly for the ring-system law, and it is free.**

Arm C1 reproduces the corpus per-ring SIZE law (TV 0.030 against arm A's 0.209)
and the ring-system SIGNATURE law (0.066 against 0.258), with bootstrap
intervals that contain the corpus values and do not overlap arm A's. Molecules
carrying a 3- or 4-ring fall 57.9% to 6.0% (z = 9.96).

It does NOT reproduce the ring-system COUNT law: it improves it (0.317 to
0.201) but under-delivers 2.13 systems per molecule against the corpus 2.61,
because 18.8% of requested systems find no host and are dropped. Arm B1, which
pins only the count, is the best arm on count (0.071) for exactly that reason.
**No single arm wins all three laws, and the gap is the realization defect, not
the factorization.**

Nothing was paid for it. Published quality is flat (0.121 vs 0.113, z = 0.22),
as are QED pass (z = 0.72), SA pass (z = 0.88), validity and uniqueness (1.000)
and diversity (0.8900 vs 0.8897). Mean QED falls 0.522 to 0.457 raw, but that is
attributable to the ring-system shortfall: adding ring-system count to the
regression collapses the arm term to -0.024 +- 0.021.

Two further findings worth reading the artifacts for: on a FULL host the model
already samples the corpus ring law and does so independently of conditioning
time (`ring_time_sensitivity_v1.json`), which confirms the diagnosis at the
model's own conditional law; and the realization failures are host exhaustion,
monotone in plan position, with the most-demanding-first ordering already
protecting the fused systems.

## Files

| file | what it is |
| --- | --- |
| `plan_prior_v1.json` | `p(R \| heavy-atom bin)`, fitted on all 500,000 GuacaMol training molecules |
| `corpus_ring_census_v1.json` | the reference ring law the arms are scored against, from the same pass |
| `arms_report_v1.json` | the scored arms |
| `arms_table_v1.txt` | the rendered comparison |
| `plan_hazard_collapse_v1.json` | why the plan cannot be realized at `t = 0` |
| `ring_time_sensitivity_v1.json` | how the ring-template law moves with conditioning time (it does not) |
| `qed_attribution_v1.json` | whether the planned arms' QED gap survives controlling for size and ring count (it does not) |

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
