# Generator results summary — unconditional + conditional

**Lane:** `claude/generator-cond-uncond`. Honest status of both generator tracks.
Marks each result **proven / measured / pending**; nothing is stated as done that
isn't. See `docs/CONDITIONAL_CONTROLLER_DESIGN.md` for the conditional method and
`docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md` for the lineage decision.

---

## 1. Base decision — Lineage B (measured)

Selected **Lineage B** (quotient-correct flexible-Graft + whole-ring-system) over
the pre-quotient pancake (A) and factorized-tree (C).

- Checkpoint: Modal `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`,
  SHA `c9d927510360ec6e…` (step-1,000 best_so_far, verified).
- **Clean dynamics (measured, zero-calibration):** 0 canonical self-events, 0
  delete-to-one collapse, 100% valid/unique — with **no calibration band-aid**
  (the pancake's `−0.5/−1.5` were pancake-specific; B needs neither).
- **Conditional-relevant:** on a matched constrained-QED controller B beat pancake
  **33.3% vs 16.7%**, mean best QED 0.881 vs 0.824, 12/12 vs 9/12 beat-lead, and
  erased pancake's no-op collapse. Clean dynamics is what lifts the conditional
  number (editing task) — decoupled from ring marginals.

## 2. B as an unconditional generator (measured, n=120 zero-calib rollout)

**Strong on:** validity 1.00, uniqueness 1.00, novelty 1.00, internal diversity 0.89.

**The one real flaw — ring taxonomy:**

| metric | B (generated) | reference | gap |
|---|---|---|---|
| small ring (3–4 mem) | **0.492** | 0.060 | ~8× excess |
| ≥3-heteroatom ring | 0.375 | 0.096 | ~4× |
| fused | 0.383 | 0.580 | under-produced |
| SA (lower=easier) | 4.48 | 2.88 | harder to synthesize |
| bridged / spiro | 0.033 / 0.017 | 0.040 / 0.035 | ~matched |

So ~half of B's molecules carry an unrealistic strained small ring. Root cause
(from replay, handoff §2.2): when only rare small-ring templates are legal, the
Boolean "ring family enabled" head renormalizes them to probability one.

## 3. The §9.3 ring fix (proven mechanism; at-scale result pending)

Fix = `rate_factorization="superposed"` + `ring_template_factorization="topology_cycle_hierarchical"`,
warm-started from B. In superposed mode the topology-group intensities enter one
global softmax and do **not** cancel, so losing common-ring support lowers total
ring hazard instead of renormalizing onto rare small rings.

- **Mechanism: proven** — `tests/test_topology_group_intensity_ring_hazard.py`
  (zero-GPU falsifier): superposed drops ring mass on support loss; hierarchical
  (the defect) does not.
- **Code: proven sound** — forward + loss + backward + compatible-init-from-B all
  run locally (113 tensors transfer, the ring-template head + new group head
  fresh-init).
- **Trained result: pending.** Modal training failed 3× on opaque cache/infra
  orchestration (not a code defect). A bounded **local** warm-start is running as
  directional evidence (small corpus, 444-template catalog — so it proves the fix
  reduces small-rings *when trained*, but is not a faithful at-scale B fix). The
  faithful at-scale checkpoint (B's 4,096 catalog + full corpus) is a follow-up.

## 4. Conditional controller — value-guided SMC (measured V0; V1 built)

Deployable method (not the best-first ceiling estimator): particle SMC over B's
rewrite CTMC, reward-tilted Feynman-Kac target, **hard Tanimoto≥0.40 fiber**,
exact per-step oracle value on real intermediates (`scripts/griddd_value_guided_smc_controller.py`).

- **V0 (reward-difference potential), 12 leads, budget 400, zero-calib B:**
  success **25%** (3/12), mean best QED **0.880**, **population diversity 0.62** —
  the diversity best-first/greedy baselines structurally lack.
- **Reference:** best-first *ceiling* 33.3% (budget 640, ~0 diversity); GrIDDD 45.1%.
- **V1 learned twist (built + tested):** `scripts/griddd_value_twist.py` (Doob
  value / reward-to-go), collector `scripts/collect_twist_trajectories.py`,
  controller `--value-twist-checkpoint`; regression test `tests/test_value_twist.py`
  passes. Ready to train + evaluate at matched budget (`scratchpad/run_after_pilot.sh`).

## 5. Proven / pending ledger

| item | status |
|---|---|
| Base = B, clean, zero-calibration | **measured** |
| B ring pathology quantified (small-ring 49%) | **measured** |
| §9.3 fix mechanism | **proven** (falsifier) |
| §9.3 fix code soundness | **proven** (local train step) |
| §9.3 trained result (directional, small corpus) | **in progress** (local pilot) |
| §9.3 faithful at-scale fixed-B | **pending** (needs full corpus; Modal blocked) |
| Conditional controller V0 (25% + 0.62 diversity) | **measured** |
| Conditional controller V1 (learned twist) | **built + tested**, eval pending |
| B native `canonical_successor_native` qualification | **pending** (Task #13) |

## 6. What "unconditional reportable" needs

B is clean on every axis except ring taxonomy. Reportable positions, in order of
strength: (a) B + §9.3 mechanism-proven + directional training evidence [~now];
(b) + faithful at-scale fixed-B rollout showing small-ring → ~reference [follow-up].
Do not claim the ring marginals are repaired until a trained rollout shows it.
