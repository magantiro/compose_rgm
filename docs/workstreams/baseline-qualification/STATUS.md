# Workstream D — status

**Status:** `DESIGN_ONLY` — qualification complete, ready for handoff.
**Branch:** `codex/compose-baseline-qualification` (base `04f1c46`).
**Held-out data opened:** **no.** No sealed panel, no confirmatory reserve, no
matched reserve was touched.
**Running now:** nothing. No Modal job was launched; no dependency was installed.
**Last completed gate:** all six priority methods qualified against primary
sources, with per-cell evidence.

## The finding to read first

Three baselines natively do something the COMPOSE Claim-4 story treats as
distinctive. None of them does the full claim, but the paper must state the
**conjunction**, never the part.

1. **REINVENT 4 changes the objective mid-run, natively.** Staged/curriculum
   learning varies the scoring function in stages and carries the agent
   checkpoint across the switch. It re-targets a *policy*, not a realized
   molecule, and it re-adapts by further RL — but "no existing method changes
   objective mid-run" is now a sentence the paper cannot write.
2. **HN-GFN changes the goal without retraining**, for scalarization weights over
   a fixed objective set, via a hypernetwork over the prediction heads.
3. **GraphXForm applies structural action masking at every step**, pre-softmax.
   Real pathwise enforcement — but only over valence, atom type, atom count and
   bonding legality, never labeled subgraphs.

And one correction that goes **against** COMPOSE: **DDSBM does support effective
atom addition and deletion** through a dummy atom type inside a padded node
array. `docs/RELATED_WORK_MATRIX.md` marks it `✗` on `var-card` and
`birth/death`; on this evidence both should be `~`.

## Verdicts

| method | verdict | why |
|---|---|---|
| MARS | `MUST_RUN` | closest iterative editor; source-conditionable with stock code; CC BY-NC license and a 2021 dependency stack are the risks |
| GraphXForm | `MUST_RUN` | exact source conditioning via `start_from_smiles`; constructive-only action space is a hard scope limit |
| GraphGA | `MUST_RUN` | PMO rank 2, MIT, RDKit-only, CPU-native, no training — the cheapest and cleanest comparator |
| REINVENT | `MUST_RUN` | PMO rank 1, and the only external arm that natively switches objective mid-run |
| HN-GFN | `CONDITIONAL` | gate: GPU authorisation **and** a working BoTorch pin (upstream issue #1 is unresolved) |
| DDSBM | `CONTEXT_ONLY` | **no LICENSE file at all**, no checkpoints, intermediates are not molecules, zero oracle calls at sampling |

## Costed smoke plan — nothing executed

| method | plan | projected CPU-core-hours | confidence |
|---|---|---:|---|
| GraphGA | 5 sources, pop 120, 10 generations | 0.1 | HIGH |
| REINVENT (A, PMO-wrapped) | 5 sources, ~1000 calls each | 0.3 | MEDIUM-HIGH |
| REINVENT (B, v4 staged learning) | 2–3 sources, 2 stages × 25 steps | 0.5–1.0 | MEDIUM |
| MARS | 5 sources, 1 chain each, 50 steps | 1.5 (incl. one-time ChEMBL vocab build) | LOW on env |
| GraphXForm | 5 sources, 3 epochs, beam 32 | 2–4 | MEDIUM |
| HN-GFN, DDSBM | not runnable CPU-only | — | — |
| **total** | | **~4.4–6.9** | |

Under USD 1 of compute at a nominal CPU rate. **The cost of this lane is
engineering time on dependency rot, not credits.**

## Next action

One bounded action: **build and smoke the GraphGA adapter** (0.1 CPU-core-hours),
because it is the cheapest end-to-end proof that the frozen-oracle shim and the
chosen oracle-counting convention work — and because §0 of `FAIRNESS_CONTRACT.md`
shows the counting convention must be decided before any comparator runs, and
GraphGA is where that decision is cheapest to test.

## Open decisions this lane deliberately did not make

- **The oracle-counting convention.** COMPOSE's rule (count everything) and
  PMO's rule (unique valid canonical SMILES) disagree, and the choice changes the
  ranking. It is claim-level and belongs to the main workstream.
- **How to report HN-GFN's surrogate.** Its 1000-call budget is spent against a
  learned proxy, not the true oracle.
- **The two `RELATED_WORK_MATRIX.md` cells** (MARS `pathwise`, DDSBM
  `var-card`/`birth/death`). Evidence recorded, file not edited — see
  `DECISION_LOG.md` for why the asymmetry is deliberate.
