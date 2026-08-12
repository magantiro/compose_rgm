# DDSBM — adapter qualification

**Status: `DESIGN_ONLY`. Nothing installed, nothing run.**

## Identity

| field | value |
|---|---|
| paper | Kim, Kim, Moon, Kim, Woo, Kim — *Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation*, ICLR 2025 |
| preprint | [arXiv:2410.01500](https://arxiv.org/abs/2410.01500) (v1 2024-10-02, v2 2025-02-28); OpenReview `tQyh0gnfqW` |
| code | <https://github.com/junhkim1226/DDSBM> — 5 commits, last `b7787042` 2025-04-15 |
| license | **NONE. There is no LICENSE file and the GitHub API reports `license: null`.** Default is all-rights-reserved. This is a hard blocker for redistribution and must be resolved with the authors before any use beyond reading. |
| checkpoint | **Not available.** The README TODO "Checkpoints update using Zenodo" is unchecked; open issue #1 requests them and the author reply did not supply them. |
| citation key | `kim2025ddsbm` (the bib entry currently records it as an arXiv preprint, year 2024; it is now ICLR 2025) |

## What was verified

- **Native object is a bridge between two distributions**, not a conditional
  generator given a target. Only the two marginals are prescribed; the coupling
  is what IMF solves for. The paper contrasts itself with baselines that "assume
  that suitable molecule pairs have already been identified" (§F.2.2).
- **Training input is a paired CSV** (`REF-SMI`, `PRB-SMI`), but the pairing is
  arbitrary — "We randomly coupled the data of initial and terminal
  distributions" (§5.1) — so in effect it needs two *marginal* datasets.
- **The objective is the terminal dataset.** The conditioning channel `y` is a
  zero-width tensor in every dataset file. A new objective means a new terminal
  distribution and a full retrain (ZINC: 300 epochs × 6 IMF iterations on four
  RTX A4000s).
- **Intermediate states are not molecules.** The sampler resamples every node and
  edge category independently at each of 100 steps with no valency check, no
  connectivity check and no sanitisation. Even the final state is not guaranteed
  connected — validity is computed on the largest fragment. Reported DBM validity
  on Polymer is 43.4%.
- **No pathwise or subgraph constraint mechanism exists.** The only structural
  pressure is the GED-shaped reference process, which is an average-case bias,
  not a per-path guarantee.
- **No mid-trajectory retargeting.** `sample_forward_bridge_batch` hard-starts at
  `t=0` with no start-time parameter and no way to inject a realized intermediate
  state; the learned generator is time-inhomogeneous and conditioned on remaining
  time to `τ`, so splicing a differently-trained model mid-run has no theoretical
  backing in the paper.
- **Zero oracle calls at sampling time.** Cost per sample is 100 network forward
  passes. The property signal lives entirely in how the training datasets were
  split.

## The one finding that goes against a COMPOSE framing

Atom addition and deletion **are** effectively supported. Every molecule is
padded to the dataset-wide maximum node count with a first-class dummy atom type
`"X"`, and the uniform CTMC transition freely flips a slot C→X (deletion) or
X→C (insertion); dummies are stripped at decode. The node *tensor* is fixed
(37 slots on their ZINC subset, mean molecule 23.7 heavy atoms); the *molecule*
size is not.

`docs/RELATED_WORK_MATRIX.md` currently marks DDSBM `var-card ✗` and
`birth/death ✗`. On this evidence both should be `~` with the padded-array
caveat stated. **Recommended to the paper lane; not edited here** — that file is
owned by the manuscript workstream. The `complete ✗` cell is confirmed correct.

## Adapter work required — and why it is probably not worth it

A faithful DDSBM comparison means **reproducing its native protocol**, not
adapting it to a COMPOSE panel:

1. Build two marginal datasets under a COMPOSE property split (the analogue of
   their logP-2 vs logP-4 sets).
2. Train from scratch: 6 IMF iterations × 300 epochs. GPU-only in practice.
3. Evaluate both DDSBM and COMPOSE with the same evaluator on the same data.

There is no cheap adapter. The blockers are the missing license, the missing
checkpoints, and the fact that the Polymer pipeline in the repo is empty
(`experiments/EXPERIMENTS.md` has a Polymer section whose code blocks are
literally blank).

## Known friction

- Dependencies include **graph-tool 2.45 (conda-only)**, `EDeN` from a git URL,
  and a C++ `orca` binary compiled with g++. Built on DiGress.
- ZINC results are on a filtered ~29,920-pair subset (no formal charges, no
  explicit H on aromatic atoms), not full ZINC250k.
- The validity metric scores disconnected outputs by their largest fragment.
- The author reply on issue #1 also confirms the paper's DiGress ZINC numbers
  were taken from the DiGress paper, not reproduced.

## CPU feasibility

**No.** Training is four-GPU; the repo's default config notes multi-GPU is not
implemented on that branch, and there are no checkpoints to do inference-only
CPU work with. Under this lane's CPU-only, no-GPU constraint, DDSBM cannot be
run at all.
