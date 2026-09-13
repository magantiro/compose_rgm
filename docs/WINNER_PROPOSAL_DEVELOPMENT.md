# Winner-informed proposal development

## Identity and scope

COMPOSE generates complete supported molecular graphs through the frozen executable
rewrite process. This development task changes the controller's proposal policy,
not the executor, reference model, molecular support, or T4 endpoint constraints.
The current hypothesis is that demonstrations can improve the WHAT distribution
over existing compositional options without requiring the final target at inference.
This is not a claim about future-value calibration or benchmark performance.

The immediate experiment reuses saved IVG executable witnesses on local CPU, with
zero new docking calls. Compare a state-conditioned imitation actor with the
existing balanced option prior and a source-balanced marginal demonstration prior.
Keep generic and every supplied applicable option at positive probability. Record
the source split before extracting labels. All these winners were already inspected;
source-held-out fitting diagnostics are retrospective, not a fresh benchmark test.

## Representation and evidence boundaries

- Inputs to the learned actor are current-molecule features and compositional
  option descriptors. No winner, remaining witness length, docking label, or
  target-difference feature is supplied at inference.
- Demonstration labels are contiguous segments of saved exact-slot traces.
  Whole-ring and carbonyl segments must match existing descriptor contracts.
  Other actions retain their ordinary primitive-family option interpretation.
- Witnesses establish executor reachability, not positive probability in every
  deployed enumerator, nor an expected continuation value under a sampled policy.
  Their labels therefore train proposals only. No artificial behavior likelihood,
  advantage, or Doob future-value target is attached to them.
- A shared fixed descriptor menu is used for the offline classification diagnostic.
  Its distractors need not all be applicable. Deployment must normalize against
  the actual production applicability row. Offline rank is not autonomous recovery.
- Broad organic chemistry remains available through generic and ordinary options.
  The ring descriptors retain their existing C/N/O, size, topology and electronic
  limits. They are not a whole-molecule template catalog.

## Decision

Measure held-out-source demonstration probability, negative log likelihood, top
choice precision, descriptor coverage, source counts and time. If conditioning
does not improve over the marginal demonstration prior, do not deploy the extra
model on the basis of training accuracy. If extraction misses compound behavior,
report that gap explicitly. A favorable result licenses a proposal-only development
comparison, not a claim of beating IVG. Docking performance still requires matched,
candidate-locked evaluation with historical-data accounting.

Current champion, active experiment and benchmark costs remain recorded in
`docs/START_HERE_ICLR.md`; this document records only this bounded development task.

## Initial result and literature check

The initial raw-witness imitation run is not promoted. Its actor underperformed
the source-balanced marginal prior on source-excluded examples and had no
whole-ring training labels. Results and the next decision are in
`diagnostics/winner_option_proposal/attempt_1/README.md`.

Inverse decomposition subsequently supplied whole-ring examples, but a fit with
the unchanged actor still underperformed the marginal prior on excluded sources.
The static marginal was then evaluated on the actual saved four-option PARP1
route: it helped the first action and suppressed the next three. The next
development unit therefore concerns coordinated option sequences and operand
choices rather than another global frequency or network-size adjustment.
Full evidence and precise conditioning: `diagnostics/inverse_ring_proposals/README.md`.

The literature supports investigating proposal improvement, but does not imply
that COMPOSE will benefit without a measured experiment. [Trust-Region Twisted
Policy Improvement](https://arxiv.org/abs/2504.06048v4), ICML 2025, adapts SMC
planning to policy improvement through constrained action sampling, terminal
handling and improved policy/value targets. [TRI-TSMC](https://arxiv.org/abs/2605.25123v1),
a May 2026 preprint, iteratively fits a twisted proposal by weighted maximum
likelihood after a path-space trust-region update. Its reported domains are
text and image diffusion, not molecular editing. The proposed COMPOSE connection
is to improve where particles go before relying on resampling to rescue sparse
successful paths. The current supervised diagnostic does not implement either
paper's full algorithm or inherit its guarantees.
