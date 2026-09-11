# Repeated online feedback did not improve best-score discovery

On this single warm-start `perindopril_mpo` development run, balanced sampling
outperformed both learned endpoint-allocation arms on final best score and
top-ten archive mean. Seven online updates changed only 3 of 28 selections with
matched parents and candidate pools relative to the frozen policy. This result
does not support scaling the current reranker as the next intervention.

Authoritative computed analysis: [audit.json](audit.json). Authoritative sealed
remote result: [result_sealed.json](result_sealed.json), physical SHA-256
`627f2e8a86107f54080902c53d8ac11fc307275c8d522541bfdc4e3d99287946`.
The audit verifies selection locks, archive curves, policy-update identities and
request accounting against all eight rounds' saved receipts. It makes no new
oracle calls or model fits.

## Measured outcome

All arms started with best score 0.4945748847 and top-ten mean 0.4742636650.
Each received 32 logical scored requests over eight rounds, with no abstentions
and 32 distinct selected canonical molecules per arm.

| Metric, higher is better | Balanced | Frozen learner | Online learner |
|---|---:|---:|---:|
| Final best | 0.522233 | 0.508913 | 0.508913 |
| Final top-ten mean | 0.500116 | 0.483070 | 0.485089 |
| Warm-window top-ten AUC | 0.484213 | 0.477011 | 0.477111 |
| Mean selected score | 0.414483 | 0.424643 | 0.422910 |

The AUC is the trapezoidal top-ten archive curve over requests 0 through 32,
including the common warm start, divided by 32. It is not official PMO AUC.
The learned policies improved mean selected score relative to balanced sampling
but did not improve the discovery metrics above. This is one development run,
not a multi-seed significance result or an IVG benchmark comparison.

The experiment used 64 new physical PMO oracle calls, within the 96-call cap.
Independently selected cross-arm duplicates reused labels while remaining
charged to each arm's logical allowance. The 228 previously charged development
calls remain part of the history, giving 292 physical charges across that history
and this experiment. No cross-arm outcome was used to train an arm. No docking,
prescreening, winner-derived training or reference-model retraining occurred.

## What the feedback actually changed

The online arm used eight distinct policy hashes, including the initial policy;
the frozen arm used one. The seven fits took 2.93 to 3.59 seconds each.
Updates therefore executed, but mostly preserved sampled decisions. Some
probability distributions changed substantially, so unchanged selections should
not be described as unchanged model parameters or universally negligible updates.

There is also an observed ranking error, not just a flat best-score curve. In
round 2, slot 0, the online policy assigned probability 0.814 to an annulation
candidate scoring 0.456435, versus 0.134 to a scaffold-extension candidate scoring
0.498224. Both outcomes were actually queried in this experiment. The frozen
policy made the same preference. This retrospective comparison uses already
charged cross-arm labels only for diagnosis; it is not an unbiased estimate of
regret over all unqueried candidates, and probabilities are not calibrated
confidence estimates.

The implementation conditions endpoint selection on task observations after
sampling at most four complete-option candidates. It does not train task-directed
primitive construction or multi-option future value. All arms do have
score-dependent archive/parent selection. The negative result is therefore not
evidence that molecular score feedback was entirely absent.

## Structural and computational observations

- Median intended region release was 0.226, 0.108 and 0.100 for balanced, frozen
  and online arms. Median realized largest changed fraction was 0.054 for all
  three. A large released region did not imply a large realized rewrite.
- Each arm selected one cycle-rank-increasing candidate, but none selected a
  candidate increasing ring-system count. These are different quantities; fused
  annulation can increase cycle rank without adding a separate ring system.
- Median selected-parent size was 37 heavy atoms in every arm. Under the frozen
  40-atom limit, 19/32, 20/32 and 23/32 parent requests respectively had fewer
  than five free atoms. New pendant-ring construction can require shrinking or
  replacing material first. This is a measured capacity constraint, not authority
  to enlarge the declared support or proof that every useful route is excluded.
- The saved proposal census includes generic, local, rebuild, cyclize, growth,
  pendant/fused construction and ring-system programs. Across 56 unique worker
  tasks, 223 of 224 complete-option draws completed and one ring-system program
  reached a support dead end. Selected endpoints retained executor replay checks.
- Remote run time was 1,168.24 seconds (19 minutes 28 seconds), excluding image
  builds and earlier failed starts. Summed worker elapsed time was 1,765.94
  seconds; median worker time was 27.25 seconds and median initialization was
  0.277 seconds. These are elapsed compute measurements, not measured billing.
- Execution was CPU-only, with at most twelve proposal workers and one driver.
  The primitive reference used float32; policy calculations used float64.
  Full configurations, software versions and seeds remain in the sealed result.

## Decision and proposed next intervention

Do not infer that more reranking rounds will close the performance gap. The
next development hypothesis should change what is generated: learn task-dependent
choices over executable options and their continuations using scored trajectories,
while retaining the generic channel, the region-scale interface and exact executor.
Credit must cover complete multi-option continuations rather than assuming every
useful intermediate has immediate high value. This is proposed work, not a trained
or validated new controller, and requires a separate bounded run contract.

Reuse the existing [answer-known T4 route diagnosis](../t4_route_diagnosis/README.md)
to locate missing support, proposal probability and value-ranking failures. It
already records a five-program witness to a known endpoint and an older guide
that misranked that endpoint. That evidence is not an actual docking verification
or autonomous recovery. Any winner-informed development or training must remain
separate from untouched blind evaluation. Molecular executability and high PMO
score also do not establish medicinal quality.

## Provenance, failures and verification

- Scientific producer: `662a2227d74ddf09c8776e442576cc5d194b5c5c`.
- Audit implementation revision: `14acbb17d38762311bcbfa8fe577eef0527fc689`.
- Contract self-hash:
  `b49390261fdd2b404162d4408468e660a2bfde2c9538a0482e72596a782f0720`.
- Run ID:
  `8aaa065247395cecc46b97f2d8805a2c0903624f15f9c87cecda4c26b82baba3`.
- Durable call: `fc-01M26Z0VM43QJD8BJ385CBFBZ7`.
- Remote receipts: volume `compose-v4-artifacts`, prefix
  `pmo_online_policy/<run_id>`, including `result.json` and
  `phase/1` through `phase/8` choices/proposals/outcomes.
- Every material local audit input is SHA-256 bound in `audit.json`. The remote
  result binds the model package, configuration and warm-start inputs. The
  numerical runtime admission receipt is retained separately as
  [runtime_check_sealed.json](runtime_check_sealed.json).
- Two earlier zero-oracle starts failed on a missing runtime dependency and an
  empty remote calibration dispatch. Their identities, cancellation and repairs
  are preserved in [launch_repairs.json](launch_repairs.json). Their time is not
  included in the completed run's 19-minute measurement.
- Checks that ran: nine initial focused tests, four empty-calibration regression
  cases, one audit/accounting test, touched-code Ruff formatting/lint checks,
  clean scientific-launch preflight and whitespace diff checks. The completed
  audit also verifies actual persisted selections and curves. No repository-wide
  suite or release/milestone completion is claimed.

To recompute the audit without molecular generation, download the result and all
eight choices/proposals receipts into the same relative snapshot layout and run:

```sh
python3 tools/pmo_online_audit.py --snapshot SNAPSHOT --output NEW_AUDIT.json
```

Use the documented Python/RDKit environment and this report's audit revision.
The output path must not already exist. Input contents are hash-bound; timestamps
and operational source-path fields vary when reducing a new local snapshot.
