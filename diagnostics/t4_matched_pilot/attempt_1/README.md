# Matched T4 development audit: no guidance benefit; closure-contract defect found

One cold-start PARP1 seed0, d=0.4 round per arm completed from clean scientific
revision `219c1cd276ed5cf0fd1c98311e408bdf9b04f41b`. Modal call:
`fc-01M1ZKZYS3HCSBWJ3Z26H5NHJC`. Neither training nor macro-weight tuning occurred.
Both candidate locks preceded all docking and their outer plans match exactly.

| Measure | Reference | Frozen committor |
|---|---:|---:|
| Docking attempts / failures | 20 / 0 | 20 / 0 |
| Feasible docked molecules | 19 | 17 |
| Best feasible docking score | -8.3 | -8.2 |
| Unique pool / unique docked | 29 / 20 | 29 / 20 |
| Selected bundles / docked owner bundles | 24 / 19 | 24 / 19 |
| Mean pairwise Morgan distance, docked | 0.54959 | 0.55433 |
| Proposal seconds | 433.479 | 442.207 |
| Docking seconds | 49.654 | 47.106 |
| All public executor applications | 12,759 | 15,046 |
| Cached law enumerations | 79 | 76 |
| Recorded sampled transitions | 120 | 120 |
| Mean recorded primitive KL | 0 | 0.24157 |

The best molecule is identical in both arms:
`CC(C)c1cc2n(c1)-c1ccc(CN(C)C)cc1CNC2=O`.
The inherited OpenBabel/QuickVina procedure is unseeded. Its different scores for
the same molecule directly preclude interpreting this 0.1 difference as a
guidance effect. Both use the same post-hoc cold-start selector, with no docking
labels available during proposals. Total remote elapsed time was 1,234.867 s
(20.58 min), including frozen runtime initialization and persistence.

## Search geometry and options

Selected options in each arm: generic 5, build_ring_system 3, grow 3, decorate 3,
rebuild 2, aromatize 2, and one each of small_ring, open, scaffold_extend,
restate, shrink, append. The fused program was product-applicable in 13/24 draws
but selected in none. No resampling forced its inclusion.

Docked intended release ranged from 0.0526 to 0.8947 in both arms. Median actual
coherent change was 0.1053 in both; maxima were 0.2632 reference and 0.2105 guided.
These are lineage-relative displacement measures, not fractions of original
scaffold replaced. Both arms docked three topology-changing candidates:
small_ring adds one cycle and one ring system with no added atoms; open and
shrink each reduce cycle rank by one while increasing ring-system count by one.
Thus ring-system count alone is not evidence of constructive ring growth.

## The important diagnosis

All three BUILD_RING_SYSTEM bundles in each arm grew eight times, then halted
at append_system before completing the 11-step program. None was docked as an
incomplete program. The saved counterfactual ledger distinguishes two causes:

1. **Verified false rejection in one bundle, in both arms.** Two already executed
   products have new six-atom pendant ring systems, disjoint from every old ring
   atom in exact persistent slots. Both replay identically under the production
   executor, are valid and connected, satisfy region admissibility, preserve
   frozen context, and have 27 active atoms. The existing SMILES-based contract
   rejects both. `append_system_closure` compares atom-index sets from separately
   canonicalized molecules; those index sets do not encode atom correspondence.
2. **Observed size shortfall in the other two bundles.** Their saved executed
   closures produce only 3–4-atom new systems, below the unchanged minimum of six.
   Eight growth steps alone do not ensure a suitable closing geometry. This is
   a finding about the observed proposal support, not proof of chemical
   unreachability.

The diagnosis required no additional model or docking calls. The initial
verification replayed two unique edits; the packaged reproduction independently
replayed those same two edits again (four verification executor calls total).
The products are counterfactual executor outcomes, not completed 11-step
offspring or measured affinity improvements.

The evidence supports repairing identity-safe option contracts before judging
the controller's ring-construction ability. It does not support increasing
kappa, retraining R_theta, changing Q(M), learning Q(o), or replacing the general
controller with ring-only search. Topology construction, atom/bond restating,
and objective-based allocation remain separate responsibilities within the same
region-option framework. No production contract repair is included in this run.

## Artifacts and reproduction

`review.json` reports lock-verified score and structural summaries.
`closure_diagnosis.json` contains the saved closure census and exact-slot
false-rejection witnesses; `closure_products.json` and `closure_verification.json`
retain the initial exploratory diagnosis. `remote_inventory.json` hashes all
29 remote JSON artifacts, totaling 40,944,412 bytes. Full executor ledgers and
parent checkpoints remain on `compose-v4-artifacts` under:

`/t4_matched_pilot/4036ff371486d2aae87c2295296440ba55865b5fdcdef9aaf1ee3c535f8b868a`

Read-only summary:

```sh
.venv/bin/python tools/t4_matched_audit.py diagnostics/t4_matched_pilot/attempt_1
```

For the closure diagnosis, download each arm's `executor_attempts_0.json` to
`reference.json` and `committor.json` in a local ledger directory, then run:

```sh
.venv/bin/python tools/t4_closure_diagnosis.py diagnostics/t4_matched_pilot/attempt_1 /path/to/ledger_directory
```

This verifies ledger hashes and uses exact states; it does not reconstruct
intermediates from SMILES or invoke a learned model or docking oracle.

## Verification limitations

90 focused tests passed before result analysis. The required full suite completed:
4,445 passed, 55 failed, 58 errors, two skips and one expected failure. Eight new
failing identities were caused by the new test fixture leaking disabled PyTorch
gradient mode, not by production model changes. Test-only commit `dd2f7be` restores
that mode; the fixture followed by all eight affected nodes passed together
(9/9). The other 105 nonpassing identities match the previous recorded suite;
matching identities does not establish identical causes or waive a gate. The
repository-wide suite is not green, and no broad milestone completion is claimed.
The running scientific code remained exactly `219c1cd`; the test isolation repair
did not alter any serialized source, input, proposal, or oracle.
