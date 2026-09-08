# Pendant-ring identity repair

The repaired production contract accepts both previously rejected six-membered
pendant closures and preserves all negative decisions in the saved comparison.
This is a verified predicate repair, not a newly sampled completed program or a
docking improvement. No model or docking calls were made.

Implementation and audit revision: `cac32a2ef784aafd18627255665e93ec5f4560cb`.
Scope and prospective acceptance: `docs/APPEND_CONTRACT_REPAIR.md`.

## Fixed-product result

| Saved arm | Closure products checked | Accepted, matching diagnosis | Rejected, matching diagnosis | False acceptances / rejections |
|---|---:|---:|---:|---:|
| Reference | 23 | 2 | 21 | 0 / 0 |
| Committor | 24 | 2 | 22 | 0 / 0 |

Coverage and precision are both 2/2 within each arm. These tiny denominators
refer only to the pre-repair, independently diagnosed saved product set. The
two positive structures occur in both arms; four positive records are not four
distinct molecules. All 47 records were checked, with 33 unique source/mark
executor replays, each exactly matching its saved product. No graph was rebuilt
from SMILES for replay.

The assay includes every executed closure product recorded at all six failed
growth endpoints. Other ledger sources and non-closure families are excluded
with counts in `attempt_1/fixture.json`. It is not exhaustive enumeration of
all reachable closures. The accepted products arise from bundle
`c808400777e6a0952c54`; the other two construction bundles per arm still have no
qualifying recorded closure. Their undersized rings remain rejected.

The complete local audit took 2.723 seconds. Predicate construction and
evaluation over the 47 records took 0.02924 s for the exact-state adapter versus
0.00834 s for the historical SMILES predicate. Those timings exclude decoding,
canonicalization, and executor replay. This is a correctness repair with a small
measured local cost, not a proposal-throughput speedup measurement.

The immutable result binds the fixture payload, exact implementation hashes,
revision, configuration, hardware/software, timestamp, and per-product results.
The fixture binds both original ledger hashes, both candidate locks, the remote
inventory, and the prior diagnosis. The original T4 candidate locks, scores,
failed programs, and diagnosis remain unchanged.

## What changed, and what did not

`state_contract_for` identifies ring systems using exact persistent atom slots.
Both T4 applicability and execution, plus the option-continuation kernel, use
this adapter. The same executor still produces every transition. The append
predicate still requires an increased system count and a system of at least six
atoms disjoint from every pre-closure ring atom. Generic remains available;
Q(M), Q(o), R_theta, kappa, committor, horizons, and all other macro predicates
are unchanged.

The historical `append_system_closure` SMILES API remains deliberately unchanged
for legacy reproduction, with an explicit warning about its identity defect.
Legacy basin applications still using that API are not repaired by this change.
They are not the audited T4 or exact-state continuation execution path.

## Ring capabilities and limits

- Pendant construction: the eleven-step BUILD_RING_SYSTEM already combines
  backbone growth, pendant closure, and restatement. This repair removes a
  demonstrated false rejection at its closure phase. Completion of the two
  subsequent restatement phases has not been tested on these saved products.
- Fused construction: the opt-in stateful program has a verified frozen-model
  five-step aromatic C6 path on PARP1 seed0, recorded in
  `diagnostics/fused_reference_profile/attempt_2/`. Its declared support is not
  arbitrary ring size or heteroatom composition.
- Small rings and other changes: small_ring, cyclize, grow, rebuild, restate,
  aromatize, open, shrink, and generic remain available. The matched T4 audit
  already sampled a three-membered pendant ring and ring-opening changes. An
  option name alone is not evidence of efficient discovery or improved affinity.
- Broader descriptor helpers expose size, composition/stoichiometry, electronic
  state, and pendant/fused requests. They are not automatically the stochastic
  region-controller programs. Their available parameters do not establish
  universal realizability or coverage of the inspected IVG winners.

The controller remains an objective-driven local/global search with ring and
non-ring moves. Winner molecules were not used as repair inputs, templates,
rewards, or option-weight supervision. No new support or training was added.

## Verification and next boundary

The final focused command in `verification.json` ran 102 tests: 101 passed and
one existing repository-wide slot-ratchet test failed. Its four reported sites
are byte-unchanged from the pre-repair revision and match the previous full-suite
failure exactly. The repair-specific tests, actual T4 preparation fixture,
continuation/fused tests, option-prior invariants, and enforced experiment-slot
scan passed. No gate or test was weakened.

An earlier broader dependency run was stopped during the unrelated legacy BRAF
feasibility-frontier search. Before that point, six new audit-fixture tests
failed because the test used a legacy action name with the Active8 codec. The
fixture was corrected to use the existing cycle_close action; all six pass in
the final focused run. The stopped run has no complete-suite result.

New Python files and the continuation module pass lint and formatting. The two
legacy production files retain their existing lint/format debt, with zero new
lint diagnostics. Strict preflight passed at the implementation revision with
zero mounted-tree drift. `git diff --check` passes for the complete staged diff;
raw test XML is retained byte-for-byte.

The repository is not green, and a full scientific milestone or controller
performance improvement is not declared. The previous full-suite failure record
remains in `diagnostics/t4_matched_pilot/`. It was not replayed for this isolated
repair. No remote job was launched and no branch was pushed.

Next: review whether to authorize one repaired, matched end-to-end development
round. It must test completed ring and non-ring proposals, objective, diversity,
intended/realized scale, and work at the same budget. Do not substitute these
counterfactual closures into the already locked candidate set or infer affinity
from the successful predicate checks.

## Reproduction

With byte-verified copies of the original arm ledgers as `reference.json` and
`committor.json` in LEDGER_DIR, publish to a fresh output directory:

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python \
  tools/t4_append_contract_audit.py diagnostics/t4_matched_pilot/attempt_1 \
  LEDGER_DIR NEW_OUTPUT_DIR
```

The committed fixture also supports offline predicate/executor regression via
`tools.t4_append_contract_audit.audit_fixture`, without the original large
ledgers, remote access, model checkpoint, or docking binary.
