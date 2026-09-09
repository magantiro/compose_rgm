# Production task-search audit: guidance did not activate

2026-09-08. **Decision: diagnose before docking.** The eight-parent audit
completed, and all 100 selected primitive edits passed exact production-executor
replay. The planning gate failed. No docking comparison or new oracle call was
launched. This is development evidence on the already inspected PARP1 seed0,
d=0.4 cell, not a benchmark superiority result.

## Measured outcome

| Quantity | Result |
| --- | ---: |
| Parents completed and persisted | 8/8 |
| Planning rollouts completed | 0/8 attempted |
| Planning executor allowance consumed | 512/512 for every parent |
| Non-reference WHERE decisions | 0/40 |
| Non-reference WHAT decisions | 0/37 |
| Non-reference HOW decisions | 0/100 |
| Unique completed-option candidates | 33 |
| Feasible, undocked candidates | 4/33, from 3 parents |
| Search executor calls | 19,648/20,000 |
| Verification executor calls | 100, reported separately |
| Proposal wall time, summed parent units | 1,272.854903 s |
| Total remote wall time | 1,439.347599 s |
| New oracle attempts | 0 |

The largest decision total-variation change from its reference was below
3e-16, numerical roundoff rather than task guidance. Seven parents reached the
2,500-call executor cap; one completed all 16 primitive edits in 2,148 calls.
Committed primitive depths were 7, 16, 10, 14, 14, 13, 13 and 13. A completed
parent checkpoint is an accounted outcome, not necessarily a completed rewrite
at the full edit horizon.

The run used one CPU, an 8-GiB allocation, torch.float32 generator parameters,
NumPy 1.26.4 and RDKit 2024.03.5. Runtime initialization and frozen-input checks
took 129.866902 s. Peak RSS is retained in the remote result in native units.
There is no accelerator allocation or claim that the development estimate was
an independently measured dollar charge.

## Chemistry, scale and diversity

Nine parameterized construction options were selected, and seven completed:
six pendant rings and one fused ring. Both five- and six-membered construction
programs appear, including aromatic and nonaromatic heterocyclic cases. The
other selected options were generic (12), open (4), restate (3), append (2),
and aromatize, cyclize, local, rebuild, scaffold_extend, shrink and small_ring
(one each). Exact option identifiers and counts are in result.json.

Across 33 unique candidates, cycle rank increased for 10 and decreased for 3.
Ring-system deltas were +1 for 8, zero for 23, and -1 for 2. These are different
topological quantities; generic/cyclize ring changes are not automatically
classified as desirable medicinal chemistry. Constraint failures overlap:
27 candidates failed source similarity, 11 failed QED and 9 failed SA.

Intended region fractions ranged from 0.028571 to 0.857143; realized coherent
fractions ranged from 0.028571 to 0.269231. Those are not interchangeable.
The four feasible candidates were:

| Option | Primitive depth | Intended fraction | Realized coherent fraction | Cycle-rank delta | Ring-system delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| restate | 1 | 0.357143 | 0.071429 | 0 | 0 |
| restate | 1 | 0.592593 | 0.074074 | 0 | 0 |
| open | 1 | 0.428571 | 0.071429 | 0 | 0 |
| pendant six-member C5N ring | 8 | 0.074074 | 0.259259 | +1 | +1 |

The last candidate adds a saturated nitrogen-containing pendant six-membered
ring through the primitive executor. Its QED is 0.667765, SA is 3.603371 and
original-source similarity is 0.458333. It is feasible but **not docked**.
Its predicted score is not an observed docking score. Another lineage completed
two successive pendant five-membered rings, but those outcomes failed T4
constraints. Valid construction capability did not establish useful task control.

## Interpretation and next action

**Measured:** each planning attempt exhausted its compute allowance before
reaching a terminal evaluation. The new guidance therefore never obtained a
completed return. Its chronological prediction result remains a separate,
retrospective finding; this experiment did not test whether active guidance
improves docking.

**Code-supported interpretation:** constructing full executed successor rows
before following a sampled path is too costly for the declared planning
allowance and 16-step terminal horizon. No ring-count bonus, new macro catalog,
generator retraining, larger KL radius or additional docking spend addresses
that measured failure directly.

**Proposed next repair:** reduce physical successor enumeration during reference
trajectory sampling, with a small law-parity and cost check on one saved parent
before another full audit. Preserve admissible support and reference probabilities;
do not interpret an interrupted trajectory as terminal reward zero. Separately,
the current planner values only the full edit horizon while the emitter accepts
earlier completed options. Any change to that stopping/value policy must be
recorded explicitly, not silently introduced as a speed optimization.

The four locked candidates and every completed parent remain reusable assets,
not an authorization to bypass the failed gate. Best observed docking remains
-9.7 at 51 prior calls. No new scientific job is running.

## Provenance and reproduction

Scientific code: `626c0a2230cdc21b7c8bb0546da14b323ebb94eb`.
Contract: configs/t4_task_search_audit.json, physical SHA-256
`ae7b9d88e7a48bb8959618fa954ba9773d5617e8476b8e54e08d37c6e9e63778`.
The self-hashed contract and run metadata retain all source/checkpoint hashes,
configuration, seed derivation and stop limits.

Modal app: genmol-t4-opt. Call: `fc-01M21SBKSMVSVVY641W68YQED3`.
Volume: compose-v4-artifacts. Immutable run prefix:

```text
t4_task_search_audit/8f0f86eb38ad5edfbb03d1b9cb3a879dafb1a75c38d03aa594573f0831312005
```

- result.json SHA-256: `fa64b8525921a0775bfd551279c079abaca8240a8c02d055940cca4b852c66cb`
- verification.json SHA-256: `cca89e67e96a89b660eb653a4c0a52e06fa150dae5f19fbad3e036a8a7075e59`
- Full candidate_lock.json SHA-256: `4696ee9ceaf2bb6642433868eeacfc9d7784b887613f82f42010531e636abedb`
- Compact review.json SHA-256: `09d2868d0017839bb62db9588680076de07a6b805c9b1cef0541d7e5c0854d5b`

The full 62-MiB lock and parent executor ledgers are retained on the volume;
local copies live in ignored cache/. The committed compact review preserves
every candidate's identity, properties, intended/realized scale and topology,
along with the physical source hashes and exact projection implementation.
Reproduce that projection without molecular computation:

```sh
python tools/t4_task_search_review.py diagnostics/t4_task_search_audit
```

Verification before launch: clean-source preflight passed; 107 focused tests
passed in 8.60 s (focused_tests.xml); touched core/launcher/test files passed
Ruff and formatting checks, and the production import chain resolved. The
large legacy Modal file compiled and retains its pre-existing lint findings;
unrelated cleanup was not performed. No repository-wide release suite was
rerun for this bounded T4 development audit. The first restricted deployment
connection failed; the subsequent network-enabled deployment succeeded before
the single scientific spawn. Changes are committed, not pushed.
