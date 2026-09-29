# PMO — black-box molecular optimization (no-prescreen)

**Status: DEVELOPMENT. There is no frozen PMO paper table.**

Every PMO number in this repo is development evidence. Nothing here is a
submitted result, and the reproducer says so on every run. That is the single
most important difference from `experiments/t4/`, which is frozen.

The question: given a task-independent initial population and a scalar oracle
under a fixed call budget, can the controller improve the top-ten score curve?

    objective   maximize AUC of the top-ten mean over charged calls, on [0, 1]
    feedback    scalar oracle only, every call counted against the budget
    budget      1,000 calls is the development horizon; 250 an intermediate
                diagnostic; 10,000 is the official published budget
    suite       23 tasks; development set gsk3b, celecoxib_rediscovery,
                perindopril_mpo

---

## 1. Recompute every number — no credentials, no PyTDC, no oracle calls

```
PYTHONPATH=src python3 tools/reproduce_pmo_tables.py
PYTHONPATH=src python3 tools/reproduce_pmo_tables.py --task celecoxib_rediscovery
```

Needs numpy (for the production AUC) and optionally RDKit for the chemistry
column; the score table is identical without RDKit, and the column is reported
`UNAVAILABLE` rather than silently dropped.

It reads **charged-call receipts** — `<ledger>/oracle/query_NNNNNN/result.json`,
one per call, carrying `index`, `endpoint`, `score`, `role`, `status` — and
recomputes `best`, the top-ten mean, and AUC through the **production**
`compose_v4.control.program_task.pmo_top_ten_auc`, imported rather than
transcribed, so it cannot drift from what the campaign scored.

A self-test runs first: a constant sequence `c` must give
`c * (1 - frequency / (2 * budget))`, the closed form of the trapezoid rule from
(0, 0). If that fails, the environment is wrong and nothing below is
trustworthy.

## 2. Three ways to misread a PMO table, all handled

**AUC is structurally depressed at short budgets.** `top_auc` trapezoids up from
(0, 0), so a run holding a constant top-ten level `c` scores
`c * (1 - frequency / (2 * budget))` — **20% removed at budget 250**, 5% at
1,000, 0.5% at the official 10,000. Every sub-official budget is labelled.
**Never scale a 250-call AUC and compare it to a published 10k figure**; the
error points toward a false negative.

**Three oracles reward leaving the drug-like manifold.** `gsk3b`, `jnk3` and
`drd2` are ML predictors over fingerprints. Measured `r(score, QED)` is **-0.705**
on gsk3b and **-0.750** on jnk3; the gsk3b leader at 0.410 carries two
hypervalent iodines and stacked hydrazines at QED 0.038. Those tasks are flagged
and a chemistry column is printed beside them. The similarity/rediscovery tasks
are measured healthy — their on-manifold best essentially equals their overall
best. **Read the SMILES before banking a number.**

**A zero-oracle gate produces a ledger shaped exactly like a scored run** — same
schema, same task name, same `oracle_protocol` — while its scores come from a
non-production scorer. One such workspace holds a celecoxib "best" of **0.9167**
against a real best of ~0.25. The reproducer excludes any ledger whose own
committed report declares `new_charged_oracle_calls == 0` or a zero-charged
`evidence_role`, detected **from that evidence, not from the folder being named
`dryrun`** — a naming convention is not a guarantee, and this found two such
ledgers rather than the one that was known.

Partial ledgers and ledgers absent from git are labelled too: a local run sees
workspaces a fresh clone will not, and reporting numbers nobody else can
reproduce is exactly the failure this directory exists to prevent.

## 3. Verify the inputs

```
python3 tools/verify_experiment_inputs.py --task pmo
```

The 71 MB of TDC oracle pickles are gitignored and declared as a documented gap:
not needed for §1, required before any gsk3b/jnk3/drd2 run. A fresh clone cannot
build a working image for those three tasks until they are present — and a
constructed oracle is **not** a scoring oracle, so a positive control must assert
a known active scores > 0 before the first charged call.

## 4. Running a scored campaign

Not runnable from a clone: it needs Modal, the pinned image, and a signed
authorization contract. Discipline that has already cost real runs here:

- **A contract budget is only real if the runtime reads it.** `execute_task`
  built its ledger from a module constant while the contract declared something
  else; the launcher's check was contract-consistency, not enforcement. Grep the
  runtime for the key the contract sets.
- **A re-pinned contract on disk does not reach a deployed app until you
  redeploy** — `add_local_file(..., copy=True)` fixes it at image build time.
- **Verify the first container reaches `campaign/round_0000` before calling an
  arm launched.** A five-campaign arm once read as live for three hours having
  charged zero calls; every container had exited `returncode: 1` on an identity
  mismatch. `provenance.json` carries `returncode`.
- Run a matched A/B in a **quiescent tree**. Contract pinning covers only the
  pinned files, and the proposal-synthesis path is not among them — a matched
  result was retracted because three proposal files were edited mid-run.

## 5. Environment — not the same kernel as T4

**python 3.11, rdkit 2023.9.6, PyTDC 1.1.15, numpy 1.26.4, setuptools 69.5.1.**

```
uv venv --python 3.11 <dir>
uv pip install --python <dir>/bin/python 'PyTDC==1.1.15' --no-deps
uv pip install --python <dir>/bin/python rdkit==2023.9.6 numpy==1.26.4 setuptools==69.5.1
```

`--no-deps` because PyTDC 1.1.15 pins `rdkit>=2023.9.5,<2024.3.1`, making an
install against 2024.3.5 unsatisfiable; production adds a `rdkit.six` shim, which
modern RDKit does not ship. `setuptools<81` because `tdc/metadata.py` imports
`pkg_resources`.

The image holds **one** rdkit, so the PMO runtime — oracle *and* COMPOSE executor
— is 2023.9.6, while T4's is 2024.3.5. A PMO number computed locally under
2024.3.5 needs a parity statement.

## 6. Information boundary

Permitted: a task-independent chemical prior, plus feedback **counted** against
the run's own budget. Online adaptation from scored observations is legitimate
no-prescreen optimization.

Not permitted: oracle internals, hidden component scores, uncounted same-task
history, same-task winner routes, prescreened vocabularies. **Computing a
property on an uncounted candidate in order to select it is objective
evaluation.**

A benchmark-declared target counts as a task input only where the evaluated task
actually supplies it. PMO black-box search supplies scalar feedback only — so
handing the controller a reference structure evaluates *reference-conditioned
construction*, which is a legitimate experiment that needs its own label, not
black-box discovery.

## 7. What is established, and what is not

Established by measurement:

- The initialization bank is good chemistry (median QED 0.763) and the search
  walks off it on every task except `qed` itself — the one task whose objective
  *is* drug-likeness. Drift is not universal noise; it tracks the objective.
- Online structural memory beat its matched control on celecoxib at every
  checkpoint to 1,000 calls (AUC 0.3138 vs 0.2489). **One task, one seed.**
- The proposal distribution, not the allocator or the budget, is the binding
  constraint: the controller reaches the 13-atom scale that 58.3% of productive
  transitions need on **0.4%** of proposals.

Not established:

- No result here is a benchmark claim. No frozen table, no 23-task suite run,
  no matched comparison at the official 10,000-call budget.
- Per-seed spread exceeds the treatment effect on every benchmark column in the
  three-seed construction-prior gate, so its means are directions, not verdicts.
