# Amendment — PMO (GenMol Table 3) oracle-compatibility gate

Recorded 2026-08-21, before any COMPOSE search on PMO. Infrastructure only.

## Why PMO is in scope

The 2026-08-15 scope lock expired on its own terms: it ran "until the frozen
64-source QED dev panel result lands", and the 128-source panel has since been
reported. That lock's own roadmap listed benchmark work as an authorised
parallel lane. PMO extends the MOLLEO benchmark lane rather than opening a new
one.

## Environment

PyTDC 0.3.6, the version the reference PMO repository recommends, installed in
an ISOLATED venv (`.venv_tdc`, gitignored). It is deliberately NOT installed
into the pinned chemistry environment, whose RDKit version is load-bearing for
the state-space definition.

Two compat shims are required on a modern stack. Both are environment fixes and
touch no oracle arithmetic:

  1. `rdkit.six` — PyTDC 0.3.6 does `from rdkit.six import iteritems`. RDKit
     removed its vendored `six` years ago. Shimmed with `dict.items`.
  2. TDC's rdkit guards are bare `except:` blocks that re-raise a misleading
     "install rdkit" ImportError, hiding the real cause. Do not trust that
     message; unmask it before diagnosing.

## RESULT: 20 of 23 oracles work; the 3 that fail are the 3 we already own

    WORKING (20)   albuterol_similarity, amlodipine_mpo, celecoxib_rediscovery,
                   deco_hop, fexofenadine_mpo, isomers_c7h8n2o2,
                   isomers_c9h10n2o2pf2cl, median1, median2,
                   mestranol_similarity, osimertinib_mpo, perindopril_mpo, qed,
                   ranolazine_mpo, scaffold_hop, sitagliptin_mpo,
                   thiothixene_rediscovery, troglitazone_rediscovery,
                   valsartan_smarts, zaleplon_mpo

    BLOCKED (3)    jnk3, gsk3b   ModuleNotFoundError sklearn.ensemble.forest
                   drd2          ModuleNotFoundError sklearn.svm.classes

The blocked three are precisely the pickle-backed oracles, and precisely the
three already frozen locally as checksummed npz in
`src/compose_v4/benchmark/oracles/`. TDC supplies the 20 we lack; our
extraction supplies the 3 TDC cannot load on any modern stack.

## The pickle wall is structural, not a setup error

Aliasing `sklearn.ensemble.forest -> sklearn.ensemble._forest` gets past the
import and then hits:

    ValueError: node array from the pickle has an incompatible dtype
    ... expected [..., 'missing_go_to_left']

The sklearn tree node struct gained a field after 0.21. The binary layout, not
the module path, is the wall. A live TDC-vs-ours parity check therefore
requires sklearn < 0.22, hence Python <= 3.8.

This independently vindicates the existing frozen-extraction design, which
`oracles/forest.py` documents: open the pickle once in a pinned environment,
write flat node tables, walk them in numpy forever after.

## Parity so far

QED, the one oracle live on both sides, agrees EXACTLY:

    max |TDC - ours| = 0.00e+00 over a 7-molecule panel,
    across different RDKit versions (2026.03.5 vs 2025.09.6).

That validates the wrapper plumbing end to end and shows QED does not suffer
the RDKit stereo-perception drift that `oracles/task3.py` records for SA.

## Remaining gates, in order

1. Verify the EXTRACTION, not the oracle. In a Python 3.8 / sklearn 0.21.3
   container, unpickle the TDC `jnk3.pkl`, `gsk3b.pkl`, `drd2.pkl` and confirm
   they reproduce the frozen npz we ship. This is stronger than an oracle
   parity check: it validates the artifact everything downstream depends on.
2. Wire the 20 TDC oracles behind the existing `OracleMeter` and verify the
   10,000-call ceiling is charged on the unit PMO expects. Do NOT relax the
   metering to imitate GenMol's uncharged-evaluation behaviour; the MOLLEO
   audit in `benchmark/molleo_task3.py` already refused to copy that leak.
3. Reproduce one composite MPO objective from its components, to check the
   wrapper reproduces PMO's objective construction and not merely its
   component predictors.
4. Only then choose a task-type-stratified pilot from the canonical suite and
   price it.

## Barred

Prescreening ZINC250k with the task oracle before the nominal 10k budget.
Reporting GenMol's 18.362 as current SOTA without checking the primary PDFs;
the InVirtuoGen 18.993 figure and the prescreen claim are both currently
unverified secondary-source numbers.
