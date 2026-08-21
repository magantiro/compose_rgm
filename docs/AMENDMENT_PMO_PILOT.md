# Amendment — PMO stratified pilot

Recorded 2026-08-21, BEFORE any PMO search has been run and before any PMO
result exists. Task selection is fixed here so it cannot be chosen after seeing
which landscapes COMPOSE happens to suit.

## The question

Not "can we beat GenMol on six rows". The pilot asks:

> Does the same COMPOSE search procedure behave sensibly across qualitatively
> different molecular objective landscapes?

The experimental invariant, which is the whole point:

    R_theta FIXED  +  the SAME generic search algorithm  +  the same oracle
    budget convention,  with ONLY the PMO objective changing.

No per-task tuning. No per-task policy. If a task needs its own optimizer, that
is a finding, not a fix.

## The six tasks, one per landscape type

    simple property     qed
    learned activity    gsk3b
    similarity          albuterol_similarity
    rediscovery         celecoxib_rediscovery
    MPO / composite     perindopril_mpo
    structural          deco_hop

Chosen to span landscape geometry, not difficulty. Two notes on selection:

  - valsartan_smarts is DELIBERATELY EXCLUDED despite being the obvious
    structural candidate. It returns ~0 for nearly every published method and
    for our own probe molecule, so it cannot discriminate between a healthy and
    an unhealthy search. Including it would add a row that says nothing.
  - gsk3b rather than jnk3 for learned activity, arbitrarily; both rest on the
    same extraction and the same Gate A. If Gate A fails, this row is pulled
    and NOT silently replaced with an easier task.

## Protocol

    budget            10,000 oracle calls per task, enforced by OracleMeter,
                      which raises rather than warns
    counting          PER_MOLECULE, canonicalised before cache lookup, repeats
                      free, matching PMO's mol_buffer
    replicates        3 stochastic runs per task
    primary metric    AUC of top-10 score against oracle calls, PMO's own
    also recorded     top-10 trajectory, calls to first improvement, distinct
                      molecules evaluated, wall clock

    initialization    objective-blind, fixed before any task is run, identical
                      across all six tasks
    controller        frozen R_theta + generic population search. NO h_phi, NO
                      twisted SMC, NO Doob. Those are barred for the pilot.

## No prescreen

The nominal 10,000 calls are the entire task-oracle budget. We do not screen a
library with the task oracle beforehand. Where published numbers were obtained
with a prescreen, that difference is stated wherever the comparison appears
rather than absorbed silently.

## Read-out, decided in advance

    most or all six improve steadily      -> run the full 23-task benchmark
    property tasks work, remote-target
      tasks (rediscovery, similarity) flat -> report the boundary honestly; the
                                             frozen process biases local search
                                             but does not navigate to distant
                                             targets
    broadly flat                          -> PMO is not a COMPOSE benchmark;
                                             say so and stop

The second outcome is a real result and must not be treated as a failure to be
engineered away by adding machinery until the numbers move.

## Barred

Per-task hyperparameters. Choosing replacement tasks after seeing results.
Adding h_phi, SMC or lookahead to rescue a flat curve during the pilot.
Reporting GenMol's 18.362 as current SOTA; the InVirtuoGen 18.993 figure and
the prescreen claim are unverified secondary-source numbers and must be checked
against primary PDFs before either appears in the paper.

## Preconditions

Gate A (estimator extraction parity for jnk3, gsk3b, drd2) must pass first.
Everything else is closed: meter and budget semantics, QED parity at 0.00e+00,
composite MPO construction re-derived at 0.00e+00, featurization parity at
0.00e+00 over 200 molecules.

After Gate A, infrastructure validation STOPS. No further oracle audits unless
an actual discrepancy surfaces during the pilot.
