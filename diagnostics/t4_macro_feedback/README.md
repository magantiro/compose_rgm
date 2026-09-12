# Completed macro-feedback episode: retrospective diagnosis

The observed bottleneck is proposal and continuation allocation, not an inability
to execute ring macros or a backlog of undocked eligible products. This is an
inference from one inspected development episode, not a causal ablation.

Run `3e8730df1962c80fcc8cfad050010c76a2e533f6191cd19b4bb25206ffc57c11`
used clean source `4367c0fa75efd3aefd8e3727d516bafd9de7ae73` on PARP1 seed0,
delta=0.4. It completed ten rounds, 160 macro attempts and 27 new dockings, with
zero docking failures. Its starting archive already included 67 attempted calls.
Best observed docking improved from -9.9 to -10.6. This is not a 27-call
cold-start benchmark and does not establish superiority to another method.

## Evidence

The authoritative reduction is
`3e8730df1962c80fcc8cfad050010c76a2e533f6191cd19b4bb25206ffc57c11/diagnosis.json`,
SHA-256 `4652b6c04785a36c39f44f2289346818407ef9eb6da8aed7ddff6daeb74bceda`.
It binds 45 input files, the run revision, 36 imported production-source hashes,
analysis implementation hash, software, thresholds and row-level evidence.

- Execution: 157/160 attempts completed, including 11/12 fused constructions
  and 22/22 pendant constructions. Completion means valid executable output,
  not endpoint feasibility, useful chemistry or docking improvement.
- Final pool: five roots plus 152 distinct generated representatives. Of the
  latter, 28 passed the endpoint screen, one was already in the source archive,
  and all 27 previously unseen eligible identities were docked. None remained
  undocked at the end. Thus 13 of the 40-call allowance remained unused.
- Among 152 generated representatives, 109 failed similarity, 87 QED, 66 SA,
  and 11 the inherited medchem screen; these counts overlap. All 11 medchem
  failures also failed benchmark constraints. Removing only that screen would
  not have added an eligible candidate in this pool.
- Of 80 executed parent slots, 42 used endpoint-ineligible parents. Their 83
  completed children were all ineligible, and their recorded later descendants
  never recovered eligibility. This does not prove recovery impossible.
- The diversity role selected ineligible parents in 25/27 slots; guided draws
  did so in 11/18, and uniform draws in 6/9. Diversity prioritizes previously
  unrepresented topology signatures and Morgan distance, not future recovery.
- Primitive horizon was not exhausted: every completed product retained at
  least 83 of its original 110 primitive slots. Maximum option ancestry was nine.
  Proposal wall time summed to 1795.85 seconds; worker proposal time summed to
  4402.45 seconds across parallel workers. These are different compute units.

## What docking feedback actually controlled

Source inspection of the recorded revision confirms that workers explicitly use
`arm="post_hoc"` and cannot call the docking guide. They draw only two complete
option attempts per selected parent. Docking feedback changes parent selection
and the next round's oracle acquisition, but does not directly choose the option,
attachment site or electronic variant inside a worker. No future-return model is
used to evaluate multi-option continuations. This limitation was declared in the
pilot, but it is material to interpreting the plateau.

The observed -10.6 lineage was pendant aromatic C5N construction (-8.5), pendant
saturated C5N construction (-9.9), carbonyl addition (-10.1), then N-to-C ring
restatement (-10.6). The -10.6 parent received six further attempts across three
rounds, all ineligible: chlorination, ring opening, another pendant ring,
`build_ring_system`, another decoration, and a carbonyl on its sidechain.

A feasible fused-ring product scored -9.0 in round seven, with QED 0.926,
SA 2.797 and similarity 0.604, but received no continuation in the remaining
three rounds. Conversely, an opened, carbonyl-modified candidate with similarity
0.141 and SA 4.635 received repeated diversity continuation. Neither example
establishes that the discarded branch would have won. They demonstrate that
selection was not estimating the downstream value of those branches.

The current guide is not simply devoid of signal. Its locked, prior-only
predictions on these 27 adaptively selected dockings had MAE 0.496 kcal/mol,
versus 0.987 for the corresponding prior-label mean, and ordered 24/29 non-tied
within-round pairs correctly. This is a small descriptive sample, not calibrated
uncertainty or evidence of reliable extrapolation. In particular, the final
guide predicts the diagnostic IVG endpoint at -8.605, versus -9.623 for our
observed best. The diagnostic does not redock IVG or verify its published score
under our pipeline.

## IVG as a diagnostic, not an optimization target

Use the already saved exact, answer-known 21-primitive witness. The five named
stages are execution programs, not five already-supported controller decisions:
the four-primitive linker remodel still lacks a compound linker option. Earlier
conditional probes demonstrate ring and carbonyl support from supplied route
states, not autonomous traversal from the original seed.

Parsing SMILES and comparing molecular graphs shows the missing combination:

| Structural feature | Current best | Diagnostic IVG endpoint |
| --- | --- | --- |
| Sidechain/linker | Retains dimethylaminomethyl branch; rings attached elsewhere | Ethylene-linked peripheral ring system |
| Added ring arrangement | Pendant pyridine and separate cyclohexanone | Benzene fused to carbonyl-bearing six-membered ring |
| Original core | Original carbonyl arrangement | Additional carbonyl-bearing core-ring atom |

None of the 152 generated representatives exactly matches any of the five
completed known-route stages. None contains the queried complete peripheral
ketone fused system or modified core. Three contain an uncarbonylated tetralin
substructure, all on ineligible molecules in a different molecular context.
The precise query SMILES, matches and exclusions are in the artifact. Absence
from this sample is not absence from executor support, and exact winner recovery
is not necessary for good optimization.

All five completed known-route stages pass the endpoint gates under matched
chemistry. The problem therefore cannot be explained solely by a hard
intermediate similarity restriction. This episode already allowed ineligible
intermediates. Its problem was weak coordinated exploration and no demonstrated
recovery-aware allocation, not permission to leave feasibility.

## Proposed next step, not implemented

Keep complete valid states, generic edits, local/global regions and frozen
endpoint constraints. Test task-aware allocation over completed macro branches
and short continuation/repair plans, with explicit exploratory allocation, rather
than another identical episode or a hard filter on all intermediate molecules.
Improve which programs/sites are explored and continued, not only their final
docking rank. The known route can localize proposal/selection failures in an
answer-known diagnostic; winner identity or similarity must not enter a claimed
winner-blind run. This cell is development data; generality needs untouched cells.
No guarantee of IVG-level performance follows from this proposal.

## Work performed

Downloaded existing ledgers only. The deterministic local reduction took 2.46
seconds excluding imports. It verified sealed payload hashes, exact run-source
closure, locked parent identities, unique molecular identities, and reproduction
of every new docked candidate's properties and prior-only prediction. Ruff lint
and format checks passed. No repository-wide suite, new molecule generation,
executor replay, docking, training or scientific launch was performed.
Controller implementation is unchanged; these analysis artifacts remain local.

Reproduce from the pinned clean run source:

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:/private/tmp/compose-macro-feedback-dev/src:. \
  OMP_NUM_THREADS=1 .venv/bin/python diagnostics/t4_macro_feedback/diagnose.py
```
