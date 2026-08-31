# T4 controller: conclusion of the broad-semantics investigation

Date 2026-08-27. Supersedes the working hypotheses in the four
`diagnostics/broad_c_*.md` files, which remain valid as records of what each
run measured.

## CORRECTION TO PRESERVE

An earlier claim in this investigation -- that the production realization rule
`anchor_rank=0` FAILED on the decisive winning state -- was **FALSE**. It came
from grepping a Modal log where the SMILES wrapped onto a filtered-out line,
and was reported without checking the artifact that had been written for
exactly that purpose.

Canonical verification against `diagnostics/replay_audit_5ht1b.json`:

    state            FC(F)(c1ccccc1)c1cccc(N2CC[NH2+]CC2)c1     (B r12 prefix-2)
    production_built 'FC(F)(c1ccc(-c2ccccc2)cc1)c1cccc(N2CC[NH2+]CC2)c1'  len=49
    production_stage None
    canonical(built) == canonical(B -11.30 target)  ->  True

**Do not reopen realization selection on the basis of that erroneous log read.**
There is no demonstrated realization defect on the winning path.

## The traced result

The B r12 -11.30 molecule was followed end to end:

    can COMPOSE make the useful ring?                     yes
    at the useful state/site?                             yes
    is the good endpoint among broad realizations?        yes (14 real., 10 canonical)
    does R_theta rank it reasonably?                      yes (index 2 of 10)
    does production anchor_rank=0 build it?               yes (exact match)
    does broad C sample the useful semantic often enough?  NO

One bottleneck: proposal probability.

    narrow B   P(linked/6/C6/aromatic) = 33.3%  ->  P(>=2 in 6 slots) ~ 33%
    broad C    P(same)                =  1.0%  ->  P(>=2 in 6 slots) ~ 0.14%

The winning prefix is `shrink -> s -> restate -> s`, i.e. TWO occurrences of s,
not four (an earlier count of four misread `prefix_len 4` on a 6-macro trace).
Two is routine at 1/3 and essentially unreachable at 1%.

## Architectural conclusion

    broad support  !=  flat granular action space

The executor/compiler genuinely supports linked/fused x sizes x exact
stoichiometries x saturated/aromatic. That capability is real and stays. What
does NOT follow is that every executable molecular detail should be an
independently learned top-level control variable: a CEM fed ~10 docking labels
per round cannot assign choice probability across hundreds of peer actions.

Narrow B is therefore NOT a workaround for broken chemistry. It is a sensible
COMPACT CONTROL VOCABULARY over a much broader executable process, and it is
the production controller for T4.

## Standing decisions

* B is the T4 production controller. Broad-C performance experiments are closed.
* Do not fix the realization layer; no defect was demonstrated.
* Persistence / hierarchical control is a legitimate research EXTENSION if
  5HT1B needs pushing past -11.3. It is no longer a fix for a known defect.
* Ring capability (exact stoichiometry, fused hetero, sizes) stays in the
  compiler and is available to any future controller.

## Independent finding: the two benchmarks want opposite things

    GriDDD QED   ring additions 0/37 (0%)   d(heavy) mean +0.3   QED 0.774 -> 0.914
    T4 docking   2-4 ring adds per winner   d(heavy) +10..+12 on small seeds

Same frozen R_theta and executor; only the objective differs. Ring machinery
buys nothing on the QED task, so the QED result is not evidence about the ring
controller in either direction.
