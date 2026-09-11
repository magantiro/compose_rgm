# The balanced option prior did not protect continued exploration

The saved complete-option SMC run provides a more specific diagnosis than its
unchanged best score. This retrospective audit uses the previously verified
result and no new oracle calls, model calls or rollouts. `survival_audit.json`
binds that result, its existing verification report, analyzer and software.

Immediate-score resampling at the first boundary changed five distinct starting
molecules into two retained starts: seven particles descended from the incumbent
and one from original root zero. That last alternative failed its next fused
construction. From boundary two onward, every surviving particle descended from
the incumbent. The seven survivors still had multiple distinct molecular states;
one retained start does not mean seven identical molecules.

Across boundaries two through six, the immediate arm completed 35 options:
six improved their parent, seven tied, and 22 worsened it. Those completed
options inserted four atoms and deleted six in total, with no completed
`construct:*` ring program. The independent arm completed 40 options in those
boundaries, including five construction programs and 41 inserted atoms. These
are correlated outcomes from one warm run, not precision estimates or evidence
of general superiority of independent sampling.

The first resampling discarded this complete pendant-ring proposal:

```
Cc1n[nH]c(C)c1CC(C)C(=O)NC1CCC(C)(C)CC1c1cc[nH]c1
```

Its actual score improved from 0.143969 to 0.354943. The exact shared reference
proposal's sampled suffix reached 0.372104. That is still below the incumbent's
0.522233. None of the five discarded first-boundary proposals had a saved
reference suffix that surpassed the incumbent. Thus the run shows lost
exploration, but not a known winning continuation incorrectly discarded.
Unobserved suffixes remain unknown.

The retained-start census uses the exact starting-parent identities and saved
resampling indices, not `node.root_id`: the incumbent is itself a historical
descendant of one original root. Every proposed parent SMILES, actual parent
score and exact chain prefix is checked against the preceding resampled state.
Failure is not counted as another discarded scored candidate. All evaluated
candidates remain in the archive, even when continuation effort is lost.

Interpretation: a positive generic/option exploration floor applies conditional
on a selected parent. It cannot restore a discarded lineage, and does not assure
space for ring growth once the live population is near the size ceiling.
The pending replacement-enabled continuation experiment directly tests whether
constructive replacement changes this behavior. Its fixed recipe is not altered
after this audit. If collapse persists, the next design decision must address
population-level exploration and proposal improvement, not simply repeat the
same scalar resampling with more calls. No such intervention has been measured
here, and this is not a claim that SMC itself is unsuitable.

Reproduce with `tools/pmo_particle_survival_audit.py RESULT --verified-report
diagnostics/pmo_option_particles/report.json --output
diagnostics/pmo_option_particles/survival_audit.json` in the pinned chemistry
environment. Analyzer execution and its full saved-ancestry checks passed;
Ruff passed. No repository-wide suite or new neural qualification was run.

Operational status: continuation source checkpoint `2d524447731a` is prepared.
Modal deployment was rejected pending explicit approval of the private source
and bundled-input upload. It has not been retried, and no new remote job or
oracle call was started during this analysis.
