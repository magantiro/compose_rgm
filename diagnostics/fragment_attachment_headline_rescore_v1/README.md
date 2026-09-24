# Frozen fragment runs, corrected official metric population

This is a retrospective rescore, not a new generation run. The authoritative
machine-readable artifact is `result.json` (SHA-256
`a9abd445709153038787434fe7ef10663ce1b3c5546364e002d5316a8eeefed9`).
It hashes all 180 frozen shards, verifies their identity and 100-attempt
denominators, reproduces every saved task-filtered official metric, then calls
the pinned InVirtuoGen evaluator on every chemically committed endpoint plus
an empty placeholder for each no-output attempt. Source molecules and failed
attempts were not changed. Each task/arm has ten prompts, three seeds, and
3,000 attempts. The summary is the unweighted prompt mean within each seed,
then the mean over seeds.

| Arm | Task | Chemical validity (%) | Uniqueness (%) | Quality (%) | Diversity | Prompt fidelity/attempt (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | Motif | 85.40 | 97.01 | 13.57 | .775 | 41.43 |
| Attachment control | Motif | 85.90 | 76.58 | 14.70 | .602 | 83.23 |
| Frozen baseline | Decoration | 93.47 | 97.93 | 37.27 | .726 | 3.50 |
| Attachment control | Decoration | 93.77 | 95.89 | 28.07 | .587 | 84.17 |
| Either arm | Superstructure | 93.63 | 97.59 | 37.73 | .726 | 93.53 |

The table's **validity is ordinary chemical output per attempt**, matching
the upstream evaluator's definition. All committed molecules were chemically
valid and connected. Prompt fidelity is separate: the old runner passed only
task-compliant emissions to the official evaluator, which made its apparent
"validity" a stricter composite. It must not be presented as the published
comparator's validity. The corrected `official_chemical` rows are computed on
the proper population. Distance in `result.json` is the pinned IVG molecular
fingerprint estimator against the dummy-stripped prompt reference; it is
separate from the released IVG evaluator and must not be assumed identical to
GenMol's published Distance without an estimator/reference parity check.

Interpretation is negative as well as positive. Attachment-aware control
substantially improves where valid edits land, but permanent interface
restriction collapses motif uniqueness on two prompts (Baricitinib about
44-56%, Lovastatin about 12-15%) and lowers overall motif diversity. Decoration
loses 9.2 quality points and .139 diversity while fidelity rises. Thus the
current attachment-controlled rows are **not** an unqualified benchmark win.
The unconstrained baseline is not a substitute for the requested task: only
3.5% of its decoration attempts satisfy the prompt. A later mechanism may
trade some excess placement fidelity for realistic diversity/quality, but it
must be chosen on new development evidence and a frozen independent test, not
by retroactively selecting among these scored rows.

The superstructure row above is the historical sampler, not the new opt-in
first-family/strict-lock pilot. That pilot's 200/200 result is development
evidence only; the full official row for the repaired sampler remains to run.
