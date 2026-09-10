# Public PMO molecule inspection, 2026-09-10

Purpose: inspect released candidates to diagnose search limitations, not import
winner structures, motifs, labels or routes into the controller. Any future
development informed by these examples is not blind evaluation on these tasks.

## Verified public artifacts

InVirtuoGen releases a raster figure with ten perindopril-MPO example molecules,
drawings, printed SMILES and reward labels in
[top_smiles_cuda:0.pdf](https://github.com/invirtuolabs/InVirtuoGen_results/blob/main/plots/tdc/perindopril_mpo/top_smiles_cuda%3A0.pdf).
The downloaded immutable Git blob is
`1dc8abb1399ee6dbf9220bc3059201541fc6cf3b`, SHA-256
`b034d0cc3292e08036de4f8933e133cb54373830421fec57dd219c371a2eab65`.
Text extraction returns no text: the PDF is one 1500x900 raster image. Printed
SMILES are not a machine-readable molecule table and have not been transcribed
or independently rescored here.

The figure labels examples around 0.80-0.81, but it is not bound to the three
no-prescreen runs by the inspected metadata. Moreover its first structure labeled
"Target" visually differs from the reference in the task's source definition.
Do not use this figure as exact evidence for a no-prescreen winner or silently
treat its target caption as authoritative.

The separate [no-prescreen result file](https://github.com/invirtuolabs/InVirtuoGen_results/blob/main/results/target_property/no_prescreen_3_runs/perindopril_mpo/results_perindopril_mpo.csv)
contains three aggregate rows, not SMILES. Reported best scores are
0.7071067812, 0.7794078749, and 0.6879057244. These are 10,000-query runs, unlike
our 100-query development pilot. The SHA-256 is
`8cd65d9c98cac09f9233ef6ce425d1cd35d8bb9c7a2851cccd14172af811346a`.
Both raw files and a source manifest are retained in the main workspace under
`diagnostics/pmo_public_molecules/ivg/`. Access is public research inspection under
the repository's CC-BY-NC-SA-4.0 terms. No model or executable external code was run.

The inspected current Git trees of Genetic GFN, the original mol_opt benchmark,
and ChemLactica did not expose a matching perindopril result table in the searched
paths. This is a bounded inventory finding, not proof that their molecules are
unavailable elsewhere. The pasted search-engine summary is not verified evidence
for its specific Mol-E/MolEditRL/Table-16 claims.

## What is useful for this controller

The [task definition](https://github.com/BenevolentAI/guacamol/blob/master/guacamol/standard_benchmarks.py#L249)
combines ECFP4 similarity to perindopril with a Gaussian preference for two
aromatic rings. It does not reward adding rings indiscriminately. The inspected
IVG examples visually retain much of the perindopril-like carbonyl/ester/acid
and saturated-ring chemistry while adding aromatic structure. This is qualitative
inspection, not an exact route or representability proof.

Our adaptive snapshot at 62 queries has 60 completed option attempts, including
14/14 completed ring-construction programs and 15 cycle-rank increases. Some
descendants combine several ring additions; max ancestry is nine decisions and
26 primitives. The best snapshot candidate (0.4516053237) follows eight options,
has two aromatic rings, and occupies all 40 supported heavy-atom slots. Hence the
observed gap is not simply absence of ring construction. State-conditional choice
of sites, heteroatoms, carbonyl placement, preservation and removal is still a
search-quality hypothesis, not established by these counts.

Next diagnostic: compare exact graphs and objective components of already locked
candidates. Separate representability, proposal probability, retention and task
value. Do not change option weights or the running recipe after looking at these
examples. A target-informed route is a development diagnostic, not autonomous
winner discovery.

## Completed development pair

Both arms finished at their locked 100-query budget. Balanced best: 0.491054734;
adaptive best: 0.460348270. Balanced top-ten AUC: 0.395920108; adaptive:
0.387317135. Adaptive allocation changed the sampling rows, but did not improve
the declared objective in this one pair. This is a negative developmental result,
not statistical evidence that adaptation cannot help. See
`diagnostics/pmo_archive_pilot_100/README.md` for immutable receipts and limits.

The appropriate next question is not whether to add more ring templates. It is
whether task feedback can distinguish useful *state-specific edits*, including
site/payload choice and replacements when the molecule fills the size limit.
Current option credit is pooled across molecular contexts and HOW remains task
blind. This is an implementation limitation to test, not proof of the cause of
the observed score gap. Preserve the balanced arm as the same-generator baseline.
