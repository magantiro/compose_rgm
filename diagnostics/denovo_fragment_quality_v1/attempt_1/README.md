# Saved-endpoint quality diagnosis (development evidence)

This analysis reuses the immutable `A` and `C1` shard records from Modal volume
`compose-denovo-eval-v1/ring_marginal/mstep1000_n200_h16p0_pde2a5f995df7_ie2886d773614/{A,C1}/`.
The per-shard SHA-256 identities, sample counts, configuration, source-code hash
and local software versions are in `report.json`. The pinned lineage-B checkpoint
is SHA-256 `c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c`.

No endpoint was filtered or generated for the A/C1 comparison. Arm A has 190
saved attempts and C1 has 150 of its intended 200. All 340 committed endpoints
are exact-valid, connected and unique in these samples. Recalculated QED and
synthetic accessibility (SA) scores, quality counts and arm means agree with
the frozen official arm report to 1e-10 despite the local RDKit 2025.09.6
being newer than the original RDKit 2024.03.5. This is not a new FCD result.

`size_support_matched_reference/report.json` compares each generated molecule
with three exactly heavy-atom-matched GuacaMol training molecules drawn from a
fixed 100,000-row BLAKE2b-ranked pool, restricted to the checkpoint's C/N/O/F
vocabulary. All 340 generated molecules have all three matches (1,020 reference
records, zero missing). This is a descriptive, not independent test cohort.

The post-ring C1 quality deficit is distributed across low SA fragment scores,
excess N/F, fewer aromatic rings and excess flexible chains. In C1's 85
QED-fail/SA-fail molecules, mean SA fragment score is 0.434 versus 1.574 in
size/support-matched training molecules; mean aromatic rings 1.824 versus
2.694; rotatable bonds 10.29 versus 5.85; and mean F atoms 1.294 versus
0.569. Twenty-eight QED-pass/SA-fail molecules contain 0.357 nonaromatic
unsaturated five/six-membered rings per molecule versus 0.036 in matched
training molecules. The global ring-size repair did not repair these local
electronic and composition patterns.

The `traces/` files replay four selected C1 seeds with the unchanged checkpoint,
source prior, ring plan and production sampler. Every replay reproduces its
saved final SMILES exactly. In three selected low-quality traces, F births came
from `atom_restate` and unusual nonaromatic unsaturated rings first appeared
under `ring_system_grow`; the quality-pass trace had neither birth. These are
selected cases, not an estimate of population-level event attribution.

The saved arm report has 319/393 requested ring systems realized (81.17%) and
74/393 refused for host support (18.83%). Quality is 7/88 for C1 molecules
whose plan was fully realized and 10/62 for incomplete plans. Host refusal is
real, but this observational contrast does not support claiming that refusal
alone causes low quality. No proposal change is promoted by this diagnosis.
