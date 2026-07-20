# COMPOSE-Lipid structural pretraining corpus audit

**Audit date:** 2026-07-20  
**Decision:** freeze a measured/observed R0 now; do not count reported or inaccessible libraries; qualify reaction-grounded enumeration separately.

## Executive decision

The local, legally attributable full-structure sources contain **15,433 unique measured/observed canonical ionizable lipids** after exact structure deduplication. Adding the released AGILE virtual-candidate file gives **26,509 unique full structures**, of which **11,076 are virtual-only additions**. No molecule was enumerated in this audit.

The 15,433-structure R0 is the defensible seed corpus. It is useful for initial structural pretraining and for measuring architecture coverage, but it is not a million-scale corpus and should not be described as one. The shortest credible route to scale is:

1. request LiGen's reported 900,351 structures and its data license;
2. independently qualify several reaction transforms from primary substrate/product examples;
3. build a 50k–100k streaming, stratified, route-certified R1 pilot;
4. expand to 500k–1M only after chemistry, coverage, duplication, throughput, and held-component leakage gates pass.

LipidBERT's reported 10M library is in-house and unavailable. Ouyang et al.'s near-20M library is a narrow fixed-template construction and its generated structure/prediction artifact is available on request, not a public general-purpose training prior. Neither is counted.

## Scope boundary

This is the **structural pretraining corpus**, not the lung oracle corpus. LNP formulation or organ-delivery labels are not inherited by these rows. The pan-lung oracle is a separate, study-indexed evidence union spanning all recoverable lung datasets; LuT is one constituent, not the target dataset. Keeping these lanes separate prevents formulation measurements from being silently assigned to a molecular structure after deduplication.

## Frozen canonicalization contract

- RDKit 2025.09.6.
- Canonical isomeric SMILES; explicit hydrogens removed.
- No tautomer normalization.
- No charge neutralization.
- No salt or fragment stripping.
- Eligibility requires a parseable, single connected component.
- Exact input paths, hashes, row counts, parse failures, and canonical collisions are recorded.

This intentionally conservative contract avoids inventing equivalence between protonation states, tautomers, stereoisomers, salts, or multicomponent formulations. A future standardization release must be versioned independently rather than silently replacing R0 identities.

## Obtained full-structure inventory

| Source partition | Local rows | Unique canonical structures | R0 status | Notes |
|---|---:|---:|---|---|
| LNPDB | 19,797 | 12,837 | included | measured/commercial, literature aggregate |
| LiON released aggregate | 13,069 | 9,194 | included | measured formulation occurrences |
| AGILE measured | 1,200 | 1,200 | included | Ugi 3CR products with A/B/C components |
| LUMI 4CR-1920 | 1,920 | 1,920 | included | measured Ugi 4CR, component codes retained |
| AGILE virtual candidates | 12,276 | 12,276 | excluded from R0 | retained for route audit and possible R1 |

All five files had zero RDKit parse failures under the frozen policy.

## Exact overlap and net contribution

- **R0 observed/measured union: 15,433 unique structures.**
- **All obtained union including AGILE virtual candidates: 26,509.**
- **AGILE virtual-only addition beyond R0: 11,076.**
- LNPDB–LiON overlap: 8,518 structures, covering 92.65% of LiON and 66.36% of LNPDB.
- AGILE measured–LNPDB overlap: 1,180 of 1,200 (98.33%). AGILE measured contributes no structure exclusive to R0 after all real sources are combined.
- AGILE measured–LiON overlap: 1,000 of 1,200 (83.33%).
- LUMI contributes 1,920 exact-structure additions and has zero exact overlap with the other obtained partitions.
- Across the all-obtained union, 17,591 structures occur in one source, 7,800 in two, 236 in three, and 882 in four.

These results show why reported row counts cannot be summed. The net observed union is substantially smaller than LNPDB plus LiON plus AGILE plus LUMI because LNPDB, LiON, and AGILE strongly overlap.

## R0 chemistry profile and kernel implications

R0 has a median of 53 heavy atoms, mean 60.25, 95th percentile 114, and maximum 282. Only 2,968 structures fit a 40-heavy-atom cap; 10,877 fit 64, 14,079 fit 96, and 14,893 fit 128. Therefore the current small-molecule kernel cannot train on the full observed lipid distribution without an explicit size-capacity decision.

- 52.66% are ring-bearing and 12.33% contain an aromatic ring.
- 3.30% have nonzero formal charge under the frozen representation.
- 74.37% contain at least one assigned or unassigned stereogenic atom.
- Element presence includes C/N/O plus F, P, S, and Si.

Before broad lipid training, the implementation must explicitly support the selected atom cap and the required elements, formal charges, stereochemistry representation, and long-chain batching. Molecules outside a pilot kernel cap must be reported as exclusions, not truncated.

## Source availability audit

- [LNPDB](https://github.com/evancollins1/LNPDB), [LiON](https://github.com/jswitten/LNP_ML), [AGILE](https://github.com/bowang-lab/AGILE), and [LUMI](https://zenodo.org/records/17771224) yielded usable full structures with attributable repository or dataset terms.
- [LiGen](https://aclanthology.org/2026.acl-long.392/) reports 900,351 virtual structures, but the official paper page and paper expose no downloadable corpus or independently licensed structure artifact. It is the highest-leverage author request, not an available count.
- [LipidBERT/METiS](https://arxiv.org/abs/2408.06150) reports a 10M in-house library and does not release a reproducible corpus.
- [Ouyang et al.](https://www.nature.com/articles/s41467-024-55072-6) report nearly 20M fixed-template products from 892 tails and 237 heads, with the second and third tails constrained equal. The generated library/predictions are available on request, and the construction is too narrow to define the general prior alone.
- [Bowen Li et al.](https://www.nature.com/articles/s41563-024-01867-3) report 584 measured and 40,000 virtual 4CR lipids. The accessible supplement does not provide a verified machine-readable full library.
- LuT component labels were recovered, but not full product structures. LuT is excluded from structural pretraining until a complete product representation or a qualified transform plus frozen components is available.

## R0 release contract

`compose_lipid_r0_observed_v1` contains one row per canonical observed structure with:

- all observed and all available source memberships;
- retained source occurrence counts and record identifiers;
- source metadata and region/component annotations where provided;
- a canonical-structure leakage group;
- frozen study, component, and reaction-family holdout fields;
- an explicit retrospective/prospective lock;
- an explicit rule prohibiting biological-label inheritance in structural pretraining.

The release is directly executable as a structural pretraining input. It is not yet a final train/validation/test assignment: the final split compiler must enforce canonical-structure, study, component, linker/scaffold, and reaction-family exclusions jointly.

## Reaction-grounded R1 plan

The reaction registry currently contains three `literature_spec` entries: primary-amine/epoxide opening, amine–acrylate aza-Michael addition, and AGILE Ugi 3CR. They deliberately contain no reaction SMARTS because no transform has yet passed the required substrate/product reconstruction.

A transform may be promoted to `qualified_for_enumeration` only after:

1. an exact primary-source locator and atom-mapped reaction SMARTS are frozen;
2. reactant handles, multiplicity, chemoselectivity, incompatibility, stereochemistry, salt, and protonation policies are explicit;
3. at least one reported positive product is reproduced exactly;
4. at least one known-negative or incompatible case is rejected;
5. executable tests and an immutable implementation hash pass;
6. the streaming enumerator refuses every non-qualified registry entry.

The first pilot should target 50k–100k unique products, stratified across reaction family, head/linker/tail architecture, size, formal charge, branching, unsaturation, degradability, and region combinations. Candidate generation must stream through deduplication and quota-based reservoir selection; a blind Cartesian product is neither necessary nor desirable.

## Split and leakage requirements

At minimum, maintain five distinct group controls:

1. exact canonical structure;
2. source study/publication/library;
3. head, linker/body, and tail building-block identity;
4. reaction family and linker/scaffold class;
5. prospective campaign lock.

Conditional/reward experiments require an additional hard boundary: any structure or close analogue used to train the lung oracle must be kept out of claimed prospective novelty and held-out generalization sets. The pan-lung dataset should use study-held-out evaluation, not only random row splits.

## Immediate next gates

1. Freeze and hash R0 (complete).
2. Run the current lipid kernel against the R0 element/charge/size/stereo profile and choose a non-truncating pilot eligibility policy.
3. Qualify one reaction transform end to end from reported examples.
4. Execute a 1k product smoke enumeration and audit validity, uniqueness, route replay, and region labels.
5. Scale to a 50k–100k stratified pilot; stop if any coverage or leakage gate fails.
6. Only then expand toward 500k–1M and train the broad COMPOSE-Lipid structural model.

## Reproducible artifacts

- `artifacts/datasets/compose_lipid_pretraining_v1/source_manifest.json`
- `artifacts/datasets/compose_lipid_pretraining_v1/canonicalization_audit.json`
- `artifacts/datasets/compose_lipid_pretraining_v1/available_source_structures.csv`
- `artifacts/datasets/compose_lipid_pretraining_v1/canonical_structure_union.csv`
- `artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv`
- `artifacts/datasets/compose_lipid_pretraining_v1/r0_release_manifest.json`
- `scripts/build_compose_lipid_pretraining_inventory.py`
- `configs/lipid_reactions/reaction_registry.schema.json`
- `configs/lipid_reactions/pilot_literature_specs_v1.json`

