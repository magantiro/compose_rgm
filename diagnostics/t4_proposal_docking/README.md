# Complete saved-proposal docking diagnostic

All 16 locked candidates were attempted once, with 16 finite docking results and
no failures. The observed best is **-9.9**, versus the inherited incumbent's
**-9.7**. Two candidates scored below the incumbent (-9.9 and -9.8). This small
difference is not evidence of a reproducible improvement: each molecule has one
unseeded docking evaluation, with no matched replicate comparison.

The run took 113.866 seconds inside one CPU container; its docking phase took
104.425 seconds. Deployment took another 54.412 seconds. No proposal generation,
R_theta initialization, training, automatic optimizer continuation or redocking
was performed. The diagnostic adds 16 calls to the 51-call source lineage (67
total), not a completed adaptive optimization round. Requested resources were
1 CPU and 2048 MiB; reported Linux peak RSS was 3,255,608 KiB (about 3.10 GiB).

The important negative result is structural: **all 16 endpoints still have cycle
rank 4 and two ring systems**. No repair increases cycle rank or ring-system
count relative to its immediate parent. One opens a cycle (rank 5 to 4). These
are local repairs of saved search outputs, not new ring-construction successes.

## What the SMILES show

The best candidate is:

```
Cc1nc(=O)nc(-c2c(F)c(Br)c3n2-c2ccc(CN(C)C)cc2CNC3=O)n1C
```

Its executed repair is `bond_reroute`, relocating fluorine from its immediate
parent without changing heavy-atom count or cyclic topology. Relative to the
older incumbent, it also carries bromine inherited from that parent. Endpoint
QED is 0.62793, SA 3.41353 and original-seed Morgan similarity 0.45588.

The answer-known IVG endpoint used only for structural diagnosis is:

```
O=C1CCCc2cc(CCc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21
```

| Molecular graph | Heavy atoms | Cycle rank | Ring systems | Composition |
|---|---:|---:|---:|---|
| Prior incumbent | 29 | 4 | 2 | C20 N6 O2 F |
| Best saved repair | 30 | 4 | 2 | C20 N6 O2 F Br |
| Known IVG endpoint, not docked here | 30 | 5 | 2 | C25 N2 O3 |

Similar size and equal ring-system counts hide different chemistry. The best
repair retains a directly attached six-membered C3N3 heterocycle, the
dimethylaminomethyl substituent and the original core ring pattern. The known
endpoint instead has a carbon-only fused 6/6 peripheral system with a carbonyl,
an ethylene connection to the main core, and additional core-carbonyl remodeling.
The per-molecule ring fragments, atom compositions and exact SMILES in
`summary.json` expose this difference without treating raw string distance as
molecular distance. These descriptions are not shortest-path claims or evidence
that copying a structural feature causes improved docking.

## What the guide got right and wrong

Across these 16 local repairs, predicted and observed score rankings have
Spearman correlation **0.79498**. The observed best ranked sixth by prediction;
the predicted best scored -9.8. Thus this batch does **not** support saying the
guide is useless for local ranking. Its reliability across the much larger
structural change toward the known winner remains unsupported; the separate
saved-route analysis ranks that endpoint worse than the incumbent.

All 16 molecules are canonically distinct. Mean pairwise Morgan fingerprint
distance is 0.44125 over 120 pairs (radius 2, 2048 bits). That descriptor diversity
does not imply broad topology coverage. Across the retained origin witnesses,
there are 10 bond-reroute marks, 3 atom deletions, 3 atom restates and 1 cycle
opening; one endpoint has two alternative witnesses, so these total 17, not 16.
The unchanged endpoint gate accepted the complete set; no new post-hoc chemistry
filter or favorable-example exclusion was applied.

## Evidence and checks

- Scientific code: clean committed `af28010f8731501f797a4f88903ff0661dab7659`.
- Modal call: `fc-01M253JDMSHCH569K5RG7B55Z4`.
- Run ID: `45b165569198a5cc7800a0017bcf03dfd7d3639c6f109634bcc1380d7f76f939`.
- Authoritative local audit: `summary.json`, SHA-256
  `8a2187326ce19bce40a6367132ff6f197fd4f5dc941135f626ebdc0e0dccd1a5`.
- Raw remote bytes: `attempt_1/run/`; physical download hashes:
  `attempt_1/download.json`; independent per-candidate start and result receipts
  are preserved alongside the complete pre-oracle candidate lock.
- Runtime verified the frozen contract, source assets, receptor and QuickVina2
  hashes. The audit verifies every returned identity against its candidate lock,
  exact-state canonicalization, endpoint properties, start receipt and result
  receipt. All 16 starts follow candidate locking. No attempted row is omitted.
- Five focused lock/retry/resource tests passed before launch; strict clean-tree
  preflight passed. The local artifact audit passed; collector/analysis Ruff
  lint and formatting passed. No repository-wide milestone completion is claimed.
- Local analysis used pinned RDKit 2024.03.5, numpy 1.26.4 and scipy 1.13.1, with
  production imports from the clean deployment worktree. Source hashes and
  scoring assets are recorded in the audit. A local provenance-path error was
  repaired before publication; it caused no remote or molecular rerun.

Next proposed step: evaluate structurally varied, completed macro sequences
with real docking feedback, rather than expanding this same repair neighborhood.
Use the known-winner route to diagnose supported operations, not to inject its
endpoint or choose proposals. This batch does not authorize a further run.
