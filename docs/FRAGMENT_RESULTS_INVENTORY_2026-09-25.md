# Fragment result versions in the September 25 paper zip

The source for the printed values is
`COMPOSE__ICLR_Main_2027_ (3).zip`, SHA-256
`1e615ddda7fd91398179faa42835d4fcfd723f06fdf269c9cf4b5fab34fe0cdc`.
Machine-readable source paths, hashes, values, and statuses are in
`diagnostics/fragment_results_inventory_v1/inventory.json`.

| Task | Printed COMPOSE row, Q / U / D / V | Completed evaluation for the named prior method | Development status |
| --- | --- | --- | --- |
| Motif extension | 43.0 / 97.5 / 0.669 / 100.0, 200 attempts, one seed | 42.633 / 93.632 / 0.664021 / 99.967, 3,000 attempts, three seeds | No additional motif ablation requested |
| Superstructure | 39.03 / 97.33 / 0.725 / 100.0, completed 3,000-attempt result | The printed row matches the completed evaluation | Completed primary reference ablation |
| Linker design | 35.0 / 91.5 / 0.548 / 100.0, 200 attempts, one seed | 28.6 / 72.4 / 0.562149 / 100.0, 3,000 attempts, three seeds | Novelty-selection development remains separate |
| Scaffold morphing | Printed as a linker-output alias | The prior evaluation likewise reuses linker outputs | A changed morphing constructor needs its own declared configuration and evaluation; do not carry forward the alias automatically |
| Scaffold decoration | 22.5 / 99.0 / 0.628 / 100.0, 200 attempts, one seed | No completed three-seed evaluation for the developing method | Source-coupled construction development remains separate |

Q, U, D, and V denote quality, uniqueness, diversity, and chemical validity.
The printed motif and linker values are not the completed three-seed values.
The superstructure value is already from a completed three-seed evaluation.
No developing linker, morphing, or decoration method is promoted into a
headline row here. Before final evaluation of a changed method, freeze its
constructor, selection rule, seeds, attempt budget, evaluator, and failure
accounting. Preserve the earlier results as separate versioned evidence.

The completed linker saved-panel analysis belongs to the prior linker
configuration only. It compares learned and uniform selection on the same
3,000 recorded eight-offer panels, giving quality 28.6% versus 23.333%,
uniqueness 72.4% versus 73.367%, and diversity 0.562149 versus 0.573898.
It does not remove the learned reference from panel construction or establish
an effect for a later linker or morphing constructor.

The original 3,000 saved linker offer panels, exact run configuration, locks,
rows, summary, and conditional selection analysis are preserved in the local
compressed archive described by
`diagnostics/fragment_linker_panel_archive_v1/manifest.json`. The manifest
pins the 48,858,886-byte archive by SHA-256. This local binary archive is
not a new run and must not be reused as evidence for a changed constructor.
