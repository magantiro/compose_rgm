# Frozen stochastic linker panel support

The support gate passed: 20 of 20 attempts emitted connected, chemically valid
endpoints with exact mapped core identity, declared attachment contacts and a
nonempty connector path. Every prompt emitted twice. The gate required at least
18 outputs, at least one per prompt, and perfect selected-output validity and
core/path fidelity. It was not changed after seeing results.

All 160 offered draws remain recorded: 143 exact compilations, comprising 139
unique model-supported candidates and four within-panel duplicates, plus 17
compiler/constraint refusals. No exactly compiled unique candidate was refused
by the native model in this bounded sample. All 20 selected endpoints were
distinct. This is support evidence, not a chemical-quality metric or a result
from an official-sized benchmark.

Selected complete programs used 8–28 primitive edits (median 18). Endpoints had
18–36 heavy atoms (median 30.5) and two to five RDKit rings (median 3.5).
Connectors contained 4–20 atoms (median 10), and their shortest boundary paths
contained 3–9 new atoms (median 4). Fourteen selected connectors contained at
least one ring. The exact distributions and all-row fixed-order grids are in
`../fragment_training_linker_support_visuals_v1/result.json`, `selected_1.png`
and `selected_2.png`. Blue and green identify the two supplied cores; orange
identifies the connector.

Both grids were inspected. Structural flags include the oxygen-to-nitrile
attachment in ERLOTINIB attempt 0, the epoxide connector in LIOTHYRONINE attempt
0, and the large phosphorus-containing connector in FUTIBATINIB attempt 0.
These are qualitative observations, not measured quality failures or evidence
of medicinal suitability. They caused no selection, support or configuration
change. All twenty selected molecules remain in the grids and artifacts.

Authoritative result: [summary.json](summary.json), binding all twenty complete
attempt receipts and the prospective [manifest.json](manifest.json). Execution
source was `322ae83d11eab3d29d55c1ead9570b18e7b01327`. The frozen manifest hash is
`96d3e048deee16140eedf70ce0ac2e22af2705e66d86ed0a3203774671958b92`.
The run used one CPU thread with the pinned Python 3.11.13 / RDKit 2024.03.5
environment. Candidate generation and selection took 261.959 seconds in total.
Oracle calls and quality evaluations were both zero.

The model, compiler, training prior, proposal law and all gate inputs stayed
unchanged. Focused linker/scorer checks passed (51 tests); lint, formatting and
diff checks passed. No repository-wide suite or complete milestone sign-off is
claimed. The next separately authorized 200-attempt development measurement
reuses this exact two-attempt prefix for every prompt, without regenerating it.
