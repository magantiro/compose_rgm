# Seed-3 linker binding comparison: exploratory only

The predeclared v1 contract failed its **source identity gate**. It pinned the
old runner SHA-256 as `b1bfe44cbeaed80c1feb54c30b2180018ed6510482c30b5dc7145d7b70ef9199`,
but the exact runner in its pinned commit `1e1f258185154c87cc05b3a2b52f92a277b77231`
is `b1bfe44cbeaed80c1feb54c30c2180018ed6510482c30b5dc7145d7b70ef9199`.
The contract and completed outputs are preserved. This is a metadata error, not
a reason to silently edit a frozen contract after seeing its result. The new
v2 contract corrects the pin, names v1 as superseded and tests untouched seed 4.

The 200-attempt-per-arm seed-3 outputs are descriptive evidence only. The old
site-matching law committed 28/200 genuine linkers, while the generic
constraint-bound law committed 159/200. Every committed endpoint in both arms
was chemically valid. The official InVirtuoGen metric adapter reported mean
quality 0.05 versus 0.18, mean uniqueness 0.5733 versus 0.6981, and mean
diversity 0.3211 versus 0.4941. A 1-atom bridge was present in both start
states; only paths longer than that bridge were counted as generated linkers.
These numbers do not pass the v1 frozen gate because its identity check fails.

Physical artifact hashes:

- `old_seed3_n20.json`: `ed0a7b9e994b9bd7cc178bf83f10d5fd4afa1e4d8a1831010c06d9c8f417145f`
- `new_seed3_n20.json`: `082f99e5abbb11bc2899b390297e02ea32548ba3b362ee559b442208bea25bb4`

This is not an official fragment benchmark row. Canonical SMILES and attempt
failures are retained in the JSON files for independent inspection.
