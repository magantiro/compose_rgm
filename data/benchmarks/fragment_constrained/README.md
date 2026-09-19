# SAFE-drug fragment benchmark input

`genmol_safe_drugs_fragments.csv` is an exact byte-for-byte copy of
`data/fragments.csv` from NVIDIA BioNeMo GenMol commit
`add09fc83b7255bd09c797e527c0f4b51f5fb7c1`.

- Upstream: https://github.com/NVIDIA-BioNeMo/genmol
- Upstream path: `data/fragments.csv`
- SHA-256: `a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9`
- Dataset identity: SAFE-DRUGS fragment-constrained evaluation inputs
- Dataset license: Creative Commons Attribution 4.0 (CC BY 4.0)
- License source: https://github.com/datamol-io/safe/blob/main/DATA_LICENSE
- Original benchmark: Noutahi et al., *Gotta be SAFE: A New Framework for
  Molecular Design*, arXiv:2310.10773.

The file contains the ten named drugs and the released inputs for linker
design, motif extension, scaffold decoration, and superstructure generation.
Following GenMol and InVirtuoGen, scaffold morphing reuses the linker-design
inputs.
