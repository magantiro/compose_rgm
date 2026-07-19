# Checkpoint diagnostics

## Step-1,000 preview

`step1000_preview100_grid.png` contains all 100 target-free ancestral CTMC
samples from `compose-v4-stage3-step1000-preview100-v1`. The source checkpoint
was the interim best model from
`compose-v4-stage3-flexible-graft-3k-1ac6f19-v1` (checkpoint SHA-256
`c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c`).
Sampling used no target and no beam search.

The preview produced 100/100 valid, connected, and unique molecules. It is a
diagnostic rather than a benchmark-sized evaluation: full-reference FCD was
28.688 and neutral-CNOF-matched FCD was 23.044 from only 100 generated
molecules.

`step1000_preview80_grid.png` is the earlier interim rendering retained for
provenance. Prefer the 100-molecule grid.

## Ring-topology comparison

`step1000_ring_topology_comparison.json` compares the step-1,000 preview with
the 2,000-sample legacy pre-quotient step-6,250 evaluation and the 5,000-molecule
held-out reference set. A pendant ring system is defined as one connected ring
system with exactly one bond crossing from the system to the rest of the
molecule.

Key molecule-level prevalences are:

| Property | Step 1,000 (n=100) | Legacy step 6,250 (n=2,000) | Reference (n=5,000) |
|---|---:|---:|---:|
| Has pendant ring system | 48.0% | 43.4% | 49.96% |
| Fused | 48.0% | 26.45% | 52.18% |
| Spiro | 3.0% | 0.55% | 2.74% |
| Bridged | 9.0% | 2.8% | 3.2% |
| Contains a 3- or 4-member ring | 51.0% | 29.9% | 5.26% |

Canonical molecular self-events fell to zero across 2,750 committed events,
and no trajectory deleted to one atom before regrowing. The new process fixes
the legacy quotienting failure and improves fused/spiro coverage, but the
step-1,000 model remains undertrained and substantially overproduces strained
small rings and bridged topology.
