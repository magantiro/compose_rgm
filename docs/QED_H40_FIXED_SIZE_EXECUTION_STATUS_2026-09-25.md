# Fixed-size QED in the paper-era H40 sampler

The size-changing-family restriction is implemented in
`modal_apps/hphi_h40head_ab_app.py::run_source` through
`fixed_size_allowed_families` in `hphi_lazy_sampler.py`. It removes
`atom_insert` and `atom_delete` before each primitive family draw, retains
same-cardinality atom, bond, and ring operations, renormalizes among the
remaining legal choices, and asserts unchanged heavy-atom count after
execution. An empty restricted fiber stops a particle without replacing an
output slot. The deployed restart arm's horizon 40, 32 particles, eight
terminal returns per source, frozen head, source-relative similarity,
resampling, and return rules are unchanged. The four focused mask and
empty-support tests passed on September 25. No 800-source fixed-size result
has been run or inferred from these tests.

The exact Editing-V2 reference checkpoint is locally readable at
`/Users/rmaganti/compose_v2_work/runs/run_v2_01/R_THETA_CHECKPOINT.pt`,
SHA-256 `c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8`.
The exact 800-source panel, `data/jin/qed_test.txt`, has SHA-256
`704103777e8050eb59f4d15d9997ca6070ba05a6b878b8e18738bfb1e706a090`.
The local scan of the paper sampler worktree and the known local
`compose_v2_work` tree found no `hphi_v2/head.pt`, `hphi_v2/norm.json`,
volume `run_inputs/RUN_PATHS.json`, or individual
`hphi_official800_k8` source records. The runtime manifest must bind the
Active8 source, gate-zero decision, and materialized scorer; substituting the
separately present local scorer without checking that binding is not allowed.

The current per-source rederivation has 798 scored sources, 446 solved, and
source indices 135 and 408 unscored. These two are missing records, not
observed failures. The full-panel rate currently printed as 446/800 is a
lower-bound accounting convention for the available records; it does not
provide a paired eight-return baseline on all 800 sources. A fixed-size arm
cannot be compared fairly against these 798 records as if the panel were
complete. Preserve the original records and attempt to recover the two
missing files. If they cannot be recovered, declare their status under a
frozen missing-run rule before presenting any all-800 paired estimate.

The missing volume is in Rahul's workspace. A prior read-only volume request
was denied by sandbox auto-review with an instruction not to retry through
another route. No such retry was made here. A custodian with access can stage
the immutable head, norm, runtime manifest and all referenced inputs, plus
the existing source records, into a fresh authorized directory with a
sorted path/size/SHA-256 manifest. The H40 comparison can then be contracted
and launched without changing the checkpoint or controller. Until then,
source-level success, the paired difference and interval, and the
both/full-only/fixed-only/neither counts remain unavailable.
