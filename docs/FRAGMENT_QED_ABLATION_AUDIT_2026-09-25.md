# Fragment-reference and fixed-size QED ablations: prelaunch audit

Status: **blocked before scored execution**. No comparative outcome was inspected to
choose a fragment task; no ablation sampling, remote upload, or scored Modal job
has been run from this branch. This is an intervention/asset audit, not a result.

## Scientific identity and frozen task choice

COMPOSE generates complete molecular graphs by executing typed primitive graph
edits. The fragment arm asks whether its learned reference improves useful
prompt-constrained molecules relative to the already defined empirical-family
reference on the same executable support. Motif extension is frozen as the sole
primary fragment task, using all ten released prompts. Linker is not selected
because it adds no cleaner reference intervention and has a separate live work
stream. The QED arm asks what the deployed 40-step, 32-particle, eight-return
sampler does when every primitive transition changing heavy-atom count is
disabled. These are substrate ablations, not new generator training.

The declared support is the existing bounded organic graph vocabulary,
at most 40 active atoms in 48 slots, exact executor validity, and the original
prompt/evaluator limits; stereochemistry is not represented. The QED criterion
is QED >= 0.90 and Morgan radius-two, 2048-bit Tanimoto >= 0.40 to the original
source. The original eight terminal returns, not internal particles or visited
states, are the only QED candidate slots.

## Fragment reference-use and lineage audit

`sample_motif_panel` obtains eight region offers from the frozen training-only
joint-completion prior. The proposer, exact compiler, protected-core check,
attachment check, and endpoint deduplication do **not** query the checkpoint.
For each surviving program, `learned_program_scores` obtains the mean native
log-*mark* probability across its executed primitive path. Model-unsupported or
nonfinite programs are removed, then `select_learned_program` draws one unique
endpoint by a softmax of these scores. Thus the learned checkpoint changes both
panel admission and selection, but not the eight initial structural offers.
Uniformizing the final draw over the model-admitted panel would retain learned
pruning and is only a conditional panel-reweighting diagnostic.

The existing empirical-family baseline is
`modal_apps/experiment1_reference_law_app.py`: editing-V2 realized training-law
family coefficients are restricted to families with nonempty legal canonical
successor sets, renormalized, and each family contributes uniformly to its
distinct canonical successors; aliases are summed. It is a **single-step
canonical-successor** law for the editing-V2 checkpoint and exact partitions.
The frozen motif method instead uses the older RingCore-V1 checkpoint, its
native mark fiber, and a length-normalized complete-program score. The
RingCore scaled-manifest family histogram is not a realized, family-balanced
training law for all families and does not supply the editing-V2 canonical
partitions. No frozen RingCore-aligned empirical-family law or documented
canonical mapping was found. Editing-V2's `qhat` is therefore not a legitimate
drop-in for this RingCore program selector. The required two-arm fragment
comparison is blocked, rather than replaced by a new or silently mismatched
baseline. The fresh-seed full motif run's self-hashed contract
`configs/fragment_motif_official_v1.json` is prepared but not evidence of
completed outputs. Its ten prompts, seeds 2/3/4, 100 attempted outputs per
prompt/seed, eight offers/attempt, and evaluator must remain fixed if a clean
reference law is eventually established.

## Deployed QED sampler and intervention

The deployed sampler is `modal_apps/hphi_h40head_ab_app.py` with H=40, N=32,
`arm=restart`, eight terminal returns per source, `hphi_v2` H24-budget-clamped
head, and `seed_for("restart", source, k)`. The source panel is the exact
800-line Jin QED file. The 2026-09-25 local code adds an optional size-fixed
family mask in `hphi_lazy_sampler`: `atom_insert`/`grow_connected`, `grow_root`,
`atom_delete`, `ring_system_grow`, and `ring_system_delete` have zero proposal
mass before family renormalization. The retained atom restate, bond reorder,
bond reroute, cycle close/open, and ring-system restate families preserve atom
count by their primitive executor definitions; a post-execution count assertion
fails loudly if that invariant is violated. Every primitive proposal uses the
mask. The empty restricted fiber returns no mark and kills that particle using
the existing no-proposal path; there is no relaxation, restart, or replacement
slot. Default unrestricted calls have no mask and retain their old law.

This is a capability-disabling intervention on the frozen deployed sampler.
The future-value head was trained under the unrestricted process, so even a
complete paired result could not establish superiority over a separately
trained fixed-size controller. The current full-run source records contain
returned endpoints and aggregate visited states, but no selected-particle
ancestry. The requested counts of successful paths with size changes and paths
that return to initial size are **unavailable** from those archives. They must
come from authentic selected-particle ancestry recorded during a mechanically
parity-checked full rerun, not inferred from all visited states.

## Input inventory and transfer boundary

The latest rederivation artifact, SHA-256
`813d575156f2687c93ce0b85b60bed061c200ce8b71e2cb7104587561a298c14`
(913 bytes), reports **798** full-arm per-source records, **446** solved among
those records, with sources **135 and 408 unscored**. This is not an 800-source
paired baseline. Both missing full-arm sources require their original H40/eight
slots before an 800-source paired analysis. The older aggregate JSON differs
on missing-source count and must not override the per-source rederivation.

Known local files:

| Material | Local source | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| frozen editing-V2 R-theta | `/Users/rmaganti/compose_v2_work/runs/run_v2_01/R_THETA_CHECKPOINT.pt` | `c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8` | 87,804,652 |
| exact QED panel | `data/jin/qed_test.txt` | `704103777e8050eb59f4d15d9997ca6070ba05a6b878b8e18738bfb1e706a090` | 35,782 |
| motif RingCore checkpoint | `/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt` | `24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4` | 24,746,132 |

Required but not locally inventoried: Rahul volume
`compose-v4-artifacts/editing_v2/r_theta_run/run_inputs/RUN_PATHS.json`
(a separate contract records expected SHA-256
`585286e0fb6dd8fb6b5a050dc3e5f8d1c7e782b105df114cbcb14cabbb5b2bc2`,
not independently checked here); every path referenced by that manifest,
including Active8 source and materialized scorer; `hphi_v2/head.pt` and
`hphi_v2/norm.json` (historical documents give only SHA prefixes
`9ea51ec4...` and `4f93f7ec...`); and the individual 798
`hphi_official800_k8` source shards. Their exact hashes, byte sizes, and
source-level runtimes remain **unknown**. Nitya's verified workspace has a
`compose-v4-artifacts` volume, but its `editing_v2/r_theta_run` tree is absent.
No cross-workspace transfer has occurred.

Modal's volume copy operation works within one volume, not between Rahul and
Nitya workspaces. A user-authorized custodian would need to inventory and
download the exact immutable Rahul files to a fresh local staging directory,
produce a sorted path/bytes/SHA-256 manifest, and present it before any Nitya
upload. Only after review should those bytes be uploaded under a new
content-addressed Nitya ablation namespace, with no existing baseline artifact
overwritten. The current H40 app's hardcoded `RUN_ROOT` and volume mount would
then need a predeclared path binding that preserves the frozen sampler. A
read-only Rahul volume inventory attempt was blocked by sandbox auto-review
despite relayed user approval. The exact attempted command was
`MODAL_PROFILE=rahul-94866 modal volume ls compose-v4-artifacts editing_v2/r_theta_run`
with `sandbox_permissions=require_escalated`. The reviewer called it outside
the authorized workspace because Rahul's containers were full and stated,
"Do not bypass this rejection through a workaround or indirect execution."
No workaround or retry was attempted, even after the user reaffirmed approval.

## Launch gate and intended analysis

No self-hashed ablation run contract or payload-specific scored-launch approval
has yet been issued. Before launch, bind the exact source commit, source panel,
checkpoint/head/norm/runtime hashes, per-source baseline shards, family mask,
seeds, H40/N32/resampling/terminal rule, output namespace, resource census,
cost ceiling, deterministic restart unit, and comparison/analysis scripts.
Keep source-level successes for all 800 sources, with missing full outputs
completed rather than silently counted as failures. Analyze paired full-minus-
fixed-size source success with a source bootstrap or paired exact interval and
the both/full-only/fixed-only/neither counts. Do not pool intermediate states.

If a valid fragment reference arm is later made available, report each prompt
and each seed separately; average within prompt across seeds, then analyze ten
paired prompt differences. Original headline quality, uniqueness, diversity,
and validity include every attempted slot and are **not** censored by prompt
fidelity. Report fidelity and prompt-faithful quality yield separately, the
latter divided by all 100 attempted outputs per prompt/seed, with uniqueness
beside it. No result, figure, caption, or claim should be filled from this
prelaunch audit.
