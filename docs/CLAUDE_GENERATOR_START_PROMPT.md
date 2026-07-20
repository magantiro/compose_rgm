# Paste this into the separate Claude Code generator chat

You are taking ownership of the COMPOSE/RGM **unconditional and conditional
molecular generator lanes only**. This is a lossless scientific and engineering
handoff, not a request to redesign the project from a short summary.

Your assigned worktree is:

`/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm_claude_generators`

Your assigned branch is:

`claude/generator-cond-uncond`

First verify `pwd`, `git branch --show-current`, `git status --short`, and the
remote. Do not work in the Codex worktree or in
`compose_rgm_claude_lipid`. The separate Claude lipid/oracle chat owns lipid
corpus construction and the pan-lung oracle matrix; do not edit its files or
silently change Paper 2.

Before changing code or launching anything, read these files **completely and
in order**:

1. `docs/HANDOFF_CLAUDE_GENERATORS.md`
2. `docs/CLAUDE_GENERATOR_HANDOFF_MANIFEST_V1.json`
3. `docs/research_plans/compose_two_paper_execution_plan.html`
4. `docs/research_plans/paper1_compose_methods.html`
5. `docs/research_plans/paper2_compose_lipid.html`
6. Every audit listed in section 2 of the handoff.

Verify the manifest hashes before relying on the snapshot. Treat the handoff as
the canonical status/index and the HTML files as the governing paper plans.
Preserve every negative result, exact denominator, run ID, hash, failure mode,
and claim gate. Do not infer efficacy from teacher loss, family accuracy,
chemical validity alone, or a smoke test.

Two remote activities exist at the handoff boundary:

- the unconditional v7 CPU path/evaluation compile is still running under the
  exact app/call IDs in the handoff; inspect it read-only and do not duplicate
  it;
- the conditional QED frozen-residual v5 run has completed and its best saved
  checkpoint is step 500; the immediate task is a real post-training
  molecule-level evaluation, not another training launch and not 800x20.

Execute the immediate queue in section 17. In particular:

- conditional: load the frozen qualified base plus the trained sidecar, rerun
  the real lead smoke, then proceed to a small fixed-lead GrIDDD-style panel
  only if molecule-level QED efficacy is positive under exact constraints and
  oracle accounting;
- unconditional: let the v7 CPU compile finish, validate exact multiplicity,
  replay, fingerprints, evaluation cache, catalog and selector ordering, then
  run the bounded topology-only pilot; promote only on matched rollout evidence;
- do not resurrect the expensive full-catalog ring-support rabbit hole or reuse
  the incomplete v5 cache as if it were lossless;
- keep the retained pancake checkpoint frozen as the comparison incumbent;
- favor root-cause fixes and bounded falsification experiments over complexity
  added without measured benefit.

The scientific thesis is Rewrite Generator Matching over executable stochastic
molecular rewrite matches: a flexible-size, non-monotone CTMC closed over valid,
connected complete molecular states, sampled ancestrally without beam search.
Do not weaken it into autoregressive construction, rejection sampling, or a
generic jump-process claim. Do not overclaim that the component fields are new;
the contribution is the executable-rewrite/Generator-Matching combination and
the capabilities and evidence detailed in the handoff.

Work autonomously and production-quality. Run focused tests before each launch
and the full suite before a milestone commit. Save raw artifacts and failure
records. Never expose tokens, `.env` contents, credentials, or private data.
Commit logical milestones to `claude/generator-cond-uncond`, push them, and
report exact commands, app/call IDs, hashes, throughput, metrics, uncertainty,
and explicit stop/go decisions. If live state has advanced since the handoff,
append an evidence-backed status audit rather than rewriting history.

Begin by returning a concise orientation confirming:

1. the verified worktree/branch and manifest;
2. the incumbent unconditional checkpoint and its main pathologies;
3. the completed conditional v5 checkpoint and immediate evaluation gate;
4. the live unconditional compile state;
5. the exact next three actions you will execute.
