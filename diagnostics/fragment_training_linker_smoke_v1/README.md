# Training-derived linker support diagnostic

The single seeded proposal compiled exactly on 9 of 10 supplied core pairs,
with connected validity and exact mapped core/interface preservation on all
9 completed programs. Every prompt had context/atom-capacity-compatible content
in the frozen split-first catalog, ranging from 57 to 9,318 entries. This count
is proposal availability, not executor or model support.

The first compiled ring-containing connector (BARICITINIB, 25 primitives) and
the first compiled ring-free connector (ELIGLUSTAT, 14 primitives) both received
finite native checkpoint scores, -6.956583290100098 and -5.674360002790179,
respectively. These are length-normalized program panel scores, not path
probabilities or quality values. The other seven compiled programs were not
model-scored under this diagnostic's two-program ceiling.

CYCLOTHIAZIDE abstained. Its sampled training connector from source row 8732
would produce 40 heavy atoms and six rings, but the unchanged compiler required
transient H=5 at slot 14 and reported `transient_hydrogen_outside_support`.
The sampled content, binding, probability and refusal remain in the result.
No replacement, rule relaxation or support reduction was performed.

The authoritative artifacts are [result.json](result.json) and
[manifest.json](manifest.json). The manifest SHA-256 is
`8540ceee3d63d025d22eb5ccb53fe0517b7c06fa446cbc43d2a0f3873d74efa8`.
It binds all material physical inputs, the shared prior and checkpoint,
configuration, seed derivation, software and hardware. Git provenance names
the base revision; scoped new source files are separately bound by their
physical hashes. This is an uncommitted development-source diagnostic, not
release or milestone verification.

The run used one CPU thread under Python 3.11.13 and RDKit 2024.03.5, took
3.826 seconds after interpreter startup and peaked at 1,059,930,112 process
resident bytes. It made zero oracle calls and calculated zero quality metrics.
It did not execute the eight-draw panel sampler or an official linker/morphing
evaluation. That sampler's exact chemistry and selection wiring have unit
coverage; broader finite checkpoint support and output efficiency remain
unmeasured.

Original command, from the isolated fragment worktree:

```sh
PYTHONPATH=.fragment_eval_deps:src:scripts:tools OMP_NUM_THREADS=1 \
  /Users/rmaganti/compose_rgm_git/.venv_pinned_chem/bin/python \
  tools/audit_fragment_training_linker.py \
  --checkpoint /Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt \
  --output-dir diagnostics/fragment_training_linker_smoke_v1
```

The command refuses an existing output directory. Preserve this result; any
new diagnostic must use a separately authorized output namespace and budget.
