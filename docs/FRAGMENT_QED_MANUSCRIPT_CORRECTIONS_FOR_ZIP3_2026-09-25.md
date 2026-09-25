# Targeted manuscript corrections for paper zip (3)

These are proposed replacements for `main.tex` in
`COMPOSE__ICLR_Main_2027_ (3).zip` (SHA-256
`1e615ddda7fd91398179faa42835d4fcfd723f06fdf269c9cf4b5fab34fe0cdc`).
No fragment headline row is changed by this document. Decoration and morphing
remain under separate development. The completed superstructure appendix
block is ready in `docs/FRAGMENT_REFERENCE_ABLATION_APPENDIX_v1.tex`.

## Numerical reference identity

The fragment contracts use the same RingCore checkpoint (SHA-256
`24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`).
The paper-era QED controller uses a distinct Editing-V2 checkpoint (SHA-256
`c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8`).
Do not describe these as one numerical $R_\theta$. They share the COMPOSE
executable-process design, not identical learned parameters.

- Abstract, line 96. Replace the final clause beginning `using the same learned process` with `using executable molecular processes whose reference parameters remained fixed within each evaluation setting.`
- Introduction contribution, lines 120-123. Replace `the reference process learns state-dependent transition preferences and transfers without retraining across` with `learned transition preferences support task-specific control across`. Do not claim one checkpoint serves every named setting.
- Learning subsection, line 208. Replace `Once trained, $R_\theta$ is fixed across all downstream design tasks.` with `Within each evaluation setting, we hold its trained reference checkpoint fixed while changing the task-specific controller.`
- Experiments introduction, lines 349-356. Replace the assertion that one GuacaMol-trained reference was kept fixed across all four benchmark families with `We evaluated reference-process learning and task-specific control in four molecular-design settings. The fragment experiments share a frozen RingCore checkpoint, whereas similarity-constrained QED editing uses a distinct frozen Editing-V2 checkpoint. Each controller operates without task-objective fine-tuning of its assigned reference model.` Check the PMO and T4 checkpoint labels against their run contracts before naming them in this paragraph.
- Appendix reference implementation, lines 1726-1735. Replace `For the checkpoint used in this paper` and `The resulting checkpoint is used ... in all experiments` with version-specific wording. The fragment subsection can name RingCore and its file hash. The QED implementation subsection can name Editing-V2 and its file hash. The RingCore file was the trainer's interim-best step-8,500 weights from a 16,000-step completed run; it was not shown to win the separate canonical-successor checkpoint-selection protocol.
- QED result paragraph, line 465, and appendix table caption, line 1999. Name the frozen Editing-V2 reference used by the H40 sampler instead of `the same GuacaMol-trained $R_\theta$`.
- Appendix T4 provenance, lines 2085-2087. Remove `the same frozen artifacts used throughout` and the cross-reference equating its numerical checkpoint with all other panels. State the exact T4 run-contract checkpoint identity if that provenance is available.

## Fragment sampler mechanisms

The current main fragment paragraph (lines 389-396) says the controller
selects only programs. It should distinguish the completed mechanisms:

```latex
The supplied structural specification restricts executable transitions and
programs. In the reported motif and linker configurations, the controller
constructs eight complete-program offers, admits their executable endpoints,
and scores the surviving programs with the frozen RingCore model. In the
reported superstructure configuration, the controller samples primitive edit
marks under retained-structure and attachment restrictions. The reported
decoration configuration also uses finite program panels. All five fragment
contracts pin the same RingCore checkpoint, while their task-specific
construction and sampling rules differ. Scaffold morphing in the current
evaluation reuses the linker outputs.
```

The appendix fragment protocol (around lines 1947-1972) should replace
`same frozen primitive executor and reference checkpoint used in the other
experiments` with `same frozen RingCore checkpoint across the named fragment
tasks`. Its first paragraph should also distinguish primitive superstructure
sampling from motif/linker finite-panel program construction. The completed
linker saved-panel result belongs to that prior version only. Do not carry
it forward to a changed linker or morphing constructor.

## QED missing-record accounting

The appendix currently says two missing source shards were counted as
failures (around line 1985). This conflates missing execution with observed
failure. The local rederivation has 798 scored sources, 446 successes, and
source indices 135 and 408 without scored records. The observed-source rate
is 446/798 = 55.89%. The available records establish a full-800 lower bound
of 446/800 = 55.75% and an upper bound of 448/800 = 56.00% if both missing
sources succeeded. Preserve the declared 800-source panel and identify the
two missing records explicitly. Do not report an all-800 paired fixed-size
effect or treat missing source outcomes as measured failures. The main-text
confidence interval should be recomputed only after the missing-run rule is
settled. The sentence stating that a size-fixed ablation was performed in
the appendix should be changed to `A fixed-size intervention is specified for
the deployed H40 sampler, but its 800-source evaluation remains pending exact
runtime assets and complete baseline accounting.`

## Headline row reconciliation

The printed motif row comes from a 200-attempt, one-seed development run;
its completed three-seed evaluation is 42.633% quality, 93.632% uniqueness,
0.664021 diversity, and 99.967% validity. The printed linker row is likewise
a 200-attempt development value; its completed prior-version three-seed
evaluation is 28.6%, 72.4%, 0.562149, and 100.0%. The printed superstructure
row already matches the completed evaluation. Preserve the printed decoration
and morphing development sources separately. Choose the final linker,
morphing, and decoration configurations before revising their headline rows
and associated comparative prose. Do not combine metrics from different
constructors into one row.
