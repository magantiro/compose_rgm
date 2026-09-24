# Comparator length controls and their COMPOSE analogue

## Evidence boundary

This is a read-only audit of exact public source revisions, not a reproduction
of either model's reported metrics. GenMol revision:
`add09fc83b7255bd09c797e527c0f4b51f5fb7c1`; InVirtuoGen revision:
`b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`. Thirty files, including licenses,
were verified against their Git blob identities and SHA-256 hashes in
`diagnostics/fragment_comparator_length_audit_v2/receipt.json`.
Downloaded pickle/checkpoint-formatted length distributions were **not loaded**.
The exact split lineage of the distributed length binaries remains unverified;
that is an evidence gap, not evidence of leakage.

## Verified source behavior

| Method/path | Length/content control | Important qualification |
| --- | --- | --- |
| GenMol `_insert_mask` | Sample a total sequence length L from `data/len.pk`; add `max(L - prompt_tokens, min_add_len)` masks | Token lengths, not heavy atoms or rings |
| GenMol motif/decoration/superstructure | Same marginal length law plus fixed fragment tokens | `fragment_completion` does not forward the configured `min_add_len`; its actual minimum is the helper default 18, including V2 settings listing 24 |
| GenMol one-step linker | Joint fragment context; minimum 30 added tokens forwarded | Two-step linker instead draws two separate lengths before combining completions |
| GenMol de novo | Marginal lengths plus forwarded minimum added tokens: 40 in V1, 60 in V2 | Not an unconstrained draw from corpus lengths alone |
| IVG fragment sampling | Sample a stored categorical length distribution above a prompt-dependent floor | Floor input is padded prompt width + 5; initial filter is `> int(1.05 * floor)`; refill uses `>= 1.05 * floor` |
| IVG exceptional length case | If the prompt floor exceeds the marginal's maximum, use a deterministic floor length | This special case does not make the usual content generator deterministic |

Source locations: GenMol `src/genmol/sampler.py:91`, `:107`, `:115`, `:130`,
`:152`; IVG `in_virtuo_gen/models/invirtuobase.py:278` and
`in_virtuo_gen/models/invirtuofm.py:374`. The IVG preprocessing utility
`create_mmap_buckets.py:183` builds a length histogram from observed token
lengths. That utility alone does not establish the origin of the published binary.

GenMol also uses task-specific temperature, randomness and context-guidance
settings in `scripts/exps/frag/hparams_v2.yaml`. IVG's de novo evaluator explicitly
scans temperature/noise pairs and records their quality-diversity tradeoff in
`evaluation/denovo.py:130`. Consequently, it is inaccurate to describe these
reported generators as merely a corpus with no sampling controls.

In the inspected fragment generation paths, QED/SA are evaluated after the
samples are produced, not used to rank individual generation candidates.
GenMol **does** apply substructure filtering in fragment completion/linking,
and its decoder drops failed decodes and takes the largest component. Those
are distinct from property-score guidance. The standard IVG fragment command
also exposes an optional `--oracle_length` flag based on the original molecule;
the flag is not enabled by default. Its mere existence does not establish use
in the reported comparator row. COMPOSE will not adopt that reference-length
input for the present experiment.

## Mapping to the frozen COMPOSE experiment

The analogous graph-level control is to sample the **whole completion budget**
conditioned on the supplied core and feasible interfaces, rather than give each
attachment an independent chance to fill the remaining atom capacity. COMPOSE's
current candidate uses the admitted training molecules' joint heavy-atom/ring
histogram, then samples stochastic training regions conditional on that cell.
It is analogous in purpose, not mathematically identical to a token-length prior.

No comparator output, benchmark answer, QED/SA label or original-drug length
enters this prior. All old reachable size/ring cells retain positive probability.
The new frozen pilot must still demonstrate an improvement. The source audit
does not justify changing its hyperparameters mid-run or claiming that size
control alone reproduces the competitors' much larger learned chemistry prior.

Future generic temperature/length operating points, if needed, require a fresh
calibration protocol and unchanged held-out evaluation. No new mechanism or
sampling sweep is launched by this source audit.
