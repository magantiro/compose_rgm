# GEM @ NeurIPS 2026 fork — status

Built 2026-08-26. **Deadline: 30 August 2026, 11:59pm AoE** (OpenReview).
Source of record remains `paper_iclr2027/`; nothing there was modified.

Build: `pdflatex main_gem.tex && bibtex main_gem && pdflatex ×2`.
Currently compiles clean: **0 undefined references, 0 undefined citations**.

## Venue constraints (gembio.ai, retrieved 2026-08-26)

| | |
|---|---|
| Short paper, dry-lab track | **up to 5 pages, excluding references AND appendix** |
| Review | double-blind, submissions must be anonymized |
| Template | NeurIPS 2026 main-conference template, "replacing the style file with the GEM version" |

**On the GEM style file.** Their instruction is "NeurIPS 2026 main conference
template, replacing the style file with the GEM version". The "GEM version" link
on gembio.ai points to a Google Drive file:
`https://drive.google.com/file/d/1Ay-EUdIlHGPawAoPqrFKRtFSZBrwDspk/view`

That file is now installed here as `gem_workshop.sty`
(sha256 `7409882ae8d1e9651039222a3328298ca65d168d89565cf5c0580403d766ad19`)
and **is what this fork builds against**. Two problems with it, both theirs:

1. **It is ICLR-lineage and names the wrong venue.** The file is adapted from
   the ICLR macros and hard-codes its running head to
   `Under review at the GEM workshop, ICLR 2026` (line 95; line 88 for
   camera-ready). GEM 2026 is a *NeurIPS* workshop — the style file was not
   updated from last cycle. We reproduce their string **verbatim**; a
   `NeurIPS 2026` variant is provided commented-out in `main_gem.tex`.
   Changing it is an authoring call — ask the organizers.

2. **Their running head never renders.** `\lhead{...}` is set inside
   `\@maketitle`, but their `\maketitle` wraps `\@maketitle` in
   `\begingroup...\endgroup` (lines 62–71). `\lhead` is a *local* assignment,
   so `\endgroup` discards it, and line 80's `\fancyhead{}` has already
   cleared the head. Net effect: **no GEM identification on any page**, for
   anyone using the file unmodified. Confirmed in a minimal document
   containing nothing but their `.sty`.
   **Author decision 2026-08-26: ship their unmodified behaviour.** The head is
   intentionally absent from the PDF. A one-line restoration (in either venue
   wording) sits commented out in `main_gem.tex`; neither variant modifies the
   supplied style file.

Because the file is ICLR-lineage, the document is driven the ICLR way:
`\usepackage{gem_workshop,times}`, and `\iclrfinalcopy` (kept commented out)
for camera-ready. The style substitutes the anonymous author block by itself,
which is the double-blind mode GEM requires.

The official NeurIPS 2026 kit is retained in this directory as
`neurips_2026.sty` / `REFERENCE_neurips_2026_template.tex` **for reference
only; it is not loaded**. Note that its own submission-mode footer is the
generic `Submitted to ... NeurIPS 2026. Do not distribute.` string and carries
no GEM identity at all — its `dblblindworkshop` option only affects
*camera-ready* output — which is precisely why GEM ships a replacement.

## Page state

Body is **6 pages** (references start on p7). One page over.

The overflow is **Figure 2, which is not drawn** — it renders as a red
placeholder box that sizes itself from its own spec text and is roughly twice
the height a drawn 2×2 panel would occupy. Body *prose* ends at line 227.
**Draw Figure 2 before trimming any prose.** That alone is likely to close the
gap.

Every edit in `sections/gem_*.tex` is one the abridgment map explicitly
specifies (KEEP / CUT / MOVE / REPLACE). No sentence was reworded to save
space. Further trimming is an authoring decision and has deliberately been
left to the author.

## Results: what is measured and what is not

| Slot | Status |
|---|---|
| §3.1 reference law, 5.37→3.90 nats, 86.7% | **MEASURED** |
| §3.2 future reachability, 26/65 → 40/65, 34.6% | **MEASURED** |
| §3.3 lead optimization, 26/30 feasible, 10 vs 13 paired, +0.07 kcal/mol | **MEASURED** |
| §3.3 QED editing, 54.7% (70/128) @ K=12, diversity 0.393 | **MEASURED but NOT protocol-matched** |
| Official GrIDDD 800×20 run | **NOT RUN** |
| Size-fixed ablation (GrIDDD's own comparable ablation: 45.1%→33.8%) | **NOT RUN** |
| Figure 2 panels | **NOT DRAWN** |
| `app:macros` appendix (the new §2.3 material) | **NOT WRITTEN** — stub only |

### Deviation from the abridgment map, and why

The map instructs §3.3 to carry "the final matched official evaluation" and
offers the interpretation *"COMPOSE improves source-conditioned editing success
under the matched protocol"*, and an abstract sentence saying COMPOSE
*"outperforms GrIDDD"*.

**Neither is written, because the run that would license them has not been
executed.** We hold a 128-source held-out panel at K=12; GrIDDD publishes 800
sources at K=20. The protocols do not match, so the comparison is context, not
a ranking. The starred row and its caveat are carried over from the full paper
unchanged. The GenMol half of §3.3 *is* measured and is written as **near
parity**, not a win — the +0.07 kcal/mol paired difference sits well below the
0.70 kcal/mol replicate variation.

### Swap point when the official 800×20 lands

1. `sections/gem_experiments.tex` — replace the starred COMPOSE row in
   `tab:external` with (n=800, K=20); delete the `$^{*}$` caveat from the
   caption and the "benchmark context rather than a direct numerical ranking"
   sentence from the prose. **Keep the diversity column either way.**
2. `sections/gem_abstract.tex` — the final sentence may then be restored to the
   map's wording, *if and only if* the result exceeds 45.1%.
3. Rewrite the interpretation to what the result actually supports.

## Reproducing the 54.7% QED number

Verified end-to-end on 2026-08-26 from a fresh pull of the raw records
(the stale `/tmp/v128` cache was bypassed):

```
candidates    1   2   3   4   5   6   7   8   9  10  11  12
cumulative   31  40  44  48  51  56  61  63  65  67  68  70    -> 70/128 = 54.7%
```

- generator: `modal_apps/hphi_h40head_ab_app.py`, `--valid` branch
- sources: `data/jin/hphi_valid_128.txt` (asserted 128; asserted disjoint from
  `dev_panel_qed_64.txt` and `hphi_train_1024.txt`)
- raw records: volume `compose-v4-artifacts`,
  `editing_v2/r_theta_run/hphi_valid128_k8/` (slots 0–7) and
  `.../hphi_valid128_k912/` (slots 8–11) — 128/128 present in both
- reader: `scripts/hphi_valid128_curve.py` → `docs/VALID128_CURVE.json`
- config: H=40 receding, `budget_max=24`, N=32 particles, frozen h_φ,
  CPU-only `cpu=(1.0,1.0)`, 4.5 GiB, `max_containers=64`
- the reader asserts slot indices are contiguous 0..11, so the two-run
  concatenation does not violate the frozen one-attempt-per-slot contract

Diversity 0.393 [0.331, 0.452]: `docs/VALID128_DIVERSITY.json`, computed with
Jin et al.'s released `scripts/diversity.py` (audited 2026-08-20).

## Cost of the official 800×20, if it is run

Measured per-slot cost from the banked run: **141 s** (median 119 s, p90 223 s),
consistent across both runs — not an extrapolation from one.

| | slots | core-hours | wall @80 containers | cost |
|---|---:|---:|---:|---:|
| 800 × K=8 | 6,400 | 251 | 3.1 h | $20.86 |
| 800 × K=12 | 9,600 | 376 | 4.7 h | $31.25 |
| **800 × K=20** (GrIDDD's protocol) | 16,000 | 627 | **7.8 h** | **$52.12** |

Rates from `docs/MODAL_COST_MODEL.md`: $0.04716/core-hour + $0.007992/GiB-hour
at 4.5 GiB ⇒ $0.0831/container-hour (CPU 57%, memory 43%).

**Memory cannot be cut.** A probe calling the app's own `run_source` measured
peak RSS of **4084 MiB against the 4.5 GiB request** after only two slots
(~91% utilization), still climbing. `cpu` is already exactly 1.0. The app is
pinned `max_containers=64`; the standing cap is 80.

**Risk if K=20 is run as one job:** 20 slots per source in a single container
has never been tested — the banked runs did 8 and 4. Shard by slot range
(k0–8, k8–16, k16–20), which is the pattern the 128-panel already used.

The official 800 is **completely clean**: 0/800 overlap with the validation,
development, and training sets. `data/jin/hphi_corpus.meta.json` records that
the official Jin test sources were excluded by construction.
