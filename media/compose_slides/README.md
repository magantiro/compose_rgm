# Ionizable-lipid generative programs — research deck

`build/lipid_programs.pptx` — 12 slides, 16:9, editable.

## Visual system

Plain flow charts and panel tables; no illustration anywhere. Every slide is a
title, one diagram or table, and at most one caption line.

- White ground, near-black ink, two greys for secondary text and rules
- **Colour is semantic, not decorative**: amber is FORGE and teal is
  COMPOSE-Lipid on every slide, so colour does the comparison work
- Calibri throughout, 32pt titles / 15pt labels / 12–14pt detail
- No accent bars, no stripes, no icons, no subtitles under titles

## Slide logic

| # | slide | form |
|---|-------|------|
| 1 | Title — the two programmes, one line each | type only |
| 2 | Two complementary generative routes | comparison rows |
| 3 | COMPOSE | fan: motif → process → endpoints |
| 4 | FORGE | converge: three roles → Ugi-3 assembly → candidate |
| 5 | COMPOSE-Lipid application | linear 5-step pipeline |
| 6 | FORGE application | linear 5-step pipeline |
| 7 | Proposed figure architecture | 5-figure arc |
| 8 | Figure 1 — the generative model | panel table |
| 9 | Figure 2 — computational validation | panel table |
| 10 | Figure 3 — synthesis, formulation, characterisation | panel table |
| 11 | Figure 4 — in vitro and in vivo reporter | panel table |
| 12 | Figure 5 — the application study | two panel lists |

Slides 2–6 describe the programmes; 7–12 lay out the intended papers. On the
panel slides a row that **spans both columns** is the same panel in both
programmes; a **split row** is where they differ.

## Accuracy

Slides 7–12 describe panels that are **proposed, not results**. No panel carries
a number and no figure reports a measurement — each row says what a panel would
show. Slide 7 states this explicitly.

Programme details (Ugi-3, A549, HeLa, intranasal, reporter IM, two hits) are the
author's own; nothing was invented here.

## Rebuilding

```bash
npm install pptxgenjs          # once
node build_deck.js             # -> build/lipid_programs.pptx

python3 -m venv .venv          # once
./.venv/bin/pip install defusedxml lxml Pillow 'markitdown[pptx]'
./.venv/bin/python qa_render.py build/lipid_programs.pptx
```

### QA

`qa_render.py` renders and audits the **emitted PPTX XML**, not the generator's
variables — a check recomputed from the code under test cannot fail. It reports
shapes off canvas, text inside the 0.5" margin, and text overflowing its box,
and writes `build/qa-N.png` for each slide.

Calibri is not installed on this machine as a readable TTF, so text is measured
with **Arial, which is wider at the same point size**. Anything that fits under
those metrics fits in Calibri: the check errs toward a false alarm, never toward
missing a real overflow.

`topdf.applescript` converts through PowerPoint itself (the real renderer) when
automation permission has been granted; it identifies its own presentation by
name so another open deck is never closed, and never saves over the source.
