# COMPOSE — animated research explainer

A 71.5 s, 1920×1080, 60 fps explainer for **COMPOSE: Molecular Generation and
Optimization with a Reusable Stochastic Rewrite Process** (under review at
ICLR 2027).

Output: `build/compose_explainer.mp4`

Every frame is a pure function of its index — no wall clock, no unseeded
randomness — so the render is deterministic and restartable.

## Re-rendering

```bash
cd media/compose_explainer

# one-time setup
python3 -m venv .venv
./.venv/bin/pip install playwright imageio-ffmpeg
./.venv/bin/python -m playwright install chromium

# molecular geometry (needs RDKit; only used for 2D display coordinates)
python3 src/build_molecules.py          # any python with rdkit installed

# render
./.venv/bin/python src/render.py probe            # timeline summary
./.venv/bin/python src/render.py frames           # build/frames/f%06d.png
./.venv/bin/python src/render.py video            # build/compose_explainer.mp4
```

`frames` accepts `--from N --to N` to re-render a range only; frames are
independent, so a partial re-render composites correctly with existing output.

For design iteration, `render.py sheet 12.5 33.6 51.8` writes full-size PNGs at
those times (seconds) into `build/review/` for inspection.

## Layout

```
src/
  index.html          timeline, scene table, cross-dissolve, renderFrame(i)
  core.js             drawing primitives, palette, type, molecule renderer
  lattice.js          the shared state-space lattice and camera
  scenes_a.js         scenes 1-3  (problem, limitation, core idea)
  canon.js            the canonicalisation inset used inside scene 3
  scenes_b.js         scenes 4-6  (mechanism, results, close)
  build_molecules.py  RDKit -> assets/molecules.{json,js}
  render.py           Playwright frame driver + ffmpeg encode
assets/
  molecules.json      2D layouts, centred, mean bond length normalised to 1
  molecules.js        same data as a script tag (Chromium blocks file:// fetch)
```

### Determinism

- `core.js` draws every pseudo-random value from a table seeded once
  (`mulberry32(20260928)`), indexed by a stable integer key. Nothing calls
  `Math.random` or reads the clock at draw time.
- `render.py` steps `window.COMPOSE.renderFrame(i)` explicitly rather than
  letting the page animate, and screenshots with `animations="disabled"`.
- Scene exceptions are recorded into `window.COMPOSE.errors` and reported by the
  driver, which exits non-zero — a scene that throws cannot silently render blank.

### Chemistry

Structures are laid out by RDKit from SMILES and drawn as skeletal diagrams
(implicit carbons, heteroatom labels knocked out of the bonds, aromatic rings
with inner arcs). RDKit is used **only** for display geometry; no number shown
in the video is computed here. All figures are transcribed from the paper.

The scene-4 program is the paper's own claim that whole-ring construction is a
structured program rather than a primitive: five `atom_insert` births followed by
one `cycle_close`, building a morpholine ring onto a primary sulfonamide. Atom
indices 14–18 and the closing bond (18,13) are the SMILES order of
`CC(=O)Nc1ccc(cc1)S(=O)(=O)N1CCOCC1` and are verified in `build_molecules.py`
output; every intermediate is a complete, valence-valid molecule.

## Timeline

| # | scene | start | dur |
|---|-------|-------|-----|
| 1 | problem — every edit changes what comes next | 0.0 | 8.5 |
| 2 | limitation — a new objective means a new model | 8.5 | 11.0 |
| 3 | core idea — executable state space, canonical successors, frozen R_θ | 19.5 | 20.0 |
| 4 | mechanism — programs, then control of the frozen process | 39.5 | 14.0 |
| 5 | results — learned-vs-uniform ablation, PMO-1K, lead optimization | 53.5 | 11.0 |
| 6 | close — the opening image resolved, title card | 64.5 | 7.0 |

Scenes cross-dissolve over 0.55 s; per-scene fades are deliberately absent so
the dissolve is the single authority on transitions.
