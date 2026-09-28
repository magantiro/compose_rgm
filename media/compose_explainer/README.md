# COMPOSE — animated research explainer

A 104.3 s, 1920×1080, 60 fps explainer for **COMPOSE: Molecular Generation and
Optimization with a Reusable Stochastic Rewrite Process** (under review at
ICLR 2027).

The film is about the **process and its control**, not about benchmark results:
no table, chart or reported figure appears in it. What it shows is the
executable state space, the frozen reference law, structured programs, and four
task controllers driving that one unchanged process.

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

| # | scene | start | dur | file |
|---|-------|-------|-----|------|
| 1 | problem — every edit changes what comes next | 0.0 | 8.0 | `scenes_a.js` |
| 2 | limitation — a new objective means a new model | 8.0 | 10.0 | `scenes_a.js` |
| 3 | core idea — executable state space, canonical successors, frozen R_θ | 18.0 | 18.0 | `scenes_a.js`, `canon.js` |
| 4 | mechanism — programs, then control of the frozen process | 36.0 | 12.0 | `scenes_b.js` |
| 5 | the process — one long trajectory, ten committed states | 48.0 | 9.8 | `scenes_trace.js` |
| 6 | four task controllers | 57.8 | 39.0 | `scenes_tasks.js` |
| 7 | close — the opening image resolved, title card | 96.8 | 7.5 | `scenes_b.js` |

Scene 6 holds four vignettes, each cross-dissolving over 0.5 s:

| vignette | dur | mechanism |
|---|---|---|
| fragment-constrained generation | 8.6 | a retained subgraph L is held byte-identical while both of its interfaces ∂L grow |
| black-box optimization | 9.6 | propose → execute → score → archive → reuse as parents, two rounds |
| protein-specific lead optimization | 10.2 | the feasible region F(x₀,δ); cheap predicates resolve before the docking oracle; a ligand settles into a cavity |
| finite-horizon future-value control | 10.6 | terminal desirability backs up the tree; the controlled kernel reweights R_θ by what stays reachable |

Scenes cross-dissolve over 0.55 s; per-scene fades are deliberately absent so
the dissolve is the single authority on transitions.

### The trajectory in scene 5

`build_molecules.py` holds the ten-state chain and **refuses to build** if any
consecutive pair differs by more than one heavy atom, or if the heavy-atom delta
disagrees with the family label the scene prints. So the claim the scene makes —
that each step is a single primitive rewrite — is enforced by the asset build
rather than asserted in a comment. The chain grows 6→9 and shrinks back to 6,
exercising `cycle_close`, `atom_insert`, `bond_reorder`, `cycle_open` and
`atom_delete`.

### On the one place numbers appear

The utility bars in the optimization vignette are drawn with **no axis and no
numeric labels**, and are illustrative of the loop's shape only. Nothing in the
film reports a measured value.
