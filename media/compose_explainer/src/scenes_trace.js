/* The process itself: one long trajectory through the state space.
 *
 * Ten committed states, each differing from the last by exactly one primitive
 * rewrite (verified in build_molecules.py, which refuses any step whose
 * heavy-atom delta is not the one its family label claims). The chain grows,
 * restructures and shrinks, which is the paper's trans-dimensional claim.
 */

'use strict';

// The lattice path: one node per committed state, its band following the
// molecule's actual heavy-atom count so the trace reads against the strata.
const TRACEPATH = (function () {
  const T = MOLECULES._trace;
  const base = T[0].heavy;
  const path = [];
  const used = new Set();
  let cur = LAT.home;
  T.forEach((s, i) => {
    const want = Math.max(-2, Math.min(5, s.heavy - base));
    if (i === 0) {
      cur = LAT.nodes.filter(n => n.band === want)
        .sort((a, b) => Math.abs(a.x - 430) - Math.abs(b.x - 430))[0] || LAT.home;
    } else {
      const next = LAT.nodes
        .filter(n => n.band === want && !used.has(n.id) && n.x > cur.x - 30)
        .sort((a, b) => Math.hypot(a.x - cur.x - 105, a.y - cur.y)
                      - Math.hypot(b.x - cur.x - 105, b.y - cur.y))[0];
      if (next) cur = next;
    }
    used.add(cur.id);
    path.push(cur.id);
  });
  return path;
})();

function sceneTrace(t, d) {
  const T = MOLECULES._trace;
  const k = camIdentity();
  const n = T.length;

  const introP = seg(t, 0.0, 0.8, ease.out);
  const headP  = seg(t, 0.35, 1.45, ease.out);
  // One state per beat, with a short dwell so each is legible.
  const START = 1.35, PER = 0.74;
  const walk = c01((t - START) / (PER * (n - 1)));
  const fi = walk * (n - 1);
  const si = Math.min(n - 1, Math.floor(fi + 1e-6));
  const local = c01(fi - si);
  const closeP = seg(t, START + PER * (n - 1) + 0.45, d - 0.35, ease.out);

  // ---- the frozen process, and the trace through it ----
  alpha(introP * 0.9, () => {
    drawLattice({
      cam: k, bands: LAT.BANDS, bandA: 0.8, nodeA: 0.8, edgeA: 0.62,
      weightP: 1, color: C.grey,
    });
  });

  // Band ticks: the trace visibly moves between strata.
  alpha(introP * 0.75, () => {
    [[3, 'n+3'], [0, 'n'], [-2, 'n−2']].forEach(([bk, s]) => {
      tag(s, 1848, k.y(LAT.bandY(bk)),
          { size: 17, color: C.ink25, font: F.serif, align: 'right' });
    });
  });

  alpha(introP, () => {
    field(G.left - 58, G.head - 124, 620, 250);
    drawPath(TRACEPATH, k, c01((fi + 1) / (n - 1)), C.teal, 2.6,
             { dots: true, head: true });
  });

  // ---- the molecule at the head of the trace ----
  // Consecutive layouts are independent, so states cross-dissolve rather than
  // pretending to morph; the changed family is named beside them.
  const px = 1330, py = 560, bond = 46;
  // One feathered field covers the structure and its read-out together, so the
  // lattice never runs through either.
  alpha(introP, () => field(px - 316, py - 246, 632, 520));

  const fade = c01((local - 0.55) / 0.45);
  const cur = MOL[T[si].key];
  alpha(introP * (1 - fade), () => {
    drawMolecule(cur, { cx: px, cy: py, bond, color: C.ink, lw: 2.3 });
  });
  if (si + 1 < n && fade > 0.01) {
    alpha(introP * fade, () => {
      drawMolecule(MOL[T[si + 1].key], { cx: px, cy: py, bond, color: C.ink, lw: 2.3 });
    });
  }

  // Step read-out: which family fired, and what it did to the atom count.
  const shown = fade > 0.5 ? Math.min(n - 1, si + 1) : si;
  alpha(introP, () => {
    const lx = px - 256, ly = py + 196;
    rule(lx, ly - 28, 512, introP, C.ink07, 1);
    const s = T[shown];
    if (s.family) {
      tag(s.family, lx, ly, { size: 21, color: C.teal });
      const w = measure(s.family, { size: 21, font: F.mono, track: 0.2 });
      tag(s.card, lx + w + 18, ly,
          { size: 19, color: s.card === '=' ? C.ink45 : C.teal });
    } else {
      tag('source state', lx, ly, { size: 21, color: C.ink45 });
    }
    tag(`${s.heavy} heavy atoms`, lx + 512, ly,
        { size: 18, color: C.ink45, align: 'right' });
  });

  // A running tally of the committed states, so the length of the trace reads.
  alpha(seg(t, 1.9, 2.7, ease.out) * introP, () => {
    const bx = G.left, by = 862;
    tag('COMMITTED STATES', bx, by - 26, { size: 15, color: C.ink45 });
    for (let i = 0; i < n; i++) {
      const on = i <= shown ? 1 : 0.14;
      alpha(on, () => {
        const x = bx + i * 30;
        circle(x, by + 8, 6.2, i <= shown ? C.teal : C.grey, null);
      });
    }
    tag('every one a complete, connected, valence-valid molecule',
        bx, by + 44, { size: 17, color: C.ink45 });
  });

  // ---- type ----
  headline('A trajectory is a', G.left, G.head, { size: 50 }, headP);
  headline('sequence of molecules.', G.left, G.head + 58, { size: 50 },
           seg(t, 0.65, 1.75, ease.out));
  alpha(seg(t, 1.25, 2.15, ease.out), () => {
    tag('it may grow, shrink and restructure within one state space',
        G.left, G.head + 108, { size: 18, color: C.ink45 });
  });

  alpha(closeP, () => {
    field(G.left - 58, 300, 660, 160);
    text('no step leaves the support', G.left, 372,
         { size: 30, weight: 500, color: C.teal });
  });
}
