/* Scenes 4-6: the mechanism (programs and control), the results, the close.
 * Claims from COMPOSE_FINAL_ICLR2027.txt sections 2.3-2.7, 3.2-3.5,
 * Tables 1/2/3/7 and appendix E.1/G.6.
 */

'use strict';

// ---- The program of scene 4 ----
// Five atom_insert births then one cycle_close, building a morpholine ring on a
// primary sulfonamide. Indices are the SMILES order of MOL.lead, verified:
// atoms 0-13 are the precursor, 14-18 the new ring, and (18,13) the closing bond.
const PROG = (function () {
  const base = [];
  for (let i = 0; i <= 13; i++) base.push(i);
  const chain = [14, 15, 16, 17, 18];
  const steps = [
    { atoms: base.slice(), name: 'source', card: '' },
    { add: 14, name: 'atom_insert', card: '+1' },
    { add: 15, name: 'atom_insert', card: '+1' },
    { add: 16, name: 'atom_insert', card: '+1' },
    { add: 17, name: 'atom_insert', card: '+1' },
    { add: 18, name: 'atom_insert', card: '+1' },
    { close: true, name: 'cycle_close', card: '=' },
  ];
  // Materialise the visible atom/bond sets at each step.
  let cur = new Set(base);
  const out = [];
  steps.forEach((s, i) => {
    if (s.add !== undefined) cur = new Set([...cur, s.add]);
    const atoms = new Set(cur);
    const bonds = new Set();
    // A bond is present once both endpoints exist, except the ring-closing bond
    // (18,13), which is committed only by the final cycle_close.
    return out.push({ atoms, bonds, name: s.name, card: s.card, closed: !!s.close });
  });
  return { steps: out, chain, closeBond: [13, 18] };
})();

/** Visible bond index set for a program step, given the molecule. */
function progBonds(mol, step) {
  const s = PROG.steps[step];
  const bonds = new Set();
  mol.bonds.forEach((b, i) => {
    const isClose = (b.a === 13 && b.b === 18) || (b.a === 18 && b.b === 13);
    if (isClose) { if (s.closed) bonds.add(i); return; }
    if (s.atoms.has(b.a) && s.atoms.has(b.b)) bonds.add(i);
  });
  return bonds;
}

// ---- Lattice paths ----
// The program climbs five bands then commits a same-cardinality ring closure.
// Each birth moves exactly one band up; the closing cycle_close stays in band.
// Nodes are chosen by proximity in the target band rather than by adjacency, so
// the climb is always six clean hops instead of stalling on a sparse neighbour.
const PROGPATH = (function () {
  const path = [LAT.home.id];
  let cur = LAT.home;
  for (let want = 1; want <= 5; want++) {
    const next = LAT.nodes
      .filter(n => n.band === want && !path.includes(n.id))
      .sort((a, b) => Math.hypot(a.x - cur.x - 96, a.y - cur.y)
                    - Math.hypot(b.x - cur.x - 96, b.y - cur.y))[0];
    if (!next) break;
    cur = next;
    path.push(cur.id);
  }
  // The final same-cardinality hop: the ring closure.
  const same = LAT.nodes
    .filter(n => n.band === cur.band && !path.includes(n.id) && n.x > cur.x)
    .sort((a, b) => (a.x - cur.x) - (b.x - cur.x))[0];
  if (same) path.push(same.id);
  return path;
})();

// Three controlled trajectories: the same grey lattice, three objectives.
const TRAJ = [
  { // structural conditioning: stays close, same cardinality region
    path: latticeWalk(LAT.home.id, 7,
      (n, f) => -Math.abs(n.band) * 3.4 + (n.x - f.x) * 0.020, 11),
    color: C.teal, glyph: 0,
    label: 'preserve a substructure', sub: 'fragment-constrained generation',
  },
  { // feedback-driven optimization: climbs, growing the molecule
    path: latticeWalk(LAT.home.id, 8,
      (n, f) => (n.band - f.band) * 3.2 + (n.x - f.x) * 0.014, 23),
    color: C.amber, glyph: 1,
    label: 'learn from scored endpoints', sub: 'black-box optimization',
  },
  { // future-value control: descends and restructures
    path: latticeWalk(LAT.home.id, 7,
      (n, f) => (f.band - n.band) * 3.0 + (n.x - f.x) * 0.018, 37),
    color: C.ink, glyph: 2,
    label: 'score the reachable future', sub: 'similarity-constrained editing',
  },
];

function headAOut0(t) { return 1 - seg(t, 6.30, 6.95, ease.inOut); }

function scene4(t, d) {
  const k = camIdentity();

  const introP  = seg(t, 0.0, 0.7, ease.out);
  // Beat A: programs.
  const headA   = seg(t, 0.35, 1.45, ease.out);
  const progP   = seg(t, 1.25, 5.45, ease.linear);
  const panelA  = pulse(t, 1.15, 1.85, 5.70, 6.45, ease.inOut);
  const ringPunch = seg(t, 5.05, 5.95, ease.out);
  // Beat B: control.
  const headB   = seg(t, 6.75, 7.75, ease.out);
  const ctlP    = [seg(t, 7.05, 8.55, ease.out),
                   seg(t, 8.35, 9.85, ease.out),
                   seg(t, 9.65, 11.15, ease.out)];
  const headC   = seg(t, 10.85, 11.85, ease.out);
  const outro   = 0; // transitions are handled by the global cross-dissolve

  save();
  ctx.globalAlpha = 1 - outro * 0.4;

  // The frozen reference process: grey, unchanging, for the whole scene.
  alpha(introP * (1 - panelA * 0.72), () => {
    drawLattice({
      cam: k, bands: LAT.BANDS, bandA: 0.85, nodeA: 0.85, edgeA: 0.8,
      weightP: 1, color: C.grey,
    });
  });

  // Soft fields sit between the lattice and the trajectories: they keep the
  // type legible without ever covering a path or the legend.
  const headAOutF = headAOut0(t);
  alpha(Math.max(headA * headAOutF, headB) * 0.97,
        () => field(G.left - 58, G.head - 124, 570, 296));

  // "R_theta fixed" stays pinned and never changes, which is the whole point.
  alpha(introP * 0.95, () => {
    const fx = 1560, fy = 962;
    text('R', fx, fy, { size: 25, font: F.serif, italic: true, color: C.ink45 });
    text('θ', fx + 16, fy + 7, { size: 16, font: F.serif, italic: true, color: C.ink45 });
    text('fixed', fx + 34, fy, { size: 21, weight: 500, color: C.ink45 });
  });

  // ---- Beat A: a coordinated transformation is a program ----
  if (progP > 0 && progP < 1.001 && panelA > 0.01) {
    const nsteps = PROG.steps.length - 1;              // 6 primitives
    const fp = c01(progP) * nsteps;
    const si = Math.min(nsteps, Math.floor(fp));
    const local = c01(fp - si);

    // The path on the lattice.
    drawPath(PROGPATH, k, c01(progP), C.teal, 2.5, { dots: true, head: true });

    // The molecule builds against an opaque field rather than a boxed panel:
    // no border, so the structure sits in the composition instead of in a card.
    alpha(panelA, () => {
      // Placed below the climbing path so it never covers the trajectory.
      const px = 1402, py = 600;
      ctx.save();
      ctx.fillStyle = C.bg;
      ctx.beginPath();
      ctx.ellipse(px, py - 10, 300, 276, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      const step = PROG.steps[si];
      const bonds = progBonds(MOL.lead, si);
      // The newest bond draws on over the step.
      const prevBonds = si > 0 ? progBonds(MOL.lead, si - 1) : bonds;
      const newBonds = [...bonds].filter(b => !prevBonds.has(b));

      drawMolecule(MOL.lead, {
        cx: px, cy: py - 34, bond: 30, color: C.ink, lw: 2.1,
        atoms: step.atoms, bonds: prevBonds,
      });
      // Draw the newly committed bond separately so it animates in.
      if (newBonds.length) {
        drawMolecule(MOL.lead, {
          cx: px, cy: py - 34, bond: 30, color: C.teal, lw: 2.3,
          atoms: step.atoms, bonds: new Set(newBonds), draw: local,
          labelColor: C.teal,
        });
      }

      // Step read-out: ordinal, family, cardinality effect, on one tight line.
      const lx0 = px - 250, ry = py + 222;
      const sub = ['₁', '₂', '₃', '₄', '₅', '₆'];
      if (si > 0) {
        tag(`a${sub[si - 1]}`, lx0, ry, { size: 18, color: C.ink45 });
        tag(PROG.steps[si].name, lx0 + 40, ry, { size: 20, color: C.teal });
        const w = measure(PROG.steps[si].name, { size: 20, font: F.mono, track: 0.2 });
        tag(PROG.steps[si].card, lx0 + 56 + w, ry,
            { size: 18, color: PROG.steps[si].card === '=' ? C.ink45 : C.teal });
      } else {
        tag('source', lx0, ry, { size: 20, color: C.ink45 });
      }
      rule(lx0, ry + 22, 500, 1, C.ink07, 1);
      tag('every intermediate is a complete molecule', lx0, ry + 48,
          { size: 16, color: C.ink45 });
    });
  }

  restore();

  // ---- Beat B: control of the frozen process ----
  TRAJ.forEach((tr, i) => {
    const p = ctlP[i];
    if (p <= 0.01) return;
    const head = drawPath(tr.path, k, p, tr.color, 2.6, { dots: true, head: true });
    // Objective glyph at the trajectory head.
    if (head && p > 0.98) {
      const last = LAT.nodes[tr.path[tr.path.length - 1]];
      objGlyph(tr.glyph, k.x(last.x) + 30, k.y(last.y) - 26, 11, tr.color, true);
    }
  });

  // ---- Type ----
  const headAOut = 1 - seg(t, 6.30, 6.95, ease.inOut);
  headline('Coordinated change', G.left, G.head, { size: 52 }, headA * headAOut);
  headline('is a program.', G.left, G.head + 60, { size: 52 },
           seg(t, 0.65, 1.75, ease.out) * headAOut);
  alpha(ringPunch * (1 - seg(t, 6.35, 6.95, ease.inOut)), () => {
    const rx = G.left, ry = G.head + 128;
    // Laid out as measured runs so the subscripts sit correctly instead of
    // falling back to literal underscores.
    runs(rx, ry, [
      { s: 'π', size: 29, font: F.serif, italic: true, color: C.ink70, gap: 10 },
      { s: '=', size: 26, font: F.serif, color: C.ink45, gap: 12 },
      { s: '(', size: 29, font: F.serif, color: C.ink70, gap: 3 },
      { s: 'a', size: 29, font: F.serif, italic: true, color: C.ink70 },
      { s: '1', size: 18, font: F.serif, color: C.ink70, dy: 7, gap: 7 },
      { s: ',', size: 26, font: F.serif, color: C.ink45, gap: 9 },
      { s: '…', size: 26, font: F.serif, color: C.ink45, gap: 9 },
      { s: ',', size: 26, font: F.serif, color: C.ink45, gap: 9 },
      { s: 'a', size: 29, font: F.serif, italic: true, color: C.ink70 },
      { s: 'L', size: 18, font: F.serif, italic: true, color: C.ink70, dy: 7, gap: 3 },
      { s: ')', size: 29, font: F.serif, color: C.ink70 },
    ]);
    tag('a ring is a program, not a primitive', rx, ry + 38, { size: 18, color: C.teal });
  });

  // Beat B's headline crossfades with beat A's rather than being painted over.
  headline('Same process.', G.left, G.head, { size: 52 }, headB);
  headline('New objective.', G.left, G.head + 60, { size: 52, color: C.amber },
           seg(t, 8.15, 9.15, ease.out));
  headline('No retraining.', G.left, G.head + 120, { size: 52, color: C.teal }, headC);
}

// ---- Closing scene: the opening image, resolved, then the card ----
function scene6(t, d) {
  const k = camIdentity();
  const hold = seg(t, 0.0, 1.0, ease.out);
  const flow = seg(t, 0.1, 2.75, ease.out);
  const collapse = seg(t, 3.05, 4.15, ease.inOut);
  const card = seg(t, 3.55, 4.70, ease.out);
  const cardB = seg(t, 3.95, 5.10, ease.out);

  // The lattice and all three trajectories together: the opening fan, resolved.
  save();
  ctx.globalAlpha = 1 - collapse;
  const squash = 1 - collapse;
  ctx.translate(0, (1 - squash) * 0);
  drawLattice({
    cam: k, bands: LAT.BANDS, bandA: 0.7 * hold, nodeA: 0.85 * hold,
    edgeA: 0.75 * hold, weightP: 1, color: C.grey,
  });
  TRAJ.forEach((tr, i) => {
    drawPath(tr.path, k, c01(flow * 1.15 - i * 0.05), tr.color, 2.6, { dots: true });
  });
  // The state the trajectories leave from, knocked out of the graph behind it.
  alpha(hold, () => {
    const hx = k.x(LAT.home.x), hy = k.y(LAT.home.y);
    const kk = molKnock(MOL.core, 16.5, 10);
    knockout(hx, hy, kk.rx, kk.ry);
    drawMolecule(MOL.core, { cx: hx, cy: hy, bond: 16.5, color: C.ink, lw: 1.65 });
  });
  restore();

  // A single rule that the lattice collapses into, carrying the eye to the card.
  const rp = seg(t, 3.20, 4.10, ease.inOut);
  if (rp > 0.01 && card < 0.99) {
    alpha(rp * (1 - card * 0.4), () => {
      line(W / 2 - 520 * rp, 618, W / 2 + 520 * rp, 618, C.ink12, 1.2);
    });
  }

  // ---- Title card ----
  if (card > 0.01) {
    alpha(card, () => {
      text('COMPOSE', W / 2, 468, {
        size: 92, weight: 600, color: C.ink, align: 'center', track: 6,
      });
    });
    alpha(cardB, () => {
      line(W / 2 - 300, 512, W / 2 + 300, 512, C.teal, 1.4);
      text('Molecular Generation and Optimization', W / 2, 578, {
        size: 31, weight: 400, color: C.ink70, align: 'center',
      });
      text('with a Reusable Stochastic Rewrite Process', W / 2, 620, {
        size: 31, weight: 400, color: C.ink70, align: 'center',
      });
    });
    alpha(seg(t, 4.35, 5.45, ease.out), () => {
      text('COntrollable Molecular Process Over Stochastic Edits', W / 2, 700, {
        size: 19, weight: 400, color: C.ink45, align: 'center', track: 0.6,
      });
    });
    alpha(seg(t, 4.70, 5.80, ease.out), () => {
      text('Under review at ICLR 2027', W / 2, 776, {
        size: 18, weight: 400, color: C.ink45, align: 'center', track: 1.2,
      });
    });
  }
}
