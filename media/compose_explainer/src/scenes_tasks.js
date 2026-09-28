/* Four task vignettes. The same frozen process, four controllers.
 *
 * Each reuses the grey lattice and a trajectory tracing through it, with a
 * task-specific mechanism in the foreground. Nothing here reports a result:
 * these are the control mechanisms of sections 2.5, 2.6 and 2.7.
 */

'use strict';

// ---- shared vignette furniture ----

/** Kicker + headline + control expression, on the standard left column. */
function vigType(u, kicker, l1, l2, caption) {
  alpha(seg(u, 0.15, 0.95, ease.out), () => {
    field(G.left - 58, G.head - 132, 660, 268);
    tag(kicker, G.left, G.head - 46, { size: 15, color: C.ink45 });
  });
  headline(l1, G.left, G.head, { size: 46 }, seg(u, 0.30, 1.25, ease.out));
  if (l2) headline(l2, G.left, G.head + 54, { size: 46 }, seg(u, 0.50, 1.45, ease.out));
  if (caption) {
    alpha(seg(u, 0.95, 1.85, ease.out), () => {
      tag(caption, G.left, G.head + (l2 ? 106 : 52), { size: 18, color: C.ink45 });
    });
  }
}

/** The frozen process, dimmed, with one controlled trajectory drawn on it. */
function vigLattice(u, path, col, p, latA) {
  const k = camIdentity();
  alpha(seg(u, 0.0, 0.7, ease.out) * (latA === undefined ? 0.55 : latA), () => {
    drawLattice({
      cam: k, bands: LAT.BANDS, bandA: 0.7, nodeA: 0.75, edgeA: 0.55,
      weightP: 1, color: C.grey,
    });
  });
  if (path) drawPath(path, k, p, col, 2.5, { dots: true, head: true });
  return k;
}

/** Atom partition of a molecule into a retained ring core and the rest,
 *  with the growth order a breadth-first walk outward from that core. */
function coreAndGrowth(mol, ringPick) {
  const adj = mol.atoms.map(() => []);
  mol.bonds.forEach(b => { adj[b.a].push(b.b); adj[b.b].push(b.a); });
  const ring = (mol.rings || []).filter(r =>
    r.atoms.every(i => mol.atoms[i].arom))[ringPick || 0] || (mol.rings || [])[0];
  const core = new Set(ring ? ring.atoms : [0]);
  const iface = [...core].filter(i => adj[i].some(j => !core.has(j)));
  const order = [];
  let frontier = [...core];
  const seen = new Set(core);
  while (frontier.length) {
    const next = [];
    frontier.forEach(i => adj[i].forEach(j => {
      if (!seen.has(j)) { seen.add(j); next.push(j); order.push(j); }
    }));
    frontier = next;
  }
  return { core, iface, order };
}

const FRAG = coreAndGrowth(MOLECULES.lead, 0);

const VIG_PATHS = [
  latticeWalk(LAT.home.id, 6, (n, f) => (n.band - f.band) * 2.0 + (n.x - f.x) * 0.02, 101),
  latticeWalk(LAT.home.id, 9, (n, f) => (n.band - f.band) * 1.4 + (n.x - f.x) * 0.02, 211),
  latticeWalk(LAT.home.id, 7, (n, f) => -Math.abs(n.band) * 2.6 + (n.x - f.x) * 0.02, 307),
  latticeWalk(LAT.home.id, 6, (n, f) => (f.band - n.band) * 2.0 + (n.x - f.x) * 0.02, 409),
];

// =====================================================================
// A. Fragment-constrained generation  (section 2.5)
// =====================================================================
function vigFragment(u, dur) {
  const grow = seg(u, 1.45, 6.10, ease.linear);
  vigLattice(u, VIG_PATHS[0], C.teal, c01((u - 1.45) / 4.4));

  const px = 1268, py = 566, bond = 36;
  const mol = MOL.lead;
  const nGrow = FRAG.order.length;
  const shown = new Set(FRAG.core);
  const kGrow = Math.floor(grow * nGrow + 1e-6);
  for (let i = 0; i < kGrow; i++) shown.add(FRAG.order[i]);

  alpha(seg(u, 0.35, 1.15, ease.out), () => {
    field(px - 336, py - 300, 672, 620);
  });

  // The retained subgraph L: drawn first, in the accent, and never altered.
  alpha(seg(u, 0.55, 1.35, ease.out), () => {
    drawMolecule(mol, {
      cx: px, cy: py, bond, color: C.teal, lw: 2.7,
      atoms: FRAG.core, bonds: new Set(mol.bonds.map((b, i) =>
        (FRAG.core.has(b.a) && FRAG.core.has(b.b)) ? i : -1).filter(i => i >= 0)),
    });
  });

  // The interfaces through which it may be extended.
  alpha(seg(u, 1.05, 1.75, ease.out) * (1 - seg(u, 5.6, 6.4, ease.inOut)), () => {
    const P = mol.atoms.map(a => ({ x: px + a.x * bond, y: py + a.y * bond }));
    FRAG.iface.forEach(i => circle(P[i].x, P[i].y, 15, null, C.teal, 1.8));
  });

  // The rest is generated around it, on both interfaces at once.
  if (kGrow > 0) {
    const bonds = new Set();
    mol.bonds.forEach((b, i) => {
      if (shown.has(b.a) && shown.has(b.b) &&
          !(FRAG.core.has(b.a) && FRAG.core.has(b.b))) bonds.add(i);
    });
    drawMolecule(mol, {
      cx: px, cy: py, bond, color: C.ink, lw: 2.3, atoms: shown, bonds,
    });
  }

  alpha(seg(u, 1.5, 2.3, ease.out), () => {
    tag('L  retained', px - 176, py + 250, { size: 19, color: C.teal });
    tag('∂L  interfaces', px + 16, py + 250, { size: 19, color: C.ink45 });
  });

  vigType(u, 'FRAGMENT-CONSTRAINED GENERATION',
          'Hold a substructure.', 'Generate the rest.',
          'the controller places mass only on programs that preserve L');
}

// =====================================================================
// B. Feedback-driven black-box optimization  (section 2.6)
// =====================================================================
const VIG_B = (function () {
  // Illustrative relative utilities only: no axis, no numbers, no claim that
  // these are measured values. They exist to show the shape of the loop.
  const rounds = [
    { cands: ['core_grow', 'core_restate', 'core_shrink'], u: [0.42, 0.66, 0.30], best: 1 },
    { cands: ['core_ring', 'core_grow', 'core_restate'],   u: [0.84, 0.51, 0.62], best: 0 },
  ];
  return rounds;
})();

function vigOptimize(u, dur) {
  vigLattice(u, VIG_PATHS[1], C.amber, c01((u - 1.2) / 5.6), 0.40);

  const mx = 1128;                 // proposed structures
  const bx = 1262, bw = 240;       // their scored utility
  const ax = 1706;                 // archive column
  const ys = [378, 552, 726];

  alpha(seg(u, 0.55, 1.25, ease.out), () => field(1008, 254, 884, 590));

  const archive = [];
  for (let r = 0; r < 2; r++) {
    const t0 = 1.25 + r * 3.5;
    const prop  = seg(u, t0,        t0 + 1.00, ease.out);
    const score = seg(u, t0 + 0.90, t0 + 1.75, ease.outQuint);
    const store = seg(u, t0 + 1.70, t0 + 2.35, ease.out);
    if (prop <= 0.01) continue;
    if (store > 0.6) archive.push({ v: VIG_B[r].u[VIG_B[r].best], r });
    // Round one hands the column over to round two rather than stacking on it.
    const live = r === 0 ? 1 - seg(u, 4.55, 5.20, ease.inOut) : 1;
    if (live <= 0.01) continue;

    const R = VIG_B[r];
    R.cands.forEach((key, i) => {
      const p = c01((prop * 3 - i * 0.45) / 1.3);
      if (p <= 0) return;
      const y = ys[i], isBest = i === R.best;
      alpha(p * live, () => {
        drawMolecule(MOL[key], {
          cx: mx, cy: y, bond: 17, color: isBest ? C.ink : C.ink45, lw: 1.7,
        });
        // Scored utility as a bare bar: no axis and no number, so nothing here
        // can be read as a measured value. Only the shape of the loop is shown.
        const w = R.u[i] * bw * score;
        if (w > 1) {
          ctx.save();
          ctx.fillStyle = isBest ? C.amber : C.grey;
          ctx.globalAlpha *= 0.9 * score;
          ctx.fillRect(bx, y - 8, w, 16);
          ctx.restore();
        }
        if (isBest && store > 0.05) {
          alpha(store, () => {
            linkP(bx + w, y, ax - 120, y, 18, 12, C.amber, 1.6, store);
            if (store > 0.9) arrowHead(ax - 120, y, 1, 0, 9, C.amber);
          });
        }
      });
    });
  }

  alpha(seg(u, 1.05, 1.85, ease.out), () => {
    tag('PROPOSED ENDPOINTS', mx - 100, 302, { size: 14, color: C.ink45 });
    tag('SCORED', bx, 302, { size: 14, color: C.ink45 });
  });

  // The archive: scored endpoints kept, and reused as parents.
  const arcP = seg(u, 2.9, 3.8, ease.out);
  if (arcP > 0.01) {
    alpha(arcP, () => {
      tag('ARCHIVE', ax - 106, 302, { size: 14, color: C.ink45 });
      rule(ax - 106, 318, 212, arcP, C.ink12, 1);
    });
    archive.slice().sort((p, q) => q.v - p.v).forEach((e, i) => {
      const p = seg(u, 3.35 + e.r * 3.5, 4.05 + e.r * 3.5, ease.out);
      alpha(p * 0.9, () => {
        const y = 362 + i * 44;
        ctx.save(); ctx.fillStyle = C.amber;
        ctx.fillRect(ax - 106, y - 9, e.v * 196, 18); ctx.restore();
      });
    });
  }

  // The loop closes: what was scored becomes the next parent.
  const loopP = seg(u, 7.55, 8.65, ease.out);
  if (loopP > 0.01) {
    alpha(loopP * 0.85, () => {
      curveP(ax - 6, 476, 1460, 892, mx - 82, 566, C.amber, 1.7, loopP);
      if (loopP > 0.92) arrowHead(mx - 82, 566, -0.9, -0.4, 9, C.amber);
    });
    alpha(c01((loopP - 0.35) / 0.65), () => {
      tag('reused as parents', 1372, 878, { size: 17, color: C.amber, align: 'center' });
    });
  }

  vigType(u, 'BLACK-BOX OPTIMIZATION',
          'Propose, execute,', 'score, remember.',
          'only scores observed during the run inform the next proposal');
  alpha(seg(u, 2.1, 2.9, ease.out), () => {
    field(G.left - 58, 700, 620, 150);
    runs(G.left, 772, [
      { s: 'D', size: 30, font: F.serif, italic: true, color: C.ink70 },
      { s: 't', size: 19, font: F.serif, italic: true, color: C.ink70, dy: 9, gap: 12 },
      { s: '=', size: 26, font: F.serif, color: C.ink45, gap: 12 },
      { s: '{ ( x , π , y , u', size: 27, font: F.serif, color: C.ink70 },
      { s: 'c', size: 18, font: F.serif, italic: true, color: C.ink70, dy: 8 },
      { s: '( y ) ) }', size: 27, font: F.serif, color: C.ink70 },
    ]);
  });
}

// =====================================================================
// C. Protein-specific lead optimization, under constraints  (section 2.6)
// =====================================================================
/**
 * The cavity wall, as a radius function of angle: one concave lobe facing the
 * mouth (pointing left) on a gently irregular body. The contour and the contact
 * marks both read this, so a contact can never be drawn outside the wall.
 */
function pocketPoint(ox, oy, th) {
  const dth = Math.atan2(Math.sin(th - Math.PI), Math.cos(th - Math.PI));
  const dent = 0.52 * Math.exp(-(dth * dth) / 0.30);
  const lobe = 1 + 0.09 * Math.cos(2 * th + 0.7) + 0.05 * Math.sin(3 * th);
  const r = 176 * lobe * (1 - dent);
  return { x: ox + Math.cos(th) * r, y: oy + Math.sin(th) * r * 0.92 };
}

function vigDocking(u, dur) {
  const k = camIdentity();
  vigLattice(u, null, null, 0, 0.42);

  // The feasible region around the starting lead: everything the controller
  // may return has to sit inside it.
  const bx = 640, by = 566, R = 218;
  const ballP = seg(u, 0.75, 1.75, ease.out);
  alpha(ballP * 0.9, () => {
    ctx.save();
    ctx.strokeStyle = C.teal; ctx.lineWidth = 1.6; ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.arc(bx, by, R * ballP, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  });
  alpha(seg(u, 1.4, 2.2, ease.out), () => {
    tag('sim ≥ δ', bx, by - R - 26, { size: 19, color: C.teal, align: 'center' });
  });

  // The starting lead, and candidates drawn around it.
  alpha(seg(u, 0.35, 1.15, ease.out), () => {
    knockout(bx, by, 150, 132);
    drawMolecule(MOL.core, { cx: bx, cy: by, bond: 21, color: C.ink, lw: 2.1 });
  });

  // Three candidates: two inside the ball, one outside that is never evaluated.
  const cand = [
    { a: -0.9, r: 150, inside: true },
    { a: 0.75, r: 162, inside: true },
    { a: -0.15, r: 296, inside: false },
  ];
  cand.forEach((c, i) => {
    const p = seg(u, 2.0 + i * 0.35, 2.9 + i * 0.35, ease.out);
    if (p <= 0.01) return;
    const x = bx + Math.cos(c.a) * c.r, y = by + Math.sin(c.a) * c.r;
    const gone = c.inside ? 0 : seg(u, 3.8, 4.6, ease.inOut);
    alpha(p * (1 - gone), () => {
      linkP(bx, by, x, y, 126, 26, c.inside ? C.teal : C.ink25, 1.6, p,
            c.inside ? null : [4, 4]);
      circle(x, y, 9, c.inside ? C.teal : null, c.inside ? null : C.ink25, 1.5);
    });
  });
  alpha(seg(u, 4.0, 4.8, ease.out) * (1 - seg(u, 7.6, 8.3, ease.inOut)), () => {
    tag('outside the region, never evaluated', bx + 172, by - 214,
        { size: 16, color: C.ink45 });
  });

  // Cheap predicates resolve before the expensive oracle is ever called.
  const gateP = seg(u, 4.4, 5.4, ease.out);
  if (gateP > 0.01) {
    alpha(gateP, () => {
      const gx = 300, gy = 862;
      field(gx - 90, gy - 88, 1060, 150);
      tag('CHECKED FIRST', gx, gy - 28, { size: 14, color: C.ink45 });
      ['similarity', 'QED', 'SA'].forEach((s, i) => {
        const p = c01((gateP * 3 - i * 0.5) / 1.2);
        alpha(p, () => {
          const x = gx + i * 132;
          tag(s, x, gy, { size: 18, color: C.teal });
        });
      });
      alpha(c01((gateP - 0.6) / 0.4), () => {
        tag('→', gx + 404, gy, { size: 22, color: C.ink25 });
        tag('then the docking oracle', gx + 438, gy, { size: 18, color: C.ink });
      });
    });
  }

  // The pocket: a thin-line 2D cavity the eligible candidate settles into.
  const dockIn = seg(u, 5.2, 6.4, ease.out);
  const settle = seg(u, 6.1, 7.5, ease.inOut);
  if (dockIn > 0.01) {
    const ox = 1470, oy = 500;
    alpha(dockIn, () => {
      field(ox - 262, oy - 250, 524, 520);
      // A smooth cavity: a closed radial contour with one concave lobe facing
      // the incoming ligand, rather than a polygon with a wedge cut out of it.
      ctx.save();
      ctx.strokeStyle = C.ink45; ctx.lineWidth = 2.0; ctx.lineJoin = 'round';
      ctx.beginPath();
      const N = 132;
      const lim = Math.max(3, Math.round(N * dockIn));
      for (let i = 0; i <= lim; i++) {
        const th = (i / N) * Math.PI * 2 + Math.PI;     // start at the mouth
        const P = pocketPoint(ox, oy, th);
        if (i === 0) ctx.moveTo(P.x, P.y); else ctx.lineTo(P.x, P.y);
      }
      ctx.stroke();
      ctx.restore();
      tag('binding pocket', ox, oy + 218, { size: 17, color: C.ink45, align: 'center' });
    });

    // The eligible molecule travels from the ball into the pocket.
    if (settle > 0.01) {
      const sx = lerp(bx + Math.cos(cand[0].a) * cand[0].r, ox + 26, ease.inOut(settle));
      const sy = lerp(by + Math.sin(cand[0].a) * cand[0].r, oy + 14, ease.inOut(settle));
      alpha(settle, () => {
        drawMolecule(MOL.core_grow, {
          cx: sx, cy: sy, bond: 15 + 4 * settle, color: C.ink, lw: 1.8,
        });
      });
      // Contacts between the ligand and the cavity wall. Each ends ON the wall,
      // because the endpoint is read from the same radius function the contour
      // is drawn from rather than being placed by hand.
      alpha(c01((settle - 0.62) / 0.38), () => {
        [-2.1, -0.75, 0.55, 1.95].forEach(th => {
          const W = pocketPoint(ox, oy, th);
          const dx = W.x - sx, dy = W.y - sy;
          const L = Math.hypot(dx, dy) || 1;
          const ux = dx / L, uy = dy / L;
          line(sx + ux * 52, sy + uy * 52,
               W.x - ux * 8, W.y - uy * 8, C.teal, 1.3, [3, 4]);
        });
      });
    }
  }

  vigType(u, 'PROTEIN-SPECIFIC LEAD OPTIMIZATION',
          'Stay inside', 'the feasible region.',
          'cheap predicates resolve before the expensive oracle is called');
  alpha(seg(u, 1.9, 2.7, ease.out), () => {
    field(G.left - 58, 316, 600, 140);
    runs(G.left, 384, [
      { s: 'F', size: 30, font: F.serif, italic: true, color: C.ink70, gap: 4 },
      { s: '( x', size: 27, font: F.serif, color: C.ink70 },
      { s: '0', size: 18, font: F.serif, color: C.ink70, dy: 8 },
      { s: ', δ )', size: 27, font: F.serif, color: C.ink70 },
    ]);
  });
}

// =====================================================================
// D. Finite-horizon future-value (Doob) control  (section 2.7)
// =====================================================================
// A small explicit tree: two immediate successors whose subtrees differ in how
// much desirable terminal state they can still reach.
const DOOB = (function () {
  const root = { x: 560, y: 508 };
  const dx = 300, spread = 120;
  const kids = [
    { x: root.x + dx, y: root.y - 112, good: false },
    { x: root.x + dx, y: root.y + 112, good: true },
  ];
  const leaves = [];
  kids.forEach((kd, i) => {
    for (let j = 0; j < 3; j++) {
      const y = kd.y + (j - 1) * (spread * 0.52);
      leaves.push({
        parent: i, x: root.x + dx * 2.0, y,
        // the lower subtree keeps more desirable continuations reachable
        g: i === 1 ? [0.55, 0.92, 0.74][j] : [0.10, 0.06, 0.22][j],
      });
    }
  });
  const h = [0, 1].map(i => {
    const ls = leaves.filter(l => l.parent === i);
    return ls.reduce((s, l) => s + l.g, 0) / ls.length;
  });
  return { root, kids, leaves, h };
})();

function vigDoob(u, dur) {
  vigLattice(u, VIG_PATHS[3], C.ink, c01((u - 6.4) / 3.0), 0.34);

  const treeP = seg(u, 0.85, 2.15, ease.out);
  const leafP = seg(u, 1.9, 3.1, ease.out);
  const backP = seg(u, 3.2, 4.6, ease.out);
  const tiltP = seg(u, 4.7, 5.9, ease.out);

  alpha(seg(u, 0.55, 1.25, ease.out), () => {
    field(DOOB.root.x - 190, 250, 1200, 520);
  });

  // The tree of what remains reachable within the horizon.
  DOOB.kids.forEach((kd, i) => {
    const p = c01((treeP * 2 - i * 0.35) / 1.3);
    if (p <= 0) return;
    // After the backward pass the edge carries the value of its subtree.
    const w = 1.9 + tiltP * (DOOB.h[i] * 7.0 - 1.2);
    const col = tiltP > 0.05 && i === 1 ? C.teal : C.ink45;
    alpha(p, () => {
      linkP(DOOB.root.x, DOOB.root.y, kd.x, kd.y, 20, 18, col, w, p);
    });
  });
  DOOB.leaves.forEach((l, j) => {
    const p = c01((leafP * 6 - j * 0.45) / 1.4);
    if (p <= 0) return;
    const kd = DOOB.kids[l.parent];
    alpha(p * 0.9, () => {
      linkP(kd.x, kd.y, l.x, l.y, 18, 20, C.ink25, 1.4, p);
    });
    // Terminal desirability g_z, shown as fill rather than as a number.
    alpha(p, () => {
      circle(l.x, l.y, 12, null, C.ink25, 1.5);
      const f = l.g * c01((leafP - 0.4) / 0.6);
      if (f > 0.02) circle(l.x, l.y, 12 * Math.sqrt(f), C.teal, null);
    });
  });

  alpha(seg(u, 1.2, 1.9, ease.out), () => {
    circle(DOOB.root.x, DOOB.root.y, 11, C.ink, null);
    tag('x', DOOB.root.x - 30, DOOB.root.y, { size: 19, color: C.ink, align: 'right' });
  });
  alpha(seg(u, 2.6, 3.4, ease.out), () => {
    tag('g', DOOB.leaves[0].x + 40, 300, { size: 19, color: C.teal });
    tag('z', DOOB.leaves[0].x + 52, 308, { size: 14, color: C.teal });
    tag('terminal desirability', DOOB.leaves[0].x + 40, 328,
        { size: 16, color: C.ink45 });
  });

  // Values propagate backward, right to left.
  if (backP > 0.01 && backP < 0.999) {
    DOOB.kids.forEach((kd, i) => {
      const q = ease.inOut(backP);
      const lx = DOOB.leaves[0].x;
      const x = lerp(lx, kd.x, q);
      alpha(1 - Math.abs(q - 0.5) * 0.8, () => {
        circle(x, lerp(kd.y + (i ? 20 : -20), kd.y, q), 5.5, C.teal, null);
      });
    });
  }
  alpha(seg(u, 4.0, 4.8, ease.out), () => {
    tag('h', DOOB.kids[1].x - 6, DOOB.kids[1].y + 58, { size: 20, color: C.teal });
    tag('b−1', DOOB.kids[1].x + 8, DOOB.kids[1].y + 66, { size: 13, color: C.teal });
    tag('backed up from the horizon', DOOB.kids[1].x - 6, DOOB.kids[1].y + 86,
        { size: 16, color: C.ink45 });
  });

  // The controlled kernel: reference law reweighted by the future value.
  alpha(seg(u, 5.6, 6.5, ease.out), () => {
    field(G.left - 58, 744, 1200, 170);
    runs(G.left, 830, [
      { s: 'P', size: 31, font: F.serif, italic: true, color: C.ink },
      { s: 'z,b', size: 17, font: F.serif, color: C.ink, dy: -12, gap: 2 },
      { s: '( y | x )', size: 31, font: F.serif, color: C.ink, gap: 18 },
      { s: '=', size: 27, font: F.serif, color: C.ink45, gap: 18 },
      { s: 'R', size: 31, font: F.serif, italic: true, color: C.ink70 },
      { s: 'θ', size: 19, font: F.serif, italic: true, color: C.ink70, dy: 9 },
      { s: '( y | x )', size: 31, font: F.serif, color: C.ink70, gap: 20 },
      { s: '·', size: 27, font: F.serif, color: C.ink45, gap: 20 },
      { s: 'h', size: 31, font: F.serif, italic: true, color: C.teal },
      { s: 'b−1', size: 17, font: F.serif, color: C.teal, dy: 9 },
      { s: '( y , z )  /  h', size: 31, font: F.serif, color: C.teal },
      { s: 'b', size: 17, font: F.serif, color: C.teal, dy: 9 },
      { s: '( x , z )', size: 31, font: F.serif, color: C.teal },
    ]);
  });
  alpha(seg(u, 6.2, 7.0, ease.out), () => {
    tag('same support, reweighted by what remains reachable', G.left, 876,
        { size: 18, color: C.ink45 });
  });

  vigType(u, 'FINITE-HORIZON FUTURE-VALUE CONTROL',
          'Choose by what', 'stays reachable.',
          'not by the desirability of the next molecule alone');
}

// =====================================================================
// Dispatcher
// =====================================================================
const VIGNETTES = [
  { fn: vigFragment, dur: 8.6 },
  { fn: vigOptimize, dur: 9.6 },
  { fn: vigDocking,  dur: 10.2 },
  { fn: vigDoob,     dur: 10.6 },
];

function sceneTasks(t, d) {
  const XF = 0.5;
  let t0 = 0;
  VIGNETTES.forEach(v => {
    const local = t - t0;
    let a = 0;
    if (local >= -XF && local < v.dur + XF) {
      a = 1;
      if (local < 0) a = 0;
      else if (local < XF && t0 > 0) a = ease.inOut(c01(local / XF));
      if (local > v.dur) a = 1 - ease.inOut(c01((local - v.dur) / XF));
    }
    if (a > 0.002) {
      save();
      ctx.globalAlpha *= a;
      v.fn(local, v.dur);
      restore();
    }
    t0 += v.dur;
  });
}
