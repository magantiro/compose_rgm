/* Scenes 1-3: the problem, the limitation of the standard approach, and the
 * core idea. All claims here are from COMPOSE_FINAL_ICLR2027.txt sections 1,
 * 2.1, 2.2, 3.1 and appendix E.1.
 */

'use strict';

let MOL = null;               // set by index.html once molecules.json is loaded
function setMolecules(m) { MOL = m; }

// Shared editorial grid.
const G = { left: 150, head: 154, sub: 202, foot: 946, footB: 984 };

// ---- Scene 1 geometry: a clean radial fan around the protagonist ----
// Angles are hand-set so no ray crosses the headline block or either structure,
// and so the committed ray lands exactly on the successor node.
const S1 = (function () {
  const cx = 880, cy = 612;
  const nx = 1135, ny = 425;                  // the successor, up and to the right
  const pickA = Math.atan2(ny - cy, nx - cx);
  const pickR = Math.hypot(nx - cx, ny - cy);

  const A = [-1.62, pickA, 0.05, 0.72, 1.45, 2.15];
  const pick = 1;
  const before = A.map((a, i) => {
    const R = i === pick ? pickR : 296 + rndr(900 + i, -16, 22);
    return { a, R, x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R };
  });

  // After the edit the available set is different: five successors, new angles.
  const A2 = [-1.78, -1.02, -0.26, 0.52, 1.28];
  const after = A2.map((a, i) => {
    const R = 246 + rndr(950 + i, -12, 20);
    return { a, x: nx + Math.cos(a) * R, y: ny + Math.sin(a) * R };
  });
  return { cx, cy, before, after, pick, nx, ny };
})();

/** Knock a soft background disc out of the fan so a structure reads cleanly. */
function knockout(cx, cy, rx, ry) {
  ctx.save();
  ctx.fillStyle = C.bg;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function scene1(t, d) {
  const mol = MOL.core, molB = MOL.core_grow;

  // The protagonist draws itself on.
  const drawIn = seg(t, 0.15, 1.85, ease.out);
  // First fan.
  const fan1 = seg(t, 1.55, 3.05, ease.out);
  // Edit commits: the chosen edge lights, the molecule becomes its successor.
  const commit = seg(t, 3.35, 4.45, ease.inOut);
  // Second fan, from the new state.
  const fan2 = seg(t, 4.70, 6.20, ease.out);
  // The first fan retires as the second arrives.
  const fade1 = seg(t, 4.45, 5.50, ease.inOut);
  const txt = seg(t, 5.95, 7.10, ease.out);
  const outro = 0;   // transitions are handled by the global cross-dissolve

  const pk = S1.before[S1.pick];

  save();
  ctx.globalAlpha = 1 - outro * 0.55;

  // ---- layer 1: all fan geometry ----
  // The first neighbourhood retires completely: what remains is the committed
  // edge and the new available set, which is the point of the scene.
  const a1 = fan1 * (1 - fade1);
  S1.before.forEach((p, i) => {
    if (i === S1.pick) return;
    const pp = c01((fan1 * 6 - i * 0.55) / 1.6);
    alpha(a1, () => {
      lineP(S1.cx, S1.cy, p.x, p.y, C.grey, 1.4, pp);
      if (pp > 0.9) circle(p.x, p.y, 4.6, C.grey, null);
    });
  });

  // The chosen edge: grey while merely enumerated, teal once committed.
  const pp = c01((fan1 * 6 - S1.pick * 0.55) / 1.6);
  alpha(fan1 * (1 - commit), () => lineP(S1.cx, S1.cy, pk.x, pk.y, C.grey, 1.4, pp));
  if (commit > 0) {
    lineP(S1.cx, S1.cy, pk.x, pk.y, C.teal, 2.5, Math.min(1, commit * 1.5));
  }

  // The second neighbourhood: a visibly different available set.
  S1.after.forEach((p, i) => {
    const q = c01((fan2 * 5 - i * 0.5) / 1.5);
    alpha(fan2, () => {
      lineP(S1.nx, S1.ny, p.x, p.y, C.grey, 1.4, q);
      if (q > 0.9) circle(p.x, p.y, 4.6, C.grey, null);
    });
  });

  // ---- layer 2: structures, knocked out of the fan so nothing crosses them ----
  // A neighbour drawn as a real molecule: successors are molecules too.
  const showNb = c01((fan1 - 0.6) / 0.35) * (1 - fade1);
  if (showNb > 0.01) {
    alpha(showNb, () => {
      knockout(S1.before[4].x, S1.before[4].y, 96, 74);
      alpha(0.5, () => drawMolecule(MOL.core_shrink, {
        cx: S1.before[4].x, cy: S1.before[4].y, bond: 15.5, color: C.ink, lw: 1.35,
      }));
    });
  }

  // The parent stays as a ghost at its own node, so the committed edge reads as
  // a move between two states rather than a line into empty space.
  const parentA = commit < 1 ? 1 - commit * 0.80 : 0.20;
  alpha(parentA, () => {
    knockout(S1.cx, S1.cy, 168, 128);
    drawMolecule(mol, {
      cx: S1.cx, cy: S1.cy, bond: lerp(34, 25, commit),
      color: C.ink, lw: 2.25, draw: drawIn,
    });
  });
  if (commit > 0) {
    const s = ease.out(commit);
    alpha(s, () => {
      knockout(S1.nx, S1.ny, 172 * s, 132 * s);
      drawMolecule(molB, {
        cx: S1.nx, cy: S1.ny, bond: lerp(22, 33, s), color: C.ink, lw: 2.25,
      });
    });
  }
  if (commit > 0.05 && commit < 0.98) {
    // A travelling mark on the committed edge.
    const q = ease.inOut(commit);
    circle(lerp(S1.cx, S1.nx, q), lerp(S1.cy, S1.ny, q), 5.4, C.teal, null);
  }

  restore();

  // --- type ---
  const exit1 = 1 - outro * 0.55;
  headline('Every edit changes', G.left, G.head, { size: 58 }, txt * exit1);
  headline('what comes next.', G.left, G.head + 66, { size: 58 },
           seg(t, 6.15, 7.30, ease.out) * exit1);
  alpha(seg(t, 6.85, 7.9, ease.out) * exit1, () => {
    tag('the reachable set is state dependent', G.left, G.head + 122,
        { size: 19, color: C.ink45 });
  });
}

// ---- Scene 2: the standard approach, and what limits its reuse ----
function scene2(t, d) {
  const row = 528;                       // y of the generative trajectory
  const xs = [310, 512, 714, 916, 1136]; // four latent/corrupted states + endpoint

  const arrive = seg(t, 0.0, 0.9, ease.inOut);
  const traj   = seg(t, 0.55, 3.30, ease.out);
  const distA  = seg(t, 2.55, 3.95, ease.out);
  const objA   = seg(t, 3.45, 4.10, ease.out);
  // The objective changes; the learned endpoint distribution has to be re-fit.
  const swap   = seg(t, 5.15, 5.75, ease.inOut);
  const dissolve = seg(t, 5.60, 6.80, ease.inOut);
  const refit  = seg(t, 6.75, 8.55, ease.out);
  const label  = seg(t, 4.05, 4.95, ease.out);
  const headP  = seg(t, 7.45, 8.55, ease.out);
  const outro  = 0;  // transitions are handled by the global cross-dissolve

  save();
  ctx.globalAlpha = (1 - outro);

  // Kicker sits over the trajectory, clear of the headline block.
  alpha(seg(t, 0.15, 1.0, ease.out), () => {
    tag('THE STANDARD APPROACH', xs[0] - 10, row - 176, { size: 16, color: C.ink45 });
    rule(xs[0] - 10, row - 158, 360, seg(t, 0.35, 1.3, ease.out), C.ink12, 1);
  });

  // --- the corrupted/masked trajectory ---
  // Partially specified states: drawn as incomplete graphs with dashed bonds
  // and open valences, which is what "not a molecule" looks like.
  const nStates = 4;
  for (let i = 0; i < nStates; i++) {
    const p = c01((traj * (nStates + 1.6) - i) / 1.3);
    if (p <= 0) continue;
    const x = xs[i], frac = 0.22 + i * 0.22;
    const nb = Math.max(2, Math.round(MOL.core.bonds.length * frac));
    const bset = new Set();
    for (let k = 0; k < nb; k++) bset.add(k);
    const aset = new Set();
    MOL.core.bonds.forEach((b, k) => { if (bset.has(k)) { aset.add(b.a); aset.add(b.b); } });

    alpha(p * (1 - dissolve * 0.55), () => {
      ctx.save();
      ctx.setLineDash([4.4, 4.6]);
      drawMolecule(MOL.core, {
        cx: x, cy: row, bond: 22.5, color: C.ink45, lw: 1.75,
        atoms: aset, bonds: bset, noLabels: true,
      });
      ctx.restore();
      // Open valences: short stubs that go nowhere.
      for (let s = 0; s < 3; s++) {
        const ang = rndr(300 + i * 11 + s, -Math.PI, Math.PI);
        const rr2 = 34 + rnd(360 + i * 7 + s) * 18;
        line(x + Math.cos(ang) * rr2 * 0.55, row + Math.sin(ang) * rr2 * 0.55,
             x + Math.cos(ang) * rr2, row + Math.sin(ang) * rr2,
             C.ink25, 1.45, [3, 3.6]);
      }
    });
    // Arrow to the next state.
    if (i < nStates) {
      const ap = c01((traj * (nStates + 1.6) - i - 0.55) / 0.8);
      const x2 = xs[i + 1];
      alpha(ap * (1 - dissolve * 0.5), () => {
        const gx1 = x + 44, gx2 = x2 - 44;
        lineP(gx1, row, gx2, row, C.ink25, 1.3, ap);
        if (ap > 0.9) arrowHead(gx2, row, 1, 0, 8, C.ink25);
      });
    }
  }

  // Endpoint: a complete molecule.
  const endP = c01((traj * (nStates + 1.6) - nStates) / 1.4);
  alpha(endP * (1 - dissolve * 0.35), () => {
    drawMolecule(MOL.core, { cx: xs[4], cy: row, bond: 23, color: C.ink, lw: 2.0 });
  });

  alpha(label, () => {
    tag('corrupted or masked states — not molecules', 300, row + 132,
        { size: 18, color: C.ink45 });
    rule(300, row + 112, 640, label, C.ink07, 1);
  });

  // --- the learned endpoint distribution, and the retrain cycle ---
  const dx = 1420, dy = row, dw = 340, dh = 132;
  const shapeA = u => Math.exp(-Math.pow((u - 0.40) * 3.1, 2)) * 0.92
                    + Math.exp(-Math.pow((u - 0.76) * 5.4, 2)) * 0.40;
  const shapeB = u => Math.exp(-Math.pow((u - 0.66) * 3.6, 2)) * 0.96
                    + Math.exp(-Math.pow((u - 0.28) * 6.2, 2)) * 0.34;

  // Current distribution: A, dissolving, then B re-fitting from flat.
  const curveAmp = distA * (1 - dissolve);
  const curveAmp2 = refit;
  const drawCurve = (fn, amp, col, p) => {
    if (amp <= 0.005 || p <= 0.005) return;
    ctx.save();
    ctx.strokeStyle = col; ctx.lineWidth = 2.1; ctx.lineJoin = 'round';
    ctx.beginPath();
    const N = 110, lim = Math.round(N * c01(p));
    for (let i = 0; i <= lim; i++) {
      const u = i / N;
      const x = dx + u * dw, y = dy + dh * 0.5 - fn(u) * dh * amp;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  };

  alpha(Math.max(distA, refit) * 0.9, () => {
    line(dx, dy + dh * 0.5, dx + dw, dy + dh * 0.5, C.ink12, 1.2);
  });
  drawCurve(shapeA, curveAmp, C.teal, distA);
  drawCurve(shapeB, curveAmp2, C.amber, refit);

  // Samples under the active curve.
  const sampleA = distA * (1 - dissolve);
  for (let i = 0; i < 7; i++) {
    const u = 0.10 + rnd(500 + i) * 0.8;
    const hA = shapeA(u), hB = shapeB(u);
    alpha(sampleA * 0.45, () => {
      circle(dx + u * dw, dy + dh * 0.5 - hA * dh * sampleA * rnd(520 + i) * 0.9, 3.0, C.teal, null);
    });
    alpha(refit * 0.45, () => {
      circle(dx + u * dw, dy + dh * 0.5 - hB * dh * refit * rnd(540 + i) * 0.9, 3.0, C.amber, null);
    });
  }

  // Objective glyph above the distribution: it swaps, and that forces the re-fit.
  const gy = dy - 118;
  alpha(objA * (1 - swap), () => objGlyph(0, dx + dw * 0.5, gy, 11, C.teal, true));
  alpha(swap, () => objGlyph(1, dx + dw * 0.5, gy, 11, C.amber, true));
  alpha(Math.max(objA, swap) * 0.85, () => {
    tag('objective', dx + dw * 0.5, gy - 34, { size: 15, color: C.ink45, align: 'center' });
  });

  // The retrain indicator: an arc that sweeps once and then stays as a ring, so
  // the label it belongs to is never left floating on its own.
  const rp = seg(t, 5.95, 7.55, ease.inOut);
  if (rp > 0.01) {
    const rx = dx + dw * 0.5, ry = dy + dh * 0.5 + 104, rad = 21;
    const sweep = Math.PI * 1.78 * rp;
    ctx.save();
    ctx.strokeStyle = C.amber; ctx.lineWidth = 2.1; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(rx, ry, rad, -Math.PI * 0.5, -Math.PI * 0.5 + sweep);
    ctx.stroke();
    ctx.restore();
    const a2 = -Math.PI * 0.5 + sweep;
    arrowHead(rx + Math.cos(a2) * rad, ry + Math.sin(a2) * rad,
              -Math.sin(a2), Math.cos(a2), 8.5, C.amber);
    alpha(seg(t, 6.3, 7.1, ease.out), () => {
      tag('retrain', rx, ry + 48, { size: 18, color: C.amber, align: 'center' });
    });
  }

  restore();

  const exitA = 1 - outro;
  headline('A new objective means', G.left, G.head, { size: 52 }, headP * exitA);
  headline('a new model.', G.left, G.head + 60, { size: 52, color: C.amber },
           seg(t, 7.75, 8.85, ease.out) * exitA);
}

// ---- Scene 3: COMPOSE. The executable state space and the frozen reference ----
const CAM3A = camOn(LAT.home, 1.52, 700, 604);
const CAM3B = camIdentity();

// Representative successors of the protagonist, one per cardinality class.
// Offsets are chosen so the three structures spread around the parent instead
// of stacking to one side, and so none of them collides with another.
const S3SUCC = (function () {
  const pick = (band, dx) => LAT.nodes
    .filter(n => n.band === band && n.id !== LAT.home.id)
    .sort((a, b) => Math.abs(a.x - LAT.home.x - dx) - Math.abs(b.x - LAT.home.x - dx))[0];
  return { up: pick(1, 195), same: pick(0, 355), down: pick(-1, -165) };
})();

const FAMILIES = [
  ['atom_insert', '+1'], ['atom_delete', '−1'], ['atom_restate', '='],
  ['bond_reorder', '='], ['bond_reroute', '='], ['cycle_close', '='],
  ['cycle_open', '='], ['ring_aromaticity_restate', '='],
];

function scene3(t, d) {
  const bandsIn = seg(t, 0.15, 1.65, ease.out);
  const molIn   = seg(t, 0.0, 0.85, ease.out);
  const headA   = seg(t, 0.75, 1.85, ease.out);
  const enumP   = seg(t, 2.05, 4.90, ease.out);
  const famP    = seg(t, 7.60, 9.30, ease.out);
  const guardP  = pulse(t, 5.15, 5.95, 7.05, 7.70, ease.inOut);
  // Canonicalisation inset.
  const canonIn = seg(t, 9.85, 10.65, ease.out);
  const canonOut= seg(t, 14.05, 14.75, ease.inOut);
  const canonA  = canonIn * (1 - canonOut);
  // Learned weights, then frozen.
  const learnP  = seg(t, 15.05, 17.35, ease.inOut);
  const freezeP = seg(t, 17.65, 18.60, ease.out);
  const pull    = seg(t, 18.05, d - 0.15, ease.inOut);

  const k = camLerp(CAM3A, CAM3B, pull);
  const bands = pull < 0.22 ? [-1, 0, 1] : LAT.BANDS;
  // Keep the visible neighbourhood tight while zoomed, then open it out on the
  // pull-back so the full state space arrives with the camera.
  const revealR = lerp(0, 430, enumP) + lerp(0, 1700, pull);

  // The lattice dims while the canonicalisation inset holds the screen.
  const latDim = 1 - canonA * 0.90;

  save();
  ctx.globalAlpha = latDim;

  drawLattice({
    cam: k, bands, bandA: bandsIn * (1 - pull * 0.35),
    nodeA: enumP * 0.95, edgeA: enumP * 0.95,
    weightP: learnP,
    revealC: LAT.home, revealR,
    color: freezeP > 0.5 ? C.grey : C.grey,
  });

  // Band labels: the strata of the state space by heavy-atom count.
  const bl = bandsIn * (1 - pull * 0.55);
  if (bl > 0.01) {
    [[1, 'n+1'], [0, 'n'], [-1, 'n−1']].forEach(([bk, s]) => {
      alpha(bl, () => {
        tag(s, 1812, k.y(LAT.bandY(bk)),
            { size: 19, color: C.ink45, font: F.serif, align: 'right' });
      });
    });
    alpha(bl * 0.8, () => {
      tag('HEAVY ATOMS', 1812, k.y(LAT.bandY(1)) - 52,
          { size: 13, color: C.ink25, align: 'right' });
    });
  }

  const hx = k.x(LAT.home.x), hy = k.y(LAT.home.y);

  // Three representative successors, one per cardinality class.
  const succ = [
    [S3SUCC.up, MOL.core_grow, 'atom_insert', '+1', 0.0],
    [S3SUCC.same, MOL.core_restate, 'atom_restate', '=', 0.30],
    [S3SUCC.down, MOL.core_shrink, 'atom_delete', '−1', 0.60],
  ];

  // The three example structures retire just before the full family list
  // arrives, so the screen carries one idea at a time.
  const succA = 1 - seg(t, 7.10, 7.95, ease.inOut);
  const bPar = 15.5 * k.s, bSucc = 13.0 * k.s;
  const kPar = molKnock(MOL.core, bPar);

  // 1. Knockouts, sized to each structure so they clear only what they must.
  alpha(molIn, () => knockout(hx, hy, kPar.rx, kPar.ry));
  succ.forEach(([node, m, , , off]) => {
    const p = c01((enumP - off * 0.5) / 0.42);
    if (p <= 0.55 || !node) return;
    const kk = molKnock(m, bSucc);
    alpha((p - 0.55) / 0.45 * succA,
          () => knockout(k.x(node.x), k.y(node.y), kk.rx, kk.ry));
  });

  // 2. Edges, inset to the same extents so they meet each structure cleanly and
  //    reach the bare node once the structure has retired.
  succ.forEach(([node, m, , , off]) => {
    const p = c01((enumP - off * 0.5) / 0.42);
    if (p <= 0.01 || !node) return;
    const kk = molKnock(m, bSucc);
    const padFar = lerp(7, Math.min(kk.rx, kk.ry) * 1.05, succA);
    alpha(p * succA, () => {
      linkP(hx, hy, k.x(node.x), k.y(node.y),
            Math.min(kPar.rx, kPar.ry) * 1.05, padFar,
            C.teal, 1.9, Math.min(1, p * 1.6));
    });
  });

  // 3. Structures.
  alpha(molIn, () => {
    drawMolecule(MOL.core, {
      cx: hx, cy: hy, bond: 15.5 * k.s, color: C.ink, lw: 1.55 * Math.min(1.4, k.s),
    });
  });
  succ.forEach(([node, m, name, card, off]) => {
    const p = c01((enumP - off * 0.5) / 0.42);
    if (p <= 0.55 || !node) return;
    const ex = k.x(node.x), ey = k.y(node.y);
    alpha((p - 0.55) / 0.45 * succA, () => {
      drawMolecule(m, {
        cx: ex, cy: ey, bond: 13.0 * k.s, color: C.ink, lw: 1.35 * Math.min(1.4, k.s),
      });
      tag(`${name}  ${card}`, ex, ey + 56 * k.s,
          { size: 15, color: C.teal, align: 'center' });
    });
  });

  // propose - validate - commit: a candidate that fails a guard never becomes a
  // state. It is absent from the support, not scored down.
  if (guardP > 0.01) {
    const gnode = { x: LAT.home.x + 118, y: LAT.home.y - 132 };
    const gx = k.x(gnode.x), gy = k.y(gnode.y);
    // The candidate is proposed, then contracts to nothing rather than being
    // marked rejected: such states are absent from the support, never
    // penalised, so nothing here may read as a score applied to them.
    // Driven by time, not by the pulse value, which is flat across its hold.
    const form = seg(t, 5.15, 5.85, ease.out);
    const vanish = seg(t, 6.35, 7.15, ease.inOut);
    const sc = 1 - vanish;
    alpha(form * (1 - vanish) * 0.9, () => {
      linkP(hx, hy, gx, gy, Math.min(kPar.rx, kPar.ry) * 1.05, 26 * k.s,
            C.amber, 1.5, form, [3.6, 4.2]);
      circle(gx, gy, 17 * k.s * sc, null, C.amber, 1.5);
    });
  }

  restore();

  // --- type layer (above the dimmed lattice) ---
  // Everything here yields while the canonicalisation inset holds the screen,
  // so only one headline is ever on frame.
  const typeA = 1 - canonA;
  headline('Every state is a', G.left, G.head, { size: 52 }, headA * typeA);
  headline('complete molecule.', G.left, G.head + 60, { size: 52 },
           seg(t, 1.05, 2.15, ease.out) * typeA);

  // The eight primitive families, with their cardinality effect. The block
  // retires once the canonicalisation inset takes over the screen.
  const famA = famP * (1 - canonIn);
  if (famA > 0.01) {
    save();
    ctx.globalAlpha *= famA;
    const bx = G.left, by = 874;
    alpha(famP, () => {
      tag('EIGHT PRIMITIVE REWRITE FAMILIES', bx, by - 34, { size: 15, color: C.ink45 });
      rule(bx, by - 18, 1240, famP, C.ink12, 1);
    });
    FAMILIES.forEach((f, i) => {
      const p = c01((famP * 8 - i * 0.62) / 1.1);  // staggered reveal
      if (p <= 0) return;
      const col = i < 4 ? 0 : 1;
      const x = bx + (i % 4) * 314;
      const y = by + 18 + Math.floor(i / 4) * 34;
      alpha(p, () => {
        tag(f[0], x, y, { size: 16.5, color: C.ink70 });
        const w = measure(f[0], { size: 16.5, font: F.mono, track: 0.2 });
        tag(f[1], x + w + 12, y, { size: 15, color: f[1] === '=' ? C.ink25 : C.teal });
      });
    });
    alpha(seg(t, 8.80, 9.60, ease.out), () => {
      tag('a primitive changes the heavy-atom count by at most one', bx, by + 96,
          { size: 17, color: C.ink45 });
    });
    restore();
  }

  // Guard caption, shown with the refused candidate.
  alpha(guardP * typeA, () => {
    tag('vocabulary · valence · connectivity · size', 1180, 288,
        { size: 17, color: C.amber });
    tag('inadmissible states are absent from the support', 1180, 318,
        { size: 17, color: C.ink45 });
  });

  // --- canonicalisation inset (paper's own example) ---
  if (canonA > 0.01) {
    alpha(canonA, () => {
      // Soft panel so the type reads over the lattice.
      ctx.save(); ctx.fillStyle = C.bg;
      ctx.globalAlpha = canonA; ctx.fillRect(0, 0, W, H); ctx.restore();
      canonInset(t - 9.85);
    });
  }

  // --- learned, then frozen ---
  if (learnP > 0.01) {
    const lx = 1256, ly = 250;
    alpha(seg(t, 15.05, 15.85, ease.out), () => {
      field(lx - 58, ly - 76, 534, 246);
      tag('CANONICAL-SUCCESSOR NLL', lx, ly, { size: 15, color: C.ink45 });
      rule(lx, ly + 16, 420, seg(t, 15.20, 16.05, ease.out), C.ink12, 1);
    });
    alpha(seg(t, 15.35, 16.15, ease.out), () => {
      // A single figure counting down as the learned weights fill in.
      // Both figures are the paper's own values. The improvement is carried by
      // the arrow and the reveal, never by counting through invented numbers.
      text('5.37', lx, ly + 76, { size: 40, weight: 400, font: F.serif, color: C.ink25 });
      alpha(c01((learnP - 0.25) / 0.35), () => {
        text('→', lx + 96, ly + 76, { size: 30, weight: 400, color: C.ink25 });
      });
      alpha(c01((learnP - 0.55) / 0.35), () => {
        text('3.90', lx + 146, ly + 76,
             { size: 46, weight: 500, font: F.serif, color: C.teal });
        tag('nats', lx + 258, ly + 70, { size: 17, color: C.ink45 });
      });
      tag('3,545 held-out transitions', lx, ly + 112, { size: 16, color: C.ink45 });
    });
  }

  if (freezeP > 0.01) {
    alpha(freezeP, () => {
      const fx = G.left, fy = 300;
      field(fx - 58, fy - 82, 628, 186);
      text('R', fx, fy, { size: 42, weight: 400, font: F.serif, italic: true, color: C.teal });
      text('θ', fx + 26, fy + 12, { size: 26, weight: 400, font: F.serif, italic: true, color: C.teal });
      text('fixed', fx + 62, fy, { size: 34, weight: 500, color: C.ink });
      tag('goal-independent · no downstream objective', fx, fy + 34,
          { size: 17, color: C.ink45 });
    });
  }
}

