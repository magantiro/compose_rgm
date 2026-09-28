/* The shared state-space lattice.
 *
 * This single object is the continuity device of the whole film: the fan in
 * scene 1, the enumerated successor set in scene 3, the program path and the
 * three controlled trajectories in scene 4, and the closing image in scene 6
 * are all the same geometry seen at different camera scales.
 *
 * Bands are strata of the state space by heavy-atom count. Band 0 is X_n; band
 * +1 is X_{n+1}; band -1 is X_{n-1}. A primitive rewrite moves at most one band
 * (birth / death / same cardinality), which is the paper's trans-dimensional
 * claim, so every lattice edge connects bands differing by at most one.
 */

'use strict';

const LAT = (function build() {
  const BAND_Y0 = 616;        // y of band 0
  const BAND_DY = 76;         // px between bands
  // Bands reach +5 so the six-primitive ring-building program of scene 4
  // (five births then one cycle_close) has somewhere to land.
  const BANDS = [-2, -1, 0, 1, 2, 3, 4, 5];
  const X0 = 250, X1 = 1670;

  const bandY = k => BAND_Y0 - k * BAND_DY;

  const nodes = [];
  let key = 1;
  BANDS.forEach(k => {
    // Fewer nodes on the outer bands so the lattice reads as a body, not a grid.
    const n = k === 0 ? 15 : (Math.abs(k) === 1 ? 14 : (Math.abs(k) === 2 ? 12 : 11));
    for (let i = 0; i < n; i++) {
      const u = n === 1 ? 0.5 : i / (n - 1);
      const jx = rndr(key++, -26, 26);
      const jy = rndr(key++, -13, 13);
      nodes.push({
        id: nodes.length,
        band: k,
        x: lerp(X0, X1, u) + jx,
        y: bandY(k) + jy,
        r: 3.1 + rnd(key++) * 1.5,
        s: key,
      });
    }
  });

  // Edges: connect each node to nearby nodes in the same or an adjacent band.
  const edges = [];
  const seen = new Set();
  nodes.forEach(a => {
    const cands = nodes
      .filter(b => b.id !== a.id && Math.abs(b.band - a.band) <= 1)
      .map(b => ({ b, d: Math.hypot(b.x - a.x, b.y - a.y) }))
      .filter(o => o.d < 190)
      .sort((p, q) => p.d - q.d)
      .slice(0, 4);
    const take = 2 + Math.floor(rnd(a.s * 7) * 2); // 2..3 edges per node
    cands.slice(0, take).forEach(o => {
      const k2 = a.id < o.b.id ? `${a.id}-${o.b.id}` : `${o.b.id}-${a.id}`;
      if (seen.has(k2)) return;
      seen.add(k2);
      edges.push({
        a: a.id, b: o.b.id,
        // Learned reference weight: a deterministic stand-in for R_theta used
        // purely to give the drawing a non-uniform, state-dependent texture.
        w: 0.18 + rnd(edges.length * 13 + 5) * 0.82,
        d: o.d,
      });
    });
  });

  // The protagonist sits on band 0, a little left of centre.
  let home = nodes.filter(n => n.band === 0).sort((p, q) =>
    Math.abs(p.x - 640) - Math.abs(q.x - 640))[0];

  // Adjacency for path walks.
  const adj = nodes.map(() => []);
  edges.forEach((e, i) => { adj[e.a].push({ to: e.b, e: i }); adj[e.b].push({ to: e.a, e: i }); });

  return { nodes, edges, adj, home, bandY, BANDS, BAND_DY, X0, X1 };
})();

/** Camera: maps lattice coords to screen. */
function cam(scale, focusX, focusY, atX, atY) {
  return {
    s: scale,
    x: (px) => atX + (px - focusX) * scale,
    y: (py) => atY + (py - focusY) * scale,
  };
}

/** Interpolate between two cameras. */
function camLerp(a, b, t) {
  const s = lerp(a.s, b.s, t);
  // Interpolate the mapping of two reference points, which is exact for an
  // affine camera and avoids the drift of interpolating parameters separately.
  const ax0 = a.x(0), bx0 = b.x(0), ay0 = a.y(0), by0 = b.y(0);
  const x0 = lerp(ax0, bx0, t), y0 = lerp(ay0, by0, t);
  return { s, x: (px) => x0 + px * s, y: (py) => y0 + py * s };
}

/** A camera that maps lattice space 1:1 to the screen. */
function camIdentity() { return { s: 1, x: p => p, y: p => p }; }

/** A camera zoomed on a lattice point, placing it at (atX, atY). */
function camOn(node, scale, atX, atY) {
  return {
    s: scale,
    x: (px) => atX + (px - node.x) * scale,
    y: (py) => atY + (py - node.y) * scale,
  };
}

/**
 * Draw the lattice.
 * opts: {cam, nodeA, edgeA, bands:[k...], bandA, weightP, color, accentEdges,
 *        revealR (px radius from a centre), revealC, nodeScale}
 */
function drawLattice(o) {
  const k = o.cam || camIdentity();
  const nodeA = o.nodeA === undefined ? 1 : o.nodeA;
  const edgeA = o.edgeA === undefined ? 1 : o.edgeA;
  const bands = o.bands || LAT.BANDS;
  const bandSet = new Set(bands);
  const wp = o.weightP === undefined ? 0 : o.weightP;  // 0 uniform -> 1 learned
  const col = o.color || C.grey;

  // Radial reveal: nodes/edges fade in with distance from a centre.
  const rc = o.revealC, rr_ = o.revealR;
  const vis = (x, y) => {
    if (rr_ === undefined || !rc) return 1;
    const d = Math.hypot(x - rc.x, y - rc.y);
    return c01((rr_ - d) / 120);
  };

  // Band rules.
  if (o.bandA > 0.001) {
    bands.forEach(bk => {
      const y = k.y(LAT.bandY(bk));
      alpha(o.bandA * (bk === 0 ? 1 : 0.62), () => {
        line(k.x(LAT.X0) - 30, y, k.x(LAT.X1) + 30, y, C.ink07, 1.2);
      });
    });
  }

  if (edgeA > 0.001) {
    LAT.edges.forEach((e, i) => {
      const a = LAT.nodes[e.a], b = LAT.nodes[e.b];
      if (!bandSet.has(a.band) || !bandSet.has(b.band)) return;
      const v = Math.min(vis(a.x, a.y), vis(b.x, b.y));
      if (v <= 0.01) return;
      // Uniform -> learned: line weight becomes state dependent.
      const w = lerp(0.9, 0.35 + e.w * 2.0, wp);
      const av = lerp(0.5, 0.2 + e.w * 0.78, wp);
      alpha(edgeA * v * av, () => {
        line(k.x(a.x), k.y(a.y), k.x(b.x), k.y(b.y), col, w * (k.s > 1.4 ? 1.25 : 1));
      });
    });
  }

  if (nodeA > 0.001) {
    LAT.nodes.forEach(n => {
      if (!bandSet.has(n.band)) return;
      const v = vis(n.x, n.y);
      if (v <= 0.01) return;
      alpha(nodeA * v, () => {
        circle(k.x(n.x), k.y(n.y), n.r * (o.nodeScale || 1) * Math.min(1.6, Math.max(0.85, k.s)),
               col, null);
      });
    });
  }

  // Accent edges drawn on top.
  (o.accentEdges || []).forEach(ae => {
    const e = LAT.edges[ae.i];
    if (!e) return;
    const a = LAT.nodes[e.a], b = LAT.nodes[e.b];
    lineP(k.x(a.x), k.y(a.y), k.x(b.x), k.y(b.y), ae.color || C.teal, ae.w || 2.2, ae.p === undefined ? 1 : ae.p);
  });
}

/**
 * Deterministic walk through the lattice, biased by a direction preference.
 * Used for the controlled trajectories: the same grey lattice, different paths.
 * pref: function(node, fromNode) -> score. Higher is preferred.
 */
function latticeWalk(startId, steps, pref, seed) {
  const path = [startId];
  const used = new Set([startId]);
  let cur = startId;
  for (let s = 0; s < steps; s++) {
    const node = LAT.nodes[cur];
    let cands = LAT.adj[cur].filter(o => !used.has(o.to));
    // Walks advance rather than doubling back, so a trajectory reads as one
    // purposeful route instead of a tangle. Relaxed only if nothing qualifies.
    const fwd = cands.filter(o => LAT.nodes[o.to].x > node.x - 18);
    if (fwd.length) cands = fwd;
    const opts = cands
      .map(o => ({ o, v: pref(LAT.nodes[o.to], node) + rnd(seed + s * 31 + o.to) * 0.30 }))
      .sort((p, q) => q.v - p.v);
    if (!opts.length) break;
    cur = opts[0].o.to;
    used.add(cur);
    path.push(cur);
  }
  return path;
}

/** Draw a path through lattice nodes, revealed to fraction p, with a head dot. */
function drawPath(path, k, p, col, lw, opts) {
  const o = opts || {};
  if (path.length < 2) return null;
  const segs = path.length - 1;
  const fp = c01(p) * segs;
  let head = null;
  for (let i = 0; i < segs; i++) {
    const local = c01(fp - i);
    if (local <= 0) break;
    const a = LAT.nodes[path[i]], b = LAT.nodes[path[i + 1]];
    const x1 = k.x(a.x), y1 = k.y(a.y), x2 = k.x(b.x), y2 = k.y(b.y);
    lineP(x1, y1, x2, y2, col, lw || 2.4, local);
    head = { x: lerp(x1, x2, local), y: lerp(y1, y2, local),
             dx: x2 - x1, dy: y2 - y1 };
    if (local >= 1 && o.dots) circle(x2, y2, (lw || 2.4) * 0.95, col, null);
  }
  if (head && o.head) circle(head.x, head.y, (lw || 2.4) * 1.5, col, null);
  return head;
}
