/* COMPOSE explainer -- drawing core.
 *
 * Deterministic 2D vector drawing on canvas. Nothing here reads the clock or
 * Math.random at draw time: every pseudo-random value is drawn from a
 * precomputed seeded table so frame N is byte-identical on every render.
 */

'use strict';

// ---- Design system ----
const W = 1920, H = 1080;

const C = {
  bg:      '#FCFCFB',
  ink:     '#16191C',
  ink70:   'rgba(22,25,28,0.70)',
  ink45:   'rgba(22,25,28,0.45)',
  ink25:   'rgba(22,25,28,0.25)',
  ink12:   'rgba(22,25,28,0.12)',
  ink07:   'rgba(22,25,28,0.07)',
  teal:    '#3E767D',   // paper's composeteal, slightly deepened for screen
  tealSoft:'rgba(62,118,125,0.16)',
  amber:   '#B4622B',
  amberSoft:'rgba(180,98,43,0.14)',
  grey:    '#A8AEB2',
  grey50:  'rgba(168,174,178,0.5)',
};

const F = {
  // System stack: no network fetch, so rendering is reproducible offline.
  sans: '"Helvetica Neue", "Inter", -apple-system, "Segoe UI", Arial, sans-serif',
  serif:'"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif',
  mono: '"SF Mono", "Menlo", "Consolas", monospace',
};

// ---- Seeded noise table (determinism) ----
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const RND = (function () {
  const r = mulberry32(20260928);
  const tbl = new Float64Array(4096);
  for (let i = 0; i < tbl.length; i++) tbl[i] = r();
  return tbl;
})();
/** Stable pseudo-random in [0,1) for a given integer key. */
function rnd(key) { return RND[((key % RND.length) + RND.length) % RND.length]; }
/** Stable pseudo-random in [lo,hi). */
function rndr(key, lo, hi) { return lo + rnd(key) * (hi - lo); }

// ---- Easing ----
const ease = {
  linear: t => t,
  inOut:  t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out:    t => 1 - Math.pow(1 - t, 3),
  outQuint: t => 1 - Math.pow(1 - t, 5),
  in:     t => t * t * t,
  outBackSoft: t => {
    const c1 = 1.05, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
};

/** Clamp to [0,1]. */
function c01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

/** Local progress of a sub-beat: 0 before `a`, 1 after `b`, eased in between. */
function seg(t, a, b, fn) {
  if (b <= a) return t >= b ? 1 : 0;
  return (fn || ease.inOut)(c01((t - a) / (b - a)));
}

/** Rises 0->1 over [a,b] then falls 1->0 over [c,d]. */
function pulse(t, a, b, c, d, fn) {
  const f = fn || ease.inOut;
  if (t < a) return 0;
  if (t < b) return f(c01((t - a) / (b - a)));
  if (t <= c) return 1;
  if (t < d) return 1 - f(c01((t - c) / (d - c)));
  return 0;
}

function lerp(a, b, t) { return a + (b - a) * t; }
function mix(p, q, t) { return { x: lerp(p.x, q.x, t), y: lerp(p.y, q.y, t) }; }

// ---- Canvas helpers ----
let ctx = null;
function setCtx(c) { ctx = c; }

function save() { ctx.save(); }
function restore() { ctx.restore(); }

function alpha(a, fn) {
  if (a <= 0.001) return;
  ctx.save(); ctx.globalAlpha *= Math.min(1, a); fn(); ctx.restore();
}

function line(x1, y1, x2, y2, col, w, dash) {
  ctx.save();
  ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round';
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.restore();
}

/**
 * Line between two points, inset by `pad1`/`pad2` at each end so it stops clear
 * of whatever occupies the endpoints, drawn to fraction p of the inset span.
 */
function linkP(x1, y1, x2, y2, pad1, pad2, col, w, p, dash) {
  const dx = x2 - x1, dy = y2 - y1;
  const L = Math.hypot(dx, dy);
  if (L <= pad1 + pad2 + 2) return;
  const ux = dx / L, uy = dy / L;
  lineP(x1 + ux * pad1, y1 + uy * pad1,
        x2 - ux * pad2, y2 - uy * pad2, col, w, p, dash);
}

/** Straight line drawn from its start to fraction p of its length. */
function lineP(x1, y1, x2, y2, col, w, p, dash) {
  if (p <= 0) return;
  const q = Math.min(1, p);
  line(x1, y1, lerp(x1, x2, q), lerp(y1, y2, q), col, w, dash);
}

function circle(x, y, r, fill, stroke, sw) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw || 1.4; ctx.stroke(); }
}

/** Quadratic curve, drawn from 0 to fraction p, via de Casteljau subdivision. */
function curveP(x1, y1, cx, cy, x2, y2, col, w, p, dash) {
  if (p <= 0.001) return;
  const q = Math.min(1, p);
  const N = 48, n = Math.max(1, Math.round(N * q));
  ctx.save();
  ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const u = (i / N) * 1; // param along full curve
    const uu = Math.min(u, q);
    const mt = 1 - uu;
    const x = mt * mt * x1 + 2 * mt * uu * cx + uu * uu * x2;
    const y = mt * mt * y1 + 2 * mt * uu * cy + uu * uu * y2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function qpt(x1, y1, cx, cy, x2, y2, u) {
  const mt = 1 - u;
  return {
    x: mt * mt * x1 + 2 * mt * u * cx + u * u * x2,
    y: mt * mt * y1 + 2 * mt * u * cy + u * u * y2,
  };
}

/** Small solid arrowhead pointing along (dx,dy). */
function arrowHead(x, y, dx, dy, size, col) {
  const L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L, px = -uy, py = ux;
  ctx.save();
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - ux * size + px * size * 0.42, y - uy * size + py * size * 0.42);
  ctx.lineTo(x - ux * size - px * size * 0.42, y - uy * size - py * size * 0.42);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

// ---- Text ----
/**
 * Draw text. opts: {size, weight, font, color, align, baseline, track, italic}
 * `track` is letter-spacing in px (uses ctx.letterSpacing where available).
 */
function text(str, x, y, opts) {
  const o = opts || {};
  ctx.save();
  const style = o.italic ? 'italic ' : '';
  ctx.font = `${style}${o.weight || 400} ${o.size || 24}px ${o.font || F.sans}`;
  ctx.fillStyle = o.color || C.ink;
  ctx.textAlign = o.align || 'left';
  ctx.textBaseline = o.baseline || 'alphabetic';
  if (o.track !== undefined && 'letterSpacing' in ctx) {
    ctx.letterSpacing = `${o.track}px`;
  }
  ctx.fillText(str, x, y);
  ctx.restore();
}

function measure(str, opts) {
  const o = opts || {};
  ctx.save();
  const style = o.italic ? 'italic ' : '';
  ctx.font = `${style}${o.weight || 400} ${o.size || 24}px ${o.font || F.sans}`;
  if (o.track !== undefined && 'letterSpacing' in ctx) ctx.letterSpacing = `${o.track}px`;
  const m = ctx.measureText(str).width;
  ctx.restore();
  return m;
}

/** Text that types on by character fraction p (no caret; clean reveal). */
function textP(str, x, y, opts, p) {
  if (p <= 0) return;
  const n = Math.round(str.length * c01(p));
  text(str.slice(0, n), x, y, opts);
}

/**
 * Headline that fades + rises into place. p in [0,1].
 */
function headline(str, x, y, opts, p) {
  if (p <= 0.001) return;
  const o = Object.assign({ size: 54, weight: 600, track: -1.0 }, opts || {});
  alpha(p, () => text(str, x, y + (1 - p) * 16, o));
}

/**
 * Lay out a row of text runs left to right, measuring each so nothing collides.
 * Each item: {s, size, font, italic, weight, color, dy, gap, track}.
 * `align` 'center' centres the whole row on x. Returns the total width.
 */
function runs(x, y, items, align) {
  let total = 0;
  items.forEach(it => { total += measure(it.s, it) + (it.gap || 0); });
  let cx = align === 'center' ? x - total / 2 : x;
  items.forEach(it => {
    text(it.s, cx, y + (it.dy || 0), it);
    cx += measure(it.s, it) + (it.gap || 0);
  });
  return total;
}

/** Thin rule that draws from its left edge. */
function rule(x, y, w, p, col, th) {
  if (p <= 0) return;
  line(x, y, x + w * c01(p), y, col || C.ink12, th || 1);
}

// ---- Molecule rendering ----
/**
 * Draw a skeletal structure.
 *
 * mol: {atoms:[{el,x,y,arom,ring}], bonds:[{a,b,order,arom,ring}], rings:[...]}
 * opts:
 *   cx, cy      centre in px
 *   bond        px per unit bond length
 *   rot         rotation in radians
 *   color       line colour
 *   lw          line width
 *   atoms       Set/array of visible atom indices (default: all)
 *   bonds       Set of visible bond indices (default: all implied by atoms)
 *   draw        0..1 progressive draw of bond lines
 *   labelScale  multiplier on heteroatom label size
 *   labelColor  colour for heteroatom labels
 *   dim         0..1 extra alpha
 *   highlight   {atoms:[], bonds:[], color} accent overlay
 *   noLabels    suppress heteroatom labels
 */
function drawMolecule(mol, opts) {
  const o = opts || {};
  const bl = o.bond || 26;
  const cx = o.cx || 0, cy = o.cy || 0;
  const rot = o.rot || 0;
  const cr = Math.cos(rot), sr = Math.sin(rot);
  const col = o.color || C.ink;
  const lw = o.lw || 2.0;
  const drawP = o.draw === undefined ? 1 : c01(o.draw);

  const visA = o.atoms ? (o.atoms instanceof Set ? o.atoms : new Set(o.atoms)) : null;
  const visB = o.bonds ? (o.bonds instanceof Set ? o.bonds : new Set(o.bonds)) : null;

  const P = mol.atoms.map(a => ({
    x: cx + (a.x * cr - a.y * sr) * bl,
    y: cy + (a.x * sr + a.y * cr) * bl,
  }));

  const shown = i => (visA ? visA.has(i) : true);
  const labelled = i => {
    const el = mol.atoms[i].el;
    return el !== 'C';
  };

  // Bond endpoints are pulled back from labelled atoms so the glyph reads clean.
  const pad = i => (labelled(i) ? bl * 0.30 : 0);

  const hlA = new Set((o.highlight && o.highlight.atoms) || []);
  const hlB = new Set((o.highlight && o.highlight.bonds) || []);
  const hlCol = (o.highlight && o.highlight.color) || C.teal;

  // Count visible bonds for staged progressive drawing.
  const order = [];
  mol.bonds.forEach((b, i) => {
    if (visB ? visB.has(i) : (shown(b.a) && shown(b.b))) order.push(i);
  });
  const nb = order.length;
  const fullBonds = drawP * nb;

  ctx.save();
  ctx.globalAlpha *= (o.dim === undefined ? 1 : o.dim);

  order.forEach((bi, k) => {
    const b = mol.bonds[bi];
    // Each bond draws over a 1-unit window, with a short overlap for flow.
    const local = c01((fullBonds - k) / 1.0);
    if (local <= 0) return;

    const pa = P[b.a], pb = P[b.b];
    let dx = pb.x - pa.x, dy = pb.y - pa.y;
    const L = Math.hypot(dx, dy) || 1;
    const ux = dx / L, uy = dy / L;
    const ax = pa.x + ux * pad(b.a), ay = pa.y + uy * pad(b.a);
    const bx = pb.x - ux * pad(b.b), by = pb.y - uy * pad(b.b);

    const thisCol = hlB.has(bi) ? hlCol : col;
    const thisLw = hlB.has(bi) ? lw * 1.7 : lw;
    const px = -uy, py = ux;

    const isArom = b.order === 4 || b.arom;
    if (isArom) {
      // Outer skeleton line plus an inner arc offset toward the ring centre,
      // the standard delocalised-ring convention.
      lineP(ax, ay, bx, by, thisCol, thisLw, local);
      const ring = mol.rings.find(r => r.atoms.includes(b.a) && r.atoms.includes(b.b));
      if (ring) {
        const rc = {
          x: cx + (ring.cx * cr - ring.cy * sr) * bl,
          y: cy + (ring.cx * sr + ring.cy * cr) * bl,
        };
        const mx = (ax + bx) / 2, my = (ay + by) / 2;
        let nx = rc.x - mx, ny = rc.y - my;
        const nl = Math.hypot(nx, ny) || 1;
        nx /= nl; ny /= nl;
        const off = bl * 0.17, sh = bl * 0.20;
        lineP(ax + nx * off + ux * sh, ay + ny * off + uy * sh,
              bx + nx * off - ux * sh, by + ny * off - uy * sh,
              thisCol, thisLw * 0.85, local);
      }
    } else if (b.order === 2) {
      const off = bl * 0.105;
      lineP(ax + px * off, ay + py * off, bx + px * off, by + py * off, thisCol, thisLw, local);
      lineP(ax - px * off, ay - py * off, bx - px * off, by - py * off, thisCol, thisLw, local);
    } else if (b.order === 3) {
      const off = bl * 0.14;
      lineP(ax, ay, bx, by, thisCol, thisLw, local);
      lineP(ax + px * off, ay + py * off, bx + px * off, by + py * off, thisCol, thisLw, local);
      lineP(ax - px * off, ay - py * off, bx - px * off, by - py * off, thisCol, thisLw, local);
    } else {
      lineP(ax, ay, bx, by, thisCol, thisLw, local);
    }
  });

  // Heteroatom labels, knocked out of the bonds by a background-coloured disc.
  if (!o.noLabels) {
    const lsz = bl * 0.80 * (o.labelScale || 1);
    mol.atoms.forEach((a, i) => {
      if (!shown(i) || !labelled(i)) return;
      // Only label once its bonds have started drawing.
      const anyBond = order.some((bi, k) => {
        const b = mol.bonds[bi];
        return (b.a === i || b.b === i) && (fullBonds - k) > 0.25;
      });
      if (nb > 0 && !anyBond) return;
      const isHl = hlA.has(i);
      circle(P[i].x, P[i].y, lsz * 0.62, C.bg, null);
      text(a.el, P[i].x, P[i].y, {
        size: lsz, weight: 500, font: F.sans,
        color: isHl ? hlCol : (o.labelColor || col),
        align: 'center', baseline: 'middle', track: 0,
      });
    });
  }

  // Accent rings on highlighted atoms.
  if (hlA.size && o.highlightRing) {
    hlA.forEach(i => {
      if (!shown(i)) return;
      circle(P[i].x, P[i].y, bl * 0.40, null, hlCol, lw * 0.9);
    });
  }

  ctx.restore();
  return P;
}

/**
 * Half-extents of the ellipse that just covers a molecule at a given bond
 * length, with a small margin. Used both to knock the structure out of the
 * graph behind it and to inset the edges that meet it, so the two always agree.
 */
function molKnock(mol, bond, margin) {
  const b = molBox(mol, bond);
  const m = margin === undefined ? 13 : margin;
  return { rx: b.w / 2 + m, ry: b.h / 2 + m };
}

/** Bounding box of a molecule in px for a given bond length (unrotated). */
function molBox(mol, bond, atomsVisible) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  mol.atoms.forEach((a, i) => {
    if (atomsVisible && !atomsVisible.has(i)) return;
    x0 = Math.min(x0, a.x); x1 = Math.max(x1, a.x);
    y0 = Math.min(y0, a.y); y1 = Math.max(y1, a.y);
  });
  return { w: (x1 - x0) * bond, h: (y1 - y0) * bond,
           cx: ((x0 + x1) / 2) * bond, cy: ((y0 + y1) / 2) * bond };
}

// ---- Composite marks ----
/** Small monospace tag, optionally with a leading cardinality glyph. */
function tag(str, x, y, opts) {
  const o = Object.assign({ size: 15, color: C.ink45, align: 'left' }, opts || {});
  text(str, x, y, { size: o.size, weight: 400, font: F.mono, color: o.color,
                    align: o.align, baseline: 'middle', track: 0.2 });
}

/** Objective glyph: 0 diamond (teal), 1 triangle (amber), 2 square (ink). */
function objGlyph(kind, x, y, r, col, fill) {
  ctx.save();
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.0;
  ctx.beginPath();
  if (kind === 0) {
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
  } else if (kind === 1) {
    const h = r * 1.08;
    ctx.moveTo(x, y - h); ctx.lineTo(x + r * 1.02, y + h * 0.72); ctx.lineTo(x - r * 1.02, y + h * 0.72);
  } else {
    const s = r * 0.92;
    ctx.rect(x - s, y - s, s * 2, s * 2);
  }
  ctx.closePath();
  if (fill) ctx.fill(); else ctx.stroke();
  ctx.restore();
}

/**
 * Soft-edged background field so type stays legible over the lattice without
 * leaving a visible rectangular cut in it. Drawn twice: a blurred pass that
 * feathers the edge, then a solid core.
 */
function field(x, y, w, h) {
  ctx.save();
  ctx.fillStyle = C.bg;
  ctx.filter = 'blur(26px)';
  rr(x + 14, y + 14, Math.max(2, w - 28), Math.max(2, h - 28), 26);
  ctx.fill();
  ctx.filter = 'none';
  rr(x + 34, y + 30, Math.max(2, w - 68), Math.max(2, h - 60), 12);
  ctx.fill();
  ctx.restore();
}

/** Rounded rect path. */
function rr(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Number formatted with fixed decimals, counting from a to b by p. */
function num(a, b, p, dp) {
  const v = lerp(a, b, c01(p));
  return v.toFixed(dp === undefined ? 2 : dp);
}
