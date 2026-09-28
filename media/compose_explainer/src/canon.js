/* The canonicalisation example, verbatim from the paper's appendix:
 * deleting any of the three symmetry-equivalent terminal methyl carbons of
 * 2-methylpropane, CC(C)C, produces propane; canonicalisation exposes propane
 * once, carrying the combined probability of those marks.
 */

'use strict';

function canonInset(u) {
  const y = 596;
  const lx = 604, rx = 1316, bond = 58;

  const inA     = seg(u, 0.00, 0.70, ease.out);
  const marks   = seg(u, 0.55, 1.50, ease.out);
  const arrows  = seg(u, 1.35, 2.75, ease.out);
  const merge   = seg(u, 2.60, 3.45, ease.inOut);
  const formula = seg(u, 3.30, 4.15, ease.out);

  const iso = MOL.isobutane;
  // The three symmetry-equivalent terminal methyls.
  const term = iso.atoms
    .map((at, i) => (iso.bonds.filter(b => b.a === i || b.b === i).length === 1 ? i : -1))
    .filter(i => i >= 0);
  const P = iso.atoms.map(at => ({ x: lx + at.x * bond, y: y + at.y * bond }));

  alpha(seg(u, 0.15, 0.95, ease.out), () => {
    headline('One molecule, one probability.', 960, 320, { size: 46, align: 'center' }, 1);
  });
  alpha(seg(u, 0.45, 1.25, ease.out), () => {
    tag('control acts on canonical successors, not on edit marks', 960, 360,
        { size: 18, color: C.ink45, align: 'center' });
  });

  // Source structure, with its three equivalent deletion sites marked.
  alpha(inA, () => {
    drawMolecule(iso, {
      cx: lx, cy: y, bond, color: C.ink, lw: 2.4,
      highlight: { atoms: marks > 0.25 ? term : [], color: C.amber },
      highlightRing: true,
    });
    tag('CC(C)C', lx, y + 136, { size: 18, color: C.ink45, align: 'center' });
  });
  alpha(marks, () => {
    tag('three distinct atom_delete marks', lx, y + 168,
        { size: 17, color: C.amber, align: 'center' });
  });

  // Three arrows that stay distinct along the way and coalesce at the target.
  const ex = rx - 86, ey = y;
  term.forEach((ti, i) => {
    const pr = c01((arrows * 3 - i * 0.5) / 1.4);
    if (pr <= 0) return;
    // Each arrow leaves its methyl heading toward the target, so none of them
    // doubles back across the structure.
    const dx0 = ex - P[ti].x, dy0 = ey - P[ti].y;
    const L0 = Math.hypot(dx0, dy0) || 1;
    const sx = P[ti].x + (dx0 / L0) * 32;
    const sy = P[ti].y + (dy0 / L0) * 32;
    // The mid-path spread collapses as `merge` rises; the endpoint is shared
    // from the start, so three marks visibly become one successor.
    const spread = 1 - merge * 0.90;
    const cym = y + (i - 1) * 104 * spread;
    alpha(arrows, () => {
      curveP(sx, sy, lerp(sx, ex, 0.52), cym, ex, ey, C.amber, 1.9, pr);
      if (pr > 0.94) arrowHead(ex, ey, 1, 0, 9.5, C.amber);
    });
  });

  // The single canonical successor.
  alpha(c01((arrows - 0.4) / 0.5), () => {
    drawMolecule(MOL.propane, { cx: rx, cy: y, bond, color: C.ink, lw: 2.4 });
    tag('CCC', rx, y + 136, { size: 18, color: C.ink45, align: 'center' });
  });
  alpha(merge, () => {
    tag('one canonical successor', rx, y + 168,
        { size: 17, color: C.teal, align: 'center' });
  });

  // The aggregation rule, laid out by measurement so nothing collides.
  alpha(formula, () => {
    runs(960, 866, [
      { s: 'R', size: 34, font: F.serif, italic: true, color: C.ink },
      { s: 'θ', size: 21, font: F.serif, italic: true, color: C.ink, dy: 10 },
      { s: '(y | x)', size: 34, font: F.serif, color: C.ink, gap: 18 },
      { s: '∝', size: 28, font: F.serif, color: C.ink45, gap: 20 },
      { s: '∑', size: 38, font: F.serif, color: C.teal, gap: 4 },
      { s: 'a ∈ G', size: 15, font: F.mono, color: C.teal, dy: 19, gap: 16 },
      { s: 'p', size: 34, font: F.serif, italic: true, color: C.teal },
      { s: 'θ', size: 21, font: F.serif, italic: true, color: C.teal, dy: 10 },
      { s: '(a | x)', size: 34, font: F.serif, color: C.teal },
    ], 'center');
  });
}
