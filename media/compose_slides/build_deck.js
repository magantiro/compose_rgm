/* Ionizable-lipid generative programs -- research deck.
 *
 * Plain flow charts, no illustration. Every slide is a title and one diagram
 * made of labelled boxes and arrows, in the manner of a methods figure.
 *
 * Colour is semantic and does the only decorative work: teal is COMPOSE-Lipid,
 * amber is FORGE, on every slide. Everything else is ink and grey.
 *
 *   node build_deck.js   ->  build/lipid_programs.pptx
 */

'use strict';

const pptxgen = require('pptxgenjs');

// ---- design system ----
const BG    = 'FFFFFF';
const INK   = '1A1D20';
const INK70 = '5A5E62';
const INK45 = '8C9094';
const RULE  = 'D8DAD9';   // box outlines and hairlines
const TEAL  = '2F6B73';   // COMPOSE-Lipid
const AMBER = 'A9581F';   // FORGE

const F = 'Calibri';

const W = 13.333, H = 7.5;
const M = 0.75;
const M0 = 0.65;   // the five-box pipeline row runs a little wider

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'Two generative routes to new ionizable lipids';

const newSlide = () => {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
};

function title(s, text, size) {
  s.addText(text, {
    x: M, y: 0.58, w: W - 2 * M, h: 1.16,
    isTextBox: true, margin: 0, fontFace: F, fontSize: size || 32, bold: true,
    color: INK, valign: 'top',
  });
}

function hairline(s, x, y, w) {
  s.addShape(pres.ShapeType.line, { x, y, w, h: 0, line: { color: RULE, width: 0.75 } });
}

/**
 * A flow-chart node: outlined rectangle, bold label, optional detail line.
 * `accent` colours the outline and the label; omit it for a neutral node.
 */
function node(s, x, y, w, h, label, detail, accent, labelH) {
  s.addShape(pres.ShapeType.rect, {
    x, y, w, h,
    fill: { color: 'FFFFFF' },
    line: { color: accent || RULE, width: accent ? 1.25 : 1 },
  });
  const hasDetail = !!detail;
  const lh = labelH || 0.34;
  s.addText(label, {
    x: x + 0.14, y: hasDetail ? y + 0.15 : y, w: w - 0.28,
    h: hasDetail ? lh : h,
    isTextBox: true, margin: 0, fontFace: F, fontSize: 15, bold: true,
    color: accent || INK, valign: hasDetail ? 'top' : 'middle',
    align: 'left',
  });
  if (hasDetail) {
    s.addText(detail, {
      x: x + 0.14, y: y + 0.15 + lh + 0.06, w: w - 0.28,
      h: h - (0.15 + lh + 0.06) - 0.12,
      isTextBox: true, margin: 0, fontFace: F, fontSize: 12,
      color: INK45, valign: 'top', align: 'left', lineSpacingMultiple: 1.1,
    });
  }
}

/** Horizontal arrow from (x,y) running right by w. */
function arrowR(s, x, y, w) {
  s.addShape(pres.ShapeType.line, {
    x, y, w, h: 0,
    line: { color: INK45, width: 1, endArrowType: 'triangle' },
  });
}

/** Elbow arrow: out to the right, up or down, then right into a target. */
function elbow(s, x0, y0, x1, y1) {
  const xm = x0 + (x1 - x0) * 0.42;
  s.addShape(pres.ShapeType.line, {
    x: x0, y: y0, w: xm - x0, h: 0, line: { color: INK45, width: 1 },
  });
  s.addShape(pres.ShapeType.line, {
    x: xm, y: Math.min(y0, y1), w: 0, h: Math.abs(y1 - y0),
    line: { color: INK45, width: 1 },
  });
  s.addShape(pres.ShapeType.line, {
    x: xm, y: y1, w: x1 - xm, h: 0,
    line: { color: INK45, width: 1, endArrowType: 'triangle' },
  });
}

/** Caption under a diagram. */
function caption(s, text, x, y, w, color) {
  s.addText(text, {
    x, y, w, h: 0.56, isTextBox: true, margin: 0, fontFace: F,
    fontSize: 12.5, color: color || INK45, valign: 'top',
    lineSpacingMultiple: 1.15,
  });
}

// =====================================================================
// 1 — title
// =====================================================================
{
  const s = newSlide();
  s.addText('Two generative routes to new ionizable lipids', {
    x: M, y: 2.22, w: 10.4, h: 1.46, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 40, bold: true, color: INK, valign: 'middle',
  });
  hairline(s, M, 3.66, 9.4);

  s.addText([
    { text: 'FORGE', options: { bold: true, color: AMBER, fontSize: 18 } },
    { text: '   synthesis-grounded flow matching within a known Ugi-3 family',
      options: { color: INK70, fontSize: 15 } },
  ], { x: M, y: 3.92, w: 11.0, h: 0.34, isTextBox: true, margin: 0,
       fontFace: F, valign: 'middle' });

  s.addText([
    { text: 'COMPOSE-Lipid', options: { bold: true, color: TEAL, fontSize: 18 } },
    { text: '   fragment-constrained transformation into novel scaffold space',
      options: { color: INK70, fontSize: 15 } },
  ], { x: M, y: 4.36, w: 11.0, h: 0.34, isTextBox: true, margin: 0,
       fontFace: F, valign: 'middle' });
}

// =====================================================================
// 2 — overview comparison
// =====================================================================
{
  const s = newSlide();
  title(s, 'Two complementary generative routes');

  const labX = M, labW = 3.30;
  const colL = 4.35, colR = 8.85, colW = 3.75;

  s.addText('FORGE', {
    x: colL, y: 1.66, w: colW, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 19, bold: true, color: AMBER, valign: 'middle',
  });
  s.addText('COMPOSE-Lipid', {
    x: colR, y: 1.66, w: colW, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 19, bold: true, color: TEAL, valign: 'middle',
  });

  const rows = [
    ['Model type',      'Flow matching',            'Controlled transformation process'],
    ['Search regime',   'Synthesis-grounded',       'Executable molecular edits'],
    ['Scaffold regime', 'Known Ugi-3 family',       'Novel scaffold space'],
    ['Chemistry prior', 'Reaction semantics',       'Fragment constraint on a motif'],
    ['Screening path',  'HeLa → reporter IM',  'A549 → intranasal lung'],
    ['Application',     'IM vaccine delivery',      'Lung editing'],
  ];

  const y0 = 2.14, dy = 0.72;
  rows.forEach((r, i) => {
    const y = y0 + i * dy;
    hairline(s, M, y, W - 2 * M);
    s.addText(r[0], {
      x: labX, y: y + 0.16, w: labW, h: 0.40, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 13, color: INK45, valign: 'middle',
    });
    s.addText(r[1], {
      x: colL, y: y + 0.16, w: colW, h: 0.40, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 15, color: INK, valign: 'middle', bold: i === 2,
    });
    s.addText(r[2], {
      x: colR, y: y + 0.16, w: colW, h: 0.40, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 15, color: INK, valign: 'middle', bold: i === 2,
    });
  });
  hairline(s, M, y0 + rows.length * dy, W - 2 * M);
}

// =====================================================================
// 3 — COMPOSE  (one constrained source, several controlled endpoints)
// =====================================================================
{
  const s = newSlide();
  title(s, 'COMPOSE');

  const bw = 2.85, bh = 1.15;
  const x1 = M, x2 = 4.55, x3 = 9.55;
  const yMid = 3.30;

  node(s, x1, yMid, bw, bh, 'Retained motif',
       'the fragment that must be preserved', TEAL);
  node(s, x2, yMid, bw, bh, 'Transformation process',
       'executable molecular edits, learned once', INK);

  arrowR(s, x1 + bw + 0.10, yMid + bh / 2, x2 - (x1 + bw) - 0.20);

  const outs = [
    ['Endpoint A', 1.98],
    ['Endpoint B', 3.42],
    ['Endpoint C', 4.86],
  ];
  outs.forEach(([lab, y]) => {
    node(s, x3, y, 3.05, 0.90, lab, null, TEAL);
    elbow(s, x2 + bw + 0.10, yMid + bh / 2, x3 - 0.10, y + 0.45);
  });

  caption(s, 'The constraint fixes what is preserved; control decides which of the '
          + 'reachable endpoints is taken.', M, 6.02, 10.8);
}

// =====================================================================
// 4 — FORGE  (three reaction roles converging on one candidate)
// =====================================================================
{
  const s = newSlide();
  title(s, 'FORGE');

  const rw = 2.85, rh = 0.90;
  const x1 = M, x2 = 4.55, x3 = 9.10;
  const yMid = 3.30;

  const roles = [['Amine role', 1.98], ['Acid role', 3.42], ['Aldehyde role', 4.86]];
  roles.forEach(([lab, y]) => {
    node(s, x1, y, rw, rh, lab, null, AMBER);
    elbow(s, x1 + rw + 0.10, y + rh / 2, x2 - 0.10, yMid + 0.575);
  });

  node(s, x2, yMid, 3.10, 1.15, 'Ugi-3 assembly',
       'generation constrained by the reaction', AMBER);
  arrowR(s, x2 + 3.10 + 0.10, yMid + 0.575, x3 - (x2 + 3.10) - 0.20);
  node(s, x3, yMid, 3.48, 1.15, 'Candidate lipid',
       'synthesisable within the known family', AMBER);

  caption(s, 'Precursor roles are preserved, so every candidate stays makeable in the '
          + 'established chemistry.', M, 6.02, 10.8);
}

// =====================================================================
// 5 — COMPOSE-Lipid application
// =====================================================================
{
  const s = newSlide();
  title(s, 'COMPOSE-Lipid: novel biodegradable linker scaffolds for lung editing', 30);

  const P = M0, bw = 2.08, gap = 0.40, bh = 1.14, y = 3.18;
  const steps = [
    ['Scaffold',   'a biodegradable linker motif'],
    ['Generation', 'novel ionizable lipids built around it'],
    ['In vitro',   'A549 screen'],
    ['In vivo',    'intranasal delivery to the lung'],
    ['Readout',    'editing across lung cell types'],
  ];
  steps.forEach((st, i) => {
    const x = P + i * (bw + gap);
    node(s, x, y, bw, bh, st[0], st[1], TEAL, 0.32);
    if (i < steps.length - 1) arrowR(s, x + bw + 0.05, y + bh / 2, gap - 0.10);
  });

  caption(s, 'The linker is held fixed and the rest is generated, so biodegradability '
          + 'is carried by construction rather than selected for downstream.',
          M, 4.74, 11.2);
}

// =====================================================================
// 6 — FORGE application
// =====================================================================
{
  const s = newSlide();
  title(s, 'FORGE: Ugi-3 lipid discovery for IM vaccine delivery', 30);

  const P = M0, bw = 2.08, gap = 0.40, bh = 1.14, y = 3.18;
  const steps = [
    ['Generation',    'candidates within the known Ugi-3 family'],
    ['In vitro',      'HeLa screen'],
    ['In vivo',       'reporter IM setting'],
    ['Hits',          'two candidates carried forward'],
    ['Vaccine study', 'advanced into the follow-on study'],
  ];
  steps.forEach((st, i) => {
    const x = P + i * (bw + gap);
    node(s, x, y, bw, bh, st[0], st[1], AMBER, 0.32);
    if (i < steps.length - 1) arrowR(s, x + bw + 0.05, y + bh / 2, gap - 0.10);
  });

  caption(s, 'Generation stays inside the established family throughout, so a hit is '
          + 'immediately actionable in the existing synthesis programme.',
          M, 4.74, 11.2);
}

pres.writeFile({ fileName: 'build/lipid_programs.pptx' })
  .then(f => console.log('wrote', f));
