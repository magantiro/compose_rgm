#!/usr/bin/env python3
"""Render and audit the deck from the EMITTED PPTX XML.

This deliberately reads the packaged artifact rather than the generator's own
variables: a check recomputed from the code under test cannot fail. Every shape
position, size, string and font size below comes out of ppt/slides/slideN.xml.

The deck sets Calibri, which is not installed here as a readable TTF, so text is
measured with Arial. Arial is WIDER than Calibri at the same point size, so
anything that fits under these metrics also fits in Calibri: the overflow check
errs toward reporting a problem that does not exist, never toward missing one.

  qa_render.py deck.pptx           # write build/qa-N.png and report defects
"""

from __future__ import annotations

import pathlib
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont
from defusedxml import ElementTree as ET

EMU = 914400.0
NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
}
SLIDE_W, SLIDE_H = 13.333, 7.5
PX = 150                      # render scale, px per inch
MARGIN_MIN = 0.5              # inches

FONTS = {
    (False,): '/System/Library/Fonts/Supplemental/Arial.ttf',
    (True,):  '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
}


def font(size_pt: float, bold: bool) -> ImageFont.FreeTypeFont:
    px = max(1, int(round(size_pt / 72.0 * PX)))
    return ImageFont.truetype(FONTS[(bold,)], px)


def emu(v) -> float:
    return float(v) / EMU if v is not None else 0.0


def solid_color(node):
    """First solid fill colour under a node, as RRGGBB, or None."""
    if node is None:
        return None
    c = node.find('.//a:srgbClr', NS)
    return c.get('val') if c is not None else None


def parse_shape(sp):
    """Geometry, style and text runs of one shape, read from its XML."""
    xfrm = sp.find('.//a:xfrm', NS)
    if xfrm is None:
        return None
    off, ext = xfrm.find('a:off', NS), xfrm.find('a:ext', NS)
    if off is None or ext is None:
        return None
    box = dict(
        x=emu(off.get('x')), y=emu(off.get('y')),
        w=emu(ext.get('cx')), h=emu(ext.get('cy')),
    )

    geom = sp.find('.//a:prstGeom', NS)
    box['geom'] = geom.get('prst') if geom is not None else 'rect'

    spPr = sp.find('.//p:spPr', NS)
    box['fill'] = solid_color(spPr.find('a:solidFill', NS)) if spPr is not None else None
    ln = spPr.find('a:ln', NS) if spPr is not None else None
    box['line'] = solid_color(ln) if ln is not None else None
    box['lineW'] = emu(ln.get('w')) * 72 if (ln is not None and ln.get('w')) else 1.0
    box['is_txbox'] = sp.find('.//p:nvSpPr/p:cNvSpPr[@txBox="1"]', NS) is not None

    # text: paragraphs of runs, each with its own size/bold/colour
    paras = []
    for p in sp.findall('.//a:p', NS):
        runs = []
        pPr = p.find('a:pPr', NS)
        algn = pPr.get('algn') if pPr is not None else None
        for r in p.findall('a:r', NS):
            t = r.find('a:t', NS)
            rPr = r.find('a:rPr', NS)
            if t is None or t.text is None:
                continue
            runs.append(dict(
                text=t.text,
                size=float(rPr.get('sz')) / 100.0 if (rPr is not None and rPr.get('sz')) else 18.0,
                bold=(rPr is not None and rPr.get('b') == '1'),
                color=solid_color(rPr) or '000000',
            ))
        if runs:
            paras.append(dict(runs=runs, algn=algn))
    box['paras'] = paras
    bodyPr = sp.find('.//a:bodyPr', NS)
    box['anchor'] = bodyPr.get('anchor') if bodyPr is not None else None
    return box


def wrap(draw, runs, max_w_px):
    """Greedy wrap of a run sequence into lines of (text, font, colour)."""
    lines, cur, cur_w = [], [], 0.0
    for r in runs:
        f = font(r['size'], r['bold'])
        for word in r['text'].split(' '):
            if not word:
                continue
            piece = word + ' '
            w = draw.textlength(piece, font=f)
            if cur and cur_w + w > max_w_px:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur.append((piece, f, r['color'], r['size']))
            cur_w += w
    if cur:
        lines.append(cur)
    return lines


def render(slide_xml: str, out: pathlib.Path, idx: int, defects: list):
    root = ET.fromstring(slide_xml)
    W, H = int(SLIDE_W * PX), int(SLIDE_H * PX)
    img = Image.new('RGB', (W, H), '#FCFCFB')
    d = ImageDraw.Draw(img)

    shapes = []
    for sp in root.findall('.//p:sp', NS) + root.findall('.//p:cxnSp', NS):
        b = parse_shape(sp)
        if b:
            shapes.append(b)

    for b in shapes:
        X, Y = b['x'] * PX, b['y'] * PX
        Wd, Ht = b['w'] * PX, b['h'] * PX

        # --- off-slide and margin checks, on the real emitted coordinates ---
        if b['x'] < -0.01 or b['y'] < -0.01 or \
           b['x'] + b['w'] > SLIDE_W + 0.01 or b['y'] + b['h'] > SLIDE_H + 0.01:
            defects.append(f"slide {idx}: shape off canvas at "
                           f"({b['x']:.2f},{b['y']:.2f}) {b['w']:.2f}x{b['h']:.2f}")
        if b['paras'] and (b['x'] < MARGIN_MIN - 0.01 or
                           b['x'] + b['w'] > SLIDE_W - MARGIN_MIN + 0.01):
            txt = ''.join(r['text'] for p in b['paras'] for r in p['runs'])[:34]
            defects.append(f"slide {idx}: text inside {MARGIN_MIN}\" margin: {txt!r}")

        # --- draw geometry ---
        if b['geom'] in ('rect', 'roundRect') and (b['fill'] or b['line']):
            if b['fill']:
                d.rectangle([X, Y, X + Wd, Y + Ht], fill='#' + b['fill'])
            if b['line']:
                d.rectangle([X, Y, X + Wd, Y + Ht], outline='#' + b['line'], width=2)
        elif b['geom'] == 'ellipse':
            d.ellipse([X, Y, X + Wd, Y + Ht],
                      fill='#' + b['fill'] if b['fill'] else None,
                      outline='#' + b['line'] if b['line'] else None,
                      width=max(1, int(b['lineW'] * PX / 72)))
        elif b['geom'] == 'line':
            d.line([X, Y, X + Wd, Y + Ht],
                   fill='#' + (b['line'] or '888888'),
                   width=max(1, int(b['lineW'] * PX / 72)))

        # --- draw and measure text ---
        if not b['paras']:
            continue
        pad = 0.05 * PX
        avail = max(10, Wd - 2 * pad)
        all_lines, line_h = [], []
        for p in b['paras']:
            ls = wrap(d, p['runs'], avail)
            for ln in ls:
                all_lines.append((ln, p['algn']))
                line_h.append(max(s for _, _, _, s in ln) / 72.0 * PX * 1.22)
        total = sum(line_h)
        if total > Ht + 2 and b['is_txbox']:
            txt = ''.join(r['text'] for p in b['paras'] for r in p['runs'])[:40]
            defects.append(f"slide {idx}: text overflows box "
                           f"({total/PX:.2f}\" of {b['h']:.2f}\"): {txt!r}")

        if b['anchor'] == 'ctr':
            ty = Y + (Ht - total) / 2
        else:
            ty = Y + 0.02 * PX
        for (ln, algn), lh in zip(all_lines, line_h):
            lw = sum(d.textlength(t, font=f) for t, f, _, _ in ln)
            if algn == 'ctr':
                tx = X + (Wd - lw) / 2
            elif algn == 'r':
                tx = X + Wd - pad - lw
            else:
                tx = X + pad
            base = ty + lh * 0.80
            for t, f, col, _ in ln:
                d.text((tx, base), t, font=f, fill='#' + col, anchor='ls')
                tx += d.textlength(t, font=f)
            ty += lh

    img.save(out)
    return shapes


def main() -> None:
    src = pathlib.Path(sys.argv[1])
    outdir = src.parent
    defects: list[str] = []
    with zipfile.ZipFile(src) as z:
        names = sorted(
            (n for n in z.namelist()
             if n.startswith('ppt/slides/slide') and n.endswith('.xml')),
            key=lambda n: int(''.join(c for c in n.rsplit('/', 1)[-1] if c.isdigit())),
        )
        for i, n in enumerate(names, 1):
            out = outdir / f'qa-{i}.png'
            render(z.read(n).decode('utf-8'), out, i, defects)
            print(f'rendered {out.name}')

    print()
    if defects:
        print(f'{len(defects)} geometry/overflow defect(s):')
        for x in defects:
            print('  -', x)
    else:
        print('no off-canvas, margin or text-overflow defects found')


if __name__ == '__main__':
    main()
