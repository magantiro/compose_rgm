"""Render a Markdown doc to ONE self-contained HTML file. No dependencies.

Written because the master plan is mostly tables and nested structure, which
Chrome shows as raw pipes when handed a .md. Everything -- CSS included -- is
inlined, so the output is a single file that opens anywhere with no network.

Supports what these documents actually use: ATX headings, pipe tables with
alignment, fenced and inline code, blockquotes, ordered/unordered lists, bold,
italic, links, horizontal rules, and the emoji markers used as section flags.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b66;--rule:#e3e2dd;--accent:#7a4a2f;
--code-bg:#f2f1ed;--th:#f6f5f2;--quote:#f7f6f2;}
@media (prefers-color-scheme:dark){:root{--bg:#17171a;--fg:#e6e5e1;--muted:#9a998f;
--rule:#33333a;--accent:#d99a6c;--code-bg:#212127;--th:#1f1f25;--quote:#1d1d23;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
.wrap{max-width:60rem;margin:0 auto;padding:3rem 1.5rem 6rem;}
h1,h2,h3,h4{line-height:1.25;text-wrap:balance;margin:2.2em 0 .6em;}
h1{font-size:2rem;letter-spacing:-.02em;border-bottom:2px solid var(--rule);padding-bottom:.4em;}
h2{font-size:1.45rem;letter-spacing:-.01em;border-bottom:1px solid var(--rule);padding-bottom:.3em;}
h3{font-size:1.15rem;} h4{font-size:1rem;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em;}
h1:first-child{margin-top:0}
p{margin:.9em 0}
a{color:var(--accent)}
code{background:var(--code-bg);padding:.12em .35em;border-radius:4px;
font:.88em/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;}
pre{background:var(--code-bg);padding:1rem 1.1rem;border-radius:8px;overflow-x:auto;
border:1px solid var(--rule);}
pre code{background:none;padding:0;font-size:.85rem;line-height:1.55;}
blockquote{margin:1.2em 0;padding:.8em 1.1em;background:var(--quote);
border-left:3px solid var(--accent);border-radius:0 6px 6px 0;}
blockquote p:first-child{margin-top:0}blockquote p:last-child{margin-bottom:0}
.tw{overflow-x:auto;margin:1.3em 0;border:1px solid var(--rule);border-radius:8px;}
table{border-collapse:collapse;width:100%;font-size:.92rem;}
th,td{padding:.55em .8em;border-bottom:1px solid var(--rule);text-align:left;
vertical-align:top;}
th{background:var(--th);font-weight:600;white-space:nowrap;}
tr:last-child td{border-bottom:none}
td code{white-space:nowrap}
hr{border:none;border-top:2px solid var(--rule);margin:2.5em 0}
ul,ol{padding-left:1.5em;margin:.9em 0}
li{margin:.3em 0}
strong{font-weight:650}
.toc{background:var(--th);border:1px solid var(--rule);border-radius:8px;
padding:1rem 1.4rem;margin:2rem 0;}
.toc h4{margin:.2em 0 .6em}
.toc a{text-decoration:none;display:block;padding:.15em 0;font-size:.93rem}
.toc a:hover{text-decoration:underline}
.toc .lvl2{padding-left:1rem}
"""

INLINE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{html.escape(m.group(1))}</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
]


def inline(text: str) -> str:
    # Code spans are escaped inside their own handler; escape everything else
    # first, then re-apply markup so tags survive.
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_]+", "-", s)


def convert(md: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md.split("\n")
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]

        if ln.startswith("```"):                      # fenced code
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", ln)        # heading
        if m:
            lvl, txt = len(m.group(1)), m.group(2).rstrip("#").strip()
            sid = slug(txt)
            if lvl <= 2:
                toc.append((lvl, txt, sid))
            out.append(f'<h{lvl} id="{sid}">{inline(txt)}</h{lvl}>')
            i += 1
            continue

        if re.match(r"^\s*\|.*\|\s*$", ln) and i + 1 < n \
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []                                  # pipe table
            while i < n and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, body = rows[0], rows[2:]
            t = ["<div class='tw'><table><thead><tr>"]
            t += [f"<th>{inline(c)}</th>" for c in header]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue

        if re.match(r"^\s*[-*]{3,}\s*$", ln):
            out.append("<hr>"); i += 1; continue

        if ln.startswith(">"):                         # blockquote
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip()); i += 1
            inner, para = [], []
            for b in buf:
                if not b:
                    if para:
                        inner.append(f"<p>{inline(' '.join(para))}</p>"); para = []
                else:
                    para.append(b)
            if para:
                inner.append(f"<p>{inline(' '.join(para))}</p>")
            out.append(f"<blockquote>{''.join(inner)}</blockquote>")
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s+", ln):      # list
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            items = []
            while i < n and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i])); i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            continue

        if not ln.strip():
            i += 1; continue

        para = []                                      # paragraph
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|```|>|\s*\|)", lines[i]) and not re.match(
                r"^\s*([-*+]|\d+\.)\s+", lines[i]) and not re.match(
                r"^\s*[-*]{3,}\s*$", lines[i]):
            para.append(lines[i].strip()); i += 1
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out), toc


def main(src: str, dst: str) -> None:
    md = Path(src).read_text()
    body, toc = convert(md)
    title = md.split("\n")[0].lstrip("# ").strip() or Path(src).stem
    nav = "".join(
        f'<a class="lvl{l}" href="#{s}">{html.escape(t)}</a>' for l, t, s in toc)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><div class='wrap'>"
           f"<div class='toc'><h4>Contents</h4>{nav}</div>{body}</div></body></html>")
    Path(dst).write_text(doc)
    print(f"{len(md):,} chars markdown -> {len(doc):,} chars self-contained HTML")
    print(f"  {len(toc)} sections in contents")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
