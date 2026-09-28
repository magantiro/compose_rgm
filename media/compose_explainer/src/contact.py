#!/usr/bin/env python3
"""Build contact sheets from rendered frames, for whole-film self-review.

Usage: contact.py [n_per_sheet] [n_sheets]
Writes build/review/sheet_NN.png -- tiled grids sampled evenly across the film,
so layout problems that only appear mid-animation are visible in one look.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import imageio_ffmpeg

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRAMES = ROOT / "build" / "frames"
OUT = ROOT / "build" / "review"


def main() -> None:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_sheets = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(FRAMES.glob("f*.png"))
    if not files:
        sys.exit("no frames in build/frames")

    total = per * n_sheets
    step = max(1, len(files) // total)
    picks = [files[min(i * step, len(files) - 1)] for i in range(total)]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cols = 4
    rows = max(1, per // cols)

    for s in range(n_sheets):
        chunk = picks[s * per:(s + 1) * per]
        if not chunk:
            break
        lst = OUT / f"_list{s}.txt"
        lst.write_text("".join(f"file '{p}'\nduration 1\n" for p in chunk))
        dest = OUT / f"sheet_{s:02d}.png"
        subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-vf", f"scale=620:349,tile={cols}x{rows}:margin=10:padding=8:color=0x666666",
             "-frames:v", "1", str(dest)],
            check=True, capture_output=True,
        )
        lst.unlink()
        print(f"wrote {dest.name}: {chunk[0].stem} .. {chunk[-1].stem}")


if __name__ == "__main__":
    main()
