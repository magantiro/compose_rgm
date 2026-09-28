#!/usr/bin/env python3
"""Verify the encoded MP4: report container properties and extract frames
from the video itself, so the final check is on the delivered artifact rather
than on the PNGs that fed the encoder.

Usage: verify.py [frame_index ...]
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import imageio_ffmpeg

ROOT = pathlib.Path(__file__).resolve().parent.parent
MP4 = ROOT / "build" / "compose_explainer.mp4"
OUT = ROOT / "build" / "verify"


def main() -> None:
    if not MP4.exists():
        sys.exit(f"missing {MP4}")
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    info = subprocess.run([ff, "-i", str(MP4)], capture_output=True, text=True).stderr
    for line in info.splitlines():
        if "Duration" in line or "Stream" in line:
            print(line.strip())

    idx = [int(x) for x in sys.argv[1:]] or [300, 1560, 2020, 2700, 3120, 4200]
    OUT.mkdir(parents=True, exist_ok=True)
    for i in idx:
        dest = OUT / f"v{i:06d}.png"
        subprocess.run(
            [ff, "-y", "-v", "error", "-i", str(MP4),
             "-vf", f"select=eq(n\\,{i})", "-vsync", "0", "-frames:v", "1", str(dest)],
            check=True,
        )
        print(f"extracted frame {i} -> {dest.name}")
    print(f"\nsize: {MP4.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
