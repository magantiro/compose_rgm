#!/usr/bin/env python3
"""Render the COMPOSE explainer frame by frame with headless Chromium.

The page exposes window.COMPOSE.renderFrame(i); frames are pure functions of
their index, so the render is deterministic and restartable.

Usage
  render.py probe                 # print timeline info
  render.py sheet  T1 T2 ...      # contact sheet at the given times (seconds)
  render.py frames [--from N] [--to N]   # write build/frames/f%06d.png
  render.py video                 # encode build/frames -> build/compose.mp4
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = ROOT / "build"
FRAMES = BUILD / "frames"
PAGE = (HERE / "index.html").as_uri()


def _browser():
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    br = pw.chromium.launch(args=[
        "--force-device-scale-factor=1",
        "--hide-scrollbars",
        "--disable-lcd-text",
        "--font-render-hinting=none",
        "--force-color-profile=srgb",
        "--disable-gpu-vsync",
    ])
    pg = br.new_page(viewport={"width": 1920, "height": 1080},
                     device_scale_factor=1)
    pg.on("console", lambda m: print(f"[console.{m.type}] {m.text}", flush=True)
          if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: print(f"[pageerror] {e}", flush=True))
    pg.goto(PAGE)
    pg.wait_for_function("window.COMPOSE && window.COMPOSE.ready", timeout=30000)
    return pw, br, pg


def _report_errors(pg) -> int:
    """Scene exceptions are recorded, not swallowed; surface them loudly."""
    errs = pg.evaluate("() => COMPOSE.errors")
    for e in errs:
        print(f"[scene-error] {e}", flush=True)
    return len(errs)


def probe() -> None:
    pw, br, pg = _browser()
    info = pg.evaluate("() => ({n: COMPOSE.nframes, fps: COMPOSE.fps, "
                       "total: COMPOSE.total, scenes: COMPOSE.scenes})")
    print(f"fps={info['fps']}  frames={info['n']}  total={info['total']:.2f}s")
    for s in info["scenes"]:
        print(f"  {s['name']:16s} t0={s['t0']:6.2f}  dur={s['dur']:5.2f}")
    br.close(); pw.stop()


def sheet(times: list[float]) -> None:
    """Write one full-size PNG per requested time, for self-review."""
    out = BUILD / "review"
    out.mkdir(parents=True, exist_ok=True)
    pw, br, pg = _browser()
    fps = pg.evaluate("() => COMPOSE.fps")
    for t in times:
        i = int(round(t * fps))
        pg.evaluate(f"() => COMPOSE.renderFrame({i})")
        p = out / f"t{t:07.3f}.png"
        pg.screenshot(path=str(p), animations="disabled")
        print(f"wrote {p.name}")
    n = _report_errors(pg)
    br.close(); pw.stop()
    if n:
        sys.exit(f"{n} scene error(s) -- fix before reviewing frames")


def frames(lo: int | None, hi: int | None) -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    pw, br, pg = _browser()
    n = pg.evaluate("() => COMPOSE.nframes")
    a = 0 if lo is None else lo
    b = n if hi is None else min(hi, n)
    print(f"rendering frames [{a}, {b}) of {n}", flush=True)
    for i in range(a, b):
        pg.evaluate(f"() => COMPOSE.renderFrame({i})")
        pg.screenshot(path=str(FRAMES / f"f{i:06d}.png"), animations="disabled")
        if (i - a) % 240 == 0:
            print(f"  {i}/{b}", flush=True)
    n = _report_errors(pg)
    br.close(); pw.stop()
    if n:
        sys.exit(f"{n} scene error(s) -- render is not clean")
    print("done", flush=True)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video() -> None:
    n = len(list(FRAMES.glob("f*.png")))
    if n == 0:
        sys.exit("no frames in build/frames")
    out = BUILD / "compose_explainer.mp4"
    cmd = [
        ffmpeg_exe(), "-y",
        "-framerate", "60",
        "-i", str(FRAMES / "f%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1920:1080:flags=lanczos",
        "-movflags", "+faststart",
        "-r", "60",
        str(out),
    ]
    print(f"encoding {n} frames -> {out}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "probe":
        probe()
    elif cmd == "sheet":
        sheet([float(x) for x in sys.argv[2:]])
    elif cmd == "frames":
        lo = hi = None
        if "--from" in sys.argv:
            lo = int(sys.argv[sys.argv.index("--from") + 1])
        if "--to" in sys.argv:
            hi = int(sys.argv[sys.argv.index("--to") + 1])
        frames(lo, hi)
    elif cmd == "video":
        video()
    else:
        sys.exit(__doc__)
