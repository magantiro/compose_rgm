"""Render every figure. `python figures/build.py` -> figures/out/*.pdf"""
import pathlib, subprocess, sys
here = pathlib.Path(__file__).parent
out = here/"out"; out.mkdir(exist_ok=True)
for s in sorted(here.glob("fig*_*.py")):
    print(f"  {s.name}")
    subprocess.run([sys.executable, str(s), str(out)], check=True)
