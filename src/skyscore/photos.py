"""Prepare observing-night photos for the app: fix rotation, shrink, and drop all metadata (GPS included).

Usage (from the project folder):
    python -m skyscore.photos <folder with the original photos> <event_id>

Writes photos/<event_id>/01.jpg, 02.jpg, ... and keeps numbering after photos already there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 1600  # px; ~300-500 KB as JPEG


def prepare(src: Path, dest: Path, max_side: int = MAX_SIDE, quality: int = 82) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    taken = [int(p.stem) for p in dest.glob("*.jpg") if p.stem.isdigit()]
    n = max(taken, default=0)
    out = []
    for f in sorted(p for p in Path(src).iterdir() if p.suffix.lower() in EXTS):
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        n += 1
        target = dest / f"{n:02d}.jpg"
        # No exif=/icc_profile= passed, so nothing from the original file (GPS, camera, time) is written.
        im.save(target, "JPEG", quality=quality, optimize=True)
        out.append(target)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="folder with the original photos")
    p.add_argument("event_id", help="e.g. E01, as in data/events.csv")
    p.add_argument("--out", default="photos", help="photos root folder (default: photos)")
    a = p.parse_args(argv)
    for f in prepare(Path(a.src), Path(a.out) / a.event_id):
        print(f"{f}  {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
