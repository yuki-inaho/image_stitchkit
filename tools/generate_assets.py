#!/usr/bin/env python3
"""Regenerate deterministic synthetic RGB assets. Python is not a runtime dependency.
Requirements: Pillow and NumPy. Committed PNGs allow CTest without either package.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "tests" / "assets" / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    width, height = 760, 280
    noise = rng.normal(0, 4, (height, width, 1))
    base = np.clip(np.full((height, width, 3), [180, 195, 210]) + noise, 0, 255).astype(np.uint8)
    scene = Image.fromarray(base)
    draw = ImageDraw.Draw(scene)
    for _ in range(450):
        x, y = int(rng.integers(8, width-28)), int(rng.integers(8, height-28))
        size = int(rng.integers(3, 17))
        color = tuple(int(c) for c in rng.integers(15, 240, 3))
        if rng.random() < .5:
            draw.ellipse((x, y, x+size, y+size), fill=color, outline=(20, 30, 40), width=1)
        else:
            draw.rectangle((x, y, x+size, y+size), fill=color, outline=(25, 35, 45), width=1)
    for y in (38, 235):
        draw.line((12, y, width-12, y), fill=(25, 25, 25), width=3)
    for x in (80, 260, 450, 660):
        draw.line((x, 18, x, 260), fill=(35, 35, 35), width=3)
    draw.arc((220, 58, 380, 210), 5, 335, fill=(250, 245, 220), width=5)
    draw.arc((440, 64, 610, 218), 20, 350, fill=(15, 50, 20), width=4)
    draw.rectangle((5, 105, 24, 175), fill=(0, 0, 0))  # Valid black is not transparency.
    scene.save(root / "scene.png")
    for i, offset in enumerate((0, 180, 360)):
        scene.crop((offset, 0, offset+400, height)).save(root / f"view{i}.png")
    scene.crop((0, 0, 580, height)).save(root / "pair_truth.png")
    yy, xx = np.mgrid[:height, :400].astype(np.float64)
    bump = np.exp(-((xx-130)/70)**2-((yy-142)/65)**2)
    map_x, map_y = xx+180+10*bump, yy+3*bump
    source = np.asarray(scene).astype(np.float64)
    x0, y0 = np.floor(map_x).astype(int), np.floor(map_y).astype(int)
    x0, y0 = np.clip(x0, 0, width-2), np.clip(y0, 0, height-2)
    fx, fy = (map_x-x0)[..., None], np.clip(map_y-y0, 0, 1)[..., None]
    warped = (source[y0, x0]*(1-fx)*(1-fy)+source[y0, x0+1]*fx*(1-fy)
              +source[y0+1, x0]*(1-fx)*fy+source[y0+1, x0+1]*fx*fy)
    Image.fromarray(np.clip(np.rint(warped), 0, 255).astype(np.uint8)).save(root / "parallax1.png")
    Image.new("RGB", (160, 120), (30, 30, 30)).save(root / "blank.png")
    metadata = {"generator_seed": 42, "image_size": [400, 280], "offsets_x": [0, 180, 360],
                "parallax": {"mapping": "source(u,v) samples scene(u+180+10*b,v+3*b)",
                             "b": "exp(-((u-130)/70)^2-((v-142)/65)^2)"}}
    (root / "geometry.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
