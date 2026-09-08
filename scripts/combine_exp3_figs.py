#!/usr/bin/env python3
"""Stitch Exp3 hard/easy + strategy PNGs into a wide combined panel."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP3 = ROOT / "outputs" / "experiments" / "exp3_difficulty"

A = EXP3 / "exp3_difficulty_all.png"
B = EXP3 / "exp3_difficulty_all_strategy.png"
OUT = EXP3 / "exp3_combined_hard_easy_and_strategy.png"

GAP_PX = 60
LABEL_PAD = 15
LABEL_FONT_SIZE = 120
LABEL_H = 110      # Minimal header space to maximize subplot area
FIG_SCALE = 2.0


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, LABEL_FONT_SIZE)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(0, 0, 0, 255), font=font)


def main() -> None:
    img_a = Image.open(A).convert("RGBA")
    img_b = Image.open(B).convert("RGBA")

    img_a = img_a.resize((int(img_a.width * FIG_SCALE), int(img_a.height * FIG_SCALE)), Image.Resampling.LANCZOS)
    img_b = img_b.resize((int(img_b.width * FIG_SCALE), int(img_b.height * FIG_SCALE)), Image.Resampling.LANCZOS)

    wa, ha = img_a.size
    wb, hb = img_b.size
    h = max(ha, hb) + LABEL_H
    w = wa + GAP_PX + wb

    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    canvas.paste(img_a, (0, LABEL_H))
    canvas.paste(img_b, (wa + GAP_PX, LABEL_H))

    draw = ImageDraw.Draw(canvas)
    _label(draw, (LABEL_PAD, LABEL_PAD), "(a)")
    _label(draw, (wa + GAP_PX + LABEL_PAD, LABEL_PAD), "(b)")

    canvas.save(OUT)
    print(f"Wrote wide stretched panel: {OUT} ({w}x{h})")


if __name__ == "__main__":
    main()