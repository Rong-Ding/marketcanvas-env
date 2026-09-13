"""One rasterization path for rendering and visibility/contrast measurements.

Opaque boxes, single-line text, fixed font and BASIC layout engine. No external assets.
Visibility is measured on glyph pixels; a button's label is intentionally inside its fill.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .models import State, ordered

FONT = Path(__file__).parent / "assets" / "DejaVuSans.ttf"


@lru_cache(maxsize=89)
def font(size):
    return ImageFont.truetype(str(FONT), size, layout_engine=ImageFont.Layout.BASIC)


def rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


def luminance(colors):
    c = np.asarray(colors, dtype=float) / 255
    linear = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    return linear @ np.array([.2126, .7152, .0722])


def contrast_ratio(first, second):
    a, b = float(luminance(rgb(first))), float(luminance(rgb(second)))
    return (max(a, b) + .05) / (min(a, b) + .05)


def render_scene(state: State, measure=True):
    canvas = Image.new("RGB", (state.width, state.height), state.background)
    layers, metrics = [], {}
    for e in ordered(state):
        opaque = np.zeros((state.height, state.width), dtype=bool)
        if e.color:
            ImageDraw.Draw(canvas).rectangle((e.x, e.y, e.x+e.width-1, e.y+e.height-1), fill=e.color)
            opaque[e.y:e.y+e.height, e.x:e.x+e.width] = True
        glyph = np.zeros_like(opaque)
        if e.content and e.type != "image":
            f = font(e.font_size)
            box = f.getbbox(e.content)
            tw, th = box[2]-box[0], box[3]-box[1]
            mask = Image.new("L", (e.width, e.height))
            ImageDraw.Draw(mask).text(((e.width-tw)//2-box[0], (e.height-th)//2-box[1]), e.content, font=f, fill=255)
            local = np.asarray(mask) > 0
            glyph[e.y:e.y+e.height, e.x:e.x+e.width] = local
            min_ratio = 0.0
            if measure and local.any():
                backgrounds = np.asarray(canvas)[glyph]
                lum = luminance(backgrounds)
                foreground = float(luminance(rgb(e.text_color)))
                ratios = (np.maximum(lum, foreground)+.05)/(np.minimum(lum, foreground)+.05)
                min_ratio = float(ratios.min())
            canvas.paste(Image.new("RGB", (e.width, e.height), e.text_color), (e.x, e.y), mask)
            metrics[e.id] = {"fits": tw <= e.width-8 and th <= e.height-8,
                             "contrast": min_ratio, "font_size": e.font_size,
                             "visible_fraction": 0.0}
            opaque |= glyph
        layers.append((e.id, opaque, glyph))
    if measure:
        covered = np.zeros((state.height, state.width), dtype=bool)
        for eid, opaque, glyph in reversed(layers):
            if eid in metrics and glyph.any():
                metrics[eid]["visible_fraction"] = float((glyph & ~covered).sum() / glyph.sum())
            covered |= opaque
    return canvas, metrics
