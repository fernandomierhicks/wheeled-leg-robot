#!/usr/bin/env python3
"""Render the RoboBlue theme images and the model bitmap.

Everything the radio displays as a picture is generated here rather than
hand-drawn, so the palette can only ever be changed in one place: PALETTE
below, which is the same set of values as THEMES/RoboBlue/theme.yml.

Outputs (all under radio/sdcard/):
    THEMES/RoboBlue/background_480x320.png   <- the TX15's actual size
    THEMES/RoboBlue/background_480x272.png   <- for portability to a TX16S etc
    THEMES/RoboBlue/logo.png                 <- theme-browser banner, 480x272
    THEMES/RoboBlue/screenshot1..3.png       <- theme-browser previews, 480x272
    IMAGES/wlrrobot.png                      <- model bitmap, 192x114

Usage:
    python make_assets.py
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
SD = REPO / "radio" / "sdcard"
THEME = SD / "THEMES" / "RoboBlue"
IMAGES = SD / "IMAGES"

# Mirrors THEMES/RoboBlue/theme.yml. Keep the two in step.
PALETTE = {
    "bg":       (0x07, 0x0A, 0x0F),
    "panel":    (0x12, 0x18, 0x21),
    "border":   (0x39, 0x43, 0x4F),
    "text":     (0xE2, 0xEC, 0xF7),
    "dim":      (0x8F, 0xA3, 0xB8),
    "accent":   (0x00, 0xA8, 0xFF),
    "cyan":     (0x00, 0xE5, 0xFF),
    "focus":    (0x00, 0x78, 0xD7),
    "warn":     (0xFF, 0x3B, 0x30),
    "ok":       (0x2E, 0xE6, 0x8A),
    "amber":    (0xFF, 0xB4, 0x00),
    "disabled": (0x4A, 0x5A, 0x6B),
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/bahnschrift.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def font(size, bold=True):
    order = FONT_CANDIDATES if bold else FONT_CANDIDATES[1:] + FONT_CANDIDATES[:1]
    for path in order:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def blend(c, other, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c, other))


# -- background ---------------------------------------------------------------

def technical_grid(w, h):
    """Deep slate ground with a faint engineering grid and one radar sweep.

    Deliberately low-contrast: this sits behind the whole UI, and anything
    bright enough to notice is bright enough to fight the text on top of it.
    """
    img = Image.new("RGB", (w, h), PALETTE["bg"])
    d = ImageDraw.Draw(img)

    # Vertical gradient, darkest at the top where the header bar sits.
    top, bottom = PALETTE["bg"], (0x0C, 0x12, 0x1B)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=blend(top, bottom, y / max(1, h - 1)))

    # Grid. Minor every 24 px, major every 96 px.
    minor = blend(PALETTE["bg"], PALETTE["border"], 0.22)
    major = blend(PALETTE["bg"], PALETTE["border"], 0.42)
    for x in range(0, w, 24):
        d.line([(x, 0), (x, h)], fill=major if x % 96 == 0 else minor)
    for y in range(0, h, 24):
        d.line([(0, y), (w, y)], fill=major if y % 96 == 0 else minor)

    # Concentric arcs anchored off the bottom-right corner -- reads as an
    # instrument bezel without occupying any usable screen area.
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = int(w * 0.86), int(h * 1.02)
    for i, r in enumerate((70, 118, 168, 220, 276)):
        shade = blend((0, 0, 0), PALETTE["accent"], 0.30 - i * 0.045)
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=shade, width=2)
    for ang in range(0, 360, 15):
        a = math.radians(ang)
        gd.line(
            [(cx + 70 * math.cos(a), cy + 70 * math.sin(a)),
             (cx + 276 * math.cos(a), cy + 276 * math.sin(a))],
            fill=blend((0, 0, 0), PALETTE["accent"], 0.06),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(1.2))
    img = Image.blend(img, Image.new("RGB", (w, h), (0, 0, 0)), 0.0)
    img = Image.composite(
        Image.blend(img, blend_image(img, glow), 1.0), img,
        Image.new("L", (w, h), 255),
    )

    # A couple of circuit traces on the left, stepped at 45 degrees.
    d = ImageDraw.Draw(img)
    trace = blend(PALETTE["bg"], PALETTE["accent"], 0.18)
    for y0 in (int(h * 0.24), int(h * 0.55), int(h * 0.80)):
        pts = [(0, y0), (46, y0), (74, y0 - 28), (132, y0 - 28), (158, y0 - 2),
               (206, y0 - 2)]
        d.line(pts, fill=trace, width=2)
        d.ellipse([203, y0 - 5, 209, y0 + 1], fill=blend(PALETTE["bg"],
                                                         PALETTE["cyan"], 0.5))
    return img


def blend_image(base, glow):
    """Additive screen blend of `glow` onto `base`, clipped."""
    out = Image.new("RGB", base.size)
    bp, gp, op = base.load(), glow.load(), out.load()
    for y in range(base.size[1]):
        for x in range(base.size[0]):
            b, g = bp[x, y], gp[x, y]
            op[x, y] = (min(255, b[0] + g[0]), min(255, b[1] + g[1]),
                        min(255, b[2] + g[2]))
    return out


# -- shared drawing helpers ---------------------------------------------------

def panel(d, box, fill=None, outline=None, radius=6, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill or PALETTE["panel"],
                        outline=outline or PALETTE["border"], width=width)


def robot_glyph(d, cx, cy, scale=1.0, colour=None):
    """A minimal two-wheel / four-bar-leg silhouette of the actual robot.

    Body box on top, a bent leg down each side, a wheel at each foot -- the
    same topology as the CLAUDE.md ASCII diagram, so the icon is recognisably
    THIS robot rather than a generic droid.
    """
    c = colour or PALETTE["cyan"]
    s = scale

    def P(x, y):
        return (cx + x * s, cy + y * s)

    # Body
    d.rounded_rectangle([P(-30, -46), P(30, -14)], radius=int(6 * s),
                        outline=c, width=max(2, int(3 * s)))
    # Sensor bar
    d.line([P(-20, -36), P(20, -36)], fill=c, width=max(1, int(2 * s)))
    d.ellipse([P(-9, -33), P(-3, -27)], fill=c)
    d.ellipse([P(3, -33), P(9, -27)], fill=c)
    # Legs: hip -> knee -> wheel centre, kinked the way the tibia actually is
    for sx in (-1, 1):
        d.line([P(sx * 22, -16), P(sx * 30, 10)], fill=c, width=max(2, int(3 * s)))
        d.line([P(sx * 30, 10), P(sx * 24, 30)], fill=c, width=max(2, int(3 * s)))
        d.ellipse([P(sx * 24 - 14, 30 - 14), P(sx * 24 + 14, 30 + 14)],
                  outline=c, width=max(2, int(3 * s)))
        d.ellipse([P(sx * 24 - 3, 30 - 3), P(sx * 24 + 3, 30 + 3)], fill=c)


def header(d, w, title, subtitle=None):
    d.rectangle([0, 0, w, 34], fill=PALETTE["panel"])
    d.line([(0, 34), (w, 34)], fill=PALETTE["accent"], width=2)
    d.text((12, 8), title, font=font(18), fill=PALETTE["text"])
    if subtitle:
        d.text((w - 12, 11), subtitle, font=font(13), fill=PALETTE["dim"],
               anchor="ra")


def gauge(d, cx, cy, r, frac, colour, label, value):
    """270-degree arc gauge, opening at the bottom."""
    start, sweep = 135, 270
    d.arc([cx - r, cy - r, cx + r, cy + r], start, start + sweep,
          fill=PALETTE["border"], width=8)
    if frac > 0:
        d.arc([cx - r, cy - r, cx + r, cy + r], start,
              start + sweep * max(0.0, min(1.0, frac)), fill=colour, width=8)
    d.text((cx, cy - 6), value, font=font(20), fill=PALETTE["text"],
           anchor="mm")
    d.text((cx, cy + 14), label, font=font(11), fill=PALETTE["dim"],
           anchor="mm")


def hbar(d, x, y, w, h, frac, colour, label, value):
    d.text((x, y - 14), label, font=font(12), fill=PALETTE["dim"])
    d.text((x + w, y - 14), value, font=font(12), fill=PALETTE["text"],
           anchor="ra")
    d.rounded_rectangle([x, y, x + w, y + h], radius=3,
                        fill=(0x0D, 0x12, 0x1A), outline=PALETTE["border"])
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw > 2:
        d.rounded_rectangle([x, y, x + fw, y + h], radius=3, fill=colour)


def attitude(d, cx, cy, r, pitch_deg, roll_deg):
    """Artificial-horizon style pitch/roll ball, robot-referenced."""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0x0A, 0x10, 0x18),
              outline=PALETTE["border"], width=2)
    a = math.radians(roll_deg)
    off = pitch_deg * (r / 30.0)
    dx, dy = math.cos(a), math.sin(a)
    px, py = cx - dy * off, cy + dx * off
    d.line([(px - dx * r * 0.9, py - dy * r * 0.9),
            (px + dx * r * 0.9, py + dy * r * 0.9)],
           fill=PALETTE["cyan"], width=3)
    for k in (-2, -1, 1, 2):
        t = k * (r / 3.2)
        lx, ly = px - dy * t, py + dx * t
        ln = r * (0.30 if abs(k) == 1 else 0.18)
        d.line([(lx - dx * ln, ly - dy * ln), (lx + dx * ln, ly + dy * ln)],
               fill=PALETTE["disabled"], width=1)
    d.line([(cx - 12, cy), (cx - 4, cy)], fill=PALETTE["amber"], width=2)
    d.line([(cx + 4, cy), (cx + 12, cy)], fill=PALETTE["amber"], width=2)
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=PALETTE["amber"])


def chip(d, x, y, text, colour, w=None):
    tw = w or (len(text) * 9 + 18)
    d.rounded_rectangle([x, y, x + tw, y + 24], radius=12,
                        fill=blend(PALETTE["bg"], colour, 0.22), outline=colour)
    d.text((x + tw / 2, y + 12), text, font=font(13), fill=colour, anchor="mm")
    return tw


# -- the individual images ----------------------------------------------------

def make_background(w, h):
    return technical_grid(w, h)


def make_logo():
    w, h = 480, 272
    img = technical_grid(w, h)
    d = ImageDraw.Draw(img)
    robot_glyph(d, 118, 128, scale=1.5, colour=PALETTE["cyan"])
    d.text((216, 96), "ROBOBLUE", font=font(40), fill=PALETTE["text"])
    d.text((218, 142), "WHEELED-LEG ROBOT", font=font(15),
           fill=PALETTE["accent"])
    d.line([(218, 166), (452, 166)], fill=PALETTE["border"], width=1)
    d.text((218, 176), "DARK INSTRUMENT THEME", font=font(12),
           fill=PALETTE["dim"])
    return img


def make_screenshot_hud():
    """Preview 1: the WLRHUD widget full screen -- the screen you leave open."""
    w, h = 480, 272
    img = technical_grid(w, h)
    d = ImageDraw.Draw(img)
    header(d, w, "WLR HUD", "LQ 98  -71dBm")

    panel(d, [8, 42, 200, 128])
    chip(d, 18, 52, "RUNNING", PALETTE["ok"], w=104)
    d.text((18, 86), "alpha", font=font(12), fill=PALETTE["dim"])
    d.text((190, 84), "0.42", font=font(17), fill=PALETTE["text"], anchor="ra")
    hbar(d, 18, 108, 172, 8, 0.42, PALETTE["accent"], "", "")

    attitude(d, 288, 86, 40, -6.0, 3.0)
    d.text((288, 134), "PITCH -6.1  ROLL 3.0", font=font(11),
           fill=PALETTE["dim"], anchor="mm")

    gauge(d, 420, 86, 36, 0.78, PALETTE["ok"], "PACK", "24.1")

    panel(d, [8, 138, 472, 232])
    hbar(d, 20, 166, 200, 10, 0.55, PALETTE["accent"], "WHEEL VEL", "0.55 m/s")
    hbar(d, 20, 200, 200, 10, 0.38, PALETTE["cyan"], "HIP TORQUE L", "2.7 Nm")
    hbar(d, 256, 166, 200, 10, 0.41, PALETTE["cyan"], "HIP TORQUE R", "2.9 Nm")
    hbar(d, 256, 200, 200, 10, 0.12, PALETTE["ok"], "LEG HEIGHT", "0.12")

    d.text((12, 244), "ESP32 LINK OK   DROPS 0   GAPS 0", font=font(12),
           fill=PALETTE["dim"])
    chip(d, 386, 240, "ARMED", PALETTE["amber"], w=82)
    return img


def make_screenshot_fault():
    """Preview 2: the fault banner -- what the theme looks like when it matters."""
    w, h = 480, 272
    img = technical_grid(w, h)
    d = ImageDraw.Draw(img)
    header(d, w, "WLR HUD", "LQ 96  -74dBm")

    panel(d, [8, 44, 472, 122], fill=(0x25, 0x0C, 0x0C),
          outline=PALETTE["warn"], width=2)
    chip(d, 20, 56, "ESTOP", PALETTE["warn"], w=92)
    d.text((124, 58), "PITCH WATCHDOG", font=font(20), fill=PALETTE["warn"])
    d.text((124, 84), "pitch outside the gain-scheduled band", font=font(12),
           fill=PALETTE["text"])
    d.text((124, 100), "for > 200 ms", font=font(12), fill=PALETTE["text"])
    chip(d, 20, 88, "REPOSITION", PALETTE["amber"], w=92)

    panel(d, [8, 132, 232, 232])
    d.text((20, 142), "RECOVERY", font=font(13), fill=PALETTE["accent"])
    for i, line in enumerate(("1  pick the robot up",
                              "2  level it, legs down",
                              "3  rescue combo to clear")):
        d.text((20, 164 + i * 22), line, font=font(12), fill=PALETTE["text"])

    panel(d, [240, 132, 472, 232])
    d.text((252, 142), "LAST KNOWN", font=font(13), fill=PALETTE["accent"])
    for i, (k, v) in enumerate((("pitch", "-41.2 deg"), ("alpha", "0.31"),
                                ("wheel", "1.84 m/s"))):
        d.text((252, 164 + i * 22), k, font=font(12), fill=PALETTE["dim"])
        d.text((460, 164 + i * 22), v, font=font(12), fill=PALETTE["text"],
               anchor="ra")
    return img


def make_screenshot_model():
    """Preview 3: model select, showing the theme's list/selection colours."""
    w, h = 480, 272
    img = technical_grid(w, h)
    d = ImageDraw.Draw(img)
    header(d, w, "Manage models", "1 model")

    d.rounded_rectangle([8, 46, 236, 152], radius=8,
                        fill=blend(PALETTE["bg"], PALETTE["focus"], 0.30),
                        outline=PALETTE["accent"], width=2)
    robot_glyph(d, 62, 96, scale=0.72, colour=PALETTE["cyan"])
    d.text((114, 66), "WLR ROBOT", font=font(16), fill=PALETTE["text"])
    d.text((114, 90), "CRSF 16CH", font=font(12), fill=PALETTE["dim"])
    d.text((114, 110), "checklist on", font=font(12), fill=PALETTE["accent"])

    for i, (x, y) in enumerate(((244, 46), (8, 160), (244, 160))):
        panel(d, [x, y, x + 228, y + 106])
        d.text((x + 14, y + 20), "-- empty --", font=font(13),
               fill=PALETTE["disabled"])
    return img


def make_model_bitmap():
    w, h = 192, 114
    img = technical_grid(w, h)
    d = ImageDraw.Draw(img)
    robot_glyph(d, 52, 54, scale=0.86, colour=PALETTE["cyan"])
    d.text((100, 34), "WLR", font=font(26), fill=PALETTE["text"])
    d.text((101, 64), "ROBOT", font=font(15), fill=PALETTE["accent"])
    d.line([(100, 84), (182, 84)], fill=PALETTE["border"])
    d.text((100, 88), "CRSF 16CH", font=font(10), fill=PALETTE["dim"])
    return img


def main():
    THEME.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    written = []
    for size in ((480, 320), (480, 272)):
        p = THEME / ("background_%dx%d.png" % size)
        make_background(*size).save(p)
        written.append(p)

    for maker, name in ((make_logo, "logo.png"),
                        (make_screenshot_hud, "screenshot1.png"),
                        (make_screenshot_fault, "screenshot2.png"),
                        (make_screenshot_model, "screenshot3.png")):
        p = THEME / name
        maker().save(p)
        written.append(p)

    p = IMAGES / "wlrrobot.png"
    make_model_bitmap().save(p)
    written.append(p)

    for p in written:
        print("wrote %s  (%d bytes)" % (p.relative_to(REPO), p.stat().st_size))


if __name__ == "__main__":
    sys.exit(main())
