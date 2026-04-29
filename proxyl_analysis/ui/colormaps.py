"""
Custom matplotlib colormaps used in the parameter map viewer.

Two LUTs live here:

- ``imagej_16_colors`` — discrete 16-step palette modeled on ImageJ's
  ``16_colors.lut`` (black → blues → cyan → green → yellow → orange → reds →
  magentas → white). Used for sequential parameter maps where the value is
  always non-negative, e.g. %Enhancement (A1/A0).
- ``nte_diverging`` — continuous diverging palette centered on black at zero,
  with blue → cyan → green for the negative side and orange → red → magenta
  for the positive side. Used for %NTE (A2/A0) which can swing either way.

Both colormaps are registered with matplotlib at import time so callers can
pass either the colormap object or its name (``"imagej_16_colors"`` /
``"nte_diverging"``) to ``imshow``.
"""

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl


# ---------------------------------------------------------------------------
# ImageJ "16_colors.lut" — 16 discrete colors.
# RGB values are based on the canonical ImageJ palette:
#   black, dark blue, blue, light blue, cyan, green, yellow-green, yellow,
#   gold, orange, red, dark red, pink, magenta, purple, white.
# ---------------------------------------------------------------------------
_IMAGEJ_16_COLORS_RGB255 = [
    (0,   0,   0),      # 0  black
    (1,   1,   171),    # 1  dark blue
    (1,   1,   224),    # 2  blue
    (0,   110, 255),    # 3  light blue
    (1,   171, 254),    # 4  cyan
    (1,   255, 1),      # 5  green
    (190, 255, 0),      # 6  yellow-green
    (255, 255, 0),      # 7  yellow
    (255, 224, 0),      # 8  gold
    (255, 141, 0),      # 9  orange
    (255, 0,   0),      # 10 red
    (224, 0,   0),      # 11 dark red
    (255, 0,   191),    # 12 pink
    (255, 0,   224),    # 13 magenta
    (224, 0,   224),    # 14 purple
    (255, 255, 255),    # 15 white
]

imagej_16_colors = ListedColormap(
    [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in _IMAGEJ_16_COLORS_RGB255],
    name="imagej_16_colors",
)


# ---------------------------------------------------------------------------
# %NTE diverging colormap — black at zero, cool greens/blues for negative,
# warm reds/magentas for positive. Continuous gradient so the transition
# through zero is smooth even at narrow data ranges.
# Key colors at evenly-spaced stops on [0, 1]; 0.5 is zero (black).
# ---------------------------------------------------------------------------
_NTE_DIVERGING_STOPS = [
    (0.000, (0.000, 0.863, 0.000)),   # bright green   (most negative)
    (1 / 6, (0.000, 0.863, 0.863)),   # cyan
    (2 / 6, (0.000, 0.314, 0.863)),   # blue
    (0.500, (0.000, 0.000, 0.000)),   # black          (zero)
    (4 / 6, (1.000, 0.549, 0.000)),   # orange
    (5 / 6, (0.863, 0.000, 0.118)),   # red
    (1.000, (1.000, 0.000, 1.000)),   # bright magenta (most positive)
]

nte_diverging = LinearSegmentedColormap.from_list(
    "nte_diverging",
    _NTE_DIVERGING_STOPS,
    N=256,
)


# Register the colormaps with matplotlib so callers can refer to them by
# name (e.g. ``cmap="imagej_16_colors"``). Skip silently if a re-import
# tries to register the same name twice.
def _register(cmap):
    try:
        # matplotlib 3.5+ API
        mpl.colormaps.register(cmap=cmap, name=cmap.name)
    except (AttributeError, ValueError):
        # Older matplotlib or already-registered: best-effort fallback.
        try:
            mpl.cm.register_cmap(cmap=cmap)
        except (AttributeError, ValueError):
            pass


_register(imagej_16_colors)
_register(nte_diverging)


__all__ = ["imagej_16_colors", "nte_diverging"]
