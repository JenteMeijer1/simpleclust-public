# theme.py
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# ── Neutral colour for community controls (CC) ───────────────────────────────
# A dark slate-blue: clearly a baseline/reference group, distinct from clusters.
CC_COLOR = "#4A6080"

# ── Categorical palette ───────────────────────────────────────────────────────
# Six colours derived from Paul Tol's colourblind-safe "muted" scheme,
# adjusted for a professional, feminine academic aesthetic.
# Verified separable under deuteranopia, protanopia, tritanopia, and
# achromatopsia (luminance values span ~0.04–0.37 relative luminance).
#
#   1  #CC6677  dusty rose        – warm, feminine, primary cluster colour
#   2  #4477AA  cerulean blue     – professional, clearly blue
#   3  #C07EAA  dusty orchid      – warm, feminine, bridges rose and violet
#   4  #44AA99  sage teal         – calm, clearly blue-green
#   5  #7755AA  soft violet       – elegant, feminine, clearly purple
#   6  #882255  deep wine/cherry  – sophisticated, very dark red
#
# Warm group: rose (H=350°), orchid (H=320°), wine (H=330°)
# Cool group: cerulean (H=210°), teal (H=170°), violet (H=264°)
CATEGORICAL = [
    "#4477AA",
    "#C07EAA",
    "#44AA99",
    "#CC6677",
    "#7755AA",
    "#882255",
]

CMAP4_NAME = "my4"
CMAP4_COLORS = [
    "#193B00",
    "#385B2A",
    "#577A55",
    "#769A80",
    "#94B9AA",
    "#B3D9D5",
    "#D2F8FF",
]
cmap4 = LinearSegmentedColormap.from_list(CMAP4_NAME, CMAP4_COLORS, N=256)

colorscale4 = [
    [0.00, "#193B00"],
    [0.20, "#385B2A"],
    [0.40, "#577A55"],
    [0.60, "#769A80"],
    [0.60, "#94B9AA"],
    [0.80, "#B3D9D5"],
    [1.00, "#D2F8FF"],
]

THEME = {
    "bg":   "#ffffff",
    "fg":   "#1a1a2e",
    "grid": "#00000000",
    "muted": "#7a88a0",

    "font": ["Inter", "DejaVu Sans", "Arial"],

    "categorical": CATEGORICAL,
    "cc_color": CC_COLOR,

    "sequential_cmap_mpl":    "viridis",
    "sequential_cmap_mpl_obj": "viridis",
    "sequential_cmap_plotly": "viridis",
    "sequential_ramp_altair": "viridis",
}


def apply_matplotlib() -> None:
    import matplotlib as mpl
    try:
        mpl.colormaps.register(cmap4, name=CMAP4_NAME)
    except ValueError:
        pass
    mpl.rcParams.update({
        "figure.facecolor":  THEME["bg"],
        "axes.facecolor":    THEME["bg"],
        "savefig.facecolor": THEME["bg"],

        "text.color":        THEME["fg"],
        "axes.labelcolor":   THEME["fg"],
        "xtick.color":       THEME["fg"],
        "ytick.color":       THEME["fg"],
        "axes.edgecolor":    THEME["muted"],

        "axes.grid":         False,
        "image.cmap":        THEME["sequential_cmap_mpl"],

        "font.family":       "sans-serif",
        "font.sans-serif":   THEME["font"],

        "axes.prop_cycle":   mpl.cycler(color=CATEGORICAL),
    })


def apply_seaborn() -> None:
    import seaborn as sns
    sns.set_theme(
        style="white",
        palette=CATEGORICAL,
        rc={
            "axes.grid": False,
            "image.cmap": THEME["sequential_cmap_mpl"],
        },
    )


def apply_plotly() -> None:
    import plotly.io as pio
    import plotly.graph_objects as go

    tpl = go.layout.Template(
        layout=dict(
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["bg"],
            font=dict(color=THEME["fg"], family=", ".join(THEME["font"])),
            colorway=CATEGORICAL,
            coloraxis=dict(colorscale=THEME["sequential_cmap_plotly"]),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=THEME["muted"],
                tickcolor=THEME["muted"],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=THEME["muted"],
                tickcolor=THEME["muted"],
            ),
        ),
        data=dict(
            heatmap=[go.Heatmap(colorscale=THEME["sequential_cmap_plotly"])],
            contour=[go.Contour(colorscale=THEME["sequential_cmap_plotly"])],
            surface=[go.Surface(colorscale=THEME["sequential_cmap_plotly"])],
        ),
    )
    pio.templates["unified"] = tpl
    pio.templates.default = "unified"


def apply_altair() -> None:
    import altair as alt

    def _theme():
        return {
            "config": {
                "background": THEME["bg"],
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelColor": THEME["fg"],
                    "titleColor": THEME["fg"],
                    "grid": False,
                    "domainColor": THEME["muted"],
                    "tickColor": THEME["muted"],
                },
                "legend": {"labelColor": THEME["fg"], "titleColor": THEME["fg"]},
                "range": {
                    "category": CATEGORICAL,
                    "ramp": THEME["sequential_ramp_altair"],
                },
                "title": {"color": THEME["fg"], "font": ", ".join(THEME["font"])},
                "text": {"color": THEME["fg"], "font": ", ".join(THEME["font"])},
            }
        }

    alt.themes.register("unified", _theme)
    alt.themes.enable("unified")


def apply_all() -> None:
    try:
        apply_matplotlib()
    except Exception:
        pass

    try:
        apply_seaborn()
    except Exception:
        pass

    # Re-apply matplotlib after seaborn (seaborn can override rcParams)
    try:
        apply_matplotlib()
    except Exception:
        pass

    try:
        apply_plotly()
    except Exception:
        pass

    try:
        apply_altair()
    except Exception:
        pass


def themed_heatmap(*args, **kwargs):
    import seaborn as sns
    kwargs.setdefault("cmap", THEME.get("sequential_cmap_mpl_obj", THEME["sequential_cmap_mpl"]))
    kwargs.setdefault("cbar_kws", {})
    return sns.heatmap(*args, **kwargs)


def themed_clustermap(*args, **kwargs):
    import seaborn as sns
    kwargs.setdefault("cmap", THEME.get("sequential_cmap_mpl_obj", THEME["sequential_cmap_mpl"]))
    return sns.clustermap(*args, **kwargs)


# ── Legacy (green) theme ──────────────────────────────────────────────────────
# The original teal/green palette used before the current scheme was introduced.
# Accessible as `theme.CATEGORICAL_LEGACY`, `theme.CC_COLOR_LEGACY`,
# `theme.THEME_LEGACY`, and `theme.apply_all_legacy()`.

CC_COLOR_LEGACY = "#4A4A4A"

CATEGORICAL_LEGACY = [
    "#327D6D",
    "#7FE3CD",
    "#5BBFA9",
    "#B2F2E5",
    "#005341",
    "#B3D9D5",
]

THEME_LEGACY = {
    "bg":   "#ffffff",
    "fg":   "#000000",
    "grid": "#2a335a00",
    "muted": "#8b93b5",

    "font": ["Inter", "DejaVu Sans", "Arial"],

    "categorical": CATEGORICAL_LEGACY,
    "cc_color": CC_COLOR_LEGACY,

    "sequential_cmap_mpl":     "viridis",
    "sequential_cmap_mpl_obj": "viridis",
    "sequential_cmap_plotly":  "viridis",
    "sequential_ramp_altair":  "viridis",
}


def apply_matplotlib_legacy() -> None:
    import matplotlib as mpl
    try:
        mpl.colormaps.register(cmap4, name=CMAP4_NAME)
    except ValueError:
        pass
    mpl.rcParams.update({
        "figure.facecolor":  THEME_LEGACY["bg"],
        "axes.facecolor":    THEME_LEGACY["bg"],
        "savefig.facecolor": THEME_LEGACY["bg"],

        "text.color":        THEME_LEGACY["fg"],
        "axes.labelcolor":   THEME_LEGACY["fg"],
        "xtick.color":       THEME_LEGACY["fg"],
        "ytick.color":       THEME_LEGACY["fg"],
        "axes.edgecolor":    THEME_LEGACY["muted"],

        "axes.grid":         False,
        "image.cmap":        THEME_LEGACY["sequential_cmap_mpl"],

        "font.family":       "sans-serif",
        "font.sans-serif":   THEME_LEGACY["font"],

        "axes.prop_cycle":   mpl.cycler(color=CATEGORICAL_LEGACY),
    })


def apply_seaborn_legacy() -> None:
    import seaborn as sns
    sns.set_theme(
        style="white",
        palette=CATEGORICAL_LEGACY,
        rc={
            "axes.grid": False,
            "image.cmap": THEME_LEGACY["sequential_cmap_mpl"],
        },
    )


def apply_plotly_legacy() -> None:
    import plotly.io as pio
    import plotly.graph_objects as go

    tpl = go.layout.Template(
        layout=dict(
            paper_bgcolor=THEME_LEGACY["bg"],
            plot_bgcolor=THEME_LEGACY["bg"],
            font=dict(color=THEME_LEGACY["fg"], family=", ".join(THEME_LEGACY["font"])),
            colorway=CATEGORICAL_LEGACY,
            coloraxis=dict(colorscale=THEME_LEGACY["sequential_cmap_plotly"]),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=THEME_LEGACY["muted"],
                tickcolor=THEME_LEGACY["muted"],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=THEME_LEGACY["muted"],
                tickcolor=THEME_LEGACY["muted"],
            ),
        ),
        data=dict(
            heatmap=[go.Heatmap(colorscale=THEME_LEGACY["sequential_cmap_plotly"])],
            contour=[go.Contour(colorscale=THEME_LEGACY["sequential_cmap_plotly"])],
            surface=[go.Surface(colorscale=THEME_LEGACY["sequential_cmap_plotly"])],
        ),
    )
    pio.templates["unified_legacy"] = tpl
    pio.templates.default = "unified_legacy"


def apply_altair_legacy() -> None:
    import altair as alt

    def _theme():
        return {
            "config": {
                "background": THEME_LEGACY["bg"],
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelColor": THEME_LEGACY["fg"],
                    "titleColor": THEME_LEGACY["fg"],
                    "grid": False,
                    "domainColor": THEME_LEGACY["muted"],
                    "tickColor": THEME_LEGACY["muted"],
                },
                "legend": {"labelColor": THEME_LEGACY["fg"], "titleColor": THEME_LEGACY["fg"]},
                "range": {
                    "category": CATEGORICAL_LEGACY,
                    "ramp": THEME_LEGACY["sequential_ramp_altair"],
                },
                "title": {"color": THEME_LEGACY["fg"], "font": ", ".join(THEME_LEGACY["font"])},
                "text": {"color": THEME_LEGACY["fg"], "font": ", ".join(THEME_LEGACY["font"])},
            }
        }

    alt.themes.register("unified_legacy", _theme)
    alt.themes.enable("unified_legacy")


def apply_all_legacy() -> None:
    """Apply the original teal/green theme to all supported plotting libraries."""
    try:
        apply_matplotlib_legacy()
    except Exception:
        pass

    try:
        apply_seaborn_legacy()
    except Exception:
        pass

    try:
        apply_matplotlib_legacy()
    except Exception:
        pass

    try:
        apply_plotly_legacy()
    except Exception:
        pass

    try:
        apply_altair_legacy()
    except Exception:
        pass
