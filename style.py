"""Shared plot style. Every plot imports colors from here (see CLAUDE.md)."""

# Okabe-Ito
SENSOR = "#404040"
TRUE_SPOT = "#000000"
WARNING = "#E69F00"

LABEL_COLOR = {
    "knock_tap": "#0072B2",
    "voice": "#009E73",
    "impact": "#E69F00",
    "machinery": "#CC79A7",
    "structure_creak": "#D55E00",
    "other": "#BBBBBB",
    "nature": "#BBBBBB",
}
# Opacity per label: highlighted vs faded vs hidden-by-default.
LABEL_ALPHA = {
    "knock_tap": 1.0,
    "voice": 1.0,
    "impact": 0.4,
    "machinery": 0.4,
    "structure_creak": 1.0,
    "other": 0.25,
    "nature": 0.25,
}
HIDDEN_BY_DEFAULT = {"other", "nature"}
HAZARD_LABELS = {"structure_creak"}
HUMAN_LIKELY = {"knock_tap", "voice"}
LABEL_ORDER = ["knock_tap", "voice", "impact", "other"]  # human-likely first
UNCERTAINTY_FILL_ALPHA = 0.18

# Detection tuning view (Plot 1)
SIGNAL_RAW = "#404040"
SIGNAL_FILT = "#56B4E9"   # sky blue
SIGNAL_ENV = "#000000"
NOISE_FLOOR = "#999999"
THRESHOLD = "#D55E00"     # vermillion
OUR_ONSET = "#0072B2"     # blue
REF_ONSET = "#999999"
CHI2_95_2D = 5.991  # 95 % ellipse: scale 1-sigma axes by sqrt(5.991)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
