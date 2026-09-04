"""Render the two result figures as SVG, from a finished run's numbers.

Two figures, because the result has two things to say and one chart cannot say
both:

* `docs/wer-by-snr.svg` -- WER per SNR band, zero-shot against fine-tuned, with
  clean audio as the rightmost band. The point of putting clean on the same axis
  is that the specialisation cost is not a separate finding: it is this curve's
  endpoint, where the noise runs out and the trade reverses sign.
* `docs/effect-intervals.svg` -- the two paired-bootstrap intervals against a
  zero line. A delta without an interval is not a result, and a reader should be
  able to see that neither interval crosses zero rather than take it on trust.

SVG rather than PNG: it is text, so it diffs, and it stays sharp at any zoom. No
plotting library -- the geometry here is a few dozen rectangles and lines, and a
matplotlib dependency for that would be the larger cost.

Light and dark variants are emitted separately and paired with `<picture>` in
the README, because GitHub strips CSS media queries out of embedded SVG. The
dark steps are chosen for the dark surface, not flipped from the light ones.

Numbers come from the run, not from this file's defaults -- pass
`--analysis results/analysis.json` to re-render after a new run.

    python -m reach_asr.plot_results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The dataviz reference palette, categorical slots 1 and 2. Both modes validated
# as a pair: worst CVD delta-E 24.7 light / 26.8 dark against an >= 8 target,
# normal-vision 33.6 / 31.8 against an >= 15 floor, both >= 3:1 on their surface.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2df",
        "zeroshot": "#2a78d6",
        "finetuned": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383734",
        "zeroshot": "#3987e5",
        "finetuned": "#d95926",
    },
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Band:
    label: str
    n: int
    zeroshot: float
    finetuned: float
    # Stated rather than derived: analyze.py computes the delta from unrounded
    # WERs, so subtracting the two displayed figures disagrees with it in the
    # last place (5.67 vs 5.66). The chart must not contradict the table.
    delta: float


# The reported run. Overridden by --analysis; kept here so the figures can be
# regenerated from a clean checkout without a results directory present.
DEFAULT_BANDS = [
    Band("-5 to 0 dB", 99, 38.19, 32.53, 5.67),
    Band("0 to 5 dB", 104, 20.93, 19.11, 1.82),
    Band("5 to 10 dB", 97, 13.16, 12.75, 0.41),
    Band("clean", 300, 4.37, 5.24, -0.87),
]
DEFAULT_EFFECTS = [
    ("degraded audio", -2.56, -3.85, -1.31),
    ("clean audio", 0.87, 0.35, 1.40),
]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_el(x: float, y: float, s: str, fill: str, size: float, anchor: str = "start",
            weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="{FONT}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def grouped_bars(bands: list[Band], theme: dict[str, str]) -> str:
    """WER by SNR band, two series per band.

    Grouped bars rather than a line: the bands are ordered categories, not a
    continuous axis, and the comparison a reader needs is within-band (which bar
    is shorter) rather than along it.
    """
    w, h = 780, 360
    pad_l, pad_r, pad_t, pad_b = 52, 16, 64, 62
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    y_max = 40.0

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - v / y_max)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="Word error rate by SNR band, '
        f'zero-shot versus fine-tuned">',
        f'<rect width="{w}" height="{h}" fill="{theme["surface"]}"/>',
        text_el(pad_l - 36, 26, "WER by SNR band", theme["text"], 15, weight="600"),
        text_el(pad_l - 36, 44, "lower is better · 300 eval utterances · whisper-base",
                theme["muted"], 11),
    ]

    # Legend up top, always present for two series; bars are also directly
    # labelled, so identity never rests on colour alone.
    lx = w - pad_r - 210
    for i, (name, key) in enumerate([("zero-shot", "zeroshot"), ("fine-tuned", "finetuned")]):
        x = lx + i * 108
        out.append(f'<rect x="{x}" y="18" width="9" height="9" rx="2" fill="{theme[key]}"/>')
        out.append(text_el(x + 14, 27, name, theme["muted"], 11))

    for gv in range(0, int(y_max) + 1, 10):
        y = y_of(gv)
        out.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        out.append(text_el(pad_l - 8, y + 4, f"{gv}%", theme["muted"], 11, anchor="end"))

    slot = plot_w / len(bands)
    bar_w = 30.0
    gap = 2.0  # surface gap between adjacent bars
    for i, band in enumerate(bands):
        cx = pad_l + slot * (i + 0.5)
        for j, (key, value) in enumerate([("zeroshot", band.zeroshot),
                                          ("finetuned", band.finetuned)]):
            x = cx - bar_w - gap / 2 + j * (bar_w + gap)
            y = y_of(value)
            bh = pad_t + plot_h - y
            out.append(
                f'<path d="M{x:.1f},{pad_t + plot_h:.1f} L{x:.1f},{y + 4:.1f} '
                f'Q{x:.1f},{y:.1f} {x + 4:.1f},{y:.1f} L{x + bar_w - 4:.1f},{y:.1f} '
                f'Q{x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + 4:.1f} '
                f'L{x + bar_w:.1f},{pad_t + plot_h:.1f} Z" fill="{theme[key]}"/>'
                if bh > 6 else
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" '
                f'fill="{theme[key]}"/>'
            )
            out.append(text_el(x + bar_w / 2, y - 6, f"{value:.1f}", theme["text"], 10.5,
                               anchor="middle", weight="600"))

        base = pad_t + plot_h
        out.append(text_el(cx, base + 18, band.label, theme["text"], 11.5, anchor="middle"))
        out.append(text_el(cx, base + 33, f"n={band.n}", theme["muted"], 10, anchor="middle"))
        # The delta is the thing being read off this chart, so it is stated
        # rather than left to be eyeballed off two bar heights.
        sign = "−" if band.delta > 0 else "+"
        out.append(text_el(cx, base + 49, f"{sign}{abs(band.delta):.2f} pp", theme["muted"], 10.5,
                           anchor="middle", weight="600"))

    out.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}" '
        f'stroke="{theme["muted"]}" stroke-width="1"/>'
    )
    out.append("</svg>")
    return "\n".join(out)


def interval_plot(effects: list[tuple[str, float, float, float]], theme: dict[str, str]) -> str:
    """Point estimate and 95% interval for each effect, against a zero line."""
    w, h = 780, 220
    pad_l, pad_r, pad_t = 132, 196, 62
    plot_w = w - pad_l - pad_r
    lo, hi = -4.5, 2.0

    def x_of(v: float) -> float:
        return pad_l + plot_w * (v - lo) / (hi - lo)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="Absolute WER change with 95 percent '
        f'confidence intervals">',
        f'<rect width="{w}" height="{h}" fill="{theme["surface"]}"/>',
        text_el(16, 26, "Absolute WER change, with 95% intervals", theme["text"], 15, weight="600"),
        text_el(16, 44, "paired bootstrap, 10,000 resamples · negative is better",
                theme["muted"], 11),
    ]

    for gv in (-4, -3, -2, -1, 0, 1, 2):
        x = x_of(gv)
        is_zero = gv == 0
        stroke = theme["muted"] if is_zero else theme["grid"]
        dash = "" if is_zero else ' stroke-dasharray="2 3"'
        out.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{h - 34}" '
            f'stroke="{stroke}" stroke-width="1"{dash}/>'
        )
        out.append(text_el(x, h - 18, f"{gv:+d}" if gv else "0", theme["muted"], 10.5,
                           anchor="middle"))
    out.append(text_el(x_of(0), pad_t - 8, "no change", theme["muted"], 10, anchor="middle"))

    row_h = (h - 34 - pad_t) / len(effects)
    for i, (name, point, low, high) in enumerate(effects):
        y = pad_t + row_h * (i + 0.5)
        key = "finetuned" if point > 0 else "zeroshot"
        out.append(text_el(pad_l - 14, y + 4, name, theme["text"], 12, anchor="end"))
        out.append(
            f'<line x1="{x_of(low):.1f}" y1="{y:.1f}" x2="{x_of(high):.1f}" y2="{y:.1f}" '
            f'stroke="{theme[key]}" stroke-width="2" stroke-linecap="round"/>'
        )
        for end in (low, high):
            out.append(
                f'<line x1="{x_of(end):.1f}" y1="{y - 5:.1f}" x2="{x_of(end):.1f}" '
                f'y2="{y + 5:.1f}" stroke="{theme[key]}" stroke-width="2"/>'
            )
        # 2px surface ring so the marker reads against the interval line.
        out.append(
            f'<circle cx="{x_of(point):.1f}" cy="{y:.1f}" r="6" fill="{theme[key]}" '
            f'stroke="{theme["surface"]}" stroke-width="2"/>'
        )
        out.append(text_el(w - pad_r + 12, y + 4,
                           f"{point:+.2f} pp  [{low:+.2f}, {high:+.2f}]",
                           theme["muted"], 10.5, weight="600"))
    out.append("</svg>")
    return "\n".join(out)


def load_from_analysis(path: Path) -> tuple[list[Band], list[tuple[str, float, float, float]]]:
    """Pull the figures' numbers out of a finished run's analysis.json."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    bands: list[Band] = []
    for label, row in data.get("by_snr", {}).items():
        bands.append(Band(label, row["n"], row["zeroshot"] * 100, row["finetuned"] * 100,
                          (row["zeroshot"] - row["finetuned"]) * 100))
    clean = data.get("specialisation")
    if clean:
        bands.append(Band("clean", data["n_utterances"],
                          clean["clean_wer_zeroshot"] * 100, clean["clean_wer_finetuned"] * 100,
                          -clean["clean_wer_change_pp"] * 100))

    boot = data["bootstrap"]["absolute_wer_reduction"]
    effects = [("degraded audio", -boot["point"] * 100, -boot["ci_high"] * 100,
                -boot["ci_low"] * 100)]
    if clean:
        cb = clean["bootstrap"]["absolute_wer_reduction"]
        effects.append(("clean audio", -cb["point"] * 100, -cb["ci_high"] * 100,
                        -cb["ci_low"] * 100))
    return bands or DEFAULT_BANDS, effects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=None,
                        help="results/analysis.json; falls back to the reported run's numbers")
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args()

    bands, effects = DEFAULT_BANDS, DEFAULT_EFFECTS
    if args.analysis is not None:
        if not args.analysis.exists():
            raise SystemExit(f"no such file: {args.analysis}")
        bands, effects = load_from_analysis(args.analysis)

    args.out.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        suffix = "" if mode == "light" else "-dark"
        for name, svg in [
            (f"wer-by-snr{suffix}.svg", grouped_bars(bands, theme)),
            (f"effect-intervals{suffix}.svg", interval_plot(effects, theme)),
        ]:
            (args.out / name).write_text(svg + "\n", encoding="utf-8")
            print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
