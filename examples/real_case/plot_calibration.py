"""Calibration figure for the README: first-order prediction vs actual change, one panel per run.

Reads `effects/*.parquet` of four runs from run_real_case.py (SGD and AdamW, each at its default step and
10x smaller) and writes light/dark SVG and PNG files. Math is rendered with Matplotlib's built-in
MathText, so no external LaTeX installation is needed. Error = max |first order - actual| /
max |actual|, per observable over all probe rows. All points are retained without jitter.

    python examples/real_case/plot_calibration.py runs/real_case/results --out docs/img
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, StrMethodFormatter  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = [
    ("sgd_80", r"SGD  ·  $\eta = 10^{-2}$"),
    ("sgd_lr1e3_80", r"SGD  ·  $\eta = 10^{-3}$"),
    ("adamw_80", r"AdamW  ·  $\eta = 1.5 \times 10^{-6}$"),
    ("adamw_lr1p5e7_80", r"AdamW  ·  $\eta = 1.5 \times 10^{-7}$"),
]
OBS = [("phi_fmt", r"$\Phi_{\mathrm{fmt}}$", "o"), ("phi_ans", r"$\Phi_{\mathrm{ans}}$", "s")]
THEMES = {
    "light": dict(surface="#ffffff", ink="#17212f", ink2="#4b586b", muted="#66758a",
                  grid="#e8edf3", axis="#b8c2cf", reference="#778494", series=["#2775cf", "#df783b"]),
    "dark": dict(surface="#0d1117", ink="#e6edf3", ink2="#bdc7d4", muted="#92a0b3",
                 grid="#26303c", axis="#536174", reference="#a6b3c3", series=["#7cb8ff", "#ffad73"]),
}


def load(results: Path, run: str) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted((results / run / "effects").glob("*.parquet")))


def draw(frames: dict[str, pd.DataFrame], theme: dict, out: Path) -> None:
    fig = plt.figure(figsize=(9.4, 10.1), dpi=220, facecolor=theme["surface"])
    # Explicit square panels leave predictable space for mathematical labels and annotations.
    panel_width = 0.345
    panel_height = panel_width * 9.4 / 10.1
    axes = [fig.add_axes([x, y, panel_width, panel_height])
            for y in (0.505, 0.115) for x in (0.13, 0.59)]
    for i, (ax, (run, title)) in enumerate(zip(axes, RUNS)):
        df = frames[run]
        ax.set_facecolor(theme["surface"])
        errs = []
        for (key, name, marker), color in zip(OBS, theme["series"]):
            d = df[df.observable == key]
            is_format = key == "phi_fmt"
            ax.scatter(d.actual, d.first_order, s=28 if is_format else 25, marker=marker,
                       facecolors=theme["surface"] if is_format else color,
                       edgecolors=color if is_format else theme["surface"],
                       linewidths=0.95 if is_format else 0.45, alpha=0.95 if is_format else 0.84,
                       label=name, zorder=4 if is_format else 3)
            errs.append(100 * (d.first_order - d.actual).abs().max() / d.actual.abs().max())
        lim = 1.08 * max(df.actual.abs().max(), df.first_order.abs().max())
        ax.plot([-lim, lim], [-lim, lim], color=theme["reference"], lw=1.15, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, steps=[1, 2, 5, 10]))
        tick_format = "{x:.2f}" if lim < 0.1 else "{x:.1f}"
        ax.xaxis.set_major_formatter(StrMethodFormatter(tick_format))
        ax.yaxis.set_major_formatter(StrMethodFormatter(tick_format))
        ax.set_axisbelow(True)
        ax.grid(color=theme["grid"], lw=0.65)
        ax.tick_params(colors=theme["muted"], labelsize=10, length=3, width=0.7, pad=5)
        for s in ax.spines.values():
            s.set_color(theme["axis"])
            s.set_linewidth(0.7)
        ax.set_title(f"({chr(97 + i)})  {title}", color=theme["ink"], fontsize=12,
                     loc="left", pad=13)
        ax.text(0.035, 0.96,
                rf"Error:  $\Phi_{{\mathrm{{fmt}}}}$ {errs[0]:.1f}%   "
                rf"$\Phi_{{\mathrm{{ans}}}}$ {errs[1]:.1f}%",
                transform=ax.transAxes, va="top", fontsize=10, color=theme["ink2"])

    fig.text(0.13, 0.965, "One-step calibration on SmolLM3-3B", ha="left", va="top",
             color=theme["ink"], fontsize=19, fontweight="semibold")
    fig.text(0.13, 0.930, "Open-MOPD · 20 independent questions × 4 answer rotations · FP32",
             ha="left", va="top", color=theme["ink2"], fontsize=10.5)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor=theme["surface"],
               markeredgecolor=theme["series"][0], markeredgewidth=1.1, label=OBS[0][1] + "  Format"),
        Line2D([], [], marker="s", linestyle="none", markersize=5.5, color=theme["series"][1],
               label=OBS[1][1] + "  Answer"),
        Line2D([], [], color=theme["reference"], lw=1.15, ls=(0, (4, 3)), label="Exact agreement"),
    ]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.118, 0.887),
                     ncol=3, frameon=False, fontsize=11, handletextpad=0.65, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(theme["ink"])
    fig.text(0.53, 0.068, r"Actual change   $\Delta\Phi = \Phi(\theta+\Delta\theta)-\Phi(\theta)$",
             ha="center", va="center", color=theme["ink"], fontsize=12)
    fig.text(0.027, 0.473,
             r"First-order prediction   $\widehat{\Delta\Phi} = \nabla_{\theta}\Phi(\theta)^{\top}\Delta\theta$",
             ha="center", va="center", rotation=90, color=theme["ink"], fontsize=12)
    fig.text(0.13, 0.040,
             r"Error $=\max_i|\widehat{\Delta\Phi}_i-\Delta\Phi_i|\,/\,\max_i|\Delta\Phi_i|$"
             "  (80 rows per observable).",
             ha="left", va="center", color=theme["ink2"], fontsize=9.5)
    fig.text(0.13, 0.008, "Axis ranges vary by panel. AdamW starts with empty optimizer moments.",
             ha="left", va="bottom", color=theme["muted"], fontsize=9)
    fig.savefig(out.with_suffix(".png"), facecolor=theme["surface"])
    fig.savefig(out.with_suffix(".svg"), facecolor=theme["surface"], metadata={"Date": None})
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="results/ directory holding the four run directories")
    p.add_argument("--out", type=Path, default=Path("docs/img"))
    a = p.parse_args()
    frames = {run: load(a.results, run) for run, _ in RUNS}
    a.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "path", "svg.hashsalt": "stepscope-calibration",
                         "axes.unicode_minus": True}):
        for name, theme in THEMES.items():
            draw(frames, theme, a.out / f"real_case_calibration_{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
