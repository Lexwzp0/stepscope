"""Calibration figure for the README: first-order prediction vs actual change, one panel per run.

Reads `effects/*.parquet` of four runs from run_real_case.py (SGD and AdamW, each at its default step and
10x smaller) and writes a light and a dark PNG. Error = max |first order - actual| / max |actual|, per
observable over all probe rows.

    python examples/real_case/plot_calibration.py runs/real_case/results --out docs/img
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = [("sgd_80", "SGD, lr 1e-2"), ("sgd_lr1e3_80", "SGD, lr 1e-3"),
        ("adamw_80", "AdamW, lr 1.5e-6"), ("adamw_lr1p5e7_80", "AdamW, lr 1.5e-7")]
OBS = [("phi_fmt", "Φ_fmt", "o"), ("phi_ans", "Φ_ans", "s")]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926"]),
}


def load(results: Path, run: str) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted((results / run / "effects").glob("*.parquet")))


def draw(frames: dict[str, pd.DataFrame], theme: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4), dpi=150, facecolor=theme["surface"])
    for ax, (run, title) in zip(axes.flat, RUNS):
        df = frames[run]
        ax.set_facecolor(theme["surface"])
        errs = []
        for (key, name, marker), color in zip(OBS, theme["series"]):
            d = df[df.observable == key]
            ax.scatter(d.actual, d.first_order, s=22, marker=marker, color=color, alpha=0.8,
                       edgecolors=theme["surface"], linewidths=0.6, label=name,
                       zorder=4 if key == "phi_fmt" else 3)  # fewer Φ_fmt points stay visible on top
            errs.append(100 * (d.first_order - d.actual).abs().max() / d.actual.abs().max())
        lim = 1.08 * max(df.actual.abs().max(), df.first_order.abs().max())
        ax.plot([-lim, lim], [-lim, lim], color=theme["muted"], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(color=theme["grid"], lw=0.6)
        ax.tick_params(colors=theme["muted"], labelsize=8)
        for s in ax.spines.values():
            s.set_color(theme["axis"])
        ax.set_title(title, color=theme["ink"], fontsize=10, loc="left")
        ax.text(0.03, 0.97, f"max error  Φ_fmt {errs[0]:.1f}%  ·  Φ_ans {errs[1]:.1f}%",
                transform=ax.transAxes, va="top", fontsize=8, color=theme["ink2"])
    for ax in axes[1]:
        ax.set_xlabel("actual change  Φ(θ+Δθ) − Φ(θ)", color=theme["ink2"], fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("first-order prediction  ∇Φ(θ)·Δθ", color=theme["ink2"], fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(theme["ink"])
    fig.suptitle("One Open-MOPD step on SmolLM3-3B: first order vs actual (80 probe rows)",
                 x=0.06, ha="left", color=theme["ink"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="results/ directory holding the four run directories")
    p.add_argument("--out", type=Path, default=Path("docs/img"))
    a = p.parse_args()
    frames = {run: load(a.results, run) for run, _ in RUNS}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        draw(frames, theme, a.out / f"real_case_calibration_{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
