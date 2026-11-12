"""Fig. 3 of the paper (task #80, Section 6.5): E2E p95 and energy per token against cluster size.

    python scripts/fig3_cluster_size.py [--rho 0.6] [--out results/scaling/fig3_cluster_size.png]

Lines: the simulator (results/scaling/sim_summary.csv, scripts/sim_scaling.py) on 3, 8 and 16
GPUs, mean over 5 seeds with the 95% confidence interval. Open markers left of 3: the same
strategies measured on the real 3-GPU cluster (results/phase5/summary.csv, experiment main).
The oracle (true output lengths, simulator only) is the dashed reference. power_of_two is left
out of the figure (eight series at most); its numbers are in sim_summary.csv.

Needs matplotlib (pip install matplotlib).
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SIZES = [3, 8, 16]
REAL_X = 2.3                         # where the real 3-GPU values are drawn, just left of x = 3
# one fixed colour and marker per strategy (validated categorical palette, light mode)
STYLE = {
    "round_robin":   ("Round robin",   "#2a78d6", "o"),
    "weighted":      ("Weighted",      "#eb6834", "s"),
    "least_load":    ("Least load",    "#1baf7a", "^"),
    "least_work":    ("Least work",    "#eda100", "D"),
    "predictive":    ("Predictive",    "#e87ba4", "v"),
    "rl":            ("RL",            "#008300", "P"),
    "marginal_cost": ("Marginal cost", "#6250d6", "X"),
}
ORACLE = ("Oracle (sim.)", "#898781")
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def read(path, experiment_filter, rho):
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        if r["workload"] == "mixed" and abs(float(r["rho"]) - rho) < 1e-9 and experiment_filter(r["experiment"]):
            out[(r["experiment"], r["strategy"])] = r
    return out


def value(r, metric, scale):
    ci = r[metric + "_ci95"]
    return float(r[metric]) * scale, (float(ci) * scale if ci not in ("", "nan") else 0.0)


def panel(ax, sim, real, metric, scale, title):
    lows, highs = [], []
    for s, (label, color, marker) in list(STYLE.items()) + [("oracle", (ORACLE[0], ORACLE[1], None))]:
        pts = [value(sim[(f"sim{k}", s)], metric, scale) for k in SIZES if (f"sim{k}", s) in sim]
        if len(pts) != len(SIZES):
            continue
        y, e = [p[0] for p in pts], [p[1] for p in pts]
        lows += [a - b for a, b in zip(y, e)]; highs += [a + b for a, b in zip(y, e)]
        if s == "oracle":
            ax.plot(SIZES, y, color=color, lw=0.9, ls=(0, (3, 1.6)), zorder=2)
            continue
        bars = ax.errorbar(SIZES, y, yerr=e, color=color, lw=1.0, marker=marker, ms=3.4, mec="#fcfcfb", mew=0.4,
                           elinewidth=0.6, capsize=0, zorder=3)
        bars[2][0].set_alpha(0.55)                     # recessive confidence intervals
        if ("main", s) in real:
            yr, er = value(real[("main", s)], metric, scale)
            lows.append(yr - er); highs.append(yr + er)
            bars = ax.errorbar([REAL_X], [yr], yerr=[er], color=color, marker=marker, ms=3.4, mfc="white", mec=color,
                               mew=0.8, elinewidth=0.6, capsize=0, ls="none", zorder=4)
            bars[2][0].set_alpha(0.55)
    ax.set_title(title, loc="left", fontsize=7, color=INK, pad=3)
    ax.set_xticks(SIZES)
    ax.set_xlim(1.5, 17.0)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=6.5, width=0.6, length=2.5)
    return min(lows), max(highs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling" / "fig3_cluster_size.png"))
    a = ap.parse_args()
    sim = read(ROOT / "results" / "scaling" / "sim_summary.csv", lambda e: e.startswith("sim"), a.rho)
    real = read(ROOT / "results" / "phase5" / "summary.csv", lambda e: e == "main", a.rho)

    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7, "text.color": INK,
                         "axes.labelcolor": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5), layout="constrained")
    lo, hi = panel(ax1, sim, real, "e2e_p95_ms", 1 / 1000, "(a) E2E latency, p95 (s)")
    ax1.set_ylim(0, hi * 1.06)
    lo, hi = panel(ax2, sim, real, "j_per_token", 1.0, "(b) Energy per token (J)")
    pad = (hi - lo) * 0.08
    ax2.set_ylim(lo - pad, hi + pad)
    fig.supxlabel("Cluster size (GPUs)", fontsize=7, color=INK2)

    handles = [Line2D([], [], color=c, marker=m, ms=3.4, lw=1.0, mec="#fcfcfb", mew=0.4) for _, c, m in STYLE.values()]
    labels = [lab for lab, _, _ in STYLE.values()]
    handles += [Line2D([], [], color=ORACLE[1], lw=0.9, ls=(0, (3, 1.6))),
                Line2D([], [], color=INK2, marker="o", ms=3.4, lw=0, mfc=INK2, mec=INK2),
                Line2D([], [], color=INK2, marker="o", ms=3.4, lw=0, mfc="white", mec=INK2, mew=0.8)]
    labels += [ORACLE[0], "Simulator", "Real cluster"]
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=6.2,
               handlelength=1.6, handletextpad=0.4, columnspacing=0.8, labelcolor=INK)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"rho {a.rho} -> {out}")


if __name__ == "__main__":
    main()
