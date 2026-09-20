import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(BASE, "outputs", "figures")
TAB = os.path.join(BASE, "outputs", "tables")

os.makedirs(FIG, exist_ok=True)

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 8.8,
        "axes.labelsize": 9.3,
        "axes.titlesize": 9.8,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#3b3b3b",
        "axes.linewidth": 0.8,
        "xtick.color": "#2f2f2f",
        "ytick.color": "#2f2f2f",
        "text.color": "#222222",
    }
)

COLORS = {
    "le": "#1f5aa6",
    "hale": "#d97b2b",
    "gap": "#2f4858",
    "gap_fill": "#cfdde6",
    "neutral": "#b8bec6",
    "axis": "#3b3b3b",
    "grid": "#e2e2e2",
    "baseline_le": "#1f5aa6",
    "baseline_sdi": "#d97b2b",
}


def polish_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(length=3, color=COLORS["axis"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.7, alpha=0.9)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_decomposition_panel(ax):
    decomp = pd.read_csv(os.path.join(TAB, "table_absolute_relative_summary_changes.csv"))
    decomp = decomp[decomp["summary_type"] == "Population-weighted country mean"].copy()
    decomp["ypos"] = [3, 2, 1]

    ax.axvline(0, color="#6f6f6f", linewidth=1.0)
    for _, row in decomp.iterrows():
        low = min(row["delta_hale"], row["delta_le"])
        high = max(row["delta_hale"], row["delta_le"])
        ax.hlines(row["ypos"], low, high, color=COLORS["neutral"], linewidth=2.4, zorder=1)
        ax.scatter(row["delta_le"], row["ypos"], s=46, color=COLORS["le"], zorder=3)
        ax.scatter(row["delta_hale"], row["ypos"], s=46, color=COLORS["hale"], zorder=3)

        le_dx = 0.10 if row["delta_le"] >= 0 else 0.06
        hale_dx = 0.10 if row["delta_hale"] >= 0 else 0.06
        ax.text(row["delta_le"] + le_dx, row["ypos"] + 0.13, f"{row['delta_le']:.2f}", color=COLORS["le"], fontsize=8.5)
        ax.text(row["delta_hale"] + hale_dx, row["ypos"] - 0.22, f"{row['delta_hale']:.2f}", color=COLORS["hale"], fontsize=8.5)

        gap_sign = "+" if row["delta_gap"] >= 0 else ""
        ax.text(high + 0.34, row["ypos"], f"Gap {gap_sign}{row['delta_gap']:.2f} y", va="center", fontsize=8.3, color=COLORS["gap"])

    ax.scatter([], [], s=46, color=COLORS["le"], label="Delta LE")
    ax.scatter([], [], s=46, color=COLORS["hale"], label="Delta HALE")
    ax.legend(frameon=False, loc="upper left", fontsize=8.2, handletextpad=0.3, borderaxespad=0.1)
    ax.set_yticks(decomp["ypos"])
    ax.set_yticklabels(decomp["period"])
    ax.set_xlabel("Change in years")
    ax.set_xlim(-2.35, 6.65)
    ax.set_ylim(0.65, 3.35)
    polish_axis(ax, grid_axis="x")


def draw_gap_uncertainty_panel(ax):
    trend = pd.read_csv(os.path.join(TAB, "table_uncertainty_population_weighted_global_trend.csv"))
    ax.plot(trend["year"], trend["gap_point"], color=COLORS["gap"], linewidth=2.4)
    ax.fill_between(trend["year"], trend["gap_low"], trend["gap_high"], color=COLORS["gap_fill"], alpha=0.85)

    point_2000 = trend.loc[trend["year"] == 2000].iloc[0]
    point_2019 = trend.loc[trend["year"] == 2019].iloc[0]
    ax.scatter([2000, 2019], [point_2000["gap_point"], point_2019["gap_point"]], s=25, color=COLORS["gap"], edgecolor="white", zorder=3)
    ax.annotate(
        "2000: 8.75",
        xy=(2000, point_2000["gap_point"]),
        xytext=(2001.0, point_2000["gap_point"] - 0.18),
        arrowprops={"arrowstyle": "-", "color": COLORS["gap"], "lw": 0.8},
        fontsize=8.1,
        color=COLORS["gap"],
    )
    ax.annotate(
        "2019: 9.60",
        xy=(2019, point_2019["gap_point"]),
        xytext=(2014.4, point_2019["gap_point"] + 0.13),
        arrowprops={"arrowstyle": "-", "color": COLORS["gap"], "lw": 0.8},
        fontsize=8.1,
        color=COLORS["gap"],
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Gap (years)")
    ax.set_xlim(1999.4, 2021.6)
    ax.set_ylim(8.45, 9.95)
    polish_axis(ax, grid_axis="y")


def draw_conversion_uncertainty_panel(ax):
    baseline_le = pd.read_csv(os.path.join(TAB, "table_uncertainty_baseline_le_tertile_conversion_2000_2019.csv"))
    baseline_sdi = pd.read_csv(os.path.join(TAB, "table_uncertainty_baseline_sdi_tertile_conversion_2000_2019.csv"))

    le_order = ["Low baseline LE", "Mid baseline LE", "High baseline LE"]
    sdi_order = ["Low baseline SDI", "Mid baseline SDI", "High baseline SDI"]
    baseline_le["order"] = baseline_le["baseline_le_group"].map({name: idx for idx, name in enumerate(le_order)})
    baseline_sdi["order"] = baseline_sdi["sdi_group"].map({name: idx for idx, name in enumerate(sdi_order)})
    baseline_le = baseline_le.sort_values("order")
    baseline_sdi = baseline_sdi.sort_values("order")

    le_x = np.arange(3)
    sdi_x = np.arange(3) + 4
    ax.errorbar(
        le_x,
        baseline_le["conversion_share_point"],
        yerr=[
            baseline_le["conversion_share_point"] - baseline_le["conversion_share_low"],
            baseline_le["conversion_share_high"] - baseline_le["conversion_share_point"],
        ],
        fmt="o",
        color=COLORS["baseline_le"],
        ecolor=COLORS["baseline_le"],
        capsize=3.5,
        markersize=5.2,
    )
    ax.errorbar(
        sdi_x,
        baseline_sdi["conversion_share_point"],
        yerr=[
            baseline_sdi["conversion_share_point"] - baseline_sdi["conversion_share_low"],
            baseline_sdi["conversion_share_high"] - baseline_sdi["conversion_share_point"],
        ],
        fmt="o",
        color=COLORS["baseline_sdi"],
        ecolor=COLORS["baseline_sdi"],
        capsize=3.5,
        markersize=5.2,
    )

    ax.axhline(1.0, color=COLORS["axis"], linewidth=1.0, linestyle="--")
    ax.set_xticks(list(le_x) + list(sdi_x))
    ax.set_xticklabels(["Low LE", "Mid LE", "High LE", "Low SDI", "Mid SDI", "High SDI"], rotation=18, ha="right")
    ax.set_ylabel("Conversion share")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.70, 1.04)
    ax.set_xlim(-0.55, 6.55)
    polish_axis(ax, grid_axis="y")


def main():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.15),
        gridspec_kw={"width_ratios": [1.18, 1.07, 1.0], "wspace": 0.36},
    )
    draw_decomposition_panel(axes[0])
    draw_gap_uncertainty_panel(axes[1])
    draw_conversion_uncertainty_panel(axes[2])

    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        add_panel_label(ax, label)

    fig.savefig(os.path.join(FIG, "fig1_main_composite.png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(os.path.join(FIG, "fig1_main_composite.png"))


if __name__ == "__main__":
    main()
