import argparse
import json
import os
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "raw")
PANEL = os.path.join(BASE, "data", "processed", "panel_master.csv")
FIG = os.path.join(BASE, "outputs", "figures")
TAB = os.path.join(BASE, "outputs", "tables")

LE60_RAW = os.path.join(RAW, "who_le60_uncertainty_raw.csv")
HALE60_RAW = os.path.join(RAW, "who_hale60_uncertainty_raw.csv")

SEX_LABELS = {
    "SEX_BTSX": "Both sexes",
    "SEX_FMLE": "Female",
    "SEX_MLE": "Male",
}

COLORS = {
    "le60": "#1f5aa6",
    "hale60": "#d97b2b",
    "gap60": "#2f4858",
    "ageing": "#5b7f3a",
    "female": "#8b4b8f",
    "male": "#2d7f8f",
    "axis": "#3b3b3b",
    "grid": "#dddddd",
}


os.makedirs(RAW, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": COLORS["axis"],
        "axes.linewidth": 0.8,
        "xtick.color": "#2f2f2f",
        "ytick.color": "#2f2f2f",
        "text.color": "#222222",
    }
)


def polish_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(length=3, color=COLORS["axis"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.7, alpha=0.8)
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


def fetch_who_indicator(indicator_code, output_path, value_name, refresh=False, top=1000):
    if os.path.exists(output_path) and not refresh:
        frame = pd.read_csv(output_path)
    else:
        rows = []
        skip = 0
        while True:
            filt = "Dim1 eq 'SEX_BTSX' or Dim1 eq 'SEX_FMLE' or Dim1 eq 'SEX_MLE'"
            url = (
                f"https://ghoapi.azureedge.net/api/{indicator_code}"
                f"?$filter={urllib.parse.quote(filt)}&$top={top}&$skip={skip}"
            )
            with urllib.request.urlopen(url, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            values = data.get("value", [])
            if not values:
                break
            rows.extend(values)
            if len(values) < top:
                break
            skip += top

        frame = pd.DataFrame(rows)
        frame = frame[["SpatialDim", "TimeDim", "Dim1", "NumericValue", "Low", "High"]].rename(
            columns={
                "SpatialDim": "iso3",
                "TimeDim": "year",
                "Dim1": "sex_code",
                "NumericValue": value_name,
                "Low": f"{value_name}_low",
                "High": f"{value_name}_high",
            }
        )
        frame["sex"] = frame["sex_code"].map(SEX_LABELS)
        frame.to_csv(output_path, index=False)

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)
    for column in [value_name, f"{value_name}_low", f"{value_name}_high"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "sex" not in frame.columns:
        frame["sex"] = frame["sex_code"].map(SEX_LABELS)
    return frame


def country_frame():
    panel = pd.read_csv(PANEL)
    country_panel = panel[panel["population_thousands"].notna()].copy()
    countries = sorted(country_panel["iso3"].dropna().unique())
    context = (
        country_panel.groupby("iso3", as_index=False)
        .agg(
            mean_old_age_share=("old_age_share", "mean"),
            population_thousands=("population_thousands", "mean"),
        )
        .dropna(subset=["mean_old_age_share"])
    )
    ranked = context["mean_old_age_share"].rank(method="first")
    context["ageing_tertile"] = pd.qcut(
        ranked,
        q=3,
        labels=["Low ageing", "Mid ageing", "High ageing"],
    )
    yearly_context = country_panel[["iso3", "year", "population_thousands", "old_age_share"]].copy()
    yearly_context["older_population_thousands"] = (
        yearly_context["population_thousands"] * yearly_context["old_age_share"] / 100.0
    )
    return countries, context[["iso3", "mean_old_age_share", "ageing_tertile"]], yearly_context


def build_age60_panel(refresh=False):
    countries, context, yearly_context = country_frame()
    le60 = fetch_who_indicator("WHOSIS_000015", LE60_RAW, "le60", refresh=refresh)
    hale60 = fetch_who_indicator("WHOSIS_000007", HALE60_RAW, "hale60", refresh=refresh)
    age60 = le60.merge(hale60, on=["iso3", "year", "sex_code", "sex"], how="inner")
    age60 = age60[age60["iso3"].isin(countries)].copy()
    age60 = age60.merge(context, on="iso3", how="left")
    age60 = age60.merge(yearly_context, on=["iso3", "year"], how="left")
    age60["gap60_years"] = age60["le60"] - age60["hale60"]
    age60["gap60_share"] = age60["gap60_years"] / age60["le60"]
    age60 = age60.sort_values(["sex_code", "iso3", "year"]).reset_index(drop=True)
    age60.to_csv(os.path.join(TAB, "table_age60_country_year_panel.csv"), index=False)
    return age60


def summarize_interval(age60, start_year=2000, end_year=2019):
    start = age60[age60["year"] == start_year].copy()
    end = age60[age60["year"] == end_year].copy()
    merged = start.merge(
        end,
        on=["iso3", "sex_code", "sex", "ageing_tertile"],
        suffixes=(f"_{start_year}", f"_{end_year}"),
        how="inner",
    )
    merged["delta_le60"] = merged[f"le60_{end_year}"] - merged[f"le60_{start_year}"]
    merged["delta_hale60"] = merged[f"hale60_{end_year}"] - merged[f"hale60_{start_year}"]
    merged["delta_gap60"] = merged[f"gap60_years_{end_year}"] - merged[f"gap60_years_{start_year}"]
    merged["conversion_share60"] = np.where(
        merged["delta_le60"] > 0,
        merged["delta_hale60"] / merged["delta_le60"],
        np.nan,
    )
    both = merged[merged["sex_code"] == "SEX_BTSX"].copy()
    ranked_le = both[f"le60_{start_year}"].rank(method="first")
    both["baseline_le60_tertile"] = pd.qcut(
        ranked_le,
        q=3,
        labels=["Low baseline LE60", "Mid baseline LE60", "High baseline LE60"],
    )
    ranked_hale = both[f"hale60_{start_year}"].rank(method="first")
    both["baseline_hale60_tertile"] = pd.qcut(
        ranked_hale,
        q=3,
        labels=["Low baseline HALE60", "Mid baseline HALE60", "High baseline HALE60"],
    )
    merged = merged.merge(
        both[["iso3", "baseline_le60_tertile", "baseline_hale60_tertile"]],
        on="iso3",
        how="left",
    )
    merged.to_csv(os.path.join(TAB, "table_age60_country_conversion_2000_2019.csv"), index=False)
    return merged


def group_summary(frame, group_col, label):
    rows = []
    for group, sub in frame.groupby(group_col, observed=True):
        rows.append(
            {
                "grouping": label,
                "group": str(group),
                "countries": int(sub["iso3"].nunique()),
                "le60_2000": sub["le60_2000"].mean(),
                "hale60_2000": sub["hale60_2000"].mean(),
                "gap60_2000": sub["gap60_years_2000"].mean(),
                "le60_2019": sub["le60_2019"].mean(),
                "hale60_2019": sub["hale60_2019"].mean(),
                "gap60_2019": sub["gap60_years_2019"].mean(),
                "delta_le60": sub["delta_le60"].mean(),
                "delta_hale60": sub["delta_hale60"].mean(),
                "delta_gap60": sub["delta_gap60"].mean(),
                "conversion_share60": sub["delta_hale60"].mean() / sub["delta_le60"].mean(),
                "mean_country_conversion_share60": sub["conversion_share60"].mean(),
                "countries_conversion_below_1": int((sub["conversion_share60"] < 1).sum()),
            }
        )
    return rows


def build_summaries(age60, interval):
    both = interval[interval["sex_code"] == "SEX_BTSX"].copy()
    sex = interval[interval["sex_code"].isin(["SEX_FMLE", "SEX_MLE"])].copy()

    rows = []
    rows.extend(group_summary(both.assign(all_countries="All countries"), "all_countries", "All"))
    rows.extend(group_summary(both, "ageing_tertile", "Ageing tertile"))
    rows.extend(group_summary(both, "baseline_le60_tertile", "Baseline LE60 tertile"))
    rows.extend(group_summary(both, "baseline_hale60_tertile", "Baseline HALE60 tertile"))
    rows.extend(group_summary(sex, "sex", "Sex"))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(TAB, "table_age60_conversion_summary_2000_2019.csv"), index=False)

    trend = (
        age60[age60["sex_code"] == "SEX_BTSX"]
        .groupby("year", as_index=False)
        .agg(
            countries=("iso3", "nunique"),
            le60=("le60", "mean"),
            hale60=("hale60", "mean"),
            gap60_years=("gap60_years", "mean"),
            gap60_share=("gap60_share", "mean"),
        )
    )
    trend["hale60_le60_ratio"] = trend["hale60"] / trend["le60"]
    trend.to_csv(os.path.join(TAB, "table_age60_global_trend_equal_country.csv"), index=False)

    weighted_base = age60[
        (age60["sex_code"] == "SEX_BTSX")
        & age60["older_population_thousands"].notna()
        & (age60["older_population_thousands"] > 0)
    ].copy()
    weighted_rows = []
    for year, sub in weighted_base.groupby("year"):
        weights = sub["older_population_thousands"].to_numpy(dtype=float)
        le60 = np.average(sub["le60"], weights=weights)
        hale60 = np.average(sub["hale60"], weights=weights)
        gap60 = le60 - hale60
        weighted_rows.append(
            {
                "year": int(year),
                "countries": int(sub["iso3"].nunique()),
                "older_population_thousands": float(weights.sum()),
                "le60_older_weighted": float(le60),
                "hale60_older_weighted": float(hale60),
                "gap60_older_weighted": float(gap60),
                "gap60_share_older_weighted": float(gap60 / le60),
                "hale60_le60_ratio_older_weighted": float(hale60 / le60),
            }
        )
    weighted_trend = pd.DataFrame(weighted_rows).sort_values("year")
    weighted_trend.to_csv(os.path.join(TAB, "table_age60_global_trend_older_weighted.csv"), index=False)

    weighted_summary_rows = []
    for start_year, end_year in [(2000, 2019), (2019, 2021), (2000, 2021)]:
        start = weighted_base[weighted_base["year"] == start_year].copy()
        end = weighted_base[weighted_base["year"] == end_year].copy()
        common = sorted(set(start["iso3"]).intersection(set(end["iso3"])))
        start = start[start["iso3"].isin(common)]
        end = end[end["iso3"].isin(common)]
        start_w = start["older_population_thousands"].to_numpy(dtype=float)
        end_w = end["older_population_thousands"].to_numpy(dtype=float)
        start_le = np.average(start["le60"], weights=start_w)
        start_hale = np.average(start["hale60"], weights=start_w)
        end_le = np.average(end["le60"], weights=end_w)
        end_hale = np.average(end["hale60"], weights=end_w)
        delta_le = end_le - start_le
        delta_hale = end_hale - start_hale
        weighted_summary_rows.append(
            {
                "period": f"{start_year}-{end_year}",
                "countries": int(len(common)),
                "le60_start_older_weighted": float(start_le),
                "hale60_start_older_weighted": float(start_hale),
                "gap60_start_older_weighted": float(start_le - start_hale),
                "le60_end_older_weighted": float(end_le),
                "hale60_end_older_weighted": float(end_hale),
                "gap60_end_older_weighted": float(end_le - end_hale),
                "delta_le60_older_weighted": float(delta_le),
                "delta_hale60_older_weighted": float(delta_hale),
                "delta_gap60_older_weighted": float((end_le - end_hale) - (start_le - start_hale)),
                "conversion_share60_older_weighted": float(delta_hale / delta_le),
            }
        )
    weighted_summary = pd.DataFrame(weighted_summary_rows)
    weighted_summary.to_csv(
        os.path.join(TAB, "table_age60_older_population_weighted_summary.csv"),
        index=False,
    )
    return summary, trend, weighted_trend, weighted_summary


def add_bar_labels(ax, bars, values, fmt="{:.1%}", pad=0.01):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            fontsize=9,
            color=COLORS["axis"],
            fontweight="bold",
        )


def build_figure(summary, trend):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    ax = axes[0, 0]
    ax.plot(trend["year"], trend["le60"], color=COLORS["le60"], lw=2.2, label="LE60")
    ax.plot(trend["year"], trend["hale60"], color=COLORS["hale60"], lw=2.2, label="HALE60")
    add_panel_label(ax, "(a)")
    ax.set_ylabel("Years after age 60")
    ax.set_xlabel("Year")
    ax.legend(frameon=False, loc="upper left")
    polish_axis(ax)

    ax = axes[0, 1]
    ax.plot(trend["year"], trend["gap60_years"], color=COLORS["gap60"], lw=2.2)
    add_panel_label(ax, "(b)")
    ax.set_ylabel("LE60-HALE60 gap (years)")
    ax.set_xlabel("Year")
    polish_axis(ax)

    ax = axes[1, 0]
    ageing = summary[summary["grouping"] == "Ageing tertile"].copy()
    order = ["Low ageing", "Mid ageing", "High ageing"]
    ageing["group"] = pd.Categorical(ageing["group"], categories=order, ordered=True)
    ageing = ageing.sort_values("group")
    bars = ax.barh(ageing["group"].astype(str), ageing["conversion_share60"], color=COLORS["ageing"], alpha=0.86)
    add_bar_labels(ax, bars, ageing["conversion_share60"])
    ax.axvline(1.0, color="#9a9a9a", lw=1.0, ls="--")
    ax.set_xlim(0.2, 1.02)
    add_panel_label(ax, "(c)")
    ax.set_xlabel("HALE60 gain share of LE60 gain")
    polish_axis(ax, grid_axis="x")

    ax = axes[1, 1]
    sex = summary[summary["grouping"] == "Sex"].copy()
    order = ["Female", "Male"]
    sex["group"] = pd.Categorical(sex["group"], categories=order, ordered=True)
    sex = sex.sort_values("group")
    colors = [COLORS["female"], COLORS["male"]]
    bars = ax.bar(sex["group"].astype(str), sex["conversion_share60"], color=colors, alpha=0.86)
    for bar, value in zip(bars, sex["conversion_share60"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["axis"],
        )
    ax.axhline(1.0, color="#9a9a9a", lw=1.0, ls="--")
    ax.set_ylim(0.2, 1.02)
    add_panel_label(ax, "(d)")
    ax.set_ylabel("HALE60 gain share of LE60 gain")
    polish_axis(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_age60_conversion_patterns.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build WHO age-60 LE/HALE secondary analysis.")
    parser.add_argument("--refresh", action="store_true", help="Re-download WHO GHO age-60 series.")
    args = parser.parse_args()

    age60 = build_age60_panel(refresh=args.refresh)
    interval = summarize_interval(age60)
    summary, trend, weighted_trend, weighted_summary = build_summaries(age60, interval)
    build_figure(summary, trend)
    print("Saved age-60 module outputs.")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(weighted_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
