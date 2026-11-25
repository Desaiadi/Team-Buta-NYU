"""Reusable plotting functions for the weather-sensitivity analysis.

Every function returns a matplotlib Figure. `save()` writes it to
reports/figures/ for the slides.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

FIG_DIR = config.REPORTS_DIR / "figures"
ACCENT = "#57068c"      # NYU violet
MUTED = "#9aa0a6"
sns.set_theme(style="whitegrid", context="notebook")


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG (200 dpi) under reports/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def weather_correlation_heatmap(panel: pd.DataFrame, weather_cols=("tmean", "precip", "snow", "wind_max")) -> plt.Figure:
    """Rank correlation between citywide daily counts of each type and the weather."""
    daily = panel.groupby(["date", "complaint_type"], observed=True)["n"].sum().unstack()
    weather = panel.groupby("date")[list(weather_cols)].first()
    ranks = daily.join(weather).rank()
    corr = ranks.corr().loc[daily.columns, list(weather_cols)].sort_values("tmean")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Spearman ρ"})
    ax.set_xticklabels(["Mean temp", "Precipitation", "Snowfall", "Max wind"][:len(weather_cols)], rotation=0)
    ax.set(xlabel="", ylabel="", title="Which complaints follow the weather? (daily, 2023–2025)")
    return fig


def heat_vs_temperature(daily: pd.DataFrame) -> plt.Figure:
    """Daily citywide HEAT/HOT WATER complaints vs overnight low, with binned means."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(daily["tmin"], daily["n"], s=10, alpha=0.35, color=MUTED, label="One day")
    bins = pd.cut(daily["tmin"], np.arange(0, 75, 5))
    means = daily.groupby(bins, observed=True).agg(t=("tmin", "mean"), n=("n", "mean"))
    ax.plot(means["t"], means["n"], "-o", color=ACCENT, lw=2.5, label="Average per 5°F bin")
    ax.set(xlabel="Overnight low (°F)", ylabel="Heat / hot-water complaints per day",
           title="Colder nights → more heat complaints (heat season, Oct–May)")
    ax.legend(frameon=False)
    return fig


def sewer_vs_rain(daily: pd.DataFrame) -> plt.Figure:
    """Average daily citywide Sewer complaints by rainfall bin."""
    bins = pd.cut(daily["precip"], [-0.01, 0.01, 0.25, 0.5, 1, 2, 10],
                  labels=["Dry", "0–0.25", "0.25–0.5", "0.5–1", "1–2", "2+"])
    g = daily.groupby(bins, observed=True)["n"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(g.index.astype(str), g["mean"], color=ACCENT)
    for i, (m, k) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m, f"{m:.0f}\n({k} days)", ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Rainfall that day (inches)", ylabel="Sewer complaints per day",
           title="Heavy rain multiplies sewer complaints")
    ax.margins(y=0.2)
    return fig


def sensitivity_map(sens: pd.DataFrame, centroids: pd.DataFrame, title: str, label: str) -> plt.Figure:
    """ZIP centroids colored by sensitivity; dot size = number of complaints."""
    d = sens.merge(centroids, on="incident_zip")
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(d["lon"], d["lat"], c=d["pct_effect"], s=np.sqrt(d["complaints"]) * 2.2,
                    cmap="magma_r", edgecolor="white", linewidth=0.5,
                    vmin=d["pct_effect"].quantile(0.05), vmax=d["pct_effect"].quantile(0.95))
    fig.colorbar(sc, ax=ax, shrink=0.7, label=label)
    ax.set_aspect(1 / np.cos(np.radians(40.7)))
    ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
    return fig


def sensitivity_ranking(sens: pd.DataFrame, n: int = 10, title: str = "", label: str = "") -> plt.Figure:
    """Most and least sensitive ZIPs with 95% confidence intervals."""
    d = pd.concat([sens.head(n), sens.tail(n)]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1))
    y = np.arange(len(d))
    colors = [ACCENT] * n + [MUTED] * n
    ax.errorbar(d["pct_effect"], y, xerr=[d["pct_effect"] - d["ci_low"], d["ci_high"] - d["pct_effect"]],
                fmt="none", ecolor="#c9c9c9", lw=2)
    ax.scatter(d["pct_effect"], y, c=colors[::-1], zorder=3)
    ax.axvline(sens["pct_effect"].median(), ls="--", color="black", lw=1, label="Median ZIP")
    ax.set_yticks(y, d["incident_zip"])
    ax.set(xlabel=label, title=title)
    ax.legend(frameon=False, loc="lower right")
    return fig


def neighbor_test_plot(results: dict[str, dict]) -> plt.Figure:
    """Observed nearest-neighbor gap vs the shuffled-pairs distribution, one panel per complaint type."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 3.8))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        ax.hist(r["permutation_gaps"], bins=40, color=MUTED, alpha=0.8, label="Random pairs (shuffled)")
        ax.axvline(r["observed_mean_gap"], color=ACCENT, lw=3, label="Actual neighbors")
        ax.set(title=f"{name}\np = {r['p_value']:.3f}", xlabel="Mean |sensitivity gap| (pct. points)", ylabel="")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Are neighboring ZIPs more alike than random ZIPs?", y=1.03)
    return fig


def building_era_plot(rate_ratios: pd.DataFrame) -> plt.Figure:
    """Rate ratios of heat complaints per apartment by building era (vs built 2000+)."""
    eras = rate_ratios.loc[rate_ratios.index.str.startswith("building_era")].copy()
    eras.index = eras.index.str.extract(r"\[T\.(.+)\]", expand=False)
    eras = pd.concat([pd.DataFrame({"rate_ratio": [1.0], "ci_low": [1.0], "ci_high": [1.0]}, index=["2000+"]), eras])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(eras))
    ax.bar(x, eras["rate_ratio"], color=[MUTED] + [ACCENT] * (len(eras) - 1))
    ax.errorbar(x, eras["rate_ratio"], yerr=[eras["rate_ratio"] - eras["ci_low"], eras["ci_high"] - eras["rate_ratio"]],
                fmt="none", ecolor="black", capsize=4)
    ax.axhline(1, color="black", lw=1)
    ax.set_xticks(x, eras.index)
    ax.set(xlabel="Year built", ylabel="× complaints per apartment vs 2000+",
           title="Older buildings: more heat complaints per apartment\n(controls: borough, building size, season)")
    return fig
