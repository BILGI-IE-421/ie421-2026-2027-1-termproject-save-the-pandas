"""
figures.py

Produces every static figure used on the site and saves it to visuals/.

The shared helpers (data loading, groups, bootstrap intervals) also live here.
The EDA notebook (scripts/eda_sonn.ipynb) imports them, so the notebook and the
site figures are built from exactly the same code and the same numbers.

Run from the repository root:
    python scripts/figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


# --------------------------------------------------
# 1. PATHS AND CONSTANTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_data.csv"
VISUALS_DIR = ROOT / "visuals"

GDP_COL = "GDP_per_capita_2015USD"
DEATH_COL = "deathsTotal"
DAMAGE_COL = "Total_Damage_2024_USD_Millions"

# Half-unit magnitude bands = "comparable magnitude" in RQ2
MAG_BINS = [0, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0]
MAG_ORDER = ["<6.0", "6.0-6.4", "6.5-6.9", "7.0-7.4", "7.5-7.9", "8.0+"]
GDP_LABELS = ["Q1 (poorest)", "Q2", "Q3", "Q4 (richest)"]

MIN_N = 10  # a band is compared only if both groups have at least this many events

# Colour-blind-safe palette; light -> dark for poorer -> richer
BLUE, ORANGE = "#0072B2", "#E69F00"
BLUES = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
ORANGES = ["#FDD9A0", "#F5A623", "#D17A00", "#8A4B00"]

SOURCE_NOTE = "Sources: NOAA Significant Earthquakes, World Bank GDP per capita, FRED CPI."


# --------------------------------------------------
# 2. DATA
# --------------------------------------------------

def load_data(path=DATA_PATH):
    """Read the cleaned data written by scripts/preprocess.py."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.py and "
            "scripts/preprocess.py first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def add_groups(df):
    """Add the derived columns used by the notebook and the figures."""
    df = df.copy()
    df["mag_band"] = pd.cut(df["eqMagnitude"], MAG_BINS, labels=MAG_ORDER, right=False)
    df["decade"] = df["year"] // 10 * 10
    # Income quarters; events without a GDP value stay empty (NaN)
    df["gdp_quartile"] = pd.qcut(df[GDP_COL], 4, labels=GDP_LABELS)
    return df


# --------------------------------------------------
# 3. BOOTSTRAP HELPERS (also imported by the EDA notebook)
# --------------------------------------------------

def median_ci(values, reps=2000, seed=421):
    """Median and 95% bootstrap interval (resampling events with replacement)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, (reps, len(v)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return np.median(v), lo, hi


def spearman_ci(x, y, reps=2000, seed=421):
    """Spearman rho and 95% bootstrap interval."""
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        k = rng.integers(0, len(x), len(x))
        boots.append(x.iloc[k].reset_index(drop=True).corr(y.iloc[k].reset_index(drop=True), method="spearman"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return x.corr(y, method="spearman"), lo, hi


def median_diff_ci(a, b, reps=2000, seed=421):
    """Median(a) - median(b) with a 95% bootstrap interval; each group is resampled separately."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = (np.median(rng.choice(a, (reps, len(a)), replace=True), axis=1)
             - np.median(rng.choice(b, (reps, len(b)), replace=True), axis=1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return np.median(a) - np.median(b), lo, hi


def spearman_cluster_ci(data, xcol, ycol, cluster="ISO3", reps=2000, seed=421):
    """Spearman rho with a 95% cluster bootstrap: whole countries are resampled, so events
    from the same country stay together."""
    groups = [g for _, g in data.groupby(cluster)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        sample = pd.concat([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        boots.append(sample[xcol].corr(sample[ycol], method="spearman"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return data[xcol].corr(data[ycol], method="spearman"), lo, hi, len(groups)


# --------------------------------------------------
# 4. FIGURES
# --------------------------------------------------

def _style(ax):
    """Light horizontal grid, no vertical grid, no top/right frame."""
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# Look of figure 1: the seaborn "whitegrid" look, set with plain matplotlib settings
# (applied only inside fig1_rq1, so other plots in the notebook are not affected)
_WHITEGRID = {
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": ".8", "axes.linewidth": 1.25,
    "axes.labelcolor": ".15", "text.color": ".15",
    "xtick.color": ".15", "ytick.color": ".15",
    "xtick.bottom": False, "ytick.left": False,
    "axes.grid": True, "grid.color": ".8", "grid.linewidth": 1, "axes.axisbelow": True,
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
}
FIG1_LABELS = ["Q1 (Poorest)", "Q2", "Q3", "Q4 (Richest)"]   # labels shown on the axis


def fig1_rq1(df):
    """RQ1: reported deaths and damage by income group (median, 95% bootstrap interval)."""
    r1 = df[df["gdp_quartile"].notna()]
    panels = [
        (r1[r1[DEATH_COL] > 0], DEATH_COL, BLUES,
         "Median Reported Deaths by GDP Group", "Median Deaths (people)"),
        (r1[r1[DAMAGE_COL] > 0], DAMAGE_COL, ORANGES,
         "Median Reported Economic Damage by GDP Group", "Median Damage (million 2024 USD)"),
    ]

    with plt.rc_context(_WHITEGRID):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, (data, col, colors, title, ylabel) in zip(axes, panels):
            stats = []
            for q in GDP_LABELS:
                values = data.loc[data["gdp_quartile"] == q, col]
                m, lo, hi = median_ci(values)
                stats.append((m, lo, hi, len(values)))
            med, lo, hi, n = (np.array(x) for x in zip(*stats))

            x = np.arange(len(GDP_LABELS))
            ax.bar(x, med, color=colors, edgecolor="black", linewidth=0.7,
                   yerr=[med - lo, hi - med], capsize=5, error_kw={"elinewidth": 1.2})
            for i in x:  # median written above the whisker
                ax.text(i, hi[i] + hi.max() * 0.02, f"{med[i]:.1f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

            ax.set_xticks(x, [f"{q}\n(n={int(k)})" for q, k in zip(FIG1_LABELS, n)])
            ax.set_ylim(0, hi.max() * 1.15)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("GDP per Capita Group")
            ax.set_ylabel(ylabel)
            ax.grid(False, axis="x")                 # no vertical grid lines
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle("Human and Economic Impacts of Earthquakes Across GDP Groups",
                     fontsize=16, fontweight="bold", y=1.03)
        fig.text(0.5, -0.04,
                 "Bars: median. Whiskers: 95% bootstrap interval of the median. Only events with a reported "
                 "value and a World Bank GDP value are included; n = number of events.\n"
                 "Sources: NOAA Significant Earthquakes, World Bank GDP per Capita, FRED CPI.",
                 ha="center", fontsize=9)
        fig.tight_layout()
    return fig



def fig2_rq2_relationship(df):
    """RQ2: relationship between earthquake magnitude and reported human losses by tsunami association."""

    rq2_relationship = df[
        (df["eqMagnitude"].between(7.0, 9.0)) &
        (df["tsunami"].isin([0, 1])) &
        (df["Death_Status"] == "Deaths Reported") &
        (df["deathsTotal"].notna()) &
        (df["deathsTotal"] > 0)
    ].copy()

    bins = np.arange(7.0, 9.01, 0.25)

    rq2_relationship["Magnitude Range"] = pd.cut(
        rq2_relationship["eqMagnitude"],
        bins=bins,
        include_lowest=True
    )

    rq2_relationship["Magnitude Midpoint"] = (
        rq2_relationship["Magnitude Range"]
        .apply(lambda x: x.mid)
        .astype(float)
    )

    summary = (
        rq2_relationship
        .groupby(["Magnitude Midpoint", "tsunami"], observed=True)
        .agg(
            mean_deaths=("deathsTotal", "mean"),
            count=("deathsTotal", "count")
        )
        .reset_index()
    )

    colors = {1: "#0072B2", 0: "#E69F00"}
    labels = {
        1: "Tsunami-associated",
        0: "No recorded tsunami association"
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for group in [1, 0]:
        data = summary[
            summary["tsunami"] == group
        ].sort_values("Magnitude Midpoint")

        ax.plot(
            data["Magnitude Midpoint"],
            data["mean_deaths"],
            color=colors[group],
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=labels[group]
        )

    ax.set_xlim(7.0, 9.0)
    ax.set_xticks(np.arange(7.0, 9.01, 0.25))
    ax.set_ylim(bottom=5)

    ax.set_xlabel("Earthquake Magnitude")
    ax.set_ylabel("Average Number of Reported Deaths (people)")
    ax.set_title(
        "Relationship Between Earthquake Magnitude and Human Losses",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(alpha=0.25)
    ax.legend(title="Tsunami Association")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Note: Y-axis starts at 5 deaths. Only earthquakes with reported positive death counts are included.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

FIGURES = {
    "fig1_rq1_income_groups.png": fig1_rq1,
    "fig2_rq2_magnitude_human_losses.png": fig2_rq2_relationship,
}


def main():
    df = add_groups(load_data())
    VISUALS_DIR.mkdir(exist_ok=True)

    for filename, make_figure in FIGURES.items():
        fig = make_figure(df)
        path = VISUALS_DIR / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
