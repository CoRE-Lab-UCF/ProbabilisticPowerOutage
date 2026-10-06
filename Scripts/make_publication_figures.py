#!/usr/bin/env python3
"""
Create a consistent set of publication figures for the outage-frequency paper.

This script consolidates the figure-generation tasks that were previously spread
across compare_feature_sets.py, compare_spatial_temporal.py,
evaluate_frequency_updated.py, make_temporal_spatial_tradeoff.py, and
plot_annual_and_spatial.py.

DEFAULT OUTPUT
--------------
<results-root>/publication_figures/
    figures/   PNG + EPS
    data/      CSV files used by the figures

DEFAULT SEASONALITY CHANGE
--------------------------
The monthly seasonality figure now shows the MEAN monthly seasonal cycle.
For each calendar month, the script first aggregates the statewide outage
county-days within each year, then averages those month-specific totals across
all available years. This reduces the influence of incomplete 2024 coverage.

Use --seasonality-mode rate only if you want the old exposure-normalized view
(days per 1,000 county-days). Use --seasonality-mode totals only if you want
whole-record cumulative month totals.

EXAMPLES
--------
python make_publication_figures.py

python make_publication_figures.py \
    --results-root results \
    --model-results results/full_partial2024 \
    --boundaries /path/to/florida_counties.geojson

Generate only selected figures:
python make_publication_figures.py --only calibration annual seasonality seasons storms spatial tradeoff
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon

from figure_config import (
    apply_style, clean_axis, panel_label, save_figure,
    DOUBLE_COLUMN_WIDTH,
    OBS_COLOR, MODEL_COLOR, SEASONAL_COLOR, CLIM_COLOR,
    MAJOR_COLOR, MODERATE_COLOR, MINOR_COLOR,
    HURRICANE_SHADE, ONE_TO_ONE_COLOR,
    FEATURE_COLORS, FEATURE_MARKERS,
)

apply_style()

# By default, keep only the plotted content and the panel labels.
# Set from the command line if metric text should be written inside panels.
ANNOTATE_METRICS = False

# =============================================================================
# Repository-relative defaults
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
MODEL_RESULTS = RESULTS_ROOT / "full_partial2024"
RUN_SUFFIX = "_partial2024"

FEATURE_SETS = [
    "minimal", "base", "static", "antecedent", "forward", "forward_plus",
    "relative", "forward_static", "antecedent_static", "full",
]

FEATURE_LABELS = {
    "minimal": "Minimal",
    "base": "Base",
    "static": "Static",
    "antecedent": "Antecedent",
    "forward": "Forward",
    "forward_plus": "Forward+",
    "relative": "Relative",
    "forward_static": "Forward + static",
    "antecedent_static": "Antecedent + static",
    "full": "Full",
}

CLASSES = [0, 1, 2, 3]
OUTAGE_CLASSES = [1, 2, 3]
CLASS_TITLES = {
    0: "None",
    1: "Minor (1--2 h)",
    2: "Moderate (2--8 h)",
    3: r"Major ($\geq$8 h)",
    "any": "All outage classes",
}

CAL_BINS = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0]
CAL_MIN_N = 30

STD_SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
STD_SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

STORMS = [
    ("TS Colin", "2016-06-06", "2016-06-07"),
    ("Hermine", "2016-09-01", "2016-09-02"),
    ("Matthew", "2016-10-06", "2016-10-08"),
    ("TS Emily", "2017-07-31", "2017-07-31"),
    ("Irma", "2017-09-09", "2017-09-11"),
    ("STS Alberto", "2018-05-27", "2018-05-28"),
    ("TS Gordon", "2018-09-03", "2018-09-04"),
    ("Michael", "2018-10-09", "2018-10-11"),
    ("Dorian", "2019-09-02", "2019-09-04"),
    ("TS Nestor", "2019-10-18", "2019-10-19"),
    ("Isaias", "2020-08-01", "2020-08-02"),
    ("Sally", "2020-09-15", "2020-09-16"),
    ("Eta", "2020-11-08", "2020-11-12"),
    ("Elsa", "2021-07-06", "2021-07-07"),
    ("TS Fred", "2021-08-16", "2021-08-16"),
    ("TS Mindy", "2021-09-08", "2021-09-09"),
    ("TS Alex", "2022-06-03", "2022-06-04"),
    ("Ian", "2022-09-28", "2022-09-30"),
    ("Nicole", "2022-11-09", "2022-11-10"),
    ("Idalia", "2023-08-29", "2023-08-30"),
    ("Debby", "2024-08-04", "2024-08-05"),
    ("Helene", "2024-09-26", "2024-09-27"),
    ("Milton", "2024-10-09", "2024-10-10"),
]
STORM_PAD_DAYS = 1

STATE_FIPS = "12"
COUNTY_IDS = {
    "Alachua": "county_1", "Baker": "county_2", "Bay": "county_3", "Bradford": "county_4",
    "Brevard": "county_5", "Broward": "county_6", "Calhoun": "county_7", "Charlotte": "county_8",
    "Citrus": "county_9", "Clay": "county_10", "Collier": "county_11", "Columbia": "county_12",
    "DeSoto": "county_13", "Dixie": "county_14", "Duval": "county_15", "Escambia": "county_16",
    "Flagler": "county_17", "Franklin": "county_18", "Gadsden": "county_19", "Gilchrist": "county_20",
    "Glades": "county_21", "Gulf": "county_22", "Hamilton": "county_23", "Hardee": "county_24",
    "Hendry": "county_25", "Hernando": "county_26", "Highlands": "county_27", "Hillsborough": "county_28",
    "Holmes": "county_29", "Indian River": "county_30", "Jackson": "county_31", "Jefferson": "county_32",
    "Lafayette": "county_33", "Lake": "county_34", "Lee": "county_35", "Leon": "county_36",
    "Levy": "county_37", "Liberty": "county_38", "Madison": "county_39", "Manatee": "county_40",
    "Marion": "county_41", "Martin": "county_42", "Miami-Dade": "county_43", "Monroe": "county_44",
    "Nassau": "county_45", "Okaloosa": "county_46", "Okeechobee": "county_47", "Orange": "county_48",
    "Osceola": "county_49", "Palm Beach": "county_50", "Pasco": "county_51", "Pinellas": "county_52",
    "Polk": "county_53", "Putnam": "county_54", "Santa Rosa": "county_55", "Sarasota": "county_56",
    "Seminole": "county_57", "St. Johns": "county_58", "St. Lucie": "county_59", "Sumter": "county_60",
    "Suwannee": "county_61", "Taylor": "county_62", "Union": "county_63", "Volusia": "county_64",
    "Wakulla": "county_65", "Walton": "county_66", "Washington": "county_67",
}

# =============================================================================
# Core helpers
# =============================================================================
def read_daily(model_results: Path) -> pd.DataFrame:
    path = model_results / "daily_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    d = pd.read_csv(path, parse_dates=["Time"])
    if "outage_data_available" in d.columns:
        d = d[d["outage_data_available"] == 1].copy()

    if "year" not in d.columns:
        d["year"] = d["Time"].dt.year
    d["month"] = d["Time"].dt.month

    # Construct one-hot observed columns if an older file does not contain them.
    if "duration" in d.columns:
        for k in CLASSES:
            c = f"p_obs_{k}"
            if c not in d.columns:
                d[c] = (d["duration"] == k).astype(float)

    for src in ("obs", "model", "seasonal", "clim"):
        cols = [f"p_{src}_{k}" for k in OUTAGE_CLASSES]
        if all(c in d.columns for c in cols):
            d[f"p_{src}_any"] = d[cols].sum(axis=1)

    return d.reset_index(drop=True)


def kge_components(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0 or np.mean(obs) == 0:
        return dict(KGE=np.nan, r=np.nan, alpha=np.nan, beta=np.nan)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return dict(KGE=float(kge), r=float(r), alpha=float(alpha), beta=float(beta))


def skill_mse(obs, sim, baseline):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    baseline = np.asarray(baseline, float)
    ok = np.isfinite(obs) & np.isfinite(sim) & np.isfinite(baseline)
    if ok.sum() < 2:
        return np.nan
    mse = np.mean((sim[ok] - obs[ok]) ** 2)
    mse_b = np.mean((baseline[ok] - obs[ok]) ** 2)
    return float(1 - mse / mse_b) if mse_b > 0 else np.nan


def add_water_year(d):
    out = d.copy()
    out["water_year"] = out["Time"].dt.year + (out["Time"].dt.month >= 10).astype(int)
    return out


def keep_complete_water_years(d):
    d = add_water_year(d)
    month_id = d["Time"].dt.year * 100 + d["Time"].dt.month
    present = d.assign(_month_id=month_id).groupby("water_year")["_month_id"].nunique()
    years = sorted(int(y) for y in present.index[present >= 12])
    if not years:
        raise ValueError("No complete October--September water years were found.")
    return d[d["water_year"].isin(years)].copy(), years


def add_standard_seasons(d):
    out = d.copy()
    out["std_season"] = out["month"].map(STD_SEASON_MAP)
    out["std_season_year"] = out["year"] + (out["month"] == 12).astype(int)
    present = (
        out.groupby(["std_season_year", "std_season"])["month"]
        .nunique().reset_index(name="n_months")
    )
    complete = present[present["n_months"] == 3][["std_season_year", "std_season"]]
    return out.merge(complete, on=["std_season_year", "std_season"], how="inner")


def bootstrap_county_totals(cy, years, obs_col, model_col, n_boot=1000, seed=42):
    """
    cy: MultiIndex [ID, period] dataframe containing obs/model columns.
    Returns model-total 95% CI for every period after resampling counties.
    """
    counties = sorted(cy.index.get_level_values(0).unique())
    idx = pd.MultiIndex.from_product([counties, years], names=cy.index.names)
    z = cy[[obs_col, model_col]].reindex(idx, fill_value=0.0)
    arr = z.to_numpy().reshape(len(counties), len(years), 2)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(years)), float)
    for b in range(n_boot):
        pick = rng.integers(0, len(counties), len(counties))
        draws[b] = arr[pick, :, 1].sum(axis=0)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


# =============================================================================
# Figure: calibration
# =============================================================================
def make_calibration(d, figdir, datadir):
    rows = []
    summary = []

    for k in CLASSES:
        p = d[f"p_model_{k}"].to_numpy(float)
        y = d[f"p_obs_{k}"].to_numpy(float)
        idx = np.digitize(p, CAL_BINS[1:-1], right=True)

        for b in range(len(CAL_BINS) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append({
                "class": k,
                "bin_low": CAL_BINS[b],
                "bin_high": CAL_BINS[b + 1],
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_frequency": float(y[m].mean()),
            })

        obs = float(y.sum())
        exp = float(p.sum())
        summary.append({"class": k, "observed": obs, "expected": exp,
                        "obs_over_expected": obs / exp if exp > 0 else np.nan})

    cal = pd.DataFrame(rows)
    cal["gap"] = cal["observed_frequency"] - cal["mean_predicted"]
    sm = pd.DataFrame(summary)

    ece = []
    for k in CLASSES:
        c = cal[cal["class"] == k]
        w = c["n"] / c["n"].sum()
        ece.append(float((w * c["gap"].abs()).sum()))
    sm["ECE"] = ece

    cal.to_csv(datadir / "calibration_bins.csv", index=False)
    sm.to_csv(datadir / "calibration_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.25))
    for letter, ax, k in zip("abcd", axes.ravel(), CLASSES):
        c = cal[(cal["class"] == k) & (cal["n"] >= CAL_MIN_N)]
        lim = max(
            c["mean_predicted"].max() if len(c) else 0,
            c["observed_frequency"].max() if len(c) else 0,
            0.02,
        ) * 1.08
        ax.plot([0, lim], [0, lim], ls="--", color=ONE_TO_ONE_COLOR, lw=0.9)
        ax.plot(c["mean_predicted"], c["observed_frequency"],
                color=MODEL_COLOR, marker="o", lw=1.4, ms=4)
        row = sm[sm["class"] == k].iloc[0]
        panel_label(ax, letter, CLASS_TITLES[k])
        if ANNOTATE_METRICS:
            ax.text(0.04, 0.94,
                    f"ECE = {row.ECE:.3f}\nObs./Exp. = {row.obs_over_expected:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        clean_axis(ax, "both")

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.94,
                        wspace=0.28, hspace=0.34)
    save_figure(fig, figdir, "Fig_calibration_reliability")


# =============================================================================
# Figure: annual variability, complete water years
# =============================================================================
def make_annual(d, figdir, datadir, n_boot=1000, seed=42):
    wd, years = keep_complete_water_years(d)
    panels = [(1, CLASS_TITLES[1]), (2, CLASS_TITLES[2]),
              (3, CLASS_TITLES[3]), ("any", CLASS_TITLES["any"])]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.2))
    records = []

    for letter, ax, (k, title) in zip("abcd", axes.ravel(), panels):
        obs_col = f"p_obs_{k}"
        mod_col = f"p_model_{k}"
        seas_col = f"p_seasonal_{k}"

        cy = wd.groupby(["ID", "water_year"])[[obs_col, mod_col] +
             ([seas_col] if seas_col in wd.columns else [])].sum()

        obs = cy[obs_col].groupby("water_year").sum().reindex(years, fill_value=0).to_numpy()
        mod = cy[mod_col].groupby("water_year").sum().reindex(years, fill_value=0).to_numpy()
        lo, hi = bootstrap_county_totals(cy, years, obs_col, mod_col, n_boot, seed)

        met = kge_components(obs, mod)
        seas = None
        skill = np.nan
        if seas_col in cy.columns:
            seas = cy[seas_col].groupby("water_year").sum().reindex(years, fill_value=0).to_numpy()
            skill = skill_mse(obs, mod, seas)

        ax.plot(years, obs, color=OBS_COLOR, marker="o", label="Observed")
        ax.errorbar(
            years, mod,
            yerr=[np.maximum(mod - lo, 0), np.maximum(hi - mod, 0)],
            color=MODEL_COLOR, marker="s", lw=1.3, ms=4,
            capsize=2.5, label="Model (95% CI)"
        )
        if seas is not None:
            ax.plot(years, seas, color=SEASONAL_COLOR, ls="--", lw=1.0,
                    label="Seasonal baseline")

        panel_label(ax, letter, title)
        txt = (f"KGE = {met['KGE']:.2f}, $r$ = {met['r']:.2f}\n"
               f"$\\alpha$ = {met['alpha']:.2f}, $\\beta$ = {met['beta']:.2f}")
        if np.isfinite(skill):
            txt += f", skill = {skill:.2f}"
        if ANNOTATE_METRICS:
            ax.text(0.03, 0.94, txt, transform=ax.transAxes,
                    va="top", ha="left", fontsize=6.5)
        ax.set_xlabel("Water year")
        ax.set_ylabel("Outage county-days")
        ax.set_xticks(years)
        clean_axis(ax, "y")

        for y, o, m, l, h in zip(years, obs, mod, lo, hi):
            records.append({
                "series": str(k), "water_year": y, "observed": o, "predicted": m,
                "ci_low": l, "ci_high": h, **met, "skill_vs_seasonal": skill
            })

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=3)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94,
                        wspace=0.28, hspace=0.34)
    pd.DataFrame(records).to_csv(datadir / "annual_water_year_totals.csv", index=False)
    save_figure(fig, figdir, "Fig_annual_observed_vs_predicted_water_year")


# =============================================================================
# Figure: monthly seasonality
# =============================================================================
def make_seasonality(d, figdir, datadir, mode="mean"):
    """Create the monthly seasonal-cycle figure.

    mode='mean' (recommended manuscript view):
        Within each calendar year, sum statewide county-days for a month, then
        average those month-specific totals across the available years. This
        prevents the incomplete final year from dominating the seasonal cycle.

    mode='totals':
        Sum all county-day observed/predicted probabilities for each calendar
        month over the entire evaluation record.

    mode='rate':
        Exposure-normalized representation in days per 1,000 county-days.
    """
    months = np.arange(1, 13)
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.0))
    panels = [(1, CLASS_TITLES[1]), (2, CLASS_TITLES[2]),
              (3, CLASS_TITLES[3]), ("any", CLASS_TITLES["any"])]

    for letter, ax, (k, title) in zip("abcd", axes.ravel(), panels):
        table = []
        for month in months:
            g = d[d["month"] == month]
            n = len(g)
            row = {"month": month, "county_days": n}
            for src in ("obs", "model", "seasonal"):
                col = f"p_{src}_{k}"
                if col not in g.columns:
                    row[src] = np.nan
                    continue
                total = float(g[col].sum())
                if mode == "totals":
                    row[src] = total
                elif mode == "rate":
                    row[src] = 1000.0 * total / n if n else np.nan
                else:  # mode == "mean"
                    # Sum statewide county-days separately within each year,
                    # then average those monthly totals across available years.
                    row[src] = float(g.groupby("year")[col].sum().mean()) if n else np.nan
            row["n_years"] = int(g["year"].nunique()) if n else 0
            table.append(row)
        tab = pd.DataFrame(table)
        tab["series"] = str(k)
        rows.extend(tab.to_dict("records"))

        # Solid very-light background avoids EPS transparency warnings.
        ax.axvspan(5.5, 11.5, facecolor=HURRICANE_SHADE, edgecolor="none", zorder=0)
        ax.plot(months, tab["obs"], color=OBS_COLOR, marker="o", label="Observed", zorder=3)
        ax.plot(months, tab["model"], color=MODEL_COLOR, marker="s", label="Model", zorder=3)
        if tab["seasonal"].notna().any():
            ax.plot(months, tab["seasonal"], color=SEASONAL_COLOR, ls="--",
                    marker="^", ms=3.4, label="Seasonal baseline", zorder=3)

        panel_label(ax, letter, title)
        ax.set_xticks(months)
        ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_xlabel("Calendar month")
        if mode == "mean":
            ax.set_ylabel("Mean monthly outage county-days")
        elif mode == "totals":
            ax.set_ylabel("Total outage county-days")
        else:
            ax.set_ylabel("Outage days per 1,000 county-days")
        clean_axis(ax, "y")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=3)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94,
                        wspace=0.28, hspace=0.34)

    out = pd.DataFrame(rows)
    out.to_csv(datadir / f"seasonal_cycle_{mode}.csv", index=False)
    stem = {
        "mean": "Fig_seasonal_cycle_mean",
        "totals": "Fig_seasonal_cycle_entire_period",
        "rate": "Fig_seasonal_cycle_rate",
    }[mode]
    save_figure(fig, figdir, stem)


# =============================================================================
# Figure: standard meteorological seasons
# =============================================================================
def standard_season_data(d, series_key, n_boot=1000, seed=42):
    sd = add_standard_seasons(d)
    rows = []
    series = {}

    for season in STD_SEASON_ORDER:
        sub = sd[sd["std_season"] == season].copy()
        years = sorted(int(y) for y in sub["std_season_year"].unique())
        if len(years) < 3:
            continue

        cols = [f"p_obs_{series_key}", f"p_model_{series_key}"]
        if f"p_seasonal_{series_key}" in sub.columns:
            cols.append(f"p_seasonal_{series_key}")
        cy = sub.groupby(["ID", "std_season_year"])[cols].sum()

        obs = cy[f"p_obs_{series_key}"].groupby("std_season_year").sum().reindex(years).to_numpy()
        mod = cy[f"p_model_{series_key}"].groupby("std_season_year").sum().reindex(years).to_numpy()
        lo, hi = bootstrap_county_totals(
            cy, years, f"p_obs_{series_key}", f"p_model_{series_key}", n_boot, seed
        )

        seas = None
        skill = np.nan
        scol = f"p_seasonal_{series_key}"
        if scol in cy.columns:
            seas = cy[scol].groupby("std_season_year").sum().reindex(years).to_numpy()
            skill = skill_mse(obs, mod, seas)

        met = kge_components(obs, mod)
        series[season] = (years, obs, mod, seas, lo, hi, met, skill)
        rows.append({
            "season": season,
            "n_seasons": len(years),
            "years_included": ",".join(map(str, years)),
            **met,
            "skill_vs_seasonal": skill,
            "observed_total": float(np.sum(obs)),
            "predicted_total": float(np.sum(mod)),
        })

    return pd.DataFrame(rows), series


def make_standard_seasons(d, figdir, datadir, series_key, stem, n_boot=1000, seed=42):
    metrics, store = standard_season_data(d, series_key, n_boot, seed)
    metrics.to_csv(datadir / f"{stem}_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.3))
    width = 0.36

    for letter, ax, season in zip("abcd", axes.ravel(), STD_SEASON_ORDER):
        if season not in store:
            ax.set_axis_off()
            continue
        years, obs, mod, seas, lo, hi, met, skill = store[season]
        x = np.arange(len(years))

        ax.bar(x - width / 2, obs, width,
               facecolor="white", edgecolor=OBS_COLOR, linewidth=0.9,
               label="Observed")
        ax.bar(x + width / 2, mod, width,
               color=MODEL_COLOR, edgecolor=MODEL_COLOR, linewidth=0.5,
               label="Model")
        ax.errorbar(x + width / 2, mod,
                    yerr=[np.maximum(mod - lo, 0), np.maximum(hi - mod, 0)],
                    fmt="none", ecolor=OBS_COLOR, elinewidth=0.8,
                    capsize=2.0, zorder=4)
        if seas is not None:
            ax.plot(x, seas, color=SEASONAL_COLOR, ls="--", marker="^",
                    ms=3.1, lw=1.0, label="Seasonal baseline")

        panel_label(ax, letter, season)
        txt = (f"KGE = {met['KGE']:.2f}, $r$ = {met['r']:.2f}\n"
               f"$\\alpha$ = {met['alpha']:.2f}, $\\beta$ = {met['beta']:.2f}")
        if np.isfinite(skill):
            txt += f", skill = {skill:.2f}"
        if ANNOTATE_METRICS:
            ax.text(0.03, 0.94, txt, transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.4)

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha="right")
        ax.set_xlabel("Season-year")
        ax.set_ylabel("Seasonal outage county-days")
        clean_axis(ax, "y")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=3)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.94,
                        wspace=0.28, hspace=0.38)
    save_figure(fig, figdir, f"Fig_{stem}")


# =============================================================================
# Figure: tropical cyclones and regimes
# =============================================================================
def auc_rank(y, score):
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tag_storms(d):
    out = d.copy()
    out["tc_day"] = False
    out["storm"] = ""
    pad = pd.Timedelta(days=STORM_PAD_DAYS)
    for name, a, b in STORMS:
        mask = out["Time"].between(pd.Timestamp(a) - pad, pd.Timestamp(b) + pad)
        out.loc[mask, "tc_day"] = True
        out.loc[mask, "storm"] = name
    return out


def make_storms_regimes(d, figdir, datadir):
    d = tag_storms(d)
    storm_rows = []
    for name, a, b in STORMS:
        g = d[d["storm"] == name]
        if g.empty:
            continue
        r = {"storm": name, "start": a, "end": b, "county_days": len(g)}
        for src in ("obs", "model", "seasonal"):
            if all(f"p_{src}_{k}" in g.columns for k in OUTAGE_CLASSES):
                r[f"{src}_any"] = float(g[[f"p_{src}_{k}" for k in OUTAGE_CLASSES]].to_numpy().sum())
            if f"p_{src}_3" in g.columns:
                r[f"{src}_major"] = float(g[f"p_{src}_3"].sum())
        storm_rows.append(r)
    storms = pd.DataFrame(storm_rows)
    storms.to_csv(datadir / "storm_by_storm.csv", index=False)

    regimes = {
        "TC windows": d["tc_day"],
        "Hurricane season\n(non-TC)": d["month"].between(6, 11) & ~d["tc_day"],
        "Off-season": ~d["month"].between(6, 11) & ~d["tc_day"],
    }
    reg_rows = []
    if "duration" in d.columns:
        yall = d["duration"].to_numpy()
    else:
        yall = np.argmax(d[[f"p_obs_{k}" for k in CLASSES]].to_numpy(), axis=1)

    for rname, mask_s in regimes.items():
        mask = mask_s.to_numpy()
        yy = yall[mask]
        row = {"regime": rname, "county_days": int(mask.sum())}
        for src in ("model", "seasonal"):
            if not all(f"p_{src}_{k}" in d.columns for k in CLASSES):
                continue
            p = d.loc[mask, [f"p_{src}_{k}" for k in CLASSES]].to_numpy()
            occ = p[:, 1:].sum(axis=1)
            out_mask = yy > 0
            row[f"occ_auc_{src}"] = auc_rank(out_mask, occ)
            row[f"major_auc_{src}"] = (
                auc_rank(yy[out_mask] == 3, p[out_mask, 3] / np.clip(occ[out_mask], 1e-12, None))
                if out_mask.sum() > 10 else np.nan
            )
        reg_rows.append(row)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(datadir / "regime_skill.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.4))
    axa, axb, axc, axd = axes.ravel()

    # (a,b) storm-window scatter
    for letter, ax, what, title in [
        ("a", axa, "any", "All outage classes"),
        ("b", axb, "major", "Major outages"),
    ]:
        if not storms.empty:
            o = storms[f"obs_{what}"].to_numpy()
            m = storms[f"model_{what}"].to_numpy()
            s = storms[f"seasonal_{what}"].to_numpy() if f"seasonal_{what}" in storms else None
            vals = [o, m] + ([s] if s is not None else [])
            lim = max(np.nanmax(v) for v in vals) * 1.08 + 1
            ax.plot([0, lim], [0, lim], ls="--", color=ONE_TO_ONE_COLOR, lw=0.8)
            ax.scatter(o, m, s=22, color=MODEL_COLOR, marker="o", label="Model")
            if s is not None:
                ax.scatter(o, s, s=24, facecolor="none", edgecolor=SEASONAL_COLOR,
                           marker="s", label="Seasonal baseline")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
        panel_label(ax, letter, title)
        ax.set_xlabel("Observed storm-window county-days")
        ax.set_ylabel("Predicted county-days")
        clean_axis(ax, "both")

    # (c) AUC by regime
    if not reg.empty and "occ_auc_model" in reg.columns:
        x = np.arange(len(reg))
        w = 0.34
        axc.bar(x - w/2, reg["occ_auc_model"], w, color=MODEL_COLOR, label="Outage vs none")
        axc.bar(x + w/2, reg["major_auc_model"], w, color=MAJOR_COLOR, label="Major vs other")
        axc.set_xticks(x)
        axc.set_xticklabels(reg["regime"])
        axc.set_ylim(0.5, 1.0)
        axc.set_ylabel("Daily AUC")
        axc.legend(loc="lower left")
    panel_label(axc, "c", "Daily discrimination by regime")
    clean_axis(axc, "y")

    # (d) MAE model vs seasonal baseline across storm windows
    if not storms.empty and "seasonal_any" in storms.columns:
        maes = {
            "All outages": [
                np.mean(np.abs(storms["model_any"] - storms["obs_any"])),
                np.mean(np.abs(storms["seasonal_any"] - storms["obs_any"])),
            ],
            "Major": [
                np.mean(np.abs(storms["model_major"] - storms["obs_major"])),
                np.mean(np.abs(storms["seasonal_major"] - storms["obs_major"])),
            ],
        }
        x = np.arange(2)
        w = 0.34
        model_mae = [maes["All outages"][0], maes["Major"][0]]
        seas_mae = [maes["All outages"][1], maes["Major"][1]]
        axd.bar(x - w/2, model_mae, w, color=MODEL_COLOR, label="Model")
        axd.bar(x + w/2, seas_mae, w, facecolor="white",
                edgecolor=SEASONAL_COLOR, linewidth=1.1, label="Seasonal baseline")
        axd.set_xticks(x)
        axd.set_xticklabels(["All outages", "Major"])
        axd.set_ylabel("Storm-window MAE (county-days)")
        axd.legend()
    panel_label(axd, "d", "Storm-window error")
    clean_axis(axd, "y")

    handles, labels = axa.get_legend_handles_labels()
    if handles:
        axa.legend(loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.94,
                        wspace=0.30, hspace=0.40)
    save_figure(fig, figdir, "Fig_tropical_cyclones_and_regimes")


# =============================================================================
# Spatial figures
# =============================================================================
def spatial_totals(d):
    cols = []
    for src in ("obs", "model", "seasonal"):
        cols.extend([c for c in [f"p_{src}_{k}" for k in OUTAGE_CLASSES] if c in d.columns])
    tot = d.groupby("ID")[cols].sum()
    tot["county_days"] = d.groupby("ID").size()

    for src in ("obs", "model", "seasonal"):
        cls = [f"p_{src}_{k}" for k in OUTAGE_CLASSES]
        if all(c in tot.columns for c in cls):
            tot[f"p_{src}_any"] = tot[cls].sum(axis=1)
    return tot


def make_spatial_scatter(d, figdir, datadir):
    tot = spatial_totals(d)
    tot.to_csv(datadir / "county_totals_observed_vs_predicted.csv")

    panels = [(1, CLASS_TITLES[1]), (2, CLASS_TITLES[2]),
              (3, CLASS_TITLES[3]), ("any", CLASS_TITLES["any"])]
    metrics = []
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.2))

    for letter, ax, (k, title) in zip("abcd", axes.ravel(), panels):
        o = tot[f"p_obs_{k}"].to_numpy()
        m = tot[f"p_model_{k}"].to_numpy()
        met = kge_components(o, m)
        metrics.append({"series": str(k), **met,
                        "observed_total": np.nansum(o),
                        "predicted_total": np.nansum(m)})

        lim = max(np.nanmax(o), np.nanmax(m)) * 1.06
        ax.plot([0, lim], [0, lim], ls="--", color=ONE_TO_ONE_COLOR, lw=0.8)
        ax.scatter(o, m, s=18, color=MODEL_COLOR, edgecolor="white", linewidth=0.25)
        panel_label(ax, letter, title)
        if ANNOTATE_METRICS:
            ax.text(0.04, 0.94,
                    f"KGE = {met['KGE']:.2f}\n$r$ = {met['r']:.2f}, "
                    f"$\\alpha$ = {met['alpha']:.2f}, $\\beta$ = {met['beta']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=6.6)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Observed county total")
        ax.set_ylabel("Predicted county total")
        clean_axis(ax, "both")

    pd.DataFrame(metrics).to_csv(datadir / "spatial_kge_metrics.csv", index=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.94,
                        wspace=0.30, hspace=0.34)
    save_figure(fig, figdir, "Fig_spatial_county_totals")


def norm_name(name):
    s = re.sub(r"\bsaint\b", "st", str(name).lower())
    return re.sub(r"[^a-z0-9]", "", re.sub(r"_?county$", "", s.strip()))


def load_counties(src):
    src = Path(src)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup = {norm_name(k): v for k, v in COUNTY_IDS.items()}
    shapes = {}
    for ft in data["features"]:
        props = ft.get("properties", {})
        fips = str(props.get("FIPS", ft.get("id", props.get("GEOID", ""))))
        state = str(props.get("STATE", props.get("STATEFP", fips[:2]))).zfill(2)
        if state != STATE_FIPS:
            continue

        cid = lookup.get(norm_name(props.get("NAME", "")))
        if cid is None:
            continue

        geom = ft["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        shapes.setdefault(cid, []).extend(np.asarray(p[0], float) for p in polys)
    return shapes


def draw_map(ax, shapes, values, cmap, norm):
    patches, colors = [], []
    for cid, rings in shapes.items():
        v = values.get(cid, np.nan)
        color = cmap(norm(v)) if np.isfinite(v) else (0.85, 0.85, 0.85, 1)
        for ring in rings:
            patches.append(Polygon(ring, closed=True))
            colors.append(color)
    pc = PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.30)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.deg2rad(28)))
    ax.set_axis_off()


def make_spatial_difference_maps(d, boundaries, figdir, datadir, diff_limit=50):
    if boundaries is None or not Path(boundaries).exists():
        print("  ! Spatial maps skipped: county boundary GeoJSON was not found.")
        return

    tot = spatial_totals(d)
    shapes = load_counties(boundaries)
    panels = [(1, CLASS_TITLES[1]), (2, CLASS_TITLES[2]),
              (3, CLASS_TITLES[3]), ("any", CLASS_TITLES["any"])]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.0))
    norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0, vmax=diff_limit)
    cmap = plt.get_cmap("RdBu_r")
    rows = []

    for letter, ax, (k, title) in zip("abcd", axes.ravel(), panels):
        o = tot[f"p_obs_{k}"]
        m = tot[f"p_model_{k}"]
        dev = 100 * (m - o) / o.replace(0, np.nan)
        rows.extend({"ID": cid, "series": str(k), "difference_pct": val}
                    for cid, val in dev.items())
        draw_map(ax, shapes, dev.to_dict(), cmap, norm)
        panel_label(ax, letter, title)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes.ravel().tolist(), orientation="horizontal",
        fraction=0.045, pad=0.045, extend="both"
    )
    cbar.set_label("Prediction difference (%)")
    pd.DataFrame(rows).to_csv(datadir / "spatial_difference_pct.csv", index=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.94,
                        wspace=0.02, hspace=0.08)
    save_figure(fig, figdir, "Fig_spatial_difference_maps")


# =============================================================================
# Feature-set temporal-spatial trade-off
# =============================================================================
def find_feature_folder(root: Path, fs: str, suffix: str):
    exact = root / f"{fs}{suffix}"
    if exact.exists():
        return exact
    matches = sorted(p for p in root.glob(f"{fs}{suffix}*") if p.is_dir())
    return matches[0] if matches else None


def spatial_kge_from_counts(path, series):
    df = pd.read_csv(path)
    if series == "any_outage":
        obs = df[[f"count_obs_{k}" for k in OUTAGE_CLASSES]].sum(axis=1)
        mod = df[[f"count_model_{k}" for k in OUTAGE_CLASSES]].sum(axis=1)
        tmp = pd.DataFrame({"ID": df["ID"], "obs": obs, "mod": mod}).groupby("ID").sum()
    else:
        k = int(series.split("_")[1])
        tmp = df.groupby("ID")[[f"count_obs_{k}", f"count_model_{k}"]].sum()
        tmp.columns = ["obs", "mod"]
    return kge_components(tmp["obs"].to_numpy(), tmp["mod"].to_numpy())["KGE"]


def build_tradeoff_points(results_root: Path, suffix: str, datadir: Path):
    # Prefer the already-created point table from the user's current workflow.
    candidates = [
        results_root / "comparison_partial2024" / "tradeoff_outputs" / "tradeoff_any_outage_class3_points.csv",
        results_root / "comparison_partial2024" / "tradeoff_outputs" / "temporal_spatial_tradeoff_summary.csv",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            rename = {}
            if "temporal_median_KGE" in df.columns:
                rename["temporal_median_KGE"] = "temporal_KGE"
            if "temporal_median_skill" in df.columns:
                rename["temporal_median_skill"] = "temporal_skill"
            if "mean_spatial_KGE" in df.columns:
                rename["mean_spatial_KGE"] = "spatial_KGE"
            df = df.rename(columns=rename)
            required = {"feature_set", "series", "temporal_KGE", "temporal_skill", "spatial_KGE"}
            if required.issubset(df.columns):
                out = df[df["series"].isin(["any_outage", "class_3"])][list(required)].copy()
                out.to_csv(datadir / "tradeoff_points_used.csv", index=False)
                return out

    rows = []
    for fs in FEATURE_SETS:
        folder = find_feature_folder(results_root, fs, suffix)
        if folder is None:
            continue

        fm = folder / "frequency_metrics.csv"
        cc = folder / "county_year_counts.csv"
        if not (fm.exists() and cc.exists()):
            continue

        met = pd.read_csv(fm)
        if "predictor" in met.columns:
            met = met[met["predictor"] == "model"]

        for series in ("any_outage", "class_3"):
            r = met[met["series"] == series]
            if r.empty:
                continue
            rows.append({
                "feature_set": fs,
                "series": series,
                "temporal_KGE": float(r.iloc[0]["KGE"]),
                "temporal_skill": float(r.iloc[0]["skill_vs_seasonal"]),
                "spatial_KGE": spatial_kge_from_counts(cc, series),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(datadir / "tradeoff_points_used.csv", index=False)
    return out


def make_tradeoff(results_root, suffix, figdir, datadir):
    df = build_tradeoff_points(results_root, suffix, datadir)
    if df.empty:
        print("  ! Feature-set trade-off skipped: no compatible completed feature-set outputs.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.35))
    for letter, ax, series, title in [
        ("a", axes[0], "any_outage", "All outage classes"),
        ("b", axes[1], "class_3", "Major outages"),
    ]:
        sub = df[df["series"] == series]
        if sub.empty:
            ax.set_axis_off()
            continue

        skill = sub["temporal_skill"].to_numpy(float)
        smin, smax = np.nanmin(skill), np.nanmax(skill)

        def point_size(v):
            if not np.isfinite(v) or np.isclose(smin, smax):
                return 55
            return 35 + 60 * (v - smin) / (smax - smin)

        for row in sub.itertuples():
            fs = row.feature_set
            ax.scatter(
                row.temporal_KGE, row.spatial_KGE,
                s=point_size(row.temporal_skill),
                color=FEATURE_COLORS.get(fs, "#777777"),
                marker=FEATURE_MARKERS.get(fs, "o"),
                edgecolor="black", linewidth=0.45,
                label=FEATURE_LABELS.get(fs, fs),
                zorder=3,
            )
        panel_label(ax, letter, title)
        ax.set_xlabel("Temporal KGE")
        ax.set_ylabel("Spatial KGE")
        clean_axis(ax, "both")

    # One shared feature-set legend.
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=6.2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.91, wspace=0.28)
    save_figure(fig, figdir, "Fig_temporal_spatial_tradeoff")


# =============================================================================
# Main
# =============================================================================
def main(args):
    global ANNOTATE_METRICS
    ANNOTATE_METRICS = args.annotate_metrics
    results_root = Path(args.results_root)
    model_results = Path(args.model_results)
    out = Path(args.output) if args.output else results_root / "publication_figures"
    figdir = out / "figures"
    datadir = out / "data"
    figdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only) if args.only else {
        "calibration", "annual", "seasonality", "seasons",
        "storms", "spatial", "tradeoff",
    }

    needs_daily = bool(selected - {"tradeoff"})
    d = read_daily(model_results) if needs_daily else None

    print(f"\nPublication figures -> {out}")
    print(f"Model results       -> {model_results}")
    if d is not None:
        print(f"Scored county-days  -> {len(d):,}")
        print(f"Evaluation period   -> {d['Time'].min().date()} to {d['Time'].max().date()}")

    if "calibration" in selected:
        print("\nCalibration")
        make_calibration(d, figdir, datadir)

    if "annual" in selected:
        print("\nAnnual variability")
        make_annual(d, figdir, datadir, args.n_boot, args.seed)

    if "seasonality" in selected:
        print(f"\nMonthly seasonality ({args.seasonality_mode})")
        make_seasonality(d, figdir, datadir, args.seasonality_mode)

    if "seasons" in selected:
        print("\nStandard seasons")
        make_standard_seasons(d, figdir, datadir, "any",
                              "standard_seasons_any_outage_totals",
                              args.n_boot, args.seed)
        make_standard_seasons(d, figdir, datadir, 3,
                              "standard_seasons_major_totals",
                              args.n_boot, args.seed)

    if "storms" in selected:
        print("\nTropical cyclones and regimes")
        make_storms_regimes(d, figdir, datadir)

    if "spatial" in selected:
        print("\nSpatial performance")
        make_spatial_scatter(d, figdir, datadir)
        make_spatial_difference_maps(d, args.boundaries, figdir, datadir, args.diff_limit)

    if "tradeoff" in selected:
        print("\nFeature-set temporal-spatial trade-off")
        make_tradeoff(results_root, args.suffix, figdir, datadir)

    print("\nDone.")
    print(f"Figures: {figdir}")
    print(f"Data:    {datadir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=str(RESULTS_ROOT),
                   help="Parent folder containing the feature-set result folders.")
    p.add_argument("--model-results", default=str(MODEL_RESULTS),
                   help="Selected model folder containing daily_predictions.csv.")
    p.add_argument("--suffix", default=RUN_SUFFIX,
                   help="Feature-set folder suffix, default _partial2024.")
    p.add_argument("--output", default=None,
                   help="Output folder; default <results-root>/publication_figures.")
    p.add_argument("--boundaries", default=None,
                   help="Florida county GeoJSON. If omitted, the difference map is skipped.")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="County-bootstrap replicates for annual/seasonal confidence intervals.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diff-limit", type=float, default=50,
                   help="Symmetric percent-difference limit for county maps.")
    p.add_argument("--seasonality-mode", choices=["mean", "totals", "rate"], default="mean",
                   help="Default mean = mean monthly seasonal cycle; totals = whole-record cumulative month totals; rate restores the per-1000 view.")
    p.add_argument("--only", nargs="*",
                   choices=["calibration", "annual", "seasonality", "seasons",
                            "storms", "spatial", "tradeoff"],
                   help="Generate only selected figure groups. Omit for all.")
    p.add_argument("--annotate-metrics", action="store_true",
                   help="Write metric text blocks inside panels. Default is off for a cleaner journal-style figure.")
    main(p.parse_args())
