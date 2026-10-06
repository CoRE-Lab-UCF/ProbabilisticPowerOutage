"""
Figures: annual variability and spatial pattern, observed vs predicted
=====================================================================

Reads daily_predictions.csv from the leave-one-year-out run and makes:

1. Fig_annual_observed_vs_predicted_water_year (PNG + PDF)
   Top row:
       observed and predicted TOTAL county-days for each complete water year
       (October-September), one panel per outage class (minor, moderate, major)
       and for all weather-related outage classes combined.

   Bottom row:
       percentage deviation = (predicted - observed) / observed * 100,
       with 95% intervals from resampling counties.

2. Fig_spatial_observed_vs_predicted (PNG + PDF)
   One row per outage class:
       observed county total,
       predicted county total,
       percentage difference,
       county-level observed-vs-predicted scatter.

Only days with outage data are used when outage_data_available exists.
Predicted totals are sums of daily class probabilities.

Usage
-----
python plot_annual_and_spatial.py
python plot_annual_and_spatial.py \
    --results results/full_partial2024
"""

import argparse
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon


# =============================================================================
# USER SETTINGS
# =============================================================================
# Repository layout used by the public code release.
# Defaults are relative to the repository; every path can be overridden by CLI.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RESULTS = os.path.join(PROJECT_ROOT, "results", "full_partial2024")

# Keep florida_counties.geojson next to this script, or pass --boundaries.
BOUNDARIES = os.path.join(
    PROJECT_ROOT, "data", "boundaries", "florida_counties.geojson"
)

STATE_FIPS = "12"
N_BOOT = 1000
SEED = 42

CLASSES = {
    1: "Minor (1-2 h)",
    2: "Moderate (2-8 h)",
    3: "Major ($\\geq$8 h)",
}

ANNUAL_PANELS = [
    (1, "Minor (1-2 h)"),
    (2, "Moderate (2-8 h)"),
    (3, "Major ($\\geq$8 h)"),
    ("any", "All outage classes"),
]

OBS_COLOR = "#3d3d3d"
MOD_COLOR = "#534AB7"
POS_COLOR = "#D85A30"  # overprediction
NEG_COLOR = "#378ADD"  # underprediction

DIFF_LIMIT = 50
ZERO_BASED_BARS = True

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


plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# =============================================================================
# Helpers
# =============================================================================
def norm_name(name):
    s = re.sub(r"\bsaint\b", "st", str(name).lower())
    return re.sub(r"[^a-z0-9]", "", re.sub(r"_?county$", "", s.strip()))


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {os.path.join(out, name)}.png / .pdf")


def load(results):
    path = os.path.join(results, "daily_predictions.csv")
    d = pd.read_csv(path, parse_dates=["Time"])

    required = [
        "Time", "ID", "year",
        "p_obs_1", "p_obs_2", "p_obs_3",
        "p_model_1", "p_model_2", "p_model_3",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns in daily_predictions.csv: {missing}")

    if "outage_data_available" in d.columns:
        d = d[d["outage_data_available"] == 1].copy()

    # Make combined weather-related outage probability/count.
    for src in ("obs", "model"):
        d[f"p_{src}_any"] = d[[f"p_{src}_{k}" for k in (1, 2, 3)]].sum(axis=1)

    return d.reset_index(drop=True)


def add_water_year(d):
    """
    Add water year using the October-September convention.

    Example:
        Oct-Dec 2015 + Jan-Sep 2016 -> water_year 2016.
    """
    d = d.copy()
    d["water_year"] = (
        d["Time"].dt.year + (d["Time"].dt.month >= 10).astype(int)
    )
    return d


def complete_water_year_subset(d):
    """
    Keep only complete October-September water years.

    Completeness requires all 12 calendar months to occur in the scored data.
    For a record beginning in 2015 and extending through September 2024,
    the expected complete sequence is WY2016-WY2024; WY2015 is incomplete
    because Oct-Dec 2014 are unavailable.
    """
    d = add_water_year(d)

    # Distinguish months by calendar year as well as month number.
    month_id = d["Time"].dt.year * 100 + d["Time"].dt.month
    present = (
        d.assign(_month_id=month_id)
         .groupby("water_year")["_month_id"]
         .nunique()
    )

    complete = sorted(
        int(y) for y in present.index[present >= 12]
    )

    if not complete:
        raise ValueError(
            "No complete October-September water years were found in "
            "daily_predictions.csv"
        )

    return d[d["water_year"].isin(complete)].copy(), complete


# =============================================================================
# Figure 1: annual variability, WATER YEARS (OCT-SEP), TOTAL county-days
# =============================================================================
def annual_figure(d, out):
    keys = [k for k, _ in ANNUAL_PANELS]
    cols = [f"p_{src}_{k}" for src in ("obs", "model") for k in keys]

    # Annual variability uses complete October-September water years.
    wy_data, years = complete_water_year_subset(d)

    print(f"  complete water years used: {years}")

    cy = wy_data.groupby(["ID", "water_year"])[cols].sum()
    counties = sorted(cy.index.get_level_values("ID").unique())

    idx = pd.MultiIndex.from_product(
        [counties, years], names=["ID", "water_year"]
    )

    # Missing county-year combinations get zero because there are no rows
    # contributing to that county-year in the scored dataset.
    C = (
        cy.reindex(idx, fill_value=0.0)
        .to_numpy()
        .reshape(len(counties), len(years), len(cols))
    )

    def totals(Cs):
        # Sum over counties -> [year, column]
        return Cs.sum(axis=0)

    point = totals(C)

    # County bootstrap for uncertainty in statewide totals/deviation.
    rng = np.random.default_rng(SEED)
    boots = np.empty((N_BOOT, len(years), len(cols)), dtype=float)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(counties), len(counties))
        boots[b] = totals(C[pick])

    rows = []
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(
        2,
        len(keys),
        height_ratios=[1.6, 1.0],
        hspace=0.12,
        wspace=0.35,
    )

    x = np.arange(len(years))

    for j, (k, title) in enumerate(ANNUAL_PANELS):
        io = cols.index(f"p_obs_{k}")
        im = cols.index(f"p_model_{k}")

        obs = point[:, io]
        mod = point[:, im]

        # Avoid divide-by-zero warnings for years/classes with zero observed total.
        dev = np.full_like(obs, np.nan, dtype=float)
        np.divide(
            100.0 * (mod - obs),
            obs,
            out=dev,
            where=obs != 0,
        )

        boot_obs = boots[:, :, io]
        boot_mod = boots[:, :, im]
        bdev = np.full_like(boot_obs, np.nan, dtype=float)
        np.divide(
            100.0 * (boot_mod - boot_obs),
            boot_obs,
            out=bdev,
            where=boot_obs != 0,
        )

        lo, hi = np.nanpercentile(bdev, [2.5, 97.5], axis=0)

        # KGE components across complete water years (October-September).
        if len(obs) >= 3 and np.nanstd(obs) > 0 and np.nanstd(mod) > 0:
            r = np.corrcoef(obs, mod)[0, 1]
            alpha = mod.std() / obs.std()
            beta = mod.mean() / obs.mean() if obs.mean() != 0 else np.nan
            kge = 1 - np.sqrt(
                (r - 1) ** 2 +
                (alpha - 1) ** 2 +
                (beta - 1) ** 2
            )
        else:
            r = alpha = beta = kge = np.nan

        # ----------------------- upper panel -----------------------
        ax = fig.add_subplot(gs[0, j])
        ax.bar(x - 0.2, obs, 0.4, color=OBS_COLOR, label="Observed")
        ax.bar(x + 0.2, mod, 0.4, color=MOD_COLOR, label="Predicted")

        top = np.nanmax(np.r_[obs, mod])
        bottom = np.nanmin(np.r_[obs, mod])

        if ZERO_BASED_BARS:
            ax.set_ylim(0, top * 1.18 if top > 0 else 1)
        else:
            span = top - bottom
            if span == 0:
                span = max(top, 1)
            ax.set_ylim(
                max(0, bottom - 0.35 * span),
                top + 0.15 * span,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_title(
            f"({'abcd'[j]}) {title}\n"
            f"KGE = {kge:.2f}, r = {r:.2f}",
            fontsize=8.5,
        )
        ax.grid(axis="y", alpha=0.3)

        if j == 0:
            ax.set_ylabel("Total county-days")
            ax.legend(frameon=False, loc="upper left", fontsize=7)

        # ----------------------- lower panel -----------------------
        ax2 = fig.add_subplot(gs[1, j])
        ax.set_xlim(-0.6, len(years) - 0.4)
        ax2.set_xlim(-0.6, len(years) - 0.4)
        ax.tick_params(axis="x", labelbottom=False)

        finite_dev = np.isfinite(dev)
        bar_colors = np.where(dev >= 0, POS_COLOR, NEG_COLOR)
        ax2.bar(
            x[finite_dev],
            dev[finite_dev],
            0.6,
            color=bar_colors[finite_dev],
            alpha=0.85,
        )

        finite_ci = finite_dev & np.isfinite(lo) & np.isfinite(hi)
        if finite_ci.any():
            ax2.errorbar(
                x[finite_ci],
                dev[finite_ci],
                yerr=[
                    dev[finite_ci] - lo[finite_ci],
                    hi[finite_ci] - dev[finite_ci],
                ],
                fmt="none",
                ecolor="k",
                lw=0.8,
                capsize=2,
            )

        ax2.axhline(0, color="k", lw=0.8)
        ax2.axhspan(-10, 10, color="grey", alpha=0.12, lw=0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(years, rotation=90)
        ax2.set_xlabel("Water year", fontsize=7.5)
        ax2.grid(axis="y", alpha=0.3)

        if j == 0:
            ax2.set_ylabel("Deviation (%)\n(pred. - obs.) / obs.")

        series_name = "any" if k == "any" else CLASSES[k].split(" (")[0].lower()
        for yi, yr in enumerate(years):
            rows.append({
                "series": series_name,
                "water_year": int(yr),
                "observed_total": float(obs[yi]),
                "predicted_total": float(mod[yi]),
                "deviation_pct": float(dev[yi]) if np.isfinite(dev[yi]) else np.nan,
                "deviation_ci_low": float(lo[yi]) if np.isfinite(lo[yi]) else np.nan,
                "deviation_ci_high": float(hi[yi]) if np.isfinite(hi[yi]) else np.nan,
            })



    save(fig, out, "Fig_annual_observed_vs_predicted_water_year")

    table = pd.DataFrame(rows)
    table.to_csv(
        os.path.join(out, "annual_observed_vs_predicted_water_year.csv"),
        index=False,
    )
    return table


# =============================================================================
# County geometry helpers
# =============================================================================
def load_counties(src):
    """Return county outlines as {ID: [rings as (N, 2) lon/lat arrays]}."""

    if str(src).startswith("http"):
        import urllib.request
        with urllib.request.urlopen(src, timeout=60) as r:
            data = json.load(r)
    else:
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"County boundary file not found: {src}. "
                "Put florida_counties.geojson next to this script or use --boundaries."
            )
        with open(src) as f:
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
        polys = (
            [geom["coordinates"]]
            if geom["type"] == "Polygon"
            else geom["coordinates"]
        )

        shapes.setdefault(cid, []).extend(
            np.asarray(p[0], float) for p in polys
        )

    missing = sorted(set(COUNTY_IDS.values()) - set(shapes))
    if missing:
        print(f"  ! counties without a boundary: {missing}")

    print(f"  county outlines loaded: {len(shapes)}")
    return shapes


def draw_map(ax, shapes, values, cmap, norm):
    """Fill each county by value; counties without values are grey."""

    patches = []
    colors = []

    for cid, rings in shapes.items():
        v = values.get(cid, np.nan)
        color = cmap(norm(v)) if np.isfinite(v) else (0.83, 0.83, 0.83, 1)

        for ring in rings:
            patches.append(Polygon(ring, closed=True))
            colors.append(color)

    pc = PatchCollection(
        patches,
        facecolor=colors,
        edgecolor="white",
        linewidth=0.25,
    )
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.deg2rad(28)))
    ax.set_axis_off()


# =============================================================================
# Figure 2: spatial pattern
# =============================================================================
def spatial_figure(d, out, boundaries):
    cols = [
        f"p_{src}_{k}"
        for src in ("obs", "model", "seasonal")
        for k in (1, 2, 3)
        if f"p_{src}_{k}" in d.columns
    ]

    tot = d.groupby("ID")[cols].sum()
    tot["county_days"] = d.groupby("ID").size()
    tot.to_csv(os.path.join(out, "county_totals_observed_vs_predicted.csv"))

    metrics = []

    for k, name in CLASSES.items():
        o = tot[f"p_obs_{k}"]
        m = tot[f"p_model_{k}"]

        dev = 100 * (m - o) / o.replace(0, np.nan)

        valid = o.notna() & m.notna()
        spatial_r = (
            np.corrcoef(o[valid], m[valid])[0, 1]
            if valid.sum() >= 2 and o[valid].std() > 0 and m[valid].std() > 0
            else np.nan
        )

        row = {
            "cls": name.split(" (")[0],
            "observed_total": float(o.sum()),
            "predicted_total": float(m.sum()),
            "spatial_r": spatial_r,
            "spatial_spearman": o.rank().corr(m.rank()),
            "median_abs_dev_pct": float(np.nanmedian(np.abs(dev))),
            "pct_counties_within_20pct": float(100 * np.nanmean(np.abs(dev) <= 20)),
        }

        if f"p_seasonal_{k}" in tot.columns:
            s = tot[f"p_seasonal_{k}"]
            vs = o.notna() & s.notna()
            row["spatial_r_seasonal_baseline"] = (
                np.corrcoef(o[vs], s[vs])[0, 1]
                if vs.sum() >= 2 and o[vs].std() > 0 and s[vs].std() > 0
                else np.nan
            )

        metrics.append(row)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(out, "spatial_metrics.csv"), index=False)

    shapes = load_counties(boundaries)
    g = tot.reset_index()

    # Slightly wider figure, compact vertical spacing, and dedicated spacer
    # before the county scatter column.
    fig = plt.figure(figsize=(9.2, 6.6))
    gs = fig.add_gridspec(
        3,
        8,
        width_ratios=[
            1.22,  # observed
            1.22,  # predicted
            0.09,  # observed/predicted colorbar
            0.08,  # spacer
            1.22,  # difference
            0.09,  # difference colorbar
            0.42,  # wider spacer before county totals
            1.05,  # county totals scatter
        ],
        wspace=0.08,
        hspace=0.04,
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.07,
    )

    for i, (k, name) in enumerate(CLASSES.items()):
        o = g[f"p_obs_{k}"]
        m = g[f"p_model_{k}"]

        # Same scale for observed and predicted maps within a class.
        vmax = np.nanpercentile(np.r_[o, m], 98)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1
        nrm = Normalize(0, vmax)

        # ---------------- observed + predicted ----------------
        for j, (col, label) in enumerate([
            (f"p_obs_{k}", "Observed"),
            (f"p_model_{k}", "Predicted"),
        ]):
            ax = fig.add_subplot(gs[i, j])
            draw_map(
                ax,
                shapes,
                dict(zip(g["ID"], g[col])),
                plt.get_cmap("YlOrRd"),
                nrm,
            )

            if i == 0:
                ax.set_title(label, fontsize=8.5)

            if j == 0:
                ax.text(
                    -0.04,
                    0.5,
                    name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8.5,
                    # intentionally not bold
                )

        # Shortened colorbar rather than full-row-height colorbar.
        cslot = fig.add_subplot(gs[i, 2])
        cslot.set_axis_off()
        cax = cslot.inset_axes([0.28, 0.16, 0.44, 0.68])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap="YlOrRd", norm=nrm),
            cax=cax,
        )
        cb.set_label("days (all available held-out days)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)

        # ---------------- difference map ----------------
        dev = 100 * (m - o) / o.replace(0, np.nan)
        dn = TwoSlopeNorm(
            vmin=-DIFF_LIMIT,
            vcenter=0,
            vmax=DIFF_LIMIT,
        )

        ax_diff = fig.add_subplot(gs[i, 4])
        draw_map(
            ax_diff,
            shapes,
            dict(zip(g["ID"], dev)),
            plt.get_cmap("RdBu_r"),
            dn,
        )
        if i == 0:
            ax_diff.set_title("Difference", fontsize=8.5)

        cslot = fig.add_subplot(gs[i, 5])
        cslot.set_axis_off()
        cax = cslot.inset_axes([0.28, 0.16, 0.44, 0.68])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap="RdBu_r", norm=dn),
            cax=cax,
            extend="both",
        )
        cb.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%")
        )
        cb.ax.tick_params(labelsize=6.5)

        # ---------------- county totals scatter ----------------
        # Use a normal GridSpec subplot rather than an inset. This keeps the
        # "County totals" title aligned with Observed/Predicted/Difference.
        ax_sc = fig.add_subplot(gs[i, 7])

        ok = o.notna() & m.notna()
        if ok.any():
            lim = max(o[ok].max(), m[ok].max()) * 1.08
        else:
            lim = 1
        if not np.isfinite(lim) or lim <= 0:
            lim = 1

        ax_sc.plot([0, lim], [0, lim], color="k", lw=0.7)
        ax_sc.scatter(
            o[ok],
            m[ok],
            s=9,
            color=MOD_COLOR,
            alpha=0.8,
            edgecolor="none",
        )
        ax_sc.set_xlim(0, lim)
        ax_sc.set_ylim(0, lim)

        mt = metrics.iloc[i]
        ax_sc.text(
            0.04,
            0.96,
            f"r = {mt.spatial_r:.2f}\n"
            f"{mt.pct_counties_within_20pct:.0f}% within $\\pm$20%",
            transform=ax_sc.transAxes,
            va="top",
            fontsize=6.5,
        )

        ax_sc.tick_params(labelsize=6)
        ax_sc.set_xlabel("observed", fontsize=7)
        ax_sc.set_ylabel("predicted", fontsize=7, labelpad=0)
        ax_sc.grid(alpha=0.3)

        if i == 0:
            ax_sc.set_title("County totals", fontsize=8.5)

    fig.text(
        0.5,
        0.02,
        "Totals over all available held-out days with outage data; difference = "
        "(predicted - observed) / observed; grey = no data.",
        ha="center",
        fontsize=7,
    )

    save(fig, out, "Fig_spatial_observed_vs_predicted")
    return metrics


# =============================================================================
# Main
# =============================================================================
def main(args):
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)

    d = load(args.results)

    print(
        f"{len(d):,} scored county-days, "
        f"{d['ID'].nunique()} counties, "
        f"calendar years present {sorted(d['year'].unique())}"
    )

    print("Annual variability figure (water year: Oct-Sep)")
    annual = annual_figure(d, out)

    display = annual.copy()
    display["deviation_pct"] = display["deviation_pct"].round(1)
    print(
        display[["series", "water_year", "deviation_pct"]]
        .pivot(index="water_year", columns="series", values="deviation_pct")
        .to_string()
    )

    print("Spatial figure")
    spatial = spatial_figure(d, out, args.boundaries)
    pd.set_option("display.width", 200)
    print(spatial.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results",
        default=RESULTS,
        help="folder containing daily_predictions.csv",
    )
    p.add_argument(
        "--boundaries",
        default=BOUNDARIES,
        help="Florida county GeoJSON file or URL",
    )
    main(p.parse_args())
