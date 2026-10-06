# -*- coding: utf-8 -*-
"""
Temporal-vs-spatial trade-off across feature sets
=================================================

For each feature-set folder under BASE_RESULTS_DIR, this script:

1. Reads county_year_counts.csv.
2. Aggregates observed and probability-summed model counts across ALL years
   for each county.
3. Computes county-to-county spatial correlation (r) for:
      - class_1 (Minor)
      - class_2 (Moderate)
      - class_3 (Major)
      - any_outage (classes 1+2+3)
      - major_share (class_3 / all outage classes)
4. Reads feature_set_comparison.csv and averages the temporal metrics across:
      class_1, class_2, class_3, any_outage, major_share
5. Creates a 2D trade-off plot:
      x = mean temporal KGE
      y = mean spatial r
      small marker with distinct shape = feature set

Expected directory structure
----------------------------
<repository-root>/
    feature_set_comparison.csv
    results_conus404_4class_p80/
        minimal/
            county_year_counts.csv
        base/
            county_year_counts.csv
        static/
            county_year_counts.csv
        antecedent/
            county_year_counts.csv
        forward/
            county_year_counts.csv
        forward_plus/
            county_year_counts.csv
        full/
            county_year_counts.csv

If county_year_counts.csv is inside a nested evaluation/figures subfolder instead,
the script searches recursively within each feature-set folder and uses the first
matching file it finds.

Outputs
-------
tradeoff_outputs/
    temporal_summary.csv
    spatial_summary.csv
    spatial_by_series.csv
    temporal_spatial_tradeoff_summary.csv
    temporal_vs_spatial_tradeoff.png
    temporal_vs_spatial_tradeoff.pdf

Usage
-----
python make_temporal_spatial_tradeoff.py

or

python make_temporal_spatial_tradeoff.py \
  --base-results "results" \
  --temporal-csv "results/comparison_partial2024/feature_set_comparison.csv" \
  --out "results/comparison_partial2024/tradeoff_outputs"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_RESULTS_DIR = ROOT_DIR / "results"
RUN_SUFFIX = "_partial2024"
COMPARISON_DIR = BASE_RESULTS_DIR / "comparison_partial2024"
TEMPORAL_CSV = COMPARISON_DIR / "feature_set_comparison.csv"
OUTPUT_DIR = COMPARISON_DIR / "tradeoff_outputs"

FEATURE_SETS = [
    "minimal",
    "base",
    "static",
    "antecedent",
    "forward",
    "forward_plus",
    "relative",
    "forward_static",
    "antecedent_static",
    "full",
]

TEMPORAL_SERIES = [
    "class_1",
    "class_2",
    "class_3",
    "any_outage",
    "major_share",
]

# Spatial KGE is summarized across the same outage targets as temporal KGE.
# Class 0 (no outage) is intentionally excluded.
SPATIAL_SERIES_FOR_MEAN = [
    "class_1",
    "class_2",
    "class_3",
    "any_outage",
    "major_share",
]

SERIES_LABELS = {
    "class_1": "Minor",
    "class_2": "Moderate",
    "class_3": "Major",
    "any_outage": "Any outage",
    "major_share": "Major share",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def safe_corr(x, y):
    """Pearson r, returning NaN if there are too few valid/nonconstant values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def find_counts_file(feature_dir: Path) -> Path:
    """Find county_year_counts.csv inside a feature-set folder."""
    direct = feature_dir / "county_year_counts.csv"
    if direct.exists():
        return direct

    matches = sorted(feature_dir.rglob("county_year_counts.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No county_year_counts.csv found under: {feature_dir}"
        )

    if len(matches) > 1:
        print(f"  ! Multiple county_year_counts.csv files under {feature_dir.name}; using:")
        print(f"    {matches[0]}")
    return matches[0]


# -----------------------------------------------------------------------------
# Temporal metrics
# -----------------------------------------------------------------------------
def build_temporal_summary(temporal_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(temporal_csv)

    required = {
        "feature_set", "description", "series",
        "KGE", "r", "alpha", "beta", "skill_vs_seasonal"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Temporal CSV is missing required columns: {sorted(missing)}"
        )

    df = df[df["series"].isin(TEMPORAL_SERIES)].copy()

    summary = (
        df.groupby(["feature_set", "description"], as_index=False)
          .agg(
              n_temporal_series=("series", "size"),
              temporal_median_KGE=("KGE", "median"),
              temporal_median_r=("r", "median"),
              temporal_median_alpha=("alpha", "median"),
              temporal_median_beta=("beta", "median"),
              temporal_median_skill=("skill_vs_seasonal", "median"),
          )
    )

    return summary


def kge_components(obs, sim):
    """KGE and components for paired 1-D arrays."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0 or np.mean(obs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta


# -----------------------------------------------------------------------------
# Spatial metrics from county-year counts
# -----------------------------------------------------------------------------
def spatial_metrics_for_feature(feature_set: str, counts_file: Path):
    df = pd.read_csv(counts_file)

    required = {"ID"}
    for k in [1, 2, 3]:
        required.update({f"count_obs_{k}", f"count_model_{k}"})
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{counts_file} is missing columns: {sorted(missing)}"
        )

    # Aggregate the entire evaluation period for every county.
    numeric_cols = [
        f"count_{src}_{k}"
        for src in ["obs", "model"]
        for k in [1, 2, 3]
    ]
    county = df.groupby("ID", as_index=False)[numeric_cols].sum()

    # Derived spatial quantities.
    county["obs_any_outage"] = county[[f"count_obs_{k}" for k in [1, 2, 3]]].sum(axis=1)
    county["model_any_outage"] = county[[f"count_model_{k}" for k in [1, 2, 3]]].sum(axis=1)

    county["obs_major_share"] = np.where(
        county["obs_any_outage"] > 0,
        county["count_obs_3"] / county["obs_any_outage"],
        np.nan,
    )
    county["model_major_share"] = np.where(
        county["model_any_outage"] > 0,
        county["count_model_3"] / county["model_any_outage"],
        np.nan,
    )

    rows = []

    # Individual outage classes.
    for k in [1, 2, 3]:
        obs = county[f"count_obs_{k}"]
        mod = county[f"count_model_{k}"]
        kge, r, alpha, beta = kge_components(obs, mod)
        rows.append({
            "feature_set": feature_set,
            "series": f"class_{k}",
            "spatial_KGE": kge,
            "spatial_r": r,
            "spatial_alpha": alpha,
            "spatial_beta": beta,
            "n_counties": int((np.isfinite(obs) & np.isfinite(mod)).sum()),
            "observed_total": float(obs.sum()),
            "predicted_total": float(mod.sum()),
        })

    # Any-outage totals.
    kge, r, alpha, beta = kge_components(county["obs_any_outage"], county["model_any_outage"])
    rows.append({
        "feature_set": feature_set,
        "series": "any_outage",
        "spatial_KGE": kge,
        "spatial_r": r,
        "spatial_alpha": alpha,
        "spatial_beta": beta,
        "n_counties": int(len(county)),
        "observed_total": float(county["obs_any_outage"].sum()),
        "predicted_total": float(county["model_any_outage"].sum()),
    })

    # Major share across counties.
    kge, r, alpha, beta = kge_components(county["obs_major_share"], county["model_major_share"])
    rows.append({
        "feature_set": feature_set,
        "series": "major_share",
        "spatial_KGE": kge,
        "spatial_r": r,
        "spatial_alpha": alpha,
        "spatial_beta": beta,
        "n_counties": int(
            (np.isfinite(county["obs_major_share"]) &
             np.isfinite(county["model_major_share"])).sum()
        ),
        "observed_total": np.nan,
        "predicted_total": np.nan,
    })

    return pd.DataFrame(rows), county


def build_spatial_summary(base_results: Path, feature_sets, run_suffix=RUN_SUFFIX):
    """
    Read county_year_counts.csv from <feature_set><run_suffix>/.

    The logical feature-set name is retained in outputs, while the physical
    folder may be e.g. forward_partial2024/.

    Also report exactly which years are present so stale/non-merged runs are
    immediately visible.
    """
    detail_frames = []
    county_frames = []

    for feature_set in feature_sets:
        feature_dir = base_results / f"{feature_set}{run_suffix}"

        if not feature_dir.exists():
            print(f"  ! Feature-set folder not found; skipping: {feature_dir}")
            continue

        counts_file = find_counts_file(feature_dir)
        raw = pd.read_csv(counts_file, nrows=0)

        # Full read is already done inside spatial_metrics_for_feature, but we
        # inspect the year column here for diagnostics.
        year_info = ""
        try:
            y = pd.read_csv(counts_file, usecols=["year"])["year"].dropna().astype(int)
            years_present = sorted(y.unique().tolist())
            year_info = f" years={years_present}"
            if 2024 not in years_present:
                year_info += "  [WARNING: 2024 MISSING]"
        except Exception:
            years_present = []
            year_info = " years=UNKNOWN"

        print(f"  {feature_set:18s} -> {counts_file}{year_info}")

        detail, county = spatial_metrics_for_feature(feature_set, counts_file)
        detail["result_folder"] = feature_dir.name
        detail["counts_file"] = str(counts_file)
        detail["years_present"] = ",".join(map(str, years_present))
        detail_frames.append(detail)

        county["feature_set"] = feature_set
        county["result_folder"] = feature_dir.name
        county["years_present"] = ",".join(map(str, years_present))
        county_frames.append(county)

    if not detail_frames:
        raise RuntimeError(
            f"No usable county_year_counts.csv files were found under "
            f"<feature_set>{run_suffix} folders in {base_results}."
        )

    detail = pd.concat(detail_frames, ignore_index=True)
    counties = pd.concat(county_frames, ignore_index=True)

    use = detail[detail["series"].isin(SPATIAL_SERIES_FOR_MEAN)].copy()
    summary = (
        use.groupby("feature_set", as_index=False)
           .agg(
               n_spatial_series=("series", "size"),
               spatial_median_KGE=("spatial_KGE", "median"),
               spatial_median_r=("spatial_r", "median"),
               spatial_median_alpha=("spatial_alpha", "median"),
               spatial_median_beta=("spatial_beta", "median"),
           )
    )

    wide_r = (
        detail.pivot(index="feature_set", columns="series", values="spatial_r")
              .rename(columns=lambda c: f"spatial_r_{c}")
              .reset_index()
    )

    wide_kge = (
        detail.pivot(index="feature_set", columns="series", values="spatial_KGE")
              .rename(columns=lambda c: f"spatial_KGE_{c}")
              .reset_index()
    )

    summary = summary.merge(wide_r, on="feature_set", how="left")
    summary = summary.merge(wide_kge, on="feature_set", how="left")

    return summary, detail, counties


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def make_tradeoff_figure(summary_by_series: pd.DataFrame, outdir: Path):
    """
    Plot two targets for each feature set:
      - any_outage : one color
      - class_3    : another color

    x-axis = temporal KGE
    y-axis = spatial KGE
    marker size = temporal skill vs seasonal baseline

    Feature sets are distinguished by marker shape.
    Targets are distinguished by color.
    """

    wanted_series = ["any_outage", "class_3"]
    plot_df = summary_by_series[
        summary_by_series["series"].isin(wanted_series)
    ].copy()

    plot_df = plot_df.dropna(
        subset=["temporal_KGE", "spatial_KGE", "temporal_skill"]
    )

    if plot_df.empty:
        raise RuntimeError("No complete any_outage/class_3 rows are available for plotting.")

    # Different marker shape for each feature set (original style).
    marker_map = {
        "minimal": "o",
        "base": "s",
        "static": "^",
        "antecedent": "v",
        "forward": "D",
        "forward_plus": "P",
        "relative": "X",
        "forward_static": "d",
        "antecedent_static": "<",
        "full": "*",
    }

    # One color for each target.
    color_map = {
        "any_outage": "C0",
        "class_3": "C3",
    }

    series_label = {
        "any_outage": "Any outage",
        "class_3": "Major outage (class 3)",
    }

    # Marker size encodes temporal skill vs seasonal baseline.
    skill = plot_df["temporal_skill"].to_numpy()
    s_min, s_max = 45.0, 150.0
    sk_min, sk_max = np.nanmin(skill), np.nanmax(skill)

    def skill_to_size(v):
        if not np.isfinite(v):
            return s_min
        if np.isclose(sk_max, sk_min):
            return (s_min + s_max) / 2
        return s_min + (v - sk_min) / (sk_max - sk_min) * (s_max - s_min)

    fig, ax = plt.subplots(figsize=(7.8, 5.9))

    feature_sets = list(dict.fromkeys(plot_df["feature_set"].tolist()))

    # Plot every feature-set x target combination.
    for _, row in plot_df.iterrows():
        fs = row["feature_set"]
        ser = row["series"]
        ax.scatter(
            row["temporal_KGE"],
            row["spatial_KGE"],
            s=skill_to_size(row["temporal_skill"]),
            marker=marker_map.get(fs, "o"),
            color=color_map[ser],
            edgecolors="black",
            linewidths=0.65,
            alpha=0.9,
            zorder=3,
        )

    # Feature-set legend: shapes only.
    feature_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_map.get(fs, "o"),
            linestyle="none",
            markerfacecolor="0.70",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7.2,
            label=("forward+" if fs == "forward_plus" else fs),
        )
        for fs in feature_sets
    ]

    leg1 = ax.legend(
        handles=feature_handles,
        title="Feature set",
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=8.5,
        handletextpad=0.6,
        borderpad=0.7,
        labelspacing=0.45,
    )
    ax.add_artist(leg1)

    # Target legend: colors only, placed on the left middle under the feature set legend.
    target_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[s],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7.2,
            label=series_label[s],
        )
        for s in wanted_series
    ]

    leg2 = ax.legend(
        handles=target_handles,
        title="Target",
        loc="center left",
        bbox_to_anchor=(0.0, 0.45),
        frameon=True,
        fontsize=8,
        title_fontsize=8.5,
        handletextpad=0.6,
        borderpad=0.7,
        labelspacing=0.5,
    )
    ax.add_artist(leg2)

    # Marker-size legend for skill.
    if not np.isclose(sk_max, sk_min):
        skill_vals = np.linspace(sk_min, sk_max, 3)
    else:
        skill_vals = np.array([sk_min])

    size_handles = [
        ax.scatter(
            [], [],
            s=skill_to_size(v),
            marker="o",
            facecolor="0.78",
            edgecolor="black",
            linewidth=0.6,
            label=f"{v:.2f}",
        )
        for v in skill_vals
    ]

    ax.legend(
        handles=size_handles,
        title="Skill vs seasonal",
        loc="lower right",
        frameon=True,
        fontsize=7.5,
        title_fontsize=8,
        borderpad=0.6,
        labelspacing=0.7,
    )

    # Reference lines at zero KGE can be useful for interpretation.
    ax.axvline(0, color="0.55", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8, alpha=0.55)

    ax.set_xlabel("Temporal reconstruction (KGE)")
    ax.set_ylabel("Spatial reconstruction (KGE)")
    ax.grid(alpha=0.25)

    # Add a little breathing room around the data.
    x = plot_df["temporal_KGE"].to_numpy()
    y = plot_df["spatial_KGE"].to_numpy()
    xspan = max(x.max() - x.min(), 0.05)
    yspan = max(y.max() - y.min(), 0.05)
    ax.set_xlim(x.min() - 0.08 * xspan, x.max() + 0.10 * xspan)
    ax.set_ylim(y.min() - 0.10 * yspan, y.max() + 0.12 * yspan)

    fig.text(
        0.5,
        0.015,
        "Marker shape identifies the feature set; blue points show any outage and red points show major outages (class 3). "
        "Marker size represents temporal skill relative to the seasonal baseline.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(outdir / "temporal_vs_spatial_tradeoff_any_major.png", dpi=300)
    fig.savefig(outdir / "temporal_vs_spatial_tradeoff_any_major.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    base_results = Path(args.base_results)
    temporal_csv = Path(args.temporal_csv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_sets = args.feature_sets or FEATURE_SETS

    print("Reading temporal comparison:")
    print(f"  {temporal_csv}")
    temporal = build_temporal_summary(temporal_csv)

    print("\nTemporal CSV feature sets and years:")
    temporal_raw = pd.read_csv(temporal_csv)
    for fs in feature_sets:
        tfs = temporal_raw[temporal_raw["feature_set"] == fs]
        if tfs.empty:
            print(f"  ! {fs:18s}: missing from temporal comparison CSV")
            continue
        if "years_included" in tfs.columns:
            vals = tfs["years_included"].dropna().astype(str).unique().tolist()
            print(f"  {fs:18s}: years_included={vals[0] if vals else 'UNKNOWN'}")
        elif "n_years" in tfs.columns:
            print(f"  {fs:18s}: n_years={tfs['n_years'].iloc[0]}")
        else:
            print(f"  {fs:18s}: year metadata not available")

    print("\nReading county-year counts and calculating spatial metrics:")
    print(f"  result-folder suffix: {args.run_suffix}")
    spatial, spatial_detail, county_totals = build_spatial_summary(
        base_results, feature_sets, run_suffix=args.run_suffix
    )

    # Build a per-series table for the two-target trade-off figure.
    # Temporal metrics come directly from feature_set_comparison.csv;
    # spatial metrics come from county_year_counts.csv aggregated across years.
    temporal_detail = pd.read_csv(temporal_csv)
    temporal_detail = temporal_detail[
        temporal_detail["series"].isin(["any_outage", "class_3"])
    ][["feature_set", "series", "KGE", "skill_vs_seasonal"]].copy()
    temporal_detail = temporal_detail.rename(columns={
        "KGE": "temporal_KGE",
        "skill_vs_seasonal": "temporal_skill",
    })

    spatial_for_plot = spatial_detail[
        spatial_detail["series"].isin(["any_outage", "class_3"])
    ][["feature_set", "series", "spatial_KGE"]].copy()

    series_summary = temporal_detail.merge(
        spatial_for_plot,
        on=["feature_set", "series"],
        how="inner",
    )

    summary = temporal.merge(spatial, on="feature_set", how="inner")

    if summary.empty:
        raise RuntimeError(
            "No feature sets matched between feature_set_comparison.csv and the result folders."
        )

    summary["temporal_rank"] = summary["temporal_median_KGE"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    summary["spatial_rank"] = summary["spatial_median_KGE"].rank(
        ascending=False, method="min"
    ).astype("Int64")

    # Sort by temporal skill for a stable output table.
    summary = summary.sort_values("temporal_median_KGE", ascending=False).reset_index(drop=True)

    temporal.to_csv(outdir / "temporal_summary.csv", index=False)
    spatial.to_csv(outdir / "spatial_summary.csv", index=False)
    spatial_detail.to_csv(outdir / "spatial_by_series.csv", index=False)
    county_totals.to_csv(outdir / "county_totals_all_years.csv", index=False)
    summary.to_csv(outdir / "temporal_spatial_tradeoff_summary.csv", index=False)
    series_summary.to_csv(outdir / "tradeoff_any_outage_class3_points.csv", index=False)

    make_tradeoff_figure(series_summary, outdir)

    show_cols = [
        "feature_set",
        "temporal_median_KGE",
        "spatial_median_KGE",
        "temporal_median_r",
        "temporal_median_alpha",
        "temporal_median_skill",
        "temporal_rank",
        "spatial_rank",
    ]

    print("\nTemporal-spatial comparison:\n")
    print(summary[show_cols].round(3).to_string(index=False))

    best_t = summary.loc[summary["temporal_median_KGE"].idxmax()]
    best_s = summary.loc[summary["spatial_median_KGE"].idxmax()]

    print("\nBest temporal reconstruction (median KGE):")
    print(
        f"  {best_t.feature_set}: median temporal KGE = {best_t.temporal_median_KGE:.3f}"
    )
    print("Best spatial reconstruction (median KGE):")
    print(
        f"  {best_s.feature_set}: median spatial KGE = {best_s.spatial_median_KGE:.3f}"
    )
    print(f"\nOutputs written to: {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base-results",
        default=str(BASE_RESULTS_DIR),
        help="Folder containing the feature-set result folders.",
    )
    p.add_argument(
        "--temporal-csv",
        default=str(TEMPORAL_CSV),
        help="feature_set_comparison.csv path.",
    )
    p.add_argument(
        "--out",
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )
    p.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        help="Optional logical feature-set names (without _partial2024).",
    )
    p.add_argument(
        "--run-suffix",
        default=RUN_SUFFIX,
        help="Suffix appended to each feature-set folder (default: _partial2024).",
    )
    main(p.parse_args())
