"""
Compare feature sets using partial-2024 LOYO results
====================================================

Reads frequency_metrics.csv and yearly_series.csv from each feature set's
*_partial2024 results folder and writes to:

    <results_root>/comparison_partial2024/

Outputs
-------
  feature_set_comparison.csv
  feature_set_summary.csv
  Fig_feature_sets_partial2024.png/.pdf
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository layout used by the public code release.
# Defaults are relative to the repository; every path can be overridden by CLI.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

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

RUN_SUFFIX = "_partial2024"

DESCRIPTIONS = {
    "minimal": "label variables only (wind, daily rain)",
    "base": "all same-day weather + season",
    "static": "base + county attributes",
    "antecedent": "base + previous days / wetness history",
    "forward": "base + following days / storm windows",
    "forward_plus": "base + forward + weather interactions (no static, no antecedent)",
    "relative": "base + county-normalized features",
    "forward_static": "base + forward + county attributes",
    "antecedent_static": "base + antecedent + county attributes",
    "full": "everything + interactions and county-relative",
}

SERIES = ["class_1", "class_2", "class_3", "any_outage", "major_share"]

TITLES = {
    "class_1": "Minor (1-2 h)",
    "class_2": "Moderate (2-8 h)",
    "class_3": "Major ($\\geq$8 h)",
    "any_outage": "All outage classes",
    "major_share": "Major share",
}

COLORS = {
    "minimal": "#BBBBBB",
    "base": "#888780",
    "static": "#E69F00",
    "antecedent": "#0072B2",
    "forward": "#009E73",
    "forward_plus": "#56B4E9",
    "relative": "#CC79A7",
    "forward_static": "#117733",
    "antecedent_static": "#332288",
    "full": "#534AB7",
}

RATE = 1000

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


def folder_for(results_root, fs, suffix):
    exact = os.path.join(results_root, fs + suffix)

    if (
        os.path.exists(os.path.join(exact, "frequency_metrics.csv"))
        and os.path.exists(os.path.join(exact, "yearly_series.csv"))
    ):
        return exact

    prefix = fs + suffix
    candidates = []

    if os.path.isdir(results_root):
        for d in sorted(os.listdir(results_root)):
            full = os.path.join(results_root, d)
            if not os.path.isdir(full):
                continue
            if d == prefix or d.startswith(prefix + "_"):
                if (
                    os.path.exists(os.path.join(full, "frequency_metrics.csv"))
                    and os.path.exists(os.path.join(full, "yearly_series.csv"))
                ):
                    candidates.append(full)

    return candidates[0] if candidates else None


def main(args):
    out = os.path.join(args.results_root, args.output_dir)
    os.makedirs(out, exist_ok=True)

    rows = []
    series_data = {}

    print("\nReading partial-2024 feature-set results:")
    for fs in FEATURE_SETS:
        d = folder_for(args.results_root, fs, args.suffix)

        if d is None:
            print(f"  ! {fs}: no completed results found for '{fs}{args.suffix}'")
            continue

        print(f"  {fs:18s} -> {os.path.basename(d)}")

        m = pd.read_csv(os.path.join(d, "frequency_metrics.csv"))
        m = m[m["predictor"] == "model"].copy()

        ys = pd.read_csv(os.path.join(d, "yearly_series.csv"))
        series_data[fs] = ys

        fold_path = os.path.join(d, "fold_diagnostics.csv")
        if os.path.exists(fold_path):
            n_years = len(pd.read_csv(fold_path))
        else:
            n_years = ys["year"].nunique() if "year" in ys.columns else np.nan

        years_included = (
            ",".join(map(str, sorted(ys["year"].dropna().astype(int).unique())))
            if "year" in ys.columns
            else ""
        )

        for s in SERIES:
            r = m[m["series"] == s]
            if r.empty:
                continue
            r = r.iloc[0]

            rows.append({
                "feature_set": fs,
                "result_folder": os.path.basename(d),
                "description": DESCRIPTIONS[fs],
                "series": s,
                "n_years": n_years,
                "years_included": years_included,
                "KGE": r.KGE,
                "r": r.r,
                "alpha": r.alpha,
                "beta": r.beta,
                "skill_vs_seasonal": r.skill_vs_seasonal,
                "mean_abs_pct_err": r.mean_abs_pct_err,
            })

    if not rows:
        raise SystemExit(
            f"No finished partial-2024 feature sets were found in {args.results_root}"
        )

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(out, "feature_set_comparison.csv"), index=False)

    available_sets = [fs for fs in FEATURE_SETS if fs in set(comp["feature_set"])]

    pd.set_option("display.width", 220)

    print("\nKGE by feature set:")
    print(
        comp.pivot(index="series", columns="feature_set", values="KGE")
        .reindex(index=SERIES, columns=available_sets)
        .round(3)
        .to_string()
    )

    print("\nMSE skill vs seasonal baseline:")
    print(
        comp.pivot(index="series", columns="feature_set", values="skill_vs_seasonal")
        .reindex(index=SERIES, columns=available_sets)
        .round(3)
        .to_string()
    )

    class_comp = comp[comp["series"].isin(["class_1", "class_2", "class_3"])]

    mean_kge = (
        class_comp.groupby("feature_set")["KGE"]
        .mean()
        .reindex(available_sets)
        .sort_values(ascending=False)
    )

    print("\nMean KGE over minor/moderate/major:")
    print(mean_kge.round(3).to_string())

    if "base" in mean_kge.index:
        gain = (mean_kge - mean_kge.loc["base"]).drop("base").sort_values(ascending=False)
        print("\nChange in mean KGE relative to base:")
        print(gain.round(3).to_string())

    summary = (
        class_comp.groupby("feature_set")
        .agg(
            mean_KGE_3classes=("KGE", "mean"),
            mean_r_3classes=("r", "mean"),
            mean_alpha_3classes=("alpha", "mean"),
            mean_beta_3classes=("beta", "mean"),
            mean_skill_vs_seasonal_3classes=("skill_vs_seasonal", "mean"),
        )
        .reindex(available_sets)
        .reset_index()
    )

    summary.to_csv(os.path.join(out, "feature_set_summary.csv"), index=False)

    sets = [fs for fs in FEATURE_SETS if fs in series_data]

    fig = plt.figure(figsize=(7.8, 5.8))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.15],
        hspace=0.78,
        wspace=0.32,
    )

    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(SERIES))
    w = 0.82 / max(len(sets), 1)

    for i, fs in enumerate(sets):
        vals = []
        for s in SERIES:
            q = comp[
                (comp["feature_set"] == fs)
                & (comp["series"] == s)
            ]["KGE"]
            vals.append(q.iloc[0] if len(q) else np.nan)

        xpos = x + (i - (len(sets) - 1) / 2) * w
        ax.bar(
            xpos, vals, w,
            color=COLORS[fs],
            label=f"{fs}: {DESCRIPTIONS[fs]}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TITLES[s] for s in SERIES], fontsize=8)
    ax.set_ylabel("KGE")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("(a) Skill by feature set", loc="left", fontsize=9)
    ax.legend(
        frameon=False,
        fontsize=6.1,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.49),
    )

    for j, s in enumerate(["class_2", "class_3", "any_outage"]):
        ax = fig.add_subplot(gs[1, j])

        first = series_data[sets[0]]
        obs = first[first["series"] == s].copy().sort_values("year")

        ax.plot(
            obs["year"], obs["obs"],
            color="black",
            marker="o",
            ms=3.5,
            lw=2,
            label="Observed",
            zorder=5,
        )

        for fs in sets:
            d = series_data[fs]
            d = d[d["series"] == s].sort_values("year")

            ax.plot(
                d["year"], d["model"],
                color=COLORS[fs],
                marker="s",
                ms=2.5,
                lw=1.15,
                label=fs,
            )

        ax.set_title(f"({'bcd'[j]}) {TITLES[s]}", loc="left", fontsize=9)

        years = sorted(obs["year"].dropna().astype(int).unique())
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=90)
        ax.grid(alpha=0.3)

        if 2024 in years:
            ax.axvspan(2023.5, 2024.5, color="0.92", zorder=0)

        if j == 0:
            ax.set_ylabel(
                "share" if s == "major_share"
                else f"days per {RATE:,} county-days"
            )
            ax.legend(frameon=False, fontsize=6.1, ncol=1, loc="best")

    if any(
        2024 in set(df["year"].dropna().astype(int))
        for df in series_data.values()
        if "year" in df.columns
    ):
        fig.text(
            0.995,
            0.01,
            "2024 is a partial held-out year (evaluated only over the available record).",
            ha="right",
            va="bottom",
            fontsize=6.5,
        )

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"Fig_feature_sets_partial2024.{ext}"))

    plt.close(fig)
    print(f"\nComparison output -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-root",
        default=RESULTS_ROOT,
        help="parent directory containing *_partial2024 result folders",
    )
    p.add_argument(
        "--suffix",
        default=RUN_SUFFIX,
        help="result-folder suffix (default: _partial2024)",
    )
    p.add_argument(
        "--output-dir",
        default="comparison_partial2024",
        help="output folder name inside results-root",
    )
    main(p.parse_args())
