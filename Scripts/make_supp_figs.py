"""
Supplementary figures S1 (time alignment) and S2 (data coverage)
================================================================

Regenerates the two supplementary figures in the same style as the main
justification figures, as PNG (300 dpi) and PDF.

  FigS1_time_alignment  : correlation between hourly weather and hourly
                          customers out at lags of -24 to +48 hours, averaged
                          over counties, with the spread across counties.
                          Built from diagnostics/time_alignment.csv and
                          time_alignment_by_county.csv, written by
                          build_conus404_outage_dataset.py.
  FigS2_data_coverage   : share of days with usable EAGLE-I data, by county and
                          year. Built from the modelling dataset
                          (outage_data_available), or from
                          diagnostics/outage_data_available_share_by_county_year.csv.

Usage
-----
  python make_supp_figs.py --diagnostics <dataset folder>/diagnostics \\
                           --data all_counties_classified_stat_all_days_conus404_4class_p80.csv \\
                           --out justification
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Repository layout used by the public code release.
# Defaults are relative to the repository; every path can be overridden by CLI.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DIAGNOSTICS = os.path.join(PROJECT_ROOT, "data", "processed", "diagnostics")
DATA = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "all_counties_classified_stat_all_days_conus404_4class_p80.csv",
)
OUT = os.path.join(PROJECT_ROOT, "results", "supplementary_figures")
LABELS = {"PREC_ACC_NC_cellmean": "Precipitation, county mean",
          "WSPD10_cellmax": "10-m wind, cell max"}
COLORS = {"PREC_ACC_NC_cellmean": "#0072B2", "WSPD10_cellmax": "#D55E00"}
FULL_W = 7.2
# =============================================================================

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {os.path.join(out, name)}.png / .pdf")


def fig_s1(diag, out):
    path = os.path.join(diag, "time_alignment.csv")
    if not os.path.exists(path):
        print(f"  ! {path} not found - rerun build_conus404_outage_dataset.py to create it")
        return
    m = pd.read_csv(path)
    per_county = None
    p2 = os.path.join(diag, "time_alignment_by_county.csv")
    if os.path.exists(p2):
        per_county = pd.read_csv(p2)

    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 2.8))
    for v, d in m.groupby("variable"):
        d = d.sort_values("lag_h")
        c = COLORS.get(v, "#555555")
        if per_county is not None:
            q = (per_county[per_county["variable"] == v]
                 .groupby("lag_h")["r"].quantile([0.25, 0.75]).unstack())
            q = q.reindex(d["lag_h"])
            ax.fill_between(d["lag_h"], q[0.25], q[0.75], color=c, alpha=0.15, lw=0)
        ax.plot(d["lag_h"], d["mean"], color=c, lw=1.6, label=LABELS.get(v, v))
        peak = d.loc[d["mean"].idxmax()]
        ax.plot(peak["lag_h"], peak["mean"], "o", color=c, ms=4)
        ax.annotate(f"{int(peak['lag_h']):+d} h", (peak["lag_h"], peak["mean"]),
                    xytext=(4, 5), textcoords="offset points", fontsize=7, color=c)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Lag (h): outage record time minus weather time")
    ax.set_ylabel("Correlation")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, out, "FigS1_time_alignment")


def fig_s2(data, diag, out):
    share_path = os.path.join(diag, "outage_data_available_share_by_county_year.csv")
    if os.path.exists(share_path):
        t = pd.read_csv(share_path, index_col=0)
        t.columns = [int(c) for c in t.columns]
    elif os.path.exists(data):
        d = pd.read_csv(data, usecols=["ID", "Time", "outage_data_available"], dtype={"ID": str})
        d["year"] = pd.to_datetime(d["Time"], format="mixed").dt.year
        t = d.groupby(["ID", "year"])["outage_data_available"].mean().unstack()
    else:
        print(f"  ! neither {share_path} nor {data} found")
        return
    t = t.loc[sorted(t.index, key=lambda s: int(s.split("_")[1]))]

    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 7.2))
    im = ax.imshow(t.to_numpy(float), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(t.columns)))
    ax.set_xticklabels(t.columns, rotation=45)
    ax.set_yticks(range(len(t.index)))
    ax.set_yticklabels([i.replace("county_", "") for i in t.index], fontsize=4.5)
    ax.set_ylabel("County (ID)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Share of days with usable outage data", fontsize=8)
    fig.tight_layout()
    save(fig, out, "FigS2_data_coverage")
    worst = t.min(axis=1).sort_values().head(5)
    print("  lowest coverage in any year: " +
          ", ".join(f"{i} {v:.2f}" for i, v in worst.items()))
    print(f"  statewide mean coverage by year: "
          f"{', '.join(f'{y} {v:.2f}' for y, v in t.mean().items())}")


def main(a):
    print("Figure S1")
    fig_s1(a.diagnostics, a.out)
    print("Figure S2")
    fig_s2(a.data, a.diagnostics, a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diagnostics", default=DIAGNOSTICS)
    p.add_argument("--data", default=DATA)
    p.add_argument("--out", default=OUT)
    main(p.parse_args())
