"""
Spatial versus temporal skill across feature sets
=================================================

Some predictor groups help the model reproduce WHICH COUNTIES have many outage
days, others help it reproduce WHICH YEARS are active. Static county attributes,
for example, are constant in time: they can only improve the spatial pattern.
This script quantifies that trade-off.

For every feature set it reads daily_predictions.csv, forms county-year rates
(days per 1,000 county-days) for each class, and computes:

  temporal skill  KGE of the statewide annual series (the headline metric)
  spatial skill   correlation of county means over the whole record, and the
                  MSE skill of those county means against a spatially flat
                  prediction
  anomaly skill   correlation of county-year anomalies (each county's yearly
                  rate minus its own mean): does the model know which counties
                  had an unusual year?

It also decomposes the mean squared error of the county-year rates into four
additive parts, writing e_cy = m + a_c + b_y + r_cy for the error in county c
and year y:

  bias^2        m^2, a constant offset
  spatial       var(a_c), each county's average error
  temporal      var(b_y), each year's average error across counties
  interaction   var(r_cy), the county-by-year remainder

MSE = bias^2 + spatial + temporal + interaction, so the bars show where each
feature set's error actually sits.

Outputs, in <results_root>/comparison/:
  spatial_temporal_metrics.csv      one row per feature set and class
  error_decomposition.csv           the four MSE components
  Fig_spatial_temporal_tradeoff.png/.pdf

Usage:  python compare_spatial_temporal.py
        python compare_spatial_temporal.py --results-root results --suffix _partial2024
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

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
RUN_SUFFIX = "_partial2024"
FEATURE_SETS = ["minimal", "base", "static", "antecedent", "forward", "forward_plus",
                "relative", "forward_static", "antecedent_static", "full"]
COLORS = {
    "minimal": "#BBBBBB",
    "base": "#888780",
    "static": "#E69F00",
    "antecedent": "#0072B2",
    "forward": "#009E73",
    "forward_plus": "#56B4E9",
    "relative": "#CC79A7",
    "forward_static": "#D55E00",
    "antecedent_static": "#999999",
    "full": "#534AB7",
}
MARKERS = {"minimal": "o", "base": "s", "static": "^", "antecedent": "v",
           "forward": "D", "forward_plus": "P", "relative": "X",
           "forward_static": "d", "antecedent_static": "<", "full": "*"}
CLASSES = [1, 2, 3]
TITLES = {1: "Minor (1-2 h)", 2: "Moderate (2-8 h)", 3: "Major ($\\geq$8 h)",
          "any": "All outage classes"}
PANELS = [1, 2, 3, "any"]
RATE = 1000
# =============================================================================

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def folder_for(root, fs, suffix=RUN_SUFFIX):
    """Find a completed result folder for one logical feature set.

    The exact <feature_set><suffix> directory is preferred so older or fast
    experiment folders are not selected accidentally.
    """
    exact = os.path.join(root, fs + suffix)
    if os.path.exists(os.path.join(exact, "daily_predictions.csv")):
        return exact

    prefix = fs + suffix
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            candidate = os.path.join(root, name)
            if name.startswith(prefix + "_") and os.path.exists(
                os.path.join(candidate, "daily_predictions.csv")
            ):
                return candidate
    return None


def county_year_rates(path, k):
    """Observed and predicted rates per 1,000 county-days, as [county, year] arrays."""
    cols = ["ID", "year", "duration"]
    d = pd.read_csv(path, usecols=lambda c: c in set(cols) | {"outage_data_available"} |
                    {f"p_model_{j}" for j in CLASSES})
    if "outage_data_available" in d.columns:
        d = d[d["outage_data_available"] == 1]
    ks = CLASSES if k == "any" else [k]
    d["_obs"] = d["duration"].isin(ks).astype(float)
    d["_mod"] = d[[f"p_model_{j}" for j in ks]].sum(axis=1)
    g = d.groupby(["ID", "year"])
    t = g[["_obs", "_mod"]].sum()
    t["n"] = g.size()
    counties = sorted(t.index.get_level_values(0).unique(), key=lambda c: int(c.split("_")[1]))
    years = sorted(t.index.get_level_values(1).unique())
    t = t.reindex(pd.MultiIndex.from_product([counties, years]), fill_value=0)
    shape = (len(counties), len(years))
    n = t["n"].to_numpy().reshape(shape).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = RATE * t["_obs"].to_numpy().reshape(shape) / n
        mod = RATE * t["_mod"].to_numpy().reshape(shape) / n
    return obs, mod, n, counties, years


def kge(obs, sim):
    if obs.std() == 0 or sim.std() == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 +
                       (sim.mean() / obs.mean() - 1) ** 2)


def analyse(obs, mod, n):
    """Spatial, temporal and anomaly skill, plus the four MSE components."""
    w = n / n.sum()
    # statewide annual series (weighted by county-days, as in the main evaluation)
    ann_obs = (obs * n).sum(axis=0) / n.sum(axis=0)
    ann_mod = (mod * n).sum(axis=0) / n.sum(axis=0)
    # county means over the record
    cty_obs = (obs * n).sum(axis=1) / n.sum(axis=1)
    cty_mod = (mod * n).sum(axis=1) / n.sum(axis=1)
    # anomalies: each county's yearly rate minus its own mean
    ao = obs - obs.mean(axis=1, keepdims=True)
    am = mod - mod.mean(axis=1, keepdims=True)

    e = mod - obs
    m = (e * w).sum()
    a_c = (e * (n / n.sum(axis=1, keepdims=True))).sum(axis=1) - m      # county effect
    b_y = (e * (n / n.sum(axis=0, keepdims=True))).sum(axis=0) - m      # year effect
    r_cy = e - m - a_c[:, None] - b_y[None, :]
    parts = dict(bias2=m ** 2,
                 spatial=float((w * (a_c[:, None] ** 2)).sum()),
                 temporal=float((w * (b_y[None, :] ** 2)).sum()),
                 interaction=float((w * r_cy ** 2).sum()))
    parts["mse_total"] = float((w * e ** 2).sum())

    flat = np.full_like(cty_obs, cty_obs.mean())      # spatially flat reference
    out = dict(
        temporal_KGE=kge(ann_obs, ann_mod),
        temporal_r=np.corrcoef(ann_obs, ann_mod)[0, 1],
        spatial_r=np.corrcoef(cty_obs, cty_mod)[0, 1],
        spatial_KGE=kge(cty_obs, cty_mod),
        spatial_skill=1 - np.mean((cty_mod - cty_obs) ** 2) / np.mean((flat - cty_obs) ** 2),
        anomaly_r=np.corrcoef(ao.ravel(), am.ravel())[0, 1],
        **parts)
    return out


def main(args):
    out = os.path.join(args.results_root, "comparison")
    os.makedirs(out, exist_ok=True)
    rows = []
    for fs in FEATURE_SETS:
        d = folder_for(args.results_root, fs, args.suffix)
        if d is None:
            print(f"  ! {fs}: no daily_predictions.csv")
            continue
        print(f"  {fs}: {os.path.basename(d)}")
        path = os.path.join(d, "daily_predictions.csv")
        for k in PANELS:
            obs, mod, n, counties, years = county_year_rates(path, k)
            rows.append(dict(feature_set=fs, series=("any" if k == "any" else f"class_{k}"),
                             n_counties=len(counties), n_years=len(years),
                             **analyse(obs, mod, n)))
    if not rows:
        raise SystemExit(f"no finished feature sets in {args.results_root}")
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(out, "spatial_temporal_metrics.csv"), index=False)
    dec = m[["feature_set", "series", "bias2", "spatial", "temporal", "interaction", "mse_total"]]
    dec.to_csv(os.path.join(out, "error_decomposition.csv"), index=False)

    pd.set_option("display.width", 200)
    sets = [f for f in FEATURE_SETS if f in set(m.feature_set)]
    for col in ["temporal_KGE", "spatial_r", "anomaly_r"]:
        print(f"\n{col}:")
        print(m.pivot(index="series", columns="feature_set", values=col)[sets]
                .reindex([f"class_{k}" for k in CLASSES] + ["any"]).round(3).to_string())
    print("\nShare of county-year MSE by component (all outage classes):")
    a = m[m.series == "any"].set_index("feature_set").loc[sets]
    share = a[["bias2", "spatial", "temporal", "interaction"]].div(a["mse_total"], axis=0)
    print(share.round(3).to_string())

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.62, wspace=0.42)
    for j, k in enumerate(PANELS):
        s = "any" if k == "any" else f"class_{k}"
        ax = fig.add_subplot(gs[0, j])
        d = m[m.series == s]
        for r in d.itertuples():
            ax.scatter(r.spatial_r, r.temporal_KGE, s=70 if r.feature_set == "full" else 45,
                       color=COLORS[r.feature_set], marker=MARKERS[r.feature_set],
                       edgecolor="k", lw=0.4, zorder=3, label=r.feature_set)
        ax.set_title(f"({'abcd'[j]}) {TITLES[k]}", loc="left", fontsize=8.5)
        ax.set_xlabel("spatial $r$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.locator_params(axis="x", nbins=4)
        if j == 0:
            ax.set_ylabel("temporal skill (KGE, annual)")
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    a = m[m.series == "any"].set_index("feature_set").loc[sets]
    share = a[["bias2", "spatial", "temporal", "interaction"]].div(a["mse_total"], axis=0)
    x = np.arange(len(sets))
    bottom = np.zeros(len(sets))
    hatch = {"bias2": "///", "spatial": "", "temporal": "", "interaction": ".."}
    cols = {"bias2": "#444444", "spatial": "#E69F00", "temporal": "#0072B2", "interaction": "#BBBBBB"}
    for comp in ["bias2", "spatial", "temporal", "interaction"]:
        ax.bar(x, share[comp], 0.6, bottom=bottom, color=cols[comp], hatch=hatch[comp],
               edgecolor="white", linewidth=0.4, label=comp.replace("bias2", "bias$^2$"))
        bottom += share[comp].to_numpy()
    for i, fs in enumerate(sets):
        ax.text(i, 1.02, f"MSE {a.loc[fs, 'mse_total']:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(sets, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of county-year MSE")
    ax.set_title("(e) Where the error sits, all outage classes", loc="left", fontsize=8.5)
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.42))

    handles, labels = fig.axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, frameon=False, ncol=7, fontsize=7.5, loc="upper center",
               bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, 0.455, "spatial skill: correlation of county mean rates over the record",
             ha="center", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"Fig_spatial_temporal_tradeoff.{ext}"))
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=RESULTS_ROOT,
                   help="parent folder containing feature-set result folders")
    p.add_argument("--suffix", default=RUN_SUFFIX,
                   help="result-folder suffix to compare (default: _partial2024)")
    main(p.parse_args())