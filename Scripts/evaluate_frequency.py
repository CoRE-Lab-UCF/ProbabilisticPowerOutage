"""
Evaluation of leave-one-year-out frequency predictions
======================================================

Works on the daily_predictions.csv written by loyo_frequency_model.py, so the
model does not need to be re-run.

  1. Water-year frequency metrics (Oct-Sep), using complete water years only
  2. Bootstrap 95% confidence intervals: counties are resampled with
     replacement and the statewide yearly series rebuilt each time
     (uncertainty from which counties happen to be in the sample)
  3. Year jackknife: metrics recomputed leaving out each evaluation year in
     turn (how much a single year drives the result)
  4. County-level check:
       anomaly_r   pooled correlation of county-year anomalies (each county's
                   yearly rate minus its own mean): does the model know WHICH
                   counties have an active year?
       median_county_r / pct_counties_r_pos   the same, county by county
       spatial_r   correlation of county mean rates: does it know which
                   counties are outage-prone on average?
       skill_vs_seasonal  MSE skill on county-year rates

  5. Calibration (reliability diagram, calibration.csv):
       for each class, predictions are grouped into bins of predicted
       probability and the observed frequency in each bin is compared with the
       mean predicted probability. Points on the 1:1 line mean the
       probabilities are honest, which is what makes the summed-probability
       frequency estimate valid. Reported with the expected calibration error
       (ECE, the frequency-weighted mean absolute gap) and the observed/expected
       ratio over all days.

  6. Seasonality (folder seasonal/):
       - seasonal cycle: mean monthly rates, observed vs predicted
       - season-by-season variability: for each season group the season totals
         of every season-year are compared (KGE, r, alpha, beta, skill, 95% CI):
           hurricane season (Jun-Nov), hurricane season without tropical-
           cyclone (TC) days, off-season (Dec-May), DJF, MAM, JJA, SON
           (December counts toward the following year's DJF / off-season;
           incomplete seasons at the ends of the record are dropped)
       - month-to-month anomalies: does the model know which months of which
         years were unusually active?
       - regimes: TC-window days vs other hurricane-season days vs off-season:
         daily skill, and storm-by-storm observed vs predicted outage days

Series: rates per 1,000 county-days of each class, any_outage (classes 1-3),
and major_share (major / all outage classes).

Usage
-----
  python evaluate_frequency.py
  python evaluate_frequency.py --results results/full_partial2024
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

RESULTS = os.path.join(PROJECT_ROOT, "results", "full_partial2024")
SKIP_YEARS = []                # partial years are already not tested by the model run
N_BOOT = 2000
SEED = 42
RATE_SCALE = 1000
CLASSES = [0, 1, 2, 3]
OUTAGE_CLASSES = [1, 2, 3]
PREDICTORS = ["model", "seasonal", "clim"]
SERIES = ["class_1", "class_2", "class_3", "any_outage", "major_share", "class_0"]
TITLES = {"class_0": "None", "class_1": "Minor (1-2 h)", "class_2": "Moderate (2-8 h)",
          "class_3": "Major (>= 8 h)", "any_outage": "Any weather-related outage",
          "major_share": "Major share = major / all outage classes"}

# Seasonality
SEASON_GROUPS = {                       # name: (months, TC-window filter)
    "Hurricane season (Jun-Nov)": ([6, 7, 8, 9, 10, 11], None),
    "Hurricane season, non-TC days": ([6, 7, 8, 9, 10, 11], False),
    "Off-season (Dec-May)": ([12, 1, 2, 3, 4, 5], None),
    "Winter (DJF)": ([12, 1, 2], None),
    "Spring (MAM)": ([3, 4, 5], None),
    "Summer (JJA)": ([6, 7, 8], None),
    "Fall (SON)": ([9, 10, 11], None),
}
SEASON_SERIES = ["class_1", "class_2", "class_3", "any_outage", "major_share"]
N_BOOT_SEASON = 500

# Standard meteorological seasons for the manuscript figure
STD_SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
STD_SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
STD_SEASON_START = 2015
STD_SEASON_END = 2024
N_BOOT_STD_SEASON = 1000
# Reliability diagram: bin edges for predicted probability. Fine near zero
# because most county-days carry small probabilities.
CAL_BINS = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
CAL_MIN_N = 30                          # bins with fewer days are not plotted
# Tropical cyclones affecting Florida, local dates (same list as the
# justification script; verify against NHC Tropical Cyclone Reports)
STORMS = [
    ("TS Colin", "2016-06-06", "2016-06-07"), ("Hermine", "2016-09-01", "2016-09-02"),
    ("Matthew", "2016-10-06", "2016-10-08"), ("TS Emily", "2017-07-31", "2017-07-31"),
    ("Irma", "2017-09-09", "2017-09-11"), ("STS Alberto", "2018-05-27", "2018-05-28"),
    ("TS Gordon", "2018-09-03", "2018-09-04"), ("Michael", "2018-10-09", "2018-10-11"),
    ("Dorian", "2019-09-02", "2019-09-04"), ("TS Nestor", "2019-10-18", "2019-10-19"),
    ("Isaias", "2020-08-01", "2020-08-02"), ("Sally", "2020-09-15", "2020-09-16"),
    ("Eta", "2020-11-08", "2020-11-12"), ("Elsa", "2021-07-06", "2021-07-07"),
    ("TS Fred", "2021-08-16", "2021-08-16"), ("TS Mindy", "2021-09-08", "2021-09-09"),
    ("TS Alex", "2022-06-03", "2022-06-04"), ("Ian", "2022-09-28", "2022-09-30"),
    ("Nicole", "2022-11-09", "2022-11-10"), ("Idalia", "2023-08-29", "2023-08-30"),
    ("Debby", "2024-08-04", "2024-08-05"), ("Helene", "2024-09-26", "2024-09-27"),
    ("Milton", "2024-10-09", "2024-10-10"),
]
STORM_PAD_DAYS = 1
# =============================================================================


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------
def read_daily(results, skip_years):
    d = pd.read_csv(os.path.join(results, "daily_predictions.csv"), parse_dates=["Time"])
    d = d[~d["year"].isin(skip_years)]
    if "outage_data_available" in d.columns:            # score observed days only
        d = d[d["outage_data_available"] == 1]
    return d.reset_index(drop=True)


def add_water_year(d):
    """
    Add U.S. water year (Oct-Sep).

    Example:
      Oct-Dec 2015 + Jan-Sep 2016 -> water year 2016.
    """
    d = d.copy()
    d["water_year"] = d["Time"].dt.year + (d["Time"].dt.month >= 10).astype(int)
    return d


def keep_complete_water_years(d):
    """
    Retain only complete Oct-Sep water years.

    Completeness is based on presence of all 12 calendar months in a water year.
    With a Jan-2015 through Sep-2024 record, WY2015 is incomplete while
    WY2016-WY2024 are complete.
    """
    d = add_water_year(d)

    month_key = d["Time"].dt.year * 100 + d["Time"].dt.month
    chk = (
        d.assign(_month_key=month_key)
         .groupby("water_year")["_month_key"]
         .nunique()
    )
    complete = chk.index[chk >= 12].astype(int).tolist()

    return d[d["water_year"].isin(complete)].copy(), complete


def counts_from(d, key):
    """Counts per county and period: C[county, period, source, class], N[county, period]."""
    sources = ["obs"] + PREDICTORS
    cols = [f"p_{s_}_{k}" for s_ in sources for k in CLASSES]
    g = d.groupby(["ID", key])[cols].sum()
    n = d.groupby(["ID", key]).size()
    counties = sorted(d["ID"].unique(), key=lambda c: int(c.split("_")[1]))
    keys = [int(y) for y in sorted(d[key].unique())]
    idx = pd.MultiIndex.from_product([counties, keys], names=["ID", key])
    g, n = g.reindex(idx, fill_value=0.0), n.reindex(idx, fill_value=0)
    C = g.to_numpy().reshape(len(counties), len(keys), len(sources), len(CLASSES))
    N = n.to_numpy().reshape(len(counties), len(keys)).astype(float)
    return C, N, counties, keys, sources


def load_counts(results, skip_years):
    """
    Counts per county WATER YEAR (Oct-Sep).

    Only complete water years are retained. This is used for all annual-
    variability metrics, bootstrap intervals, jackknife analyses, county-year
    anomaly checks, and the annual observed-vs-predicted figure.
    """
    d = read_daily(results, skip_years)
    d, complete_wy = keep_complete_water_years(d)

    if not complete_wy:
        raise ValueError("No complete Oct-Sep water years were found in daily_predictions.csv")

    print(f"Complete water years used for annual variability: {complete_wy}")
    return counts_from(d, "water_year")


def series_from_counts(C, N):
    """C[..., source, class], N[...] -> dict series -> array[..., source]."""
    n = N[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {f"class_{k}": RATE_SCALE * C[..., k] / n for k in CLASSES}
        tot = C[..., OUTAGE_CLASSES].sum(axis=-1)
        out["any_outage"] = RATE_SCALE * tot / n
        out["major_share"] = np.where(tot > 0, C[..., 3] / tot, np.nan)
    return out


def metrics(obs, sim, seas, clim):
    """KGE parts and skill for a 1-D sequence of evaluation periods."""
    ok = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim, seas, clim = obs[ok], sim[ok], seas[ok], clim[ok]
    if len(obs) < 3 or obs.std() == 0 or sim.std() == 0:
        return dict(KGE=np.nan, r=np.nan, alpha=np.nan, beta=np.nan,
                    skill_vs_seasonal=np.nan, skill_vs_clim=np.nan)
    r = np.corrcoef(obs, sim)[0, 1]
    a, b = sim.std() / obs.std(), sim.mean() / obs.mean()
    mse = np.mean((sim - obs) ** 2)
    ms, mc = np.mean((seas - obs) ** 2), np.mean((clim - obs) ** 2)
    return dict(KGE=1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), r=r, alpha=a, beta=b,
                skill_vs_seasonal=1 - mse / ms if ms > 0 else np.nan,
                skill_vs_clim=1 - mse / mc if mc > 0 else np.nan)


def statewide_metrics(C, N, sources):
    s = series_from_counts(C.sum(axis=0), N.sum(axis=0))
    i = {name: sources.index(name) for name in sources}
    rows = {}
    for name in SERIES:
        v = s[name]
        rows[name] = metrics(v[:, i["obs"]], v[:, i["model"]], v[:, i["seasonal"]], v[:, i["clim"]])
    return rows


# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------
def bootstrap(C, N, sources, n_boot=None):
    rng = np.random.default_rng(SEED)
    nc = C.shape[0]
    draws = {name: [] for name in SERIES}
    for _ in range(n_boot or N_BOOT):
        pick = rng.integers(0, nc, nc)
        m = statewide_metrics(C[pick], N[pick], sources)
        for name in SERIES:
            draws[name].append(m[name])
    rows = []
    for name in SERIES:
        d = pd.DataFrame(draws[name])
        for col in d.columns:
            rows.append(dict(series=name, metric=col, ci_low=d[col].quantile(0.025),
                             ci_high=d[col].quantile(0.975), boot_median=d[col].median()))
    return pd.DataFrame(rows)


def year_jackknife(C, N, years, sources):
    """Leave-one-water-year-out sensitivity analysis."""
    rows = []
    for j, y in enumerate(years):
        keep = [k for k in range(len(years)) if k != j]
        m = statewide_metrics(C[:, keep], N[:, keep], sources)
        for name in SERIES:
            rows.append(dict(left_out_water_year=y, series=name, **m[name]))
    jk = pd.DataFrame(rows)
    summary = jk.groupby("series")[["KGE", "r", "alpha", "skill_vs_seasonal"]].agg(["min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    worst = jk.loc[
        jk.groupby("series")["KGE"].idxmax(),
        ["series", "left_out_water_year", "KGE"]
    ]
    worst = worst.rename(columns={
        "left_out_water_year": "water_year_whose_removal_helps_most",
        "KGE": "KGE_without_it"
    })
    return jk, summary.reset_index().merge(worst, on="series")


def county_level(C, N, counties, sources):
    s = series_from_counts(C, N)          # [county, year, source]
    i = {name: sources.index(name) for name in sources}
    rows, per_county = [], []
    for name in [x for x in SERIES if x != "class_0"]:
        v = s[name]
        for pred in ["model", "seasonal"]:
            obs, sim = v[:, :, i["obs"]], v[:, :, i[pred]]
            ok = ~(np.isnan(obs) | np.isnan(sim))
            oa = obs - np.nanmean(np.where(ok, obs, np.nan), axis=1, keepdims=True)
            sa = sim - np.nanmean(np.where(ok, sim, np.nan), axis=1, keepdims=True)
            m = ok & ~np.isnan(oa) & ~np.isnan(sa)
            anomaly_r = np.corrcoef(oa[m], sa[m])[0, 1]
            cr = []
            for c in range(len(counties)):
                mc = m[c]
                if mc.sum() >= 3 and obs[c, mc].std() > 0 and sim[c, mc].std() > 0:
                    cr.append(np.corrcoef(obs[c, mc], sim[c, mc])[0, 1])
                else:
                    cr.append(np.nan)
            cr = np.array(cr)
            spatial_r = np.corrcoef(np.nanmean(np.where(ok, obs, np.nan), axis=1),
                                    np.nanmean(np.where(ok, sim, np.nan), axis=1))[0, 1]
            seas = v[:, :, i["seasonal"]]
            mse = np.nanmean(np.where(m, (sim - obs) ** 2, np.nan))
            mse_s = np.nanmean(np.where(m, (seas - obs) ** 2, np.nan))
            rows.append(dict(series=name, predictor=pred, anomaly_r=anomaly_r,
                             median_county_r=np.nanmedian(cr),
                             pct_counties_r_pos=100 * np.nanmean(cr > 0),
                             spatial_r=spatial_r,
                             skill_vs_seasonal=1 - mse / mse_s if pred == "model" and mse_s > 0 else np.nan))
            if pred == "model":
                bias = 100 * (np.nansum(np.where(m, sim, 0), axis=1) / np.nansum(np.where(m, obs, 0), axis=1) - 1)
                per_county.append(pd.DataFrame({"ID": counties, "series": name, "r": cr, "bias_pct": bias}))
    return pd.DataFrame(rows), pd.concat(per_county, ignore_index=True), s


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_yearly(C, N, years, sources, point, ci, path, skip_years):
    s = series_from_counts(C.sum(axis=0), N.sum(axis=0))
    i = {name: sources.index(name) for name in sources}
    style = {"obs": dict(color="black", marker="o", lw=2.2, label="Observed"),
             "model": dict(color="#534AB7", marker="s", lw=1.8, label="Model (sum of probabilities)"),
             "seasonal": dict(color="#0F6E56", ls="--", lw=1.2, label="Seasonal baseline"),
             "clim": dict(color="#888780", ls="--", lw=1.0, label="Climatology baseline")}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, name in zip(axes.flat, SERIES):
        for src, st in style.items():
            ax.plot(years, s[name][:, i[src]], **st)
        p = point[name]
        k = ci[(ci["series"] == name) & (ci["metric"] == "KGE")].iloc[0]
        ax.set_title(f"{TITLES[name]}\nKGE={p['KGE']:.2f} [{k.ci_low:.2f}, {k.ci_high:.2f}]  "
                     f"r={p['r']:.2f}  \u03b1={p['alpha']:.2f}  \u03b2={p['beta']:.2f}", fontsize=10)
        ax.set_ylabel("share" if name == "major_share" else f"days per {RATE_SCALE:,} county-days")
        ax.set_xticks(years)
        ax.set_xlabel("Water year (Oct-Sep)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    skipped = f" (excluding {', '.join(map(str, skip_years))})" if skip_years else ""
    fig.suptitle(
        f"Annual variability by water year (Oct-Sep): observed vs predicted{skipped}; "
        f"KGE with 95% bootstrap CI",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_county(s, sources, per_county, path):
    i = {name: sources.index(name) for name in sources}
    names = ["class_1", "class_2", "class_3", "any_outage"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for j, name in enumerate(names):
        v = s[name]
        obs, sim = v[:, :, i["obs"]], v[:, :, i["model"]]
        oa = obs - np.nanmean(obs, axis=1, keepdims=True)
        sa = sim - np.nanmean(sim, axis=1, keepdims=True)
        ax = axes[0, j]
        ax.scatter(oa.ravel(), sa.ravel(), s=8, alpha=0.4, color="#534AB7")
        lim = np.nanmax(np.abs(np.concatenate([oa.ravel(), sa.ravel()])))
        ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8)
        ok = ~(np.isnan(oa) | np.isnan(sa))
        ax.set_title(f"{TITLES[name]}: county-year anomalies\nr = {np.corrcoef(oa[ok], sa[ok])[0, 1]:.2f}",
                     fontsize=10)
        ax.set_xlabel("observed anomaly"); ax.set_ylabel("predicted anomaly"); ax.grid(alpha=0.3)
        ax = axes[1, j]
        r = per_county.loc[per_county["series"] == name, "r"].dropna()
        ax.hist(r, bins=np.linspace(-1, 1, 21), color="#1D9E75", edgecolor="white")
        ax.axvline(r.median(), color="k", ls="--", lw=1)
        ax.set_title(f"Per-county r across years (median {r.median():.2f})", fontsize=10)
        ax.set_xlabel("r"); ax.set_ylabel("counties")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
def calibration(d, out):
    """Reliability of the predicted probabilities, one panel per class."""
    rows = []
    for k in CLASSES:
        p = d[f"p_model_{k}"].to_numpy()
        y = (d["duration"].to_numpy() == k).astype(float)
        idx = np.digitize(p, CAL_BINS[1:-1], right=True)
        for b in range(len(CAL_BINS) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append(dict(cls=k, bin_low=CAL_BINS[b], bin_high=CAL_BINS[b + 1],
                             n=int(m.sum()), mean_predicted=p[m].mean(),
                             observed_frequency=y[m].mean()))
    cal = pd.DataFrame(rows)
    cal["gap"] = cal["observed_frequency"] - cal["mean_predicted"]
    cal.to_csv(os.path.join(out, "calibration.csv"), index=False)

    summary = []
    for k in CLASSES:
        c = cal[cal["cls"] == k]
        w = c["n"] / c["n"].sum()
        obs = (d["duration"].to_numpy() == k).sum()
        exp = d[f"p_model_{k}"].sum()
        summary.append(dict(cls=k, name=TITLES[f"class_{k}"], n_days=int(c["n"].sum()),
                            ECE=float((w * c["gap"].abs()).sum()),
                            observed=int(obs), expected=float(exp),
                            obs_over_expected=obs / exp if exp > 0 else np.nan))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(out, "calibration_summary.csv"), index=False)

    fig, axes = plt.subplots(2, len(CLASSES), figsize=(3.7 * len(CLASSES), 6.2),
                             gridspec_kw=dict(height_ratios=[3, 1]), sharex="col",
                             constrained_layout=True)
    for j, k in enumerate(CLASSES):
        c = cal[(cal["cls"] == k) & (cal["n"] >= CAL_MIN_N)]
        ax = axes[0, j]
        lim = max(c["mean_predicted"].max(), c["observed_frequency"].max()) * 1.1 if len(c) else 1
        ax.plot([0, lim], [0, lim], color="k", lw=0.8, zorder=1)
        ax.plot(c["mean_predicted"], c["observed_frequency"], marker="o", ms=4, lw=1.4,
                color="#534AB7", zorder=3)
        r = summary.iloc[j]
        ax.set_title(f"{TITLES[f'class_{k}']}\nECE = {r.ECE:.3f},  obs/exp = {r.obs_over_expected:.3f}",
                     fontsize=10)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("observed frequency")
        ax2 = axes[1, j]
        cc = cal[cal["cls"] == k]
        w = np.diff(CAL_BINS)[cc.index - cc.index.min()] if len(cc) else 1
        ax2.bar(cc["mean_predicted"], cc["n"], width=np.maximum(np.asarray(w) * 0.6, lim / 50),
                color="#888780")
        ax2.set_yscale("log")
        ax2.set_xlim(0, lim)
        ax2.set_xlabel("predicted probability")
        ax2.grid(alpha=0.3)
        if j == 0:
            ax2.set_ylabel("county-days")
    #fig.suptitle("Reliability of predicted class probabilities (all held-out years)", fontsize=12)
    fig.savefig(os.path.join(out, "calibration_reliability.png"), dpi=200)
    fig.savefig(os.path.join(out, "calibration_reliability.pdf"))
    plt.close(fig)
    return summary


# -----------------------------------------------------------------------------
# Seasonality
# -----------------------------------------------------------------------------
def tag_days(d):
    d = d.copy()
    d["month"] = d["Time"].dt.month
    d["season_year"] = d["year"] + (d["month"] == 12).astype(int)
    d["tc_day"] = False
    d["storm"] = ""
    pad = pd.Timedelta(days=STORM_PAD_DAYS)
    for name, a_, b_ in STORMS:
        m = d["Time"].between(pd.Timestamp(a_) - pad, pd.Timestamp(b_) + pad)
        d.loc[m, "tc_day"] = True
        d.loc[m, "storm"] = name
    return d


def season_subset(d, months, tc):
    sub = d[d["month"].isin(months)]
    # complete seasons only: every month of the season present
    present = sub.groupby("season_year")["month"].nunique()
    sub = sub[sub["season_year"].isin(present.index[present == len(set(months))])]
    if tc is not None:
        sub = sub[sub["tc_day"] == tc]
    return sub


def add_standard_seasons(d):
    """
    Add the four standard meteorological seasons.

    DJF is labeled by January/February year, so December belongs to the
    following year (e.g., Dec 2023 + Jan-Feb 2024 = DJF 2024).

    Only a season-year that contains all three calendar months is retained.
    This means that a partial calendar year such as 2024 can still contribute
    every season that is actually complete (e.g., DJF, MAM, and JJA when data
    extend through August).
    """
    d = d.copy()
    d["month"] = d["Time"].dt.month
    d["std_season"] = d["month"].map(STD_SEASON_MAP)
    d["std_season_year"] = d["year"] + (d["month"] == 12).astype(int)

    present = (
        d.groupby(["std_season_year", "std_season"])["month"]
        .nunique()
        .reset_index(name="n_months")
    )
    complete = present[present["n_months"] == 3][["std_season_year", "std_season"]]
    return d.merge(complete, on=["std_season_year", "std_season"], how="inner")


def seasonal_total_series(C):
    """
    Convert county-period probability sums to statewide TOTAL seasonal counts.

    Unlike series_from_counts(), this does NOT normalize to days per 1,000
    county-days.  Class series and any_outage are whole-period statewide
    counts (observed counts or summed predicted probabilities). major_share
    remains a dimensionless share.

    Input
    -----
    C : array [..., source, class]
        Probability/count sums returned by counts_from().
    """
    out = {f"class_{k}": C[..., k] for k in CLASSES}
    tot = C[..., OUTAGE_CLASSES].sum(axis=-1)
    out["any_outage"] = tot
    with np.errstate(invalid="ignore", divide="ignore"):
        out["major_share"] = np.where(tot > 0, C[..., 3] / tot, np.nan)
    return out


def seasonal_total_metrics(C, sources, series_name):
    """KGE components and skill using statewide seasonal TOTALS."""
    s = seasonal_total_series(C.sum(axis=0))
    i = {name: sources.index(name) for name in sources}
    v = s[series_name]
    return metrics(v[:, i["obs"]], v[:, i["model"]],
                   v[:, i["seasonal"]], v[:, i["clim"]])


def bootstrap_standard_season_totals(C, sources, series_name, predictor="model",
                                      n_boot=N_BOOT_STD_SEASON, seed=SEED):
    """
    County-bootstrap 95% CI for statewide seasonal TOTAL predicted counts.

    Counties are sampled with replacement, preserving the same number of
    counties in each replicate. For major_share the bootstrap applies to the
    statewide share rather than a count.
    """
    rng = np.random.default_rng(seed)
    nc = C.shape[0]
    i = {name: sources.index(name) for name in sources}
    draws = np.full((n_boot, C.shape[1]), np.nan)

    for b in range(n_boot):
        pick = rng.integers(0, nc, nc)
        s = seasonal_total_series(C[pick].sum(axis=0))
        draws[b] = s[series_name][:, i[predictor]]

    return (
        np.nanpercentile(draws, 2.5, axis=0),
        np.nanpercentile(draws, 97.5, axis=0),
    )


def plot_standard_season_years(d, out, series_name="any_outage",
                               start_year=STD_SEASON_START, end_year=STD_SEASON_END,
                               n_boot=N_BOOT_STD_SEASON):
    """
    Plot observed and predicted TOTALS for all complete DJF/MAM/JJA/SON
    seasons whose season-year falls within 2015-2024.

    - Uses whole-season statewide counts, not rates per 1,000 county-days.
    - Includes every COMPLETE meteorological season available in the record.
    - The seasonal analysis remains calendar/meteorological-season based; only
      the annual-variability analysis is converted to Oct-Sep water years.
    - With data through September 2024, DJF/MAM/JJA 2024 are complete; SON 2024
      is not complete because October-November 2024 are unavailable.
    - Model bars have 95% county-bootstrap confidence intervals.
    - KGE, r, alpha, beta, and skill are calculated from seasonal totals and
      printed below each panel rather than in a figure title.
    """
    sdir = os.path.join(out, "seasonal")
    os.makedirs(sdir, exist_ok=True)

    d = add_standard_seasons(d)
    d = d[(d["std_season_year"] >= start_year) &
          (d["std_season_year"] <= end_year)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    axes = axes.ravel()
    rows = []

    for ax, season in zip(axes, STD_SEASON_ORDER):
        sub = d[d["std_season"] == season].copy()
        years_available = sorted(sub["std_season_year"].unique())

        if len(years_available) < 3:
            ax.text(0.5, 0.5, f"{season}\ninsufficient complete seasons",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        C, N, counties, years, sources = counts_from(sub, "std_season_year")
        totals = seasonal_total_series(C.sum(axis=0))
        i = {name: sources.index(name) for name in sources}

        obs = totals[series_name][:, i["obs"]]
        mod = totals[series_name][:, i["model"]]
        seas = totals[series_name][:, i["seasonal"]]

        m = seasonal_total_metrics(C, sources, series_name)
        lo, hi = bootstrap_standard_season_totals(
            C, sources, series_name, predictor="model", n_boot=n_boot, seed=SEED
        )

        x = np.arange(len(years))
        width = 0.38

        ax.bar(x - width / 2, obs, width=width,
               facecolor="white", edgecolor="black", linewidth=1.1,
               label="Observed" if season == "DJF" else None, zorder=2)
        ax.bar(x + width / 2, mod, width=width,
               color="#534AB7", alpha=0.9,
               label="Predicted" if season == "DJF" else None, zorder=2)
        ax.errorbar(x + width / 2, mod,
                    yerr=[np.maximum(mod - lo, 0), np.maximum(hi - mod, 0)],
                    fmt="none", ecolor="#2E276E", elinewidth=1.2,
                    capsize=3, zorder=3)
        ax.plot(x, seas, color="#0F6E56", linestyle="--", linewidth=1.2,
                marker="o", markersize=3,
                label="Seasonal baseline" if season == "DJF" else None,
                zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("share" if series_name == "major_share"
                      else "seasonal total county-days")

        # Keep subplot-specific titles like the example figure, but remove any
        # overall figure title.
        metric_text = (
            f"KGE={m['KGE']:.2f}, r={m['r']:.2f}, "
            f"α={m['alpha']:.2f}, β={m['beta']:.2f}, "
            f"skill={m['skill_vs_seasonal']:.2f}"
        )
        ax.set_title(f"{season}\n{metric_text}", fontsize=11)

        rows.append(dict(
            season=season,
            series=series_name,
            n_seasons=len(years),
            years_included=",".join(map(str, years)),
            start_year=int(min(years)),
            end_year=int(max(years)),
            KGE=m["KGE"], r=m["r"], alpha=m["alpha"], beta=m["beta"],
            skill_vs_seasonal=m["skill_vs_seasonal"],
            skill_vs_clim=m["skill_vs_clim"],
            observed_total_all_seasons=float(np.nansum(obs)),
            predicted_total_all_seasons=float(np.nansum(mod)),
        ))

    axes[0].legend(fontsize=8, loc="upper right")

    # No overall figure title.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.09,
                        hspace=0.32, wspace=0.18)

    stem = f"standard_seasons_{series_name}_{start_year}_{end_year}_totals"
    fig.savefig(os.path.join(sdir, stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(sdir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(sdir, stem + "_metrics.csv"), index=False)
    return met

def auc_(y, s_):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s_).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def seasonal_analysis(d, out):
    sdir = os.path.join(out, "seasonal")
    os.makedirs(sdir, exist_ok=True)
    d = tag_days(d)
    i_src = {s_: j for j, s_ in enumerate(["obs"] + PREDICTORS)}

    # 1. season-by-season variability ------------------------------------------
    rows, series_store = [], {}
    for gname, (months, tc) in SEASON_GROUPS.items():
        sub = season_subset(d, months, tc)
        if sub["season_year"].nunique() < 3:
            print(f"  {gname}: fewer than 3 complete seasons, skipped")
            continue
        C, N, counties, keys, sources = counts_from(sub, "season_year")
        point = statewide_metrics(C, N, sources)
        ci = bootstrap(C, N, sources, N_BOOT_SEASON)
        sv = series_from_counts(C.sum(axis=0), N.sum(axis=0))
        series_store[gname] = (keys, sv)
        for name in SEASON_SERIES:
            c = ci[ci["series"] == name].set_index("metric")
            rows.append(dict(season=gname, series=name, n_seasons=len(keys), **point[name],
                             KGE_ci_low=c.loc["KGE", "ci_low"], KGE_ci_high=c.loc["KGE", "ci_high"],
                             skill_ci_low=c.loc["skill_vs_seasonal", "ci_low"],
                             skill_ci_high=c.loc["skill_vs_seasonal", "ci_high"],
                             obs_mean_rate=np.nanmean(sv[name][:, i_src["obs"]]),
                             model_mean_rate=np.nanmean(sv[name][:, i_src["model"]])))
    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(sdir, "season_metrics.csv"), index=False)

    # 2. seasonal cycle and month-to-month anomalies ---------------------------
    Cm, Nm, _, months_k, sources = counts_from(d, "month")
    cyc = series_from_counts(Cm.sum(axis=0), Nm.sum(axis=0))
    cyc_tab = pd.DataFrame({"month": months_k})
    for name in ["class_1", "class_2", "class_3", "any_outage"]:
        for src in ["obs", "model", "seasonal"]:
            cyc_tab[f"{name}_{src}"] = cyc[name][:, i_src[src]]
    cyc_tab.to_csv(os.path.join(sdir, "seasonal_cycle.csv"), index=False)

    d["ym"] = d["year"] * 100 + d["month"]
    Cy, Ny, _, ym, sources = counts_from(d, "ym")
    my = series_from_counts(Cy.sum(axis=0), Ny.sum(axis=0))
    mon = np.array(ym) % 100
    anom_rows = []
    for name in ["class_1", "class_2", "class_3", "any_outage"]:
        for src in ["model", "seasonal"]:
            o = pd.Series(my[name][:, i_src["obs"]]); p = pd.Series(my[name][:, i_src[src]])
            oa = o - o.groupby(mon).transform("mean"); pa = p - p.groupby(mon).transform("mean")
            anom_rows.append(dict(series=name, predictor=src, month_years=len(o),
                                  anomaly_r=np.corrcoef(oa, pa)[0, 1],
                                  seasonal_cycle_r=np.corrcoef(cyc[name][:, i_src["obs"]],
                                                               cyc[name][:, i_src[src]])[0, 1]))
    anom = pd.DataFrame(anom_rows)
    anom.to_csv(os.path.join(sdir, "monthly_anomaly_skill.csv"), index=False)

    # 3. regimes: TC days / other hurricane-season days / off-season -------------
    regimes = {"Tropical-cyclone days": d["tc_day"],
               "Hurricane season, other days": d["month"].between(6, 11) & ~d["tc_day"],
               "Off-season (Dec-May)": ~d["month"].between(6, 11) & ~d["tc_day"]}
    reg_rows = []
    y = d["duration"].to_numpy()
    for rname, m in regimes.items():
        m = m.to_numpy()
        if m.sum() == 0:
            continue
        yy = y[m]
        r_ = dict(regime=rname, county_days=int(m.sum()))
        for k, nm in [(1, "minor"), (2, "moderate"), (3, "major")]:
            r_[f"obs_{nm}_per_1000"] = RATE_SCALE * (yy == k).mean()
            r_[f"model_{nm}_per_1000"] = RATE_SCALE * d.loc[m, f"p_model_{k}"].mean()
        for src in ["model", "seasonal"]:
            p = d.loc[m, [f"p_{src}_{k}" for k in CLASSES]].to_numpy()
            occ = p[:, 1:].sum(axis=1)
            out_m = yy > 0
            r_[f"occ_auc_{src}"] = auc_(out_m, occ)
            r_[f"major_auc_{src}"] = (auc_(yy[out_m] == 3, p[out_m, 3] / np.clip(occ[out_m], 1e-12, None))
                                      if out_m.sum() > 10 else np.nan)
            r_[f"logloss_{src}"] = -np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-15, 1)))
        reg_rows.append(r_)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(os.path.join(sdir, "regime_skill.csv"), index=False)

    # 4. storm by storm ----------------------------------------------------------
    st_rows = []
    for name, a_, b_ in STORMS:
        m = d["storm"] == name
        if not m.any():
            continue
        g = d[m]
        r_ = dict(storm=name, start=a_, end=b_, county_days=len(g))
        for src in ["obs", "model", "seasonal"]:
            r_[f"{src}_any"] = g[[f"p_{src}_{k}" for k in OUTAGE_CLASSES]].to_numpy().sum()
            r_[f"{src}_major"] = g[f"p_{src}_3"].sum()
        st_rows.append(r_)
    storms = pd.DataFrame(st_rows)
    storms.to_csv(os.path.join(sdir, "storm_by_storm.csv"), index=False)

    plot_seasonal(cyc_tab, sm, series_store, reg, storms, sdir)
    return sm, anom, reg, storms


def plot_seasonal(cyc_tab, sm, series_store, reg, storms, sdir):
    col = {"obs": dict(color="black", marker="o", lw=2, label="Observed"),
           "model": dict(color="#534AB7", marker="s", lw=1.6, label="Model"),
           "seasonal": dict(color="#0F6E56", ls="--", lw=1.2, label="Seasonal baseline")}

    # seasonal cycle
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, name in zip(axes, ["class_1", "class_2", "class_3", "any_outage"]):
        ax.axvspan(5.5, 11.5, color="#FAC775", alpha=0.25, lw=0)
        for src, st in col.items():
            ax.plot(cyc_tab["month"], cyc_tab[f"{name}_{src}"], **st)
        ax.set_xticks(range(1, 13)); ax.set_xticklabels("JFMAMJJASOND")
        ax.set_title(TITLES[name], fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"days per {RATE_SCALE:,} county-days")
    axes[0].legend(fontsize=8)
    fig.suptitle("Seasonal cycle (shaded: hurricane season)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "seasonal_cycle.png"), dpi=200); plt.close(fig)

    # season-by-season series
    groups = list(series_store)
    fig, axes = plt.subplots(len(groups), 2, figsize=(11, 2.3 * len(groups)), squeeze=False)
    for r, gname in enumerate(groups):
        keys, sv = series_store[gname]
        for c, name in enumerate(["any_outage", "class_3"]):
            ax = axes[r, c]
            j = {"obs": 0, "model": 1, "seasonal": 2}
            for src, st in col.items():
                ax.plot(keys, sv[name][:, j[src]], **{**st, "ms": 3})
            m = sm[(sm["season"] == gname) & (sm["series"] == name)].iloc[0]
            ax.set_title(f"{gname}: {TITLES[name]}\nKGE={m.KGE:.2f} [{m.KGE_ci_low:.2f}, {m.KGE_ci_high:.2f}]"
                         f"  r={m.r:.2f}  \u03b1={m.alpha:.2f}  skill={m.skill_vs_seasonal:.2f}", fontsize=8)
            ax.set_xticks(keys); ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "season_by_season.png"), dpi=200); plt.close(fig)

    # skill summary heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(groups) + 1.8))
    for ax, metric, title in zip(axes, ["KGE", "skill_vs_seasonal"],
                                 ["KGE (1 = perfect)", "MSE skill vs seasonal baseline (0 = no gain)"]):
        piv = sm.pivot(index="season", columns="series", values=metric).reindex(index=groups,
                                                                                columns=SEASON_SERIES)
        im = ax.imshow(piv.to_numpy(float), cmap="RdYlBu", vmin=-0.5, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SEASON_SERIES)))
        ax.set_xticklabels([TITLES[x].split(" = ")[0] for x in SEASON_SERIES], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "season_skill_summary.png"), dpi=200); plt.close(fig)

    # tropical cyclones and regimes
    if len(storms) or len(reg):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, what, title in zip(axes[:2], ["any", "major"], ["Any weather-related outage days", "Major outage days"]):
            if not len(storms):
                ax.axis("off"); continue
            o = storms[f"obs_{what}"].to_numpy()
            lim = max(o.max(), storms[[f"model_{what}", f"seasonal_{what}"]].to_numpy().max()) * 1.15 + 1
            ax.plot([0, lim], [0, lim], color="k", lw=0.8)
            ax.scatter(o, storms[f"model_{what}"], color="#534AB7", s=30, label="Model", zorder=3)
            ax.scatter(o, storms[f"seasonal_{what}"], color="#0F6E56", marker="x", s=30,
                       label="Seasonal baseline", zorder=3)
            for rr in storms.itertuples():
                ax.annotate(rr.storm, (getattr(rr, f"obs_{what}"), getattr(rr, f"model_{what}")),
                            fontsize=6.5, xytext=(3, 3), textcoords="offset points")
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_xlabel("observed county-days in storm window")
            ax.set_ylabel("predicted county-days")
            ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        ax = axes[2]
        if len(reg):
            x = np.arange(len(reg))
            ax.bar(x - 0.2, reg["occ_auc_model"], 0.4, color="#534AB7", label="outage vs none")
            ax.bar(x + 0.2, reg["major_auc_model"], 0.4, color="#D85A30", label="major vs other outages")
            ax.set_xticks(x); ax.set_xticklabels(reg["regime"], fontsize=8, rotation=15)
            ax.set_ylim(0.5, 1); ax.set_ylabel("daily AUC (model)")
            ax.set_title("Daily skill by regime", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(sdir, "tropical_cyclones_and_regimes.png"), dpi=200)
        plt.close(fig)


# -----------------------------------------------------------------------------
def main(args):
    skip = args.skip_years
    out = os.path.join(args.results, "evaluation" + ("_no" + "_".join(map(str, skip)) if skip else "_allyears"))
    os.makedirs(out, exist_ok=True)

    # Optional fast path: generate only the 2x2 standard meteorological-season figures.
    if args.standard_seasons_only:
        d_all = read_daily(args.results, skip)
        print("\nStandard meteorological-season figures only ...")
        std_any = plot_standard_season_years(d_all, out, series_name="any_outage")
        std_major = plot_standard_season_years(d_all, out, series_name="class_3")
        std_share = plot_standard_season_years(d_all, out, series_name="major_share")
        print("\nStandard seasons: any_outage")
        print(std_any.round(3).to_string(index=False))
        print("\nStandard seasons: class_3")
        print(std_major.round(3).to_string(index=False))
        print("\nStandard seasons: major_share")
        print(std_share.round(3).to_string(index=False))
        print(f"\n-> {os.path.join(out, 'seasonal')}")
        return

    C, N, counties, years, sources = load_counts(args.results, skip)
    print(
        f"{len(counties)} counties, complete water years {years} "
        f"(calendar years explicitly skipped before aggregation: {skip or 'none'})"
    )

    point = statewide_metrics(C, N, sources)
    pt = pd.DataFrame(point).T.rename_axis("series").reset_index()
    print(f"Bootstrapping {N_BOOT} county resamples ...")
    ci = bootstrap(C, N, sources)
    jk, jk_summary = year_jackknife(C, N, years, sources)
    cl, per_county, s = county_level(C, N, counties, sources)

    # headline table: point estimate with CI
    wide = ci.pivot(index="series", columns="metric", values=["ci_low", "ci_high"])
    head = pt.set_index("series")
    for mcol in ["KGE", "r", "alpha", "beta", "skill_vs_seasonal"]:
        head[f"{mcol}_CI"] = [f"[{wide.loc[s_, ('ci_low', mcol)]:.2f}, {wide.loc[s_, ('ci_high', mcol)]:.2f}]"
                              for s_ in head.index]
    head = head[["KGE", "KGE_CI", "r", "r_CI", "alpha", "alpha_CI", "beta", "beta_CI",
                 "skill_vs_seasonal", "skill_vs_seasonal_CI"]].reindex(SERIES)

    pt.to_csv(os.path.join(out, "statewide_metrics.csv"), index=False)
    ci.to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)
    head.reset_index().to_csv(os.path.join(out, "headline_table.csv"), index=False)
    jk.to_csv(os.path.join(out, "water_year_jackknife.csv"), index=False)
    jk_summary.to_csv(os.path.join(out, "water_year_jackknife_summary.csv"), index=False)
    cl.to_csv(os.path.join(out, "county_level_metrics.csv"), index=False)
    per_county.to_csv(os.path.join(out, "per_county_metrics.csv"), index=False)
    plot_yearly(
        C, N, years, sources, point, ci,
        os.path.join(out, "observed_vs_predicted_by_water_year.png"),
        skip
    )
    plot_county(s, sources, per_county, os.path.join(out, "county_level.png"))

    pd.set_option("display.width", 220)
    print("\nStatewide metrics with 95% bootstrap CI (county resampling):")
    print(head.round(3).to_string())
    print("\nWater-year jackknife (range when leaving out one complete Oct-Sep water year):")
    print(jk_summary.round(3).to_string(index=False))
    print("\nCounty-level:")
    print(cl.round(3).to_string(index=False))
    d_all = read_daily(args.results, skip)
    print("\nCalibration ...")
    cal = calibration(d_all, out)
    print(cal.round(3).to_string(index=False))

    print("\nSeasonality ...")
    sm, anom, reg, storms = seasonal_analysis(d_all, out)

    print("\nStandard meteorological-season figures ...")
    std_any = plot_standard_season_years(d_all, out, series_name="any_outage")
    std_major = plot_standard_season_years(d_all, out, series_name="class_3")
    std_share = plot_standard_season_years(d_all, out, series_name="major_share")

    print("\nStandard seasons: any_outage")
    print(std_any.round(3).to_string(index=False))
    print("\nStandard seasons: class_3")
    print(std_major.round(3).to_string(index=False))
    print("\nStandard seasons: major_share")
    print(std_share.round(3).to_string(index=False))
    show = sm[sm["series"].isin(["any_outage", "class_3", "major_share"])][
        ["season", "series", "n_seasons", "KGE", "KGE_ci_low", "KGE_ci_high", "r", "alpha", "beta",
         "skill_vs_seasonal"]]
    print("\nSeason-by-season skill (statewide season totals):")
    print(show.round(3).to_string(index=False))
    print("\nMonth-to-month anomalies (r of observed vs predicted departures from each month's mean):")
    print(anom.round(3).to_string(index=False))
    print("\nDaily skill by regime:")
    print(reg.round(3).to_string(index=False))
    if len(storms):
        tot = storms[["obs_any", "model_any", "seasonal_any", "obs_major", "model_major", "seasonal_major"]].sum()
        print(f"\nTropical-cyclone windows ({len(storms)} storms): outage days observed {tot.obs_any:,.0f}, "
              f"model {tot.model_any:,.0f}, seasonal baseline {tot.seasonal_any:,.0f}; major observed "
              f"{tot.obs_major:,.0f}, model {tot.model_major:,.0f}, seasonal {tot.seasonal_major:,.0f}")
    print(f"\n-> {out}\n-> {os.path.join(out, 'seasonal')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=RESULTS, help="folder containing daily_predictions.csv")
    p.add_argument("--skip-years", nargs="*", type=int, default=SKIP_YEARS,
                   help="calendar years to remove before evaluation (empty = keep all); "
                        "annual variability is then aggregated to complete Oct-Sep water years")
    p.add_argument("--standard-seasons-only", action="store_true",
                   help="generate only the DJF/MAM/JJA/SON standard-season figures and metrics")
    main(p.parse_args())
