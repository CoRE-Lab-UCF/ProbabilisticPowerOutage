"""
Figures and tables justifying the weather-related outage definition
=================================================================

Input : daily_base_all_days.csv from build_conus404_outage_dataset.py
        (every county-day with CONUS404 daily weather and raw outage hours).
        Only days with outage data (outage_data_available = 1) are analysed:
        zero outage hours filled into reporting gaps are not observations.
Output: publication figures (PNG 300 dpi + vector PDF) and tables (CSV + one
        Markdown summary with the numbers to cite in the methods section)

Questions answered
------------------
  Which weather variables define a weather-related outage day?
  Which percentile threshold, which threshold basis, same day or day before?
  Which durations separate minor / moderate / major outages?

There is no ground truth for "weather-caused", so candidates are compared on
transparent criteria, all computed among outage days:
  J_8h     share of >= 8 h outage days kept  minus  share of < 8 h days kept
  J_storm  share of outage days in tropical-cyclone windows kept minus share
           of other outage days kept           (Youden index, 0 = random)
  score    mean(J_8h, J_storm)                 -> ranking criterion
  plus sample size (share kept) and stability across years.
Duration classes are compared on class sizes (total and per year) and on how
well weather separates adjacent classes (AUC of the best weather variable).

Figures
  Fig1  outage record: duration distribution, yearly and monthly rates
  Fig2  screening of daily weather variables
  Fig3  all filter candidates: selectivity vs sample size
  Fig4  variable sets at the chosen percentile: selectivity and sample size
  Fig5  percentile sensitivity of the chosen variable set
  Fig6  stability of the kept share across years
  Fig7  duration thresholds
  Fig8  weather by final class
  Fig9  final class frequencies by year and month
  FigS1 time alignment of weather and outage records (from build diagnostics)
  FigS2 outage-data availability by county and year

Usage:  python threshold_justification.py
        python threshold_justification.py --base path/to/daily_base_all_days.csv
"""

import argparse
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# CONFIGURATION
# =============================================================================
# Repository layout used by the public code release.
# Defaults are relative to the repository; every path can be overridden by CLI.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# Preferred input: the unlabelled base table produced by the build script.
BASE_FILE = os.path.join(DATA_DIR, "daily_base_all_days.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "threshold_justification")

RAW = "duration_raw_hours"
CUSTOMERS = "customers_out_peak"
RATE = 1000                                   # per 1,000 county-days

# Candidate filter variables (daily CONUS404 aggregates)
V = {
    "wind": "WSPD10_cellmax_dmax",            # strongest hourly 10-m wind in any cell
    "wind_area": "WSPD10_cellmean_dmax",      # strongest hourly county-mean wind
    "rain_total": "PREC_ACC_NC_cellmean_dsum",   # county-average daily rain
    "rain_total_cellmax": "PREC_ACC_NC_cellmax_dsum",  # sum of the wettest cell per hour
    "rain_intensity": "PREC_ACC_NC_cellmax_dmax",  # heaviest hourly rain in any cell
    "graupel": "GRAUPEL_ACC_NC_cellmax_dsum",  # lightning proxy
    "cape": "MLCAPE_cellmax_dmax",
}
PRETTY = {
    "wind": "wind", "wind_area": "county-mean wind", "rain_total": "daily rain",
    "rain_total_cellmax": "cell-max rain sum", "rain_intensity": "hourly rain intensity",
    "graupel": "graupel", "cape": "CAPE",
}
UNITS = {
    "WSPD10_cellmax_dmax": "Max 10-m wind (m s$^{-1}$)",
    "PREC_ACC_NC_cellmean_dsum": "County-mean daily rain (mm)",
    "PREC_ACC_NC_cellmax_dmax": "Max hourly rain (mm h$^{-1}$)",
    "WSPD10_cellmean_dmax": "County-mean wind (m s$^{-1}$)",
    "customers_out_peak": "Peak customers out",
}
FILTER_VAR_SETS = {
    "wind": ["wind"],
    "rain_total": ["rain_total"],
    "rain_intensity": ["rain_intensity"],
    "wind+rain_total": ["wind", "rain_total"],
    "wind+rain_intensity": ["wind", "rain_intensity"],
    "wind+rain_total_cellmax": ["wind", "rain_total_cellmax"],
    "wind+rain_total+rain_intensity": ["wind", "rain_total", "rain_intensity"],
    "wind_area+rain_total": ["wind_area", "rain_total"],
    "wind+rain_total+graupel": ["wind", "rain_total", "graupel"],
    "wind+rain_total+cape": ["wind", "rain_total", "cape"],
    "wind+rain_total+graupel+cape": ["wind", "rain_total", "graupel", "cape"],
}
PERCENTILES = [0.75, 0.80, 0.85, 0.90, 0.95]
BASES = ["outage_days", "all_days"]
LAGS = [0, 1]

# Chosen configuration (must match build_conus404_outage_dataset.py)
CHOSEN_FILTER = "wind+rain_total|p80|outage_days|lag1"
CHOSEN_A, CHOSEN_B = 2.0, 8.0                 # minor < A <= moderate < B <= major
REFERENCE_FILTERS = {                         # shown for comparison in the tables
    "strict (p90, same day)": "wind+rain_total|p90|outage_days|lag0",
    "all drivers (p90, same day)": "wind+rain_total+graupel+cape|p90|outage_days|lag0",
}

# Shortlists
MIN_PCT_DAYS_KEPT = 0.15
MAX_YEARLY_SHARE_CV = 0.12
MINOR_UPPER = [1.5, 2, 3, 4, 5, 6]
MAJOR_LOWER = [3, 4, 5, 6, 7, 8, 12, 24]
MIN_CLASS_SHARE = 0.10
MIN_CLASS_PER_YEAR = 30
SEPARATION_VARS = ["WSPD10_cellmax_dmax", "WSPD10_cellmax_dmax_fwd3", "WSPD10_cellmean_dmax",
                   "PREC_ACC_NC_cellmean_dsum", "PREC_ACC_NC_cellmean_dsum_fwd3",
                   "PREC_ACC_NC_cellmax_dmax"]

# Tropical cyclones affecting Florida, local dates (verify against NHC TCRs)
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
DURATION_BINS = [1, 2, 4, 8, 12, 24, np.inf]

CLASS_COLORS = {0: "#BBBBBB", 1: "#56B4E9", 2: "#E69F00", 3: "#D55E00"}
CLASS_NAMES = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#E69F00", "#009E73", "#7F7F7F", "#D55E00"
# =============================================================================

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
FULL_W = 7.2                                   # double-column width (inches)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def panel(ax, letter, x=-0.14, y=1.03):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def auc(pos, neg):
    """P(random pos > random neg), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def shift_days(df, col, k):
    key = df.set_index(["ID", "Time"])[col]
    idx = pd.MultiIndex.from_arrays([df["ID"], df["Time"] + pd.Timedelta(days=k)])
    return key.reindex(idx).to_numpy()


def parse_key(key):
    name, p, basis, lag = key.split("|")
    return name, int(p[1:]) / 100, basis, int(lag[3:])


def key_label(key):
    name, p, basis, lag = parse_key(key)
    vars_ = " + ".join(PRETTY[s] for s in FILTER_VAR_SETS[name])
    return f"{vars_}, p{int(p * 100)}, {basis.replace('_', ' ')}, {'day or day before' if lag else 'same day'}"


VAR_WORDS = {"PREC_ACC_NC": "Precipitation", "GRAUPEL_ACC_NC": "Graupel", "WSPD10": "10-m wind",
             "T2": "2-m temperature", "TD2": "2-m dew point", "SMOIS_TOP": "Top soil moisture",
             "MLCAPE": "ML CAPE", "PBLH": "PBL height", "PWAT": "Precipitable water",
             "CANWAT": "Canopy water"}
SPACE_WORDS = {"cellmax": "cell max", "cellmean": "county mean", "areawtd": "area-weighted"}
TIME_WORDS = {"dmax": "daily max", "dmean": "daily mean", "dsum": "daily total"}


def readable(col):
    """PREC_ACC_NC_cellmean_dsum -> 'Precipitation, county mean, daily total'."""
    m = re.match(r"^(.*)_(cellmax|cellmean|areawtd)_(dmax|dmean|dsum)(_fwd3)?$", col)
    if not m:
        return col
    txt = f"{VAR_WORDS.get(m.group(1), m.group(1))}, {SPACE_WORDS[m.group(2)]}, {TIME_WORDS[m.group(3)]}"
    return txt + (", next 3 days" if m.group(4) else "")


def md_table(df, floatfmt=3):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).to_numpy()]
    return "\n".join([head, sep, *rows]) + "\n"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path, parse_dates=["Time"])
    df = df.sort_values(["ID", "Time"]).reset_index(drop=True)
    if "outage_data_available" in df.columns:
        n0 = len(df)
        df = df[df["outage_data_available"] == 1].reset_index(drop=True)
        print(f"  analysing {len(df):,} of {n0:,} county-days with outage data "
              f"({n0 - len(df):,} flagged days without outage data excluded)")
    df["year"] = df["Time"].dt.year
    df["month"] = df["Time"].dt.month
    df["outage"] = df[RAW] > 0
    for col, how in [(V["wind"], "max"), (V["rain_total"], "sum")]:
        if col in df.columns:
            w = np.column_stack([shift_days(df, col, k) for k in range(3)])
            df[f"{col}_fwd3"] = np.nanmax(w, axis=1) if how == "max" else np.nansum(w, axis=1)
    df["storm_day"] = False
    pad = pd.Timedelta(days=STORM_PAD_DAYS)
    for _, a, b in STORMS:
        df.loc[df["Time"].between(pd.Timestamp(a) - pad, pd.Timestamp(b) + pad), "storm_day"] = True
    return df


def weather_columns(df):
    return [c for c in df.columns if re.search(r"_(dmax|dmean|dsum)(_fwd3)?$", c)]


# -----------------------------------------------------------------------------
# Filter evaluation
# -----------------------------------------------------------------------------
def exceed_table(df):
    cache = {}
    for short in sorted({v for vs in FILTER_VAR_SETS.values() for v in vs}):
        col = V[short]
        if col not in df.columns:
            continue
        x = df[col].to_numpy()
        for basis in BASES:
            src = df[df["outage"]] if basis == "outage_days" else df
            for p in PERCENTILES:
                thr = src.groupby("ID")[col].quantile(p).reindex(df["ID"]).to_numpy()
                ex = np.where(thr == 0, x > 0, x > thr)
                ex = np.where(np.isnan(x) | np.isnan(thr), False, ex)
                cache[(short, basis, p, 0)] = ex
                prev = shift_days(df.assign(_e=ex.astype(float)), "_e", -1)
                cache[(short, basis, p, 1)] = ex | (np.nan_to_num(prev) > 0)
    return cache


def flag_for(key, cache):
    name, p, basis, lag = parse_key(key)
    flag = np.zeros(len(next(iter(cache.values()))), bool)
    for s in FILTER_VAR_SETS[name]:
        flag |= cache[(s, basis, p, lag)]
    return flag


def evaluate_flag(df, flag):
    o, h = df["outage"].to_numpy(), df[RAW].to_numpy()
    storm, years = df["storm_day"].to_numpy(), df["year"].to_numpy()
    kept, dropped = o & flag, o & ~flag
    share_y = pd.Series(kept[o]).groupby(years[o]).mean()
    r = dict(kept_days=int(kept.sum()), pct_days_kept=kept.sum() / o.sum(),
             pct_hours_kept=h[kept].sum() / h[o].sum(),
             capture_8h=kept[o & (h >= 8)].mean(), capture_lt8h=kept[o & (h < 8)].mean(),
             storm_capture=kept[o & storm].mean() if (o & storm).any() else np.nan,
             nonstorm_capture=kept[o & ~storm].mean(),
             sep_auc=auc(h[kept], h[dropped]),
             flag_rate_all_days=flag.mean(),
             yearly_share_cv=share_y.std() / share_y.mean() if len(share_y) > 1 else np.nan)
    r["J_8h"] = r["capture_8h"] - r["capture_lt8h"]
    r["J_storm"] = r["storm_capture"] - r["nonstorm_capture"]
    r["score"] = np.nanmean([r["J_8h"], r["J_storm"]])
    return r, share_y


def filter_grid(df, cache):
    rows, yearly = [], {}
    for name, shorts in FILTER_VAR_SETS.items():
        if any((s, BASES[0], PERCENTILES[0], 0) not in cache for s in shorts):
            print(f"  skipping '{name}': variable missing from data")
            continue
        for basis in BASES:
            for p in PERCENTILES:
                for lag in LAGS:
                    key = f"{name}|p{int(round(p * 100))}|{basis}|lag{lag}"
                    r, sy = evaluate_flag(df, flag_for(key, cache))
                    rows.append(dict(filter=key, variables=name, percentile=p, basis=basis, lag=lag, **r))
                    yearly[key] = sy
    g = pd.DataFrame(rows)
    g["passes"] = (g["pct_days_kept"] >= MIN_PCT_DAYS_KEPT) & (g["yearly_share_cv"].fillna(0) <= MAX_YEARLY_SHARE_CV)
    g = g.sort_values(["passes", "score"], ascending=[False, False]).reset_index(drop=True)
    g.insert(0, "rank", pd.Series(np.where(g["passes"], np.arange(1, len(g) + 1), np.nan)).astype("Int64"))
    return g, pd.DataFrame(yearly).T


# -----------------------------------------------------------------------------
# Duration thresholds
# -----------------------------------------------------------------------------
def classify(h, a, b):
    return np.where(h >= b, 3, np.where(h >= a, 2, 1))


def threshold_grid(df, flag):
    kept = df[df["outage"].to_numpy() & flag]
    years = df.groupby("year").size()
    full_years = years.index[years >= 0.9 * years.max()]   # ignore partial first/last year
    sep = [c for c in SEPARATION_VARS if c in df.columns]
    rows = []
    for a in MINOR_UPPER:
        for b in [x for x in MAJOR_LOWER if x > a]:
            cls = classify(kept[RAW].to_numpy(), a, b)
            r = dict(A=a, B=b)
            for k, nm in [(1, "minor"), (2, "moderate"), (3, "major")]:
                m = cls == k
                py = pd.Series(m).groupby(kept["year"].to_numpy()).sum().reindex(full_years, fill_value=0)
                r[f"{nm}_days"], r[f"{nm}_share"] = int(m.sum()), m.mean()
                r[f"{nm}_min_per_year"] = int(py.min())
            for lo, hi, tag in [(1, 2, "mod_vs_minor"), (2, 3, "major_vs_mod")]:
                aucs = {v: auc(kept.loc[cls == hi, v], kept.loc[cls == lo, v]) for v in sep}
                best = max(aucs, key=lambda v: -1 if np.isnan(aucs[v]) else aucs[v])
                r[f"auc_{tag}"], r[f"best_var_{tag}"] = aucs[best], best
            r["mean_adjacent_auc"] = (r["auc_mod_vs_minor"] + r["auc_major_vs_mod"]) / 2
            major_cty = pd.Series(cls == 3).groupby(kept["ID"].to_numpy()).sum()
            r["counties_with_5_major"] = int((major_cty >= 5).sum())
            rows.append(r)
    g = pd.DataFrame(rows)
    ok = np.ones(len(g), bool)
    for nm in ("minor", "moderate", "major"):
        ok &= (g[f"{nm}_share"] >= MIN_CLASS_SHARE) & (g[f"{nm}_min_per_year"] >= MIN_CLASS_PER_YEAR)
    g["passes"] = ok
    g = g.sort_values(["passes", "mean_adjacent_auc"], ascending=[False, False]).reset_index(drop=True)
    g.insert(0, "rank", pd.Series(np.where(g["passes"], np.arange(1, len(g) + 1), np.nan)).astype("Int64"))
    return g, kept


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def fig1_outages(df, out):
    o = df[df["outage"]]
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    # exceedance curve: durations are multiples of 0.25 h, so a histogram on a
    # log axis would show artificial gaps
    h = np.sort(o[RAW].to_numpy())
    exc = 1 - np.arange(len(h)) / len(h)
    ax[0].step(h, exc, where="post", color=BLUE, lw=1.4)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    for t, c in [(CHOSEN_A, CLASS_COLORS[1]), (CHOSEN_B, CLASS_COLORS[3])]:
        share = (h >= t).mean()
        ax[0].axvline(t, color=c, lw=1.2, ls="--")
        ax[0].text(t * 1.08, share * 1.3, f"{share:.0%} $\\geq$ {t:g} h", color=c, fontsize=7)
    ax[0].set_xlabel("Daily outage duration (h)")
    ax[0].set_ylabel("Exceedance share")

    y = df.groupby("year").agg(n=("outage", "size"), o=("outage", "sum"),
                               long=(RAW, lambda s: (s >= CHOSEN_B).sum()), counties=("ID", "nunique"))
    ax[1].bar(y.index, RATE * y["o"] / y["n"], color=BLUE, alpha=0.85, label="all outage days")
    ax[1].bar(y.index, RATE * y["long"] / y["n"], color=RED, alpha=0.9, label=f"$\\geq${CHOSEN_B:g} h")
    ax[1].set_ylabel(f"Days per {RATE:,} county-days"); ax[1].set_xlabel("Year")
    ax[1].legend(frameon=False, loc="upper left", ncol=2, fontsize=6.5)
    ax[1].set_xticks(y.index); ax[1].tick_params(axis="x", rotation=45)
    ax[1].set_ylim(0, RATE * (y["o"] / y["n"]).max() * 1.35)
    m = df.groupby("month").agg(n=("outage", "size"), o=("outage", "sum"),
                                long=(RAW, lambda s: (s >= CHOSEN_B).sum()))
    ax[2].bar(m.index, RATE * m["o"] / m["n"], color=BLUE, alpha=0.85)
    ax[2].bar(m.index, RATE * m["long"] / m["n"], color=RED, alpha=0.9)
    ax[2].set_xticks(range(1, 13)); ax[2].set_xticklabels("JFMAMJJASOND")
    ax[2].set_xlabel("Month")
    for a, l in zip(ax, "abc"):
        panel(a, l)
    fig.tight_layout()
    save(fig, out, "Fig1_outage_record")
    return y


def fig2_screening(scr, out, n=18):
    s = scr.dropna(subset=["auc_ge8h_vs_shorter"]).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 0.17 * len(s) + 0.9))
    yy = np.arange(len(s))
    ax.barh(yy + 0.2, s["auc_outage_vs_none"], height=0.38, color=GREY, label="outage vs no outage")
    ax.barh(yy - 0.2, s["auc_ge8h_vs_shorter"], height=0.38, color=RED, label=f"$\\geq${CHOSEN_B:g} h vs shorter")
    ax.set_yticks(yy); ax.set_yticklabels([readable(v) for v in s["variable"]], fontsize=6.5)
    ax.axvline(0.5, color="k", lw=0.8)
    lo = min(0.45, s[["auc_outage_vs_none", "auc_ge8h_vs_shorter"]].min().min() - 0.02)
    ax.set_xlim(lo, s[["auc_outage_vs_none", "auc_ge8h_vs_shorter"]].max().max() + 0.03)
    ax.set_xlabel("AUC (0.5 = no relation)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, out, "Fig2_variable_screening")


def fig3_tradeoff(g, out):
    names = list(FILTER_VAR_SETS)
    cmap = plt.get_cmap("tab20")
    colors = {n: cmap(i % 20) for i, n in enumerate(names)}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.4), sharey=True)
    for ax, basis, l in zip(axes, BASES, "ab"):
        d = g[g["basis"] == basis]
        for (name, lag), s in d.groupby(["variables", "lag"]):
            ax.scatter(s["pct_days_kept"], s["score"], s=8 + 260 * (s["percentile"] - 0.72),
                       color=colors[name], marker="o" if lag == 0 else "^", alpha=0.8, lw=0.3,
                       edgecolor="k")
        c = g[g["filter"] == CHOSEN_FILTER]
        if len(c) and basis == parse_key(CHOSEN_FILTER)[2]:
            ax.scatter(c["pct_days_kept"], c["score"], marker="*", s=240, color="gold",
                       edgecolor="k", lw=0.8, zorder=5)
            ax.annotate("chosen", (c["pct_days_kept"].iloc[0], c["score"].iloc[0]),
                        xytext=(8, -12), textcoords="offset points", fontsize=7)
        ax.set_title(f"Percentile of {basis.replace('_', ' ')}")
        ax.set_xlabel("Share of outage days kept")
        panel(ax, l)
    axes[0].set_ylabel("Selectivity score  (mean of $J_{8h}$, $J_{storm}$)")
    handles = [plt.Line2D([], [], marker="s", ls="", color=colors[n], label=n.replace("_", " "))
               for n in names if n in set(g["variables"])]
    handles += [plt.Line2D([], [], marker="o", ls="", color="k", mfc="none", label="same day"),
                plt.Line2D([], [], marker="^", ls="", color="k", mfc="none", label="day or day before")]
    handles += [plt.scatter([], [], s=8 + 260 * (p - 0.72), color="grey", label=f"p{int(p * 100)}")
                for p in (PERCENTILES[0], PERCENTILES[-1])]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, out, "Fig3_filter_candidates")


def fig4_variable_sets(g, out):
    _, p, basis, lag = parse_key(CHOSEN_FILTER)
    d = g[(g["percentile"] == p) & (g["basis"] == basis) & (g["lag"] == lag)].set_index("variables")
    d = d.reindex([n for n in FILTER_VAR_SETS if n in d.index])
    chosen = parse_key(CHOSEN_FILTER)[0]
    lab = [n.replace("_", " ").replace("+", " + ") for n in d.index]
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.2), sharey=True)
    y = np.arange(len(d))
    ax[0].barh(y + 0.2, d["J_8h"], 0.38, color=RED, label="$J_{8h}$")
    ax[0].barh(y - 0.2, d["J_storm"], 0.38, color=BLUE, label="$J_{storm}$")
    ax[0].set_xlabel("Youden index (higher = more selective)")
    ax[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax[1].barh(y + 0.2, d["pct_days_kept"], 0.38, color=GREY, label="all outage days")
    ax[1].barh(y - 0.2, d["capture_8h"], 0.38, color=RED, label=f"$\\geq${CHOSEN_B:g} h days")
    ax[1].set_xlabel("Share kept")
    ax[1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=6.8)
    for a in ax:
        if chosen in d.index:
            i = list(d.index).index(chosen)
            a.axhspan(i - 0.5, i + 0.5, color="gold", alpha=0.25, zorder=0)
    ax[0].invert_yaxis()
    panel(ax[0], "a", x=-0.62, y=1.1); panel(ax[1], "b", x=-0.05, y=1.1)
    fig.tight_layout()
    save(fig, out, "Fig4_variable_sets")
    return d.reset_index()


def fig5_percentile(g, out):
    name, _, basis_c, lag_c = parse_key(CHOSEN_FILTER)
    d = g[g["variables"] == name]
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.4))
    styles = {("outage_days", 0): dict(color=BLUE, ls="--", marker="o"),
              ("outage_days", 1): dict(color=BLUE, ls="-", marker="o"),
              ("all_days", 0): dict(color=ORANGE, ls="--", marker="s"),
              ("all_days", 1): dict(color=ORANGE, ls="-", marker="s")}
    for (basis, lag), st in styles.items():
        s = d[(d["basis"] == basis) & (d["lag"] == lag)].sort_values("percentile")
        lab = f"{basis.replace('_', ' ')}, {'day or day before' if lag else 'same day'}"
        x = 100 * s["percentile"]
        ax[0].plot(x, s["pct_days_kept"], ms=3, lw=1.2, label=lab, **st)
        ax[1].plot(x, s["capture_8h"], ms=3, lw=1.2, **st)
        ax[2].plot(x, s["score"], ms=3, lw=1.2, **st)
    ax[0].set_ylabel("Share of outage days kept")
    ax[1].set_ylabel(f"Share of $\\geq${CHOSEN_B:g} h outage days kept")
    ax[2].set_ylabel("Selectivity score")
    p_c = 100 * parse_key(CHOSEN_FILTER)[1]
    for a, l in zip(ax, "abc"):
        a.set_xlabel("Percentile threshold")
        a.axvline(p_c, color="gold", lw=6, alpha=0.35, zorder=0)
        a.set_xticks([75, 80, 85, 90, 95])
        panel(a, l)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "Fig5_percentile_sensitivity")


def fig6_stability(yearly, g, out):
    keys = [CHOSEN_FILTER] + [k for k in REFERENCE_FILTERS.values() if k in yearly.index]
    _, p, basis, lag = parse_key(CHOSEN_FILTER)
    for extra in ["wind+rain_total+rain_intensity", "wind+rain_total+graupel+cape"]:
        k = f"{extra}|p{int(p * 100)}|{basis}|lag{lag}"
        if k in yearly.index and k not in keys:
            keys.append(k)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.75, 3.4))
    for i, k in enumerate(keys):
        s = yearly.loc[k]
        ax.plot(s.index.astype(int), s.to_numpy(float), marker="o", ms=3,
                lw=2 if k == CHOSEN_FILTER else 1, color="k" if k == CHOSEN_FILTER else plt.get_cmap("tab10")(i),
                label=("chosen: " if k == CHOSEN_FILTER else "") + key_label(k))
    ax.set_ylabel("Share of outage days kept"); ax.set_xlabel("Year")
    ax.set_xticks(sorted(int(y) for y in yearly.columns))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    fig.tight_layout()
    save(fig, out, "Fig6_temporal_stability")


def fig7_thresholds(tg, out):
    cols = [("mean_adjacent_auc", "Mean adjacent-class AUC", "viridis", ".2f"),
            ("major_share", "Major share of kept days", "magma", ".2f"),
            ("major_min_per_year", "Fewest major days in a year", "cividis", ".0f")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0))
    for ax, (c, title, cm, f), l in zip(axes, cols, "abc"):
        piv = tg.pivot(index="A", columns="B", values=c)
        z = piv.to_numpy(float)
        im = ax.imshow(z, cmap=cm, aspect="auto", origin="lower")
        lo, hi = np.nanmin(z), np.nanmax(z)
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{x:g}" for x in piv.columns])
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{x:g}" for x in piv.index])
        ax.set_xlabel("B: major if $\\geq$ B h"); ax.set_ylabel("A: minor if < A h")
        ax.set_title(f"({l}) {title}", loc="left", fontsize=8.5); ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                if pd.notna(v):
                    light = (v - lo) / (hi - lo + 1e-12) > 0.6
                    ax.text(j, i, format(v, f), ha="center", va="center", fontsize=5,
                            color="k" if light else "w")
        if CHOSEN_A in piv.index and CHOSEN_B in piv.columns:
            i, j = list(piv.index).index(CHOSEN_A), list(piv.columns).index(CHOSEN_B)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="gold", lw=2))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, out, "Fig7_duration_thresholds")


def fig8_class_weather(kept_cls, out):
    vars_ = [c for c in [V["wind"], V["rain_total"], V["rain_intensity"], V["wind_area"]] if c in kept_cls.columns]
    fig, axes = plt.subplots(1, len(vars_), figsize=(FULL_W, 2.4))
    for ax, v, l in zip(np.atleast_1d(axes), vars_, "abcd"):
        data = [kept_cls.loc[kept_cls["cls"] == k, v].dropna().to_numpy() for k in (0, 1, 2, 3)]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="k", lw=1))
        for patch, k in zip(bp["boxes"], (0, 1, 2, 3)):
            patch.set_facecolor(CLASS_COLORS[k]); patch.set_alpha(0.9)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["other*", "minor", "moderate", "major"], fontsize=6.5, rotation=35, ha="right")
        ax.set_ylabel(UNITS.get(v, readable(v)), fontsize=7.5)
        panel(ax, l, x=-0.3)
    fig.text(0.01, -0.03, "* outage days not classified as weather-related (dropped by the filter)", fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "Fig8_weather_by_class")


def fig9_final(df, out):
    cls = df["final_class"]
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.5))
    for key, a in [("year", ax[0]), ("month", ax[1])]:
        t = pd.crosstab(df[key], cls, normalize="index") * RATE
        bottom = np.zeros(len(t))
        for k in (1, 2, 3):
            v = t[k].to_numpy() if k in t.columns else np.zeros(len(t))
            a.bar(t.index, v, bottom=bottom, color=CLASS_COLORS[k], label=CLASS_NAMES[k])
            bottom += v
        a.set_ylabel(f"Days per {RATE:,} county-days")
    ax[0].set_xlabel("Year"); ax[0].set_xticks(sorted(df["year"].unique())); ax[0].tick_params(axis="x", rotation=45)
    ax[1].set_xlabel("Month"); ax[1].set_xticks(range(1, 13)); ax[1].set_xticklabels("JFMAMJJASOND")
    ax[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=7)
    panel(ax[0], "a", x=-0.2); panel(ax[1], "b", x=-0.2)
    fig.tight_layout()
    save(fig, out, "Fig9_final_class_frequency")


def figS1_alignment(out, data_dir):
    path = os.path.join(data_dir, "diagnostics", "time_alignment.csv")
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.55, 2.4))
    for (v, d), st in zip(m.groupby("variable"), ["-", "--"]):
        ax.plot(d["lag_h"], d["mean"], st, color=BLUE, lw=1.4, label=v)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("Lag (h): outage record time minus weather time")
    ax.set_ylabel("Mean correlation")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "FigS1_time_alignment")
    return m.loc[m.groupby("variable")["mean"].idxmax(), ["variable", "lag_h", "mean"]]


def figS2_coverage(full, out):
    """Share of days with EAGLE-I outage data, by county and year (all days are in the dataset)."""
    if "outage_data_available" not in full.columns:
        return
    t = full.groupby(["ID", full["Time"].dt.year])["outage_data_available"].mean().unstack()
    t = t.loc[sorted(t.index, key=lambda s_: int(s_.split("_")[1]))]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.55, 6.0))
    im = ax.imshow(t.to_numpy(), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(t.columns))); ax.set_xticklabels(t.columns, rotation=45)
    ax.set_yticks(range(len(t.index))); ax.set_yticklabels(t.index, fontsize=4.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, label="share of days with outage data")
    fig.tight_layout()
    save(fig, out, "FigS2_data_coverage")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    out = args.out
    tab = os.path.join(out, "tables")
    os.makedirs(tab, exist_ok=True)
    print(f"Reading {args.base}")
    full = pd.read_csv(args.base, usecols=lambda c: c in {"ID", "Time", "outage_data_available"},
                       parse_dates=["Time"])
    df = load(args.base)
    o = df[df["outage"]]
    md = ["# Justification of the weather-related outage definition\n",
          f"Data: {len(df):,} county-days, {df['ID'].nunique()} counties, "
          f"{df['Time'].min().date()} to {df['Time'].max().date()}\n"]

    # --- Table 1: data summary --------------------------------------------------
    h = o[RAW]
    t1 = pd.DataFrame([
        ("Counties", df["ID"].nunique()), ("Period", f"{df['Time'].min().date()} to {df['Time'].max().date()}"),
        ("County-days in the dataset", f"{len(full):,}"),
        ("County-days with outage data (analysed)", f"{len(df):,}"),
        ("Outage days (any duration >= 1 h)", f"{len(o):,}"),
        (f"Outage days per {RATE:,} county-days", f"{RATE * len(o) / len(df):.1f}"),
        ("Median daily outage duration (h)", f"{h.median():.2f}"),
        ("90th / 99th percentile duration (h)", f"{h.quantile(.9):.2f} / {h.quantile(.99):.2f}"),
        (f"Share of outage days >= {CHOSEN_B:g} h", f"{(h >= CHOSEN_B).mean():.3f}"),
        ("Outage days in tropical-cyclone windows", f"{int(o['storm_day'].sum()):,}"),
    ], columns=["Quantity", "Value"])
    t1.to_csv(os.path.join(tab, "Table1_data_summary.csv"), index=False)
    md += ["## Table 1. Data summary\n", md_table(t1)]
    print("Fig 1: outage record")
    yearly_rates = fig1_outages(df, out)

    # --- Table 2: screening ------------------------------------------------------
    print("Fig 2: variable screening")
    rows = []
    for c in weather_columns(df):
        rows.append(dict(variable=c,
                         auc_outage_vs_none=auc(df.loc[df["outage"], c], df.loc[~df["outage"], c]),
                         auc_ge8h_vs_shorter=auc(o.loc[o[RAW] >= CHOSEN_B, c], o.loc[o[RAW] < CHOSEN_B, c]),
                         spearman_hours=o[[c, RAW]].corr(method="spearman").iloc[0, 1],
                         pct_missing=100 * df[c].isna().mean()))
    scr = pd.DataFrame(rows).sort_values("auc_ge8h_vs_shorter", ascending=False)
    scr.insert(1, "description", scr["variable"].map(readable))
    scr.to_csv(os.path.join(tab, "Table2_variable_screening.csv"), index=False)
    fig2_screening(scr, out)
    md += ["## Table 2. Screening of daily weather variables (top 15)\n", md_table(scr.head(15))]

    # --- Filter grid -------------------------------------------------------------
    print("Figs 3-6: filter candidates")
    cache = exceed_table(df)
    g, yearly = filter_grid(df, cache)
    g.to_csv(os.path.join(tab, "TableS_filter_grid_all.csv"), index=False)
    yearly.to_csv(os.path.join(tab, "TableS_filter_yearly_kept_share.csv"))
    show = ["rank", "filter", "pct_days_kept", "capture_8h", "storm_capture", "J_8h", "J_storm",
            "score", "yearly_share_cv"]
    refs = [CHOSEN_FILTER] + list(REFERENCE_FILTERS.values())
    t3 = pd.concat([g.head(10), g[g["filter"].isin(refs) & ~g.index.isin(range(10))]])[show]
    t3.insert(1, "note", t3["filter"].map({CHOSEN_FILTER: "chosen",
                                           **{v: k for k, v in REFERENCE_FILTERS.items()}}).fillna(""))
    t3.to_csv(os.path.join(tab, "Table3_filter_candidates.csv"), index=False)
    fig3_tradeoff(g, out)
    t4 = fig4_variable_sets(g, out)
    t4 = t4[["variables", "pct_days_kept", "capture_8h", "storm_capture", "J_8h", "J_storm", "score",
             "yearly_share_cv"]]
    t4.to_csv(os.path.join(tab, "Table4_variable_sets.csv"), index=False)
    fig5_percentile(g, out)
    name_c, _, basis_c, lag_c = parse_key(CHOSEN_FILTER)
    t5 = g[(g["variables"] == name_c)].sort_values(["basis", "lag", "percentile"])[
        ["basis", "lag", "percentile", "pct_days_kept", "capture_8h", "storm_capture", "score",
         "yearly_share_cv", "kept_days"]]
    t5.to_csv(os.path.join(tab, "Table5_percentile_sensitivity.csv"), index=False)
    fig6_stability(yearly, g, out)
    ch = g[g["filter"] == CHOSEN_FILTER].iloc[0]
    md += ["## Table 3. Filter candidates (top 10, chosen, references)\n",
           f"Ranking: score = mean(J_8h, J_storm) among candidates keeping >= {MIN_PCT_DAYS_KEPT:.0%} "
           f"of outage days with yearly-share CV <= {MAX_YEARLY_SHARE_CV}. "
           f"{int(g['passes'].sum())} of {len(g)} candidates pass.\n", md_table(t3),
           "## Table 4. Variable sets at the chosen percentile, basis and lag\n", md_table(t4),
           "## Table 5. Percentile sensitivity of the chosen variable set\n", md_table(t5)]

    # --- Duration thresholds -----------------------------------------------------
    print("Figs 7-8: duration thresholds")
    flag = flag_for(CHOSEN_FILTER, cache)
    tg, kept = threshold_grid(df, flag)
    tg.to_csv(os.path.join(tab, "TableS_threshold_grid_all.csv"), index=False)
    t6cols = ["rank", "A", "B", "minor_days", "moderate_days", "major_days", "major_min_per_year",
              "auc_mod_vs_minor", "auc_major_vs_mod", "mean_adjacent_auc", "counties_with_5_major"]
    t6 = pd.concat([tg.head(10), tg[(tg["A"] == CHOSEN_A) & (tg["B"] == CHOSEN_B) & ~tg.index.isin(range(10))]])[t6cols]
    t6.to_csv(os.path.join(tab, "Table6_duration_thresholds.csv"), index=False)
    fig7_thresholds(tg, out)
    oc = o.copy()
    oc["cls"] = 0
    kmask = flag[df["outage"].to_numpy()]
    oc.loc[kmask, "cls"] = classify(oc.loc[kmask, RAW].to_numpy(), CHOSEN_A, CHOSEN_B)
    fig8_class_weather(oc, out)
    t8 = []
    for v in [c for c in [V["wind"], V["rain_total"], V["rain_intensity"], V["wind_area"], CUSTOMERS]
              if c in oc.columns]:
        for k in (0, 1, 2, 3):
            s = oc.loc[oc["cls"] == k, v].dropna()
            t8.append(dict(variable=v, cls=CLASS_NAMES[k] if k else "not weather-related",
                           n=len(s), median=s.median(), q25=s.quantile(.25), q75=s.quantile(.75)))
    t8 = pd.DataFrame(t8)
    t8.to_csv(os.path.join(tab, "Table8_weather_by_class.csv"), index=False)
    chosen_row = tg[(tg["A"] == CHOSEN_A) & (tg["B"] == CHOSEN_B)]
    md += ["## Table 6. Duration thresholds for the chosen filter\n",
           f"Classes: minor [1, A) h, moderate [A, B) h, major >= B h. Shortlist: every class >= "
           f"{MIN_CLASS_SHARE:.0%} of kept days and >= {MIN_CLASS_PER_YEAR} per year; ranked by the mean "
           f"adjacent-class AUC. {int(tg['passes'].sum())} of {len(tg)} pass.\n", md_table(t6),
           "## Table 8. Weather by class (median, IQR)\n", md_table(t8)]

    # --- Final classes -------------------------------------------------------------
    print("Fig 9: final classes")
    h_all = df[RAW].to_numpy()
    kept_all = df["outage"].to_numpy() & flag
    df["final_class"] = np.where(~kept_all, 0, classify(h_all, CHOSEN_A, CHOSEN_B))
    fig9_final(df, out)
    t7 = pd.crosstab(df["year"], df["final_class"]).rename(columns=CLASS_NAMES)
    t7["county_days"] = t7.sum(axis=1)
    for k in ("minor", "moderate", "major"):
        if k in t7.columns:
            t7[f"{k}_per_{RATE}"] = (RATE * t7[k] / t7["county_days"]).round(1)
    t7 = t7.reset_index()
    t7.to_csv(os.path.join(tab, "Table7_final_class_counts.csv"), index=False)
    md += ["## Table 7. Final class counts by year\n", md_table(t7)]

    # --- Supplementary -------------------------------------------------------------
    print("Supplementary figures")
    peaks = figS1_alignment(out, os.path.dirname(args.base))
    figS2_coverage(full, out)

    # --- Numbers to cite -------------------------------------------------------------
    rank = int(ch["rank"]) if pd.notna(ch["rank"]) else "not shortlisted"
    best = g.iloc[0]
    tot = {CLASS_NAMES[k]: int((df["final_class"] == k).sum()) for k in (0, 1, 2, 3)}
    cite = [
        "## Numbers for the methods section\n",
        f"- Chosen filter: {key_label(CHOSEN_FILTER)}; rank {rank} of {len(g)} "
        f"(score {ch['score']:.3f}; best candidate {best['filter']} with {best['score']:.3f}).",
        f"- It keeps {ch['pct_days_kept']:.1%} of outage days, {ch['capture_8h']:.1%} of outage days "
        f">= {CHOSEN_B:g} h and {ch['storm_capture']:.1%} of outage days in tropical-cyclone windows.",
        f"- Year-to-year CV of the kept share {ch['yearly_share_cv']:.3f}.",
        f"- Duration classes: minor [1, {CHOSEN_A:g}) h, moderate [{CHOSEN_A:g}, {CHOSEN_B:g}) h, "
        f"major >= {CHOSEN_B:g} h; rank "
        f"{int(chosen_row['rank'].iloc[0]) if len(chosen_row) and pd.notna(chosen_row['rank'].iloc[0]) else 'n/a'} "
        f"of {len(tg)} threshold pairs.",
        f"- Final counts: " + ", ".join(f"{k} {v:,}" for k, v in tot.items()) + ".",
    ]
    if peaks is not None:
        cite.append("- Time alignment: correlation peaks at " + ", ".join(
            f"{r.variable} {r.lag_h:+d} h" for r in peaks.itertuples()) +
            " (both sources in UTC; outages follow the weather within hours).")
    md += cite
    with open(os.path.join(out, "justification_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(cite))
    print(f"\nFigures -> {out}\nTables  -> {tab}\nSummary -> {os.path.join(out, 'justification_summary.md')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=BASE_FILE,
                   help="daily_base_all_days.csv from build_conus404_outage_dataset.py")
    p.add_argument("--out", default=OUT_DIR,
                   help="directory for figures, tables, and the justification summary")
    main(p.parse_args())
