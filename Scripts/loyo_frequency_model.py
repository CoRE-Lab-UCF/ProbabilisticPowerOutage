"""
Leave-one-year-out (LOYO) model for weather-related outage FREQUENCY, 4 classes
==============================================================================

Data: CONUS404 daily weather + EAGLE-I outages from build_conus404_outage_dataset.py
(all_counties_classified_stat_all_days_conus404_4class_p80.csv). Every county
has every day; days without EAGLE-I data (outage_data_available = 0) are
PREDICTED but never used for training or evaluation.

Target (column "duration"):
  0 none      no weather-related outage
  1 minor     1 h <= duration < 2 h
  2 moderate  2 h <= duration < 8 h
  3 major     duration >= 8 h
A day is weather-related if 10-m wind (WSPD10_cellmax_dmax) or county-mean
daily rain (PREC_ACC_NC_cellmean_dsum) exceeded the county's 80th percentile of
outage days on that day or the day before.

Goal: hindcast / climate-projection of how many days of each class occur in a
year, and its year-to-year variability.

For each year Y: train a LightGBM multiclass model on the other years
(log-loss objective, no class weights, so probabilities stay calibrated;
hyper-parameters and number of trees chosen by an inner leave-year-out loop
over the training years), predict daily class probabilities for Y, and count
expected days per class = sum of probabilities (argmax counts saved for
contrast).

Feature sets. Except for "full", each set is BASE PLUS ONE GROUP, so every
comparison isolates what that group contributes:
  minimal    : the two label-defining variables only (max 10-m wind and
               county-mean daily rain, same day) - the floor
  base       : all same-day daily weather variables + season
  static     : base + static county attributes (land use, soil type,
               vegetation, elevation, area)
  antecedent : base + backward-looking weather (previous days, 3-day backward
               windows, rain over the previous 3-30 days, wetness index,
               antecedent soil moisture and its monthly anomaly)
  forward    : base + forward-looking weather (next 1-2 days, 3- and 5-day
               forward windows, 5- and 7-day centred windows, high-wind day
               counts) - valid for hindcasting and climate-model output
  forward_plus : base + forward + the interactions that use only same-day or
               forward inputs (full without the static attributes and without
               any backward-looking weather)
  relative   : base + county-NORMALIZED features (soil moisture anomalies,
               wind-exceedance counts, wind and rain divided by the county's own
               90th percentile). These describe how unusual a day is FOR THAT
               COUNTY, so they carry timing but not absolute level.
  forward_static    : base + forward + static attributes
  antecedent_static : base + antecedent + static attributes
               (multi-day windows are regionally coherent; the static attributes
               let the model keep each county's absolute level)
  full       : all of the above, plus interactions (soil / vegetation / forest
               / rain x wind) and county-relative wind and rain (value divided
               by that county's 90th percentile)
Interactions and county-relative variables appear only in "full", so their
contribution is not isolated separately.

Nothing derived from outage data is used as a feature.

Baselines (trained on the other years only): climatology (overall class
proportions) and seasonal (county x month proportions).

Series evaluated across the held-out years (rates per 1,000 county-days):
  class_0 .. class_3, any_outage (classes 1-3), major_share (class 3 / 1-3)
Metrics: KGE, r, alpha, beta, bias, errors, MSE skill vs both baselines.

Usage
-----
  python loyo_frequency_model.py                     # all test years, one after another
  python loyo_frequency_model.py --test-year 2018    # one test year (Slurm array task)
  python loyo_frequency_model.py --merge             # combine finished years -> metrics
  python loyo_frequency_model.py --list-test-years   # years the array should cover
  python loyo_frequency_model.py --list-jobs         # "<feature set> <year>" for every
                                                     # feature set (job list for the array)
  Finished years are saved in <results>/folds/ and skipped on a rerun, so an
  interrupted run continues where it stopped.
  python loyo_frequency_model.py --fast              # one parameter set, no grid
  python loyo_frequency_model.py --drop-vars cape    # also drop CAPE as a feature
  python loyo_frequency_model.py --features static   # base + county attributes
  python loyo_frequency_model.py --skip-test-years 2025   # skip extra test years
                         (partial years are detected and skipped automatically)
  For HPC use, run separate --test-year jobs and call --merge after all
  requested years are finished.
"""

import argparse
import os
import time
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Repository layout used by the public code release.
# Defaults are relative to the repository; every path can be overridden by CLI.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "all_counties_classified_stat_all_days_conus404_4class_p80.csv",
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
AVAIL = "outage_data_available"        # 1 = EAGLE-I data that day (train / evaluate)
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))   # 0 = all cores

TARGET = "duration"
CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
OUTAGE_CLASSES = [1, 2, 3]
RATE_SCALE = 1000                      # rates per 1,000 county-days

# Never features: identifiers, target, and anything derived from outage data
# (extreme_weather uses thresholds computed on outage days)
EXCLUDE = ["Time", "ID", "year", TARGET, "duration_hours", "duration_raw_hours",
           "duration_class", "total_max", "extreme_weather", "extreme_flag",
           AVAIL, "county_id", "county_name", "weather_hours",
           # category codes, not quantities (the _frac columns carry the information)
           "IVGTYP_dominant", "ISLTYP_dominant"]
FORBIDDEN_PATTERNS = ["duration", "total_max", "extreme", "outage", "customers_out"]

# Variables used for time-shifted features; "sum" vars are accumulated in
# windows, all others take the window maximum. Missing columns (e.g. Counts,
# removed from the data) are skipped automatically.
SHIFT_VARS = ["WSPD10_cellmax_dmax", "PREC_ACC_NC_cellmean_dsum", "PREC_ACC_NC_cellmax_dmax",
              "MLCAPE_cellmax_dmax", "GRAUPEL_ACC_NC_cellmax_dsum"]
LEAD_EXTRA = []                        # water level removed from the dataset
SUM_VARS = {"PREC_ACC_NC_cellmean_dsum", "PREC_ACC_NC_cellmax_dsum", "GRAUPEL_ACC_NC_cellmax_dsum"}
WIND_VAR = "WSPD10_cellmax_dmax"       # label-defining wind
PRCP_VAR = "PREC_ACC_NC_cellmean_dsum" # label-defining daily rain
SOIL_VAR = "SMOIS_TOP_cellmean_dmean"
WIND_EXCEED_PCT = 0.95                 # county percentile of daily wind (weather only)

FEATURE_SETS = ["minimal", "base", "static", "antecedent", "forward", "forward_plus",
                "relative", "forward_static", "antecedent_static", "full"]
# Groups each set adds on top of base (same-day weather + season)
SET_GROUPS = {
    "minimal": set(),
    "base": set(),
    "static": {"static"},
    "antecedent": {"antecedent"},
    "forward": {"forward"},
    # full without the static attributes and without any backward-looking weather:
    # keeps only the interactions that do not depend on antecedent inputs
    "forward_plus": {"forward", "derived"},
    # county-normalized features on their own, to show what they do
    "relative": {"relative"},
    # window features together with the static attributes: multi-day windows are
    # regionally coherent and pool counties together, so without something that
    # identifies the county the absolute county level drifts
    "forward_static": {"forward", "static"},
    "antecedent_static": {"antecedent", "static"},
    "full": {"static", "antecedent", "forward", "relative", "derived", "derived_ante",
             "derived_static"},
}
MINIMAL_VARS = ["WSPD10_cellmax_dmax", "PREC_ACC_NC_cellmean_dsum"]   # the label variables
# Static county attributes (county_static file), matched by name or prefix
STATIC_PATTERNS = ["IVGTYP_", "ISLTYP_", "SHDMAX", "SHDMIN", "SHD_range", "HGT",
                   "county_km2", "forest_frac"]
SEASON_COLS = ["month", "doy_sin", "doy_cos"]
# Suffixes identifying derived features (all are built; each set selects some)
ANTECEDENT_SUFFIXES = ("_lag1", "_lag3", "_back3", "_ante3", "_ante7", "_ante14", "_ante30",
                       "_api", "_change3", "_ante3_max", "_ante7_mean")
FORWARD_SUFFIXES = ("_lead1", "_lead2", "_fwd3", "_fwd5", "_mid5", "_mid7")
# County-NORMALIZED features: each is expressed relative to that county's own
# distribution, so it carries timing information but not the county's absolute
# level. Kept in a group of their own: on their own they push every county
# towards a common level and distort the spatial pattern, and they are useful
# only alongside something that identifies the county (static attributes or the
# county-relative scalings).
RELATIVE_NAMES = ("wind_exceed_days_mid5", "wind_exceed_days_fwd5")
RELATIVE_SUFFIXES = ("_anom", "_z", "_rel", "_rel_max2d")
# Interactions and county-relative variables are grouped by the information their
# INPUTS carry, not by their name, so that excluding a group really excludes it.
DERIVED_MARKERS = ("_x_", "_rel")
DERIVED_ANTE_NAMES = ("soil_ante7_x_wind_fwd3", "soil_z_x_wind_fwd3", "api_x_wind_fwd3")
DERIVED_STATIC_NAMES = ("forest_x_wind_fwd3", "veg_x_wind_fwd3")   # use static attributes
SOIL_LAYERS = ["SMOIS_TOP_cellmean_dmean"]        # CONUS404 top soil layer
AREA_PRCP = "PREC_ACC_NC_cellmean_dsum"           # county-average daily rain
AREA_WIND = "WSPD10_cellmean_dmax"                # county-average wind (extent)
# Static vegetation used in the wind interactions (county_static file)
VEG_VAR = "SHDMAX"                                # max green vegetation fraction (%)
FOREST_CLASSES = [1, 2, 3, 4, 5]                  # IGBP/MODIS forest land-use classes
ANTE_PRCP_DAYS = [3, 7, 14, 30]                   # rain in the previous N days
API_DECAY = 0.9                                   # antecedent precipitation index
MIN_COVERAGE = 0.7                                # min share of days for window sums

# County-relative versions of the label-defining variables ("full" set):
# value / county percentile over all days (weather-only climatology)
RELATIVE_VARS = ["WSPD10_cellmax_dmax", "PREC_ACC_NC_cellmean_dsum"]
RELATIVE_PCT = 0.90

LAI_ALIASES = {"max_lai_mean": "lai_max", "mean_lai_mean": "lai_mean"}
LAI_REDUNDANT = ["max_lai_max", "mean_lai_max"]

PARAMS = dict(objective="multiclass", num_class=len(CLASSES), metric="multi_logloss",
              learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=42, num_threads=N_THREADS)
GRID = [dict(num_leaves=31, min_data_in_leaf=100, lambda_l2=1.0),
        dict(num_leaves=63, min_data_in_leaf=50, lambda_l2=5.0),
        dict(num_leaves=127, min_data_in_leaf=30, lambda_l2=5.0)]
MAX_ROUNDS = 2000
EARLY_STOP = 50
# Tuning: training years are split into this many groups of whole years (inner
# cross-validation). 3 instead of leave-one-year-out cuts tuning time ~3x.
INNER_FOLDS = 3

# Extra years not used as test years (still used for training). Partial years
# (< 90% of a full year's county-days, e.g. a first or last year) are detected
# and skipped automatically.
SKIP_TEST_YEARS = []

# Partial final years may be evaluated explicitly even when they contain
# less than 90% of a full year's county-days.
ALLOW_PARTIAL_TEST_YEARS = [2024]

PREDICTORS = ["model", "argmax", "seasonal", "clim"]
SERIES = [f"class_{k}" for k in CLASSES] + ["any_outage", "major_share"]
# =============================================================================


# -----------------------------------------------------------------------------
# Data and features
# -----------------------------------------------------------------------------
def load_data(path, feature_set):
    df = pd.read_csv(path, parse_dates=["Time"])
    for alt, std in LAI_ALIASES.items():
        if alt in df.columns:
            df[std] = df[std].fillna(df[alt]) if std in df.columns else df[alt]
    df = df.drop(columns=[c for c in list(LAI_ALIASES) + LAI_REDUNDANT if c in df.columns])
    df = df.sort_values(["ID", "Time"]).reset_index(drop=True)
    forest = [f"IVGTYP_{k}_frac" for k in FOREST_CLASSES if f"IVGTYP_{k}_frac" in df.columns]
    if forest:
        df["forest_frac"] = df[forest].sum(axis=1)
    df["year"] = df["Time"].dt.year
    df["month"] = df["Time"].dt.month
    doy = df["Time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return add_features(df, feature_set)


def make_grid(df, cols):
    """Every county on a complete daily calendar, so shifts/windows use real days."""
    parts = []
    for cid, g in df.groupby("ID", sort=False):
        days = pd.date_range(g["Time"].min(), g["Time"].max(), freq="D")
        parts.append(g.set_index("Time")[cols].reindex(days).rename_axis("Time").assign(ID=cid))
    return pd.concat(parts).reset_index().set_index(["ID", "Time"])


def add_features(df, feature_set=None):
    """
    Builds every time-shifted and derived feature; feature_columns() then picks
    the ones a given set uses. Offsets are calendar days relative to day t
    (negative = before, positive = after). Windows never cross into another
    county; gaps give NaN rather than a value from the wrong day.
    """
    wanted = SHIFT_VARS + LEAD_EXTRA + SOIL_LAYERS + [AREA_PRCP, AREA_WIND]
    cols = [c for c in dict.fromkeys(wanted) if c in df.columns]
    grid = make_grid(df, cols)
    by_county = grid.groupby(level="ID", sort=False)

    def at(col, k):                                   # value on day t+k
        return by_county[col].shift(-k)

    def window(col, start, end, how=None):            # days t+start .. t+end
        how = how or ("sum" if col in SUM_VARS or col == AREA_PRCP else "max")
        n = end - start + 1
        if how == "sum":   # mean x n, so a few missing days do not bias the total
            fn = lambda x: x.shift(-end).rolling(
                n, min_periods=max(1, int(np.ceil(MIN_COVERAGE * n)))).mean() * n
        elif how == "mean":
            fn = lambda x: x.shift(-end).rolling(
                n, min_periods=max(1, int(np.ceil(MIN_COVERAGE * n)))).mean()
        else:
            fn = lambda x: x.shift(-end).rolling(n, min_periods=1).max()
        return by_county[col].transform(fn)

    new = {}
    shift_vars = [c for c in SHIFT_VARS if c in cols]
    lead_vars = [c for c in SHIFT_VARS + LEAD_EXTRA if c in cols]

    # --- antecedent: previous days and backward windows ----------------------
    for v in shift_vars:
        new[f"{v}_lag1"] = at(v, -1)
        new[f"{v}_back3"] = window(v, -2, 0)
    for v in [c for c in SOIL_LAYERS if c in cols]:
        new[f"{v}_lag1"] = at(v, -1)
        new[f"{v}_lag3"] = at(v, -3)
        new[f"{v}_ante7_mean"] = window(v, -7, -1, "mean")
        new[f"{v}_ante3_max"] = window(v, -3, -1, "max")
        new[f"{v}_change3"] = grid[v] - at(v, -3)      # wetting / drying trend
    if AREA_PRCP in cols:
        for n in ANTE_PRCP_DAYS:
            new[f"{AREA_PRCP}_ante{n}"] = window(AREA_PRCP, -n, -1, "sum")
        new[f"{AREA_PRCP}_api"] = by_county[AREA_PRCP].transform(
            lambda x: x.shift(1).ewm(alpha=1 - API_DECAY, adjust=False, ignore_na=True).mean()
            / (1 - API_DECAY))

    # --- forward: persistence after day t, and centred windows ---------------
    for v in lead_vars:
        new[f"{v}_lead1"] = at(v, 1)
        new[f"{v}_lead2"] = at(v, 2)
        new[f"{v}_fwd3"] = window(v, 0, 2)
        new[f"{v}_mid5"] = window(v, -2, 2)
    if AREA_PRCP in cols:
        new[f"{AREA_PRCP}_fwd5"] = window(AREA_PRCP, 0, 4, "sum")
        new[f"{AREA_PRCP}_mid7"] = window(AREA_PRCP, -3, 3, "sum")
    for v in [c for c in [WIND_VAR, AREA_WIND] if c in cols]:
        new[f"{v}_fwd5"] = window(v, 0, 4, "max")
        new[f"{v}_mid7"] = window(v, -3, 3, "max")
    thr = grid.groupby(level="ID")[WIND_VAR].transform(lambda x: x.quantile(WIND_EXCEED_PCT))
    grid["_wind_exceed"] = (grid[WIND_VAR] > thr).astype(float).where(grid[WIND_VAR].notna())
    by_county = grid.groupby(level="ID", sort=False)
    new["wind_exceed_days_mid5"] = window("_wind_exceed", -2, 2, "sum")
    new["wind_exceed_days_fwd5"] = window("_wind_exceed", 0, 4, "sum")

    feats = pd.DataFrame(new, index=grid.index)
    rows = pd.MultiIndex.from_frame(df[["ID", "Time"]])
    df = pd.concat([df, feats.reindex(rows).set_axis(df.index)], axis=1)

    # --- antecedent: soil moisture anomaly vs the county's normal for that month
    # (weather-only climatology; for projections use the historical period)
    for v in [c for c in SOIL_LAYERS if c in df.columns]:
        grp = df.groupby(["ID", "month"])[v]
        df[f"{v}_anom"] = df[v] - grp.transform("mean")
        df[f"{v}_z"] = df[f"{v}_anom"] / grp.transform("std")

    # --- derived: interactions and county-relative variables ("full" only) ---
    wind_fwd = df[f"{WIND_VAR}_fwd3"]
    df["soil_x_wind"] = df[SOIL_VAR] * df[WIND_VAR]
    df["soil_x_wind_fwd3"] = df[SOIL_VAR] * wind_fwd
    if "forest_frac" in df.columns:
        df["forest_x_wind_fwd3"] = df["forest_frac"] * wind_fwd
    if VEG_VAR in df.columns:
        df["veg_x_wind_fwd3"] = df[VEG_VAR] * wind_fwd
    df["soil_ante7_x_wind_fwd3"] = df[f"{SOIL_VAR}_ante7_mean"] * wind_fwd
    df["soil_z_x_wind_fwd3"] = df[f"{SOIL_VAR}_z"] * wind_fwd
    df["api_x_wind_fwd3"] = df[f"{AREA_PRCP}_api"] * wind_fwd
    df["rain_x_wind_fwd3"] = df[f"{PRCP_VAR}_fwd3"] * wind_fwd
    for v in [c for c in RELATIVE_VARS if c in df.columns]:
        q = df.groupby("ID")[v].transform(lambda x: x.quantile(RELATIVE_PCT)).where(lambda x: x > 0)
        df[f"{v}_rel"] = df[v] / q
        df[f"{v}_rel_max2d"] = np.fmax(df[f"{v}_rel"], df[f"{v}_lag1"] / q)
    return df


def is_static(c):
    return any(c == p or c.startswith(p) for p in STATIC_PATTERNS)


def group_of(c):
    """relative, static, antecedent, forward, derived*, or base (same-day weather + season)."""
    if c in RELATIVE_NAMES or c.endswith(RELATIVE_SUFFIXES):
        return "relative"
    if any(k in c for k in DERIVED_MARKERS):          # interactions: grouped by their inputs
        if c in DERIVED_ANTE_NAMES:
            return "derived_ante"
        if c in DERIVED_STATIC_NAMES:
            return "derived_static"
        return "derived"
    if is_static(c):
        return "static"
    if c.endswith(FORWARD_SUFFIXES):
        return "forward"
    if c.endswith(ANTECEDENT_SUFFIXES):
        return "antecedent"
    return "base"


def feature_columns(df, drop_vars=(), feature_set="full"):
    """Features for one set: base (same-day weather + season) plus its groups."""
    def dropped(c):
        return any(c == v or c.startswith(v + "_") for v in drop_vars)

    usable = [c for c in df.columns
              if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c]) and not dropped(c)]
    if feature_set == "minimal":
        feats = [c for c in usable if c in MINIMAL_VARS]
    else:
        groups = SET_GROUPS[feature_set] | {"base"}
        feats = [c for c in usable if group_of(c) in groups]
    leaked = [c for c in feats if any(p in c.lower() for p in FORBIDDEN_PATTERNS)]
    if leaked:
        raise ValueError(f"Outage-derived columns would be used as features: {leaked}")
    if not feats:
        raise ValueError(f"no features selected for set '{feature_set}'")
    return feats


def prepare(path, feature_set=None):
    df = load_data(path, feature_set)
    if AVAIL not in df.columns:
        df[AVAIL] = 1
    found = sorted(df[TARGET].unique())
    if found != CLASSES:
        raise ValueError(f"expected classes {CLASSES} in '{TARGET}', found {found}. "
                         "Use the _4class file from build_outage_dataset.py")
    return df


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def tune_and_fit(X, y, years, grid, tag=""):
    """
    Inner cross-validation over groups of whole training years picks the
    parameters and number of trees; the final model is refit on all training years.
    """
    uy = np.unique(years)
    group = {yr: i % INNER_FOLDS for i, yr in enumerate(uy)}       # interleaved year groups
    g = np.array([group[yr] for yr in years])
    best = None
    for params in grid:
        losses, iters = [], []
        for k in range(min(INNER_FOLDS, len(uy))):
            t1 = time.time()
            tr, va = g != k, g == k
            dtr = lgb.Dataset(X[tr], y[tr])
            dva = lgb.Dataset(X[va], y[va], reference=dtr)
            m = lgb.train({**PARAMS, **params}, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            losses.append(m.best_score["valid_0"]["multi_logloss"])
            iters.append(m.best_iteration)
            print(f"    {tag} leaves={params['num_leaves']} inner fold {k + 1}/{INNER_FOLDS}: "
                  f"{m.best_iteration} trees, logloss {losses[-1]:.4f} ({time.time() - t1:.0f}s)", flush=True)
        score = np.mean(losses)
        if best is None or score < best[0]:
            best = (score, params, max(int(np.median(iters)), 10))
    score, params, rounds = best
    t1 = time.time()
    model = lgb.train({**PARAMS, **params}, lgb.Dataset(X, y), num_boost_round=rounds)
    print(f"    {tag} final model: {params}, {rounds} trees ({time.time() - t1:.0f}s)", flush=True)
    return model, dict(params=str(params), n_trees=rounds, inner_logloss=score)


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
def climatology_proba(train, n):
    p = train[TARGET].value_counts(normalize=True).reindex(CLASSES, fill_value=0).values
    return np.tile(p, (n, 1))


def seasonal_proba(train, test, prior_weight=20):
    """County x month proportions, shrunk toward the month-wide proportions."""
    onehot = pd.get_dummies(train[TARGET]).reindex(columns=CLASSES, fill_value=0).astype(float)
    month_p = onehot.groupby(train["month"]).mean()
    cm = onehot.groupby([train["ID"], train["month"]])
    key = pd.MultiIndex.from_arrays([test["ID"], test["month"]])
    s = cm.sum().reindex(key).fillna(0).values
    n = cm.size().reindex(key).fillna(0).values[:, None]
    prior = month_p.reindex(test["month"]).fillna(1 / len(CLASSES)).values
    return (s + prior_weight * prior) / (n + prior_weight)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def log_loss(y, p):
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)))


def auc(y, s):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def day_level_scores(y, probs):
    """Daily skill: overall log loss, outage occurrence, and major vs other outages."""
    out = {}
    outage = y > 0
    for name, p in probs.items():
        occ = p[:, OUTAGE_CLASSES].sum(axis=1)
        maj = p[outage, 3] / np.clip(occ[outage], 1e-12, None)
        out[f"logloss_{name}"] = log_loss(y, p)
        out[f"occ_auc_{name}"] = auc(outage, occ)
        out[f"major_auc_{name}"] = auc(y[outage] == 3, maj)
    return out


def kge_parts(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[ok], sim[ok]
    if len(obs) < 3 or obs.std() == 0 or sim.std() == 0:
        return (np.nan,) * 4
    r = np.corrcoef(obs, sim)[0, 1]
    alpha, beta = sim.std() / obs.std(), sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), r, alpha, beta


def summarize(daily, keys):
    """Per group: class counts, and every series for obs and each predictor."""
    names = ["obs"] + PREDICTORS
    g = daily.groupby(keys)
    s = g[[f"p_{m}_{k}" for m in names for k in CLASSES]].sum()
    s["n_days"] = g.size()
    frames = []
    for m in names:
        c = {k: s[f"p_{m}_{k}"] for k in CLASSES}
        out_total = sum(c[k] for k in OUTAGE_CLASSES)
        v = {f"class_{k}": RATE_SCALE * c[k] / s["n_days"] for k in CLASSES}
        v["any_outage"] = RATE_SCALE * out_total / s["n_days"]
        v["major_share"] = c[3] / out_total.replace(0, np.nan)
        frames.append(pd.DataFrame(v).assign(predictor=m))
    long = pd.concat(frames).reset_index().melt(id_vars=keys + ["predictor"], var_name="series")
    series = long.pivot(index=keys + ["series"], columns="predictor", values="value").reset_index()
    series = series[keys + ["series", "obs"] + PREDICTORS]
    counts = s.rename(columns=lambda c: c.replace("p_", "count_") if c.startswith("p_") else c)
    return series, counts.reset_index()


def frequency_metrics(yearly):
    rows = []
    for name in SERIES:
        d = yearly[yearly["series"] == name]
        obs = d["obs"].to_numpy()
        mse_ref = {b: np.mean((d[b].to_numpy() - obs) ** 2) for b in ("clim", "seasonal")}
        for m in PREDICTORS:
            sim = d[m].to_numpy()
            kge, r, a, b = kge_parts(obs, sim)
            mse = np.mean((sim - obs) ** 2)
            rows.append(dict(
                series=name, predictor=m, KGE=kge, r=r, alpha=a, beta=b,
                bias_pct=100 * (sim.mean() - obs.mean()) / obs.mean(),
                mean_abs_err=np.mean(np.abs(sim - obs)),
                mean_abs_pct_err=100 * np.mean(np.abs(sim - obs) / obs),
                skill_vs_clim=1 - mse / mse_ref["clim"] if mse_ref["clim"] > 0 else np.nan,
                skill_vs_seasonal=1 - mse / mse_ref["seasonal"] if mse_ref["seasonal"] > 0 else np.nan))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main LOYO loop
# -----------------------------------------------------------------------------
def year_counts(path):
    """County-days per year, read from the Time column only and cached next to the data."""
    cache = os.path.splitext(path)[0] + ".yearcounts.csv"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        c = pd.read_csv(cache, index_col=0).iloc[:, 0]
        c.index = c.index.astype(int)
        return c
    # take the 4-digit year wherever it sits in the string, so both
    # "2015-09-30" and "9/30/2015" work; much faster than parsing every date
    txt = pd.read_csv(path, usecols=["Time"], dtype=str)["Time"]
    t = txt.str.extract(r"(\d{4})", expand=False)
    if t.isna().any():
        bad = txt[t.isna()].head(3).tolist()
        raise ValueError(f"could not read a year from the Time column, e.g. {bad}")
    t = t.astype(int)
    c = t.value_counts().sort_index()
    try:
        c.to_frame("county_days").to_csv(cache)
    except OSError:
        pass
    return c


def test_years_of(path, skip_extra, allow_partial=None):
    """
    Complete years plus explicitly allowed partial years, minus extra skipped years.
    """
    per_year = year_counts(path)
    partial = [int(y) for y in per_year.index[per_year < 0.9 * per_year.max()]]

    allow_partial = set(allow_partial or [])
    skip_extra = set(skip_extra or [])

    auto_skipped_partial = set(partial) - allow_partial
    skip = sorted(auto_skipped_partial | skip_extra)

    test_years = [int(y) for y in per_year.index if int(y) not in skip]
    allowed_partial = sorted(set(test_years) & set(partial))
    return test_years, partial, skip, allowed_partial


def run_name_of(args):
    # Keep partial-2024 runs separate so old fold files are not silently reused.
    partial_tag = "".join(f"_partial{y}" for y in (args.allow_partial_test_years or []))
    return (args.features
            + "".join(f"_no{v}" for v in args.drop_vars)
            + "".join(f"_skip{y}" for y in (args.skip_test_years or []))
            + partial_tag
            + ("_fast" if args.fast else ""))


def run_fold(df, feats, test_year, grid, t0, partial_test_years=()):
    # Every observed day from every OTHER year is used for training.
    # Thus, when testing 2015-2023, available 2024 data are included in training.
    train = df[(df["year"] != test_year) & (df[AVAIL] == 1)]

    test = df[df["year"] == test_year].copy()

    # For an allowed partial year such as 2024, predict only through the last
    # date with observed outage data.
    if test_year in set(partial_test_years):
        available = test[test[AVAIL] == 1]
        if available.empty:
            raise ValueError(
                f"{test_year} is an allowed partial test year but has no rows with {AVAIL}=1"
            )
        cutoff = available["Time"].max()
        test = test[test["Time"] <= cutoff].copy()
        print(
            f"  {test_year} partial-year test cutoff: {cutoff.date()} "
            f"({len(test):,} county-days predicted)",
            flush=True,
        )

    ok = (test[AVAIL] == 1).to_numpy()
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    model, info = tune_and_fit(train[feats], y_tr, train["year"].to_numpy(), grid, tag=str(test_year))
    probs = {"model": model.predict(test[feats]),
             "seasonal": seasonal_proba(train, test),
             "clim": climatology_proba(train, len(test))}
    fold = dict(year=test_year, n_days=int(ok.sum()), n_days_predicted=len(test), **info,
                **day_level_scores(y_te[ok], {k: v[ok] for k, v in probs.items()}))
    imp = pd.DataFrame(dict(year=test_year, feature=feats, gain=model.feature_importance("gain")))
    out = test[["Time", "ID", "year", AVAIL, TARGET]].copy()
    argmax = probs["model"].argmax(axis=1)
    for k in CLASSES:
        out[f"p_obs_{k}"] = (y_te == k).astype(float)
        out[f"p_argmax_{k}"] = (argmax == k).astype(float)
        for name, p in probs.items():
            out[f"p_{name}_{k}"] = p[:, k]
    print(f"  {test_year} done: {info['n_trees']} trees {info['params']} | logloss model "
          f"{fold['logloss_model']:.4f} vs seasonal {fold['logloss_seasonal']:.4f} | "
          f"outage AUC {fold['occ_auc_model']:.3f} | major AUC {fold['major_auc_model']:.3f} "
          f"(total {time.time() - t0:.0f}s)", flush=True)
    return out, fold, imp


def run(args):
    t0 = time.time()
    test_years, partial, skip, allowed_partial = test_years_of(
        args.data, args.skip_test_years, args.allow_partial_test_years
    )
    if args.list_test_years:
        print("\n".join(map(str, test_years)))
        return
    if args.list_jobs:
        sets = args.feature_sets or FEATURE_SETS
        print("\n".join(f"{fs} {y}" for fs in sets for y in test_years))
        return
    run_name = run_name_of(args)
    out_dir = args.out or os.path.join(RESULTS_DIR, run_name)
    fold_dir = os.path.join(out_dir, "folds")
    os.makedirs(fold_dir, exist_ok=True)

    def done(y):
        return all(os.path.exists(os.path.join(fold_dir, f"{p}_{y}.csv")) for p in ("daily", "fold", "importance"))

    todo = [] if args.merge else ([args.test_year] if args.test_year else test_years)
    if args.test_year and args.test_year not in test_years:
        print(f"{args.test_year} is not a test year ({test_years}); nothing to do")
        return
    todo = [y for y in todo if args.overwrite or not done(y)]

    if todo:
        print(f"[{time.strftime('%H:%M:%S')}] loading {args.data} and building "
              f"'{args.features}' features (a few minutes) ...", flush=True)
        df = prepare(args.data, args.features)
        feats = feature_columns(df, args.drop_vars, args.features)
        print(f"[{time.strftime('%H:%M:%S')}] data ready ({time.time() - t0:.0f}s)", flush=True)
        grid = GRID[1:2] if args.fast else GRID
        counts = df.loc[df[AVAIL] == 1, TARGET].value_counts().sort_index()
        print(f"data: {args.data}\n{len(df):,} rows, {df['ID'].nunique()} counties, "
              f"threads={N_THREADS or 'all'}")
        print(f"test years: {test_years}; not tested but used for training: {skip or 'none'} "
              f"(partial years detected: {partial or 'none'}; "
              f"partial years allowed as tests: {allowed_partial or 'none'})")
        print(f"this run: {todo}  (already finished: {[y for y in test_years if done(y)] or 'none'})")
        print(f"days without outage data (predicted, not trained on or scored): "
              f"{int((df[AVAIL] == 0).sum()):,} of {len(df):,}")
        print("class counts: " + ", ".join(f"{CLASS_NAMES[k]}={counts.get(k, 0):,}" for k in CLASSES))
        by_group = pd.Series([group_of(c) for c in feats]).value_counts().to_dict()
        print(f"feature set '{args.features}': {len(feats)} features {by_group}")
        print(f"{feats}\n", flush=True)
        for y in todo:
            out, fold, imp = run_fold(
                df, feats, y, grid, t0, partial_test_years=allowed_partial
            )
            out.to_csv(os.path.join(fold_dir, f"daily_{y}.csv"), index=False)
            pd.DataFrame([fold]).to_csv(os.path.join(fold_dir, f"fold_{y}.csv"), index=False)
            imp.to_csv(os.path.join(fold_dir, f"importance_{y}.csv"), index=False)
    elif not args.merge:
        print("all requested years already finished (use --overwrite to redo)")

    if args.test_year:                                         # array task: merge later
        return
    missing = [y for y in test_years if not done(y)]
    if missing:
        print(f"\n! not finished yet: {missing}. Metrics use the finished years only; "
              f"rerun (or --merge) after they finish.")
    finished = [y for y in test_years if done(y)]
    if len(finished) < 3:
        print("fewer than 3 finished test years - nothing to summarize yet")
        return
    finalize(fold_dir, finished, out_dir, run_name, t0)


def finalize(fold_dir, years, out_dir, run_name, t0):
    daily = pd.concat([pd.read_csv(os.path.join(fold_dir, f"daily_{y}.csv"), parse_dates=["Time"])
                       for y in years], ignore_index=True)
    fold_rows = pd.concat([pd.read_csv(os.path.join(fold_dir, f"fold_{y}.csv")) for y in years])
    importances = pd.concat([pd.read_csv(os.path.join(fold_dir, f"importance_{y}.csv")) for y in years])
    scored = daily[daily[AVAIL] == 1].copy()                   # evaluation: observed days only
    yearly, yearly_counts = summarize(scored, ["year"])
    county_year, county_year_counts = summarize(scored, ["ID", "year"])
    metrics = frequency_metrics(yearly)
    metrics.insert(0, "run", run_name)

    daily.to_csv(os.path.join(out_dir, "daily_predictions.csv"), index=False)
    yearly.to_csv(os.path.join(out_dir, "yearly_series.csv"), index=False)
    yearly_counts.to_csv(os.path.join(out_dir, "yearly_counts.csv"), index=False)
    county_year.to_csv(os.path.join(out_dir, "county_year_series.csv"), index=False)
    county_year_counts.to_csv(os.path.join(out_dir, "county_year_counts.csv"), index=False)
    metrics.to_csv(os.path.join(out_dir, "frequency_metrics.csv"), index=False)
    fold_rows.to_csv(os.path.join(out_dir, "fold_diagnostics.csv"), index=False)
    (importances.groupby("feature")["gain"].mean().sort_values(ascending=False)
       .reset_index().to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False))
    plot_yearly(yearly, metrics, run_name, os.path.join(out_dir, "observed_vs_predicted_by_year.png"))

    pd.set_option("display.width", 200)
    show = metrics[metrics["predictor"].isin(["model", "seasonal"])].drop(columns="run")
    print(f"\nFrequency metrics across {len(years)} held-out years (rates per 1,000 county-days):")
    print(show.round(3).to_string(index=False))
    m = metrics[(metrics["predictor"] == "model") & metrics["series"].isin(["class_1", "class_2", "class_3"])]
    print(f"\nMean KGE over minor/moderate/major: {m['KGE'].mean():.3f}")
    print(f"Done in {time.time() - t0:.0f}s -> {out_dir}")


def plot_yearly(yearly, metrics, run_name, path):
    style = {"obs": dict(color="black", marker="o", lw=2.2, label="Observed"),
             "model": dict(color="#534AB7", marker="s", lw=1.8, label="Model (sum of probabilities)"),
             "argmax": dict(color="#534AB7", ls=":", marker="x", lw=1.2, label="Model (argmax counts)"),
             "seasonal": dict(color="#0F6E56", ls="--", lw=1.2, label="Seasonal baseline"),
             "clim": dict(color="#888780", ls="--", lw=1.0, label="Climatology baseline")}
    titles = {"class_0": "None", "class_1": "Minor (1-2 h)", "class_2": "Moderate (2-8 h)",
              "class_3": "Major (>= 8 h)", "any_outage": "Any weather-related outage",
              "major_share": "Major share = major / all outage classes"}
    order = ["class_1", "class_2", "class_3", "any_outage", "major_share", "class_0"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, name in zip(axes.flat, order):
        d = yearly[yearly["series"] == name]
        for m, st in style.items():
            if m == "argmax" and name == "major_share":
                continue
            ax.plot(d["year"], d[m], **st)
        r = metrics[(metrics["series"] == name) & (metrics["predictor"] == "model")].iloc[0]
        ax.set_title(f"{titles[name]}\nKGE={r.KGE:.2f}  r={r.r:.2f}  \u03b1={r.alpha:.2f}  "
                     f"\u03b2={r.beta:.2f}", fontsize=10)
        ax.set_ylabel("share" if name == "major_share" else f"days per {RATE_SCALE:,} county-days")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Leave-one-year-out: observed vs predicted ({run_name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", choices=FEATURE_SETS, default="full")
    p.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=None,
                   help="feature sets for --list-jobs (default: all)")
    p.add_argument("--data", default=DATA_PATH, help="modeling CSV written by build_conus404_outage_dataset.py")
    p.add_argument("--out", default=None, help="optional output folder; default: <repo>/results/<feature_set><run suffix>")
    p.add_argument("--fast", action="store_true", help="one parameter set (no grid)")
    p.add_argument("--skip-test-years", nargs="*", type=int, default=SKIP_TEST_YEARS,
                   help="extra years not tested (still used for training); partial years "
                        "are skipped unless explicitly allowed")
    p.add_argument("--allow-partial-test-years", nargs="*", type=int,
                   default=ALLOW_PARTIAL_TEST_YEARS,
                   help="partial years that may still be held out and tested "
                        "(default: 2024); prediction stops at the last date with "
                        "outage_data_available=1")
    p.add_argument("--test-year", type=int, default=None, help="run only this test year (array task)")
    p.add_argument("--merge", action="store_true", help="only combine finished years into metrics")
    p.add_argument("--list-test-years", action="store_true", help="print the test years and exit")
    p.add_argument("--list-jobs", action="store_true",
                   help="print '<feature set> <year>' for every feature set and exit")
    p.add_argument("--overwrite", action="store_true", help="redo years that already finished")
    p.add_argument("--drop-vars", nargs="+", default=[], metavar="VAR",
                   help="exclude these variables and all features derived from them")
    run(p.parse_args())