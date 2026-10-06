"""
Daily county dataset of weather-related power outages (CONUS404 + EAGLE-I)
=========================================================================

Replaces the old daily weather files with CONUS404 hourly county weather and
reads the updated EAGLE-I export (one CSV per Florida county, 2015 onward).
The OUTAGE PROCESSING is the same as in build_outage_dataset.py.

Stages
------
  1. Weather   hourly CONUS404 county parquet files -> daily per county, on
               LOCAL calendar days (America/New_York):
                 every variable : daily max  (_dmax) and daily mean (_dmean)
                 accumulations  : also daily total (_dsum); for precipitation
                                  PREC_ACC_NC_cellmean_dsum is the county-average
                                  daily rain, PREC_ACC_NC_cellmax_dsum sums the
                                  wettest cell of each hour (an upper bound)
  2. Outages   per county (same method as build_outage_dataset.py):
                 - records with customers out above the county's 85th percentile
                 - records less than 4 h apart form one event
                 - event duration = number of 15-min records x 0.25 h
                 - events shorter than 1 h are dropped
                 - daily outage hours = sum of events STARTING that local day
  3. Coverage  EVERY county gets EVERY day of the study period (no day is
               dropped). Days without usable outage data are kept, with zero
               outage hours, and flagged outage_data_available = 0:
                 - before a county's first / after its last EAGLE-I record
                 - gaps with no EAGLE-I record for > 7 days in a county
                 - days when < 20% of counties report anything (system outage)
                   and this county has no record either
               Exclude flagged days when training / evaluating, or keep them
               for prediction only: their zero outage is not an observation.
  4. Base      weather + outage hours + static attributes for every county-day
               -> daily_base_all_days.csv  (input of threshold_justification.py)
  5. Labels    weather-related outage classes from the chosen configuration
               -> modeling files (all days and outage days only)

Diagnostics (written to diagnostics/): time alignment of weather and outage
timestamps, data coverage, per-county outage summary.

Important data-handling choices
-------------------------------
  * Only rows with a missing outage count are dropped. The total-customer field
    is not required for event detection.
  * Zero-customer records are excluded from the 85th-percentile outage threshold
    so a reporting-convention change does not shift the event threshold.
  * Both source timestamps are treated as UTC, then aggregated to local Florida
    calendar days so weather and outage observations are aligned consistently.

Usage
-----
  python build_conus404_outage_dataset.py                 # all stages
  python build_conus404_outage_dataset.py --stage label   # relabel from cached base
  python build_conus404_outage_dataset.py --max-counties 5   # quick test
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# Repository layout used by the public code release.
# Defaults are relative to the repository; every path can be overridden by CLI.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Expected repository layout. Override these paths on the command line when the
# raw data are stored elsewhere.
OUTAGE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "eagle_i")
WEATHER_GLOB = os.path.join(PROJECT_ROOT, "data", "raw", "conus404", "*.parquet")
STATIC_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "conus404", "county_static.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Time handling
WEATHER_TZ = "UTC"                 # CONUS404 timestamps
OUTAGE_TZ = "UTC"                  # EAGLE-I run_start_time
DAY_TZ = "America/New_York"        # calendar days used for daily aggregation
MIN_HOURS_PER_DAY = 12             # only used to trim the FIRST and LAST day of the
                                   # weather record (partial because of the UTC-to-
                                   # local shift, e.g. 5 h on the day before the record
                                   # starts). Every day in between is kept; the last day
                                   # (19 of 24 h) is kept and marked by weather_hours
START_DATE = None                  # e.g. "2017-01-01" to restrict the study period
END_DATE = None                    # default: full days covered by both sources

# Weather aggregation: variables whose name contains this get a daily total too
SUM_PATTERN = "_ACC_"
WEATHER_ID_COLS = ["time", "county_id", "county_name"]

# Outage processing (same as build_outage_dataset.py)
OUTAGE_QUANTILE = 0.85
EVENT_GAP_HOURS = 4
POINT_DURATION_HOURS = 0.25
MIN_EVENT_HOURS = 1.0
EXCLUDE_ZERO_RECORDS_FROM_QUANTILE = True

# Coverage flags (days are kept; these only set outage_data_available = 0)
MAX_RECORD_GAP_DAYS = 7            # longer gaps without any record -> no data
MIN_STATEWIDE_REPORTING_SHARE = 0.20

# Weather-related outage labels (update after threshold_justification.py)
FILTER_VARS = ["WSPD10_cellmax_dmax", "PREC_ACC_NC_cellmean_dsum"]
FILTER_PERCENTILE = 0.80
FILTER_BASIS = "outage_days"       # percentile over the county's outage days
FILTER_LAG_DAYS = 1                # exceedance on day t or up to 1 day before
MINOR_UPPER = 2.0                  # minor: 1 h <= d < 2 h
MAJOR_LOWER = 8.0                  # moderate: 2 <= d < 8 h, major: d >= 8 h
CLASS_NAMES = {0: "none", 1: "minor", 2: "moderate", 3: "major"}

# Time-alignment diagnostic
ALIGN_VARS = ["PREC_ACC_NC_cellmean", "WSPD10_cellmax"]
ALIGN_LAGS = list(range(-24, 49))  # hours; positive = outage after weather

# Outcome columns kept in the base file for analysis but removed from the
# modeling file (they are derived from outage records)
OUTCOME_DIAGNOSTIC_COLS = ["n_outage_events", "customers_out_peak", "outage_records"]

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


def norm_name(name):
    s = str(name).lower()
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"_?county$", "", s.strip())
    return re.sub(r"[^a-z0-9]", "", s)


NAME_LOOKUP = {norm_name(c): c for c in COUNTY_IDS}


def to_local_date(ts, source_tz):
    """Naive timestamps in source_tz -> naive local calendar date (midnight)."""
    ts = pd.to_datetime(ts)
    return ts.dt.tz_localize(source_tz).dt.tz_convert(DAY_TZ).dt.tz_localize(None).dt.normalize()


def in_period(dates):
    keep = pd.Series(True, index=dates.index)
    if START_DATE:
        keep &= dates >= pd.Timestamp(START_DATE)
    if END_DATE:
        keep &= dates <= pd.Timestamp(END_DATE)
    return keep


# -----------------------------------------------------------------------------
# 1. Weather: hourly -> daily (local days)
# -----------------------------------------------------------------------------
def aggregate_weather(files, max_counties=None):
    """
    Each file is reduced to partial daily sums / maxima / hour counts, then the
    partials are combined, so local days that straddle two files (year ends)
    are aggregated correctly.
    """
    partials, hourly, names = [], [], {}
    for i, f in enumerate(files, 1):
        t0 = time.time()
        d = pd.read_parquet(f)
        d["county_id"] = d["county_id"].astype(str).str.zfill(5)
        if max_counties:
            d = d[d["county_id"].isin(sorted(d["county_id"].unique())[:max_counties])]
        if "county_name" in d.columns:
            names.update(d.drop_duplicates("county_id").set_index("county_id")["county_name"].to_dict())
        num = [c for c in d.columns if c not in WEATHER_ID_COLS and pd.api.types.is_numeric_dtype(d[c])]
        d["Time"] = to_local_date(d["time"], WEATHER_TZ)
        g = d.groupby(["county_id", "Time"])
        part = pd.concat([g[num].sum().add_suffix("__sum"), g[num].max().add_suffix("__max"),
                          g.size().rename("__n")], axis=1)
        partials.append(part)
        keep = [c for c in ALIGN_VARS if c in d.columns]
        hourly.append(d[["county_id", "time"] + keep].astype({"county_id": "category",
                                                              **{k: "float32" for k in keep}}))
        print(f"  weather file {i}/{len(files)}: {os.path.basename(f)}  {len(d):,} rows "
              f"({time.time() - t0:.0f}s)", flush=True)

    allp = pd.concat(partials)
    sum_cols = [c for c in allp.columns if c.endswith("__sum")] + ["__n"]
    max_cols = [c for c in allp.columns if c.endswith("__max")]
    comb = pd.concat([allp[sum_cols].groupby(level=[0, 1]).sum(),
                      allp[max_cols].groupby(level=[0, 1]).max()], axis=1)

    out = pd.DataFrame(index=comb.index)
    base_vars = [c[:-5] for c in comb.columns if c.endswith("__max")]
    for v in base_vars:
        out[f"{v}_dmax"] = comb[f"{v}__max"]
        out[f"{v}_dmean"] = comb[f"{v}__sum"] / comb["__n"]
        if SUM_PATTERN in v:
            out[f"{v}_dsum"] = comb[f"{v}__sum"]
    out["weather_hours"] = comb["__n"]
    out = out.reset_index()
    # trim only the partial first / last day of the whole record
    full = out.groupby("Time")["weather_hours"].max() >= MIN_HOURS_PER_DAY
    first, last = full[full].index.min(), full[full].index.max()
    out = out[(out["Time"] >= first) & (out["Time"] <= last)]
    out["county_name"] = out["county_id"].map(names)
    short = int((out["weather_hours"] < MIN_HOURS_PER_DAY).sum())
    print(f"  daily weather: {len(out):,} county-days, {out['county_id'].nunique()} counties, "
          f"{first.date()} to {last.date()} (kept {short} interior county-days with "
          f"< {MIN_HOURS_PER_DAY} hours; see weather_hours)")
    hourly = pd.concat(hourly, ignore_index=True)
    return out, hourly, names


# -----------------------------------------------------------------------------
# 2. Outages (method of build_outage_dataset.py)
# -----------------------------------------------------------------------------
def read_outage_file(path):
    d = pd.read_csv(path, usecols=lambda c: c in {"fips_code", "county", "state", "sum",
                                                   "run_start_time", "total_customers"})
    if "state" in d.columns:
        d = d[d["state"].astype(str).str.lower() == "florida"]
    d["time"] = pd.to_datetime(d["run_start_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    d["sum"] = pd.to_numeric(d["sum"], errors="coerce")
    d = d.dropna(subset=["time", "sum"])                 # only rows missing what we need
    d["county_id"] = d["fips_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    return d[["county_id", "county", "time", "sum", "total_customers"]]


def detect_events(d):
    """Events for one county: records above the county's 85th percentile of customers out."""
    basis = d.loc[d["sum"] > 0, "sum"] if EXCLUDE_ZERO_RECORDS_FROM_QUANTILE else d["sum"]
    thr = basis.quantile(OUTAGE_QUANTILE)
    a = d[d["sum"] > thr].sort_values("time")
    if a.empty:
        return pd.DataFrame(columns=["start", "duration", "peak"]), thr
    eid = (a["time"].diff() > pd.Timedelta(hours=EVENT_GAP_HOURS)).cumsum()
    ev = a.groupby(eid).agg(start=("time", "min"), n=("time", "size"), peak=("sum", "max"))
    ev["duration"] = ev["n"] * POINT_DURATION_HOURS
    return ev.loc[ev["duration"] >= MIN_EVENT_HOURS, ["start", "duration", "peak"]], thr


def county_missing_days(times):
    """Local days strictly inside gaps longer than MAX_RECORD_GAP_DAYS without any record."""
    t = np.sort(times.unique())
    gaps = np.where(np.diff(t) > np.timedelta64(MAX_RECORD_GAP_DAYS, "D"))[0]
    days = []
    for i in gaps:
        a = to_local_date(pd.Series([t[i]]), OUTAGE_TZ).iloc[0] + pd.Timedelta(days=1)
        b = to_local_date(pd.Series([t[i + 1]]), OUTAGE_TZ).iloc[0] - pd.Timedelta(days=1)
        if b >= a:
            days.append(pd.date_range(a, b, freq="D"))
    return days[0].append(days[1:]) if days else pd.DatetimeIndex([])


def process_outages(files, fips_keep=None):
    daily, summary, reporting, spans, missing, hourly = [], [], {}, {}, {}, []
    for i, f in enumerate(files, 1):
        d = read_outage_file(f)
        if fips_keep is not None:
            d = d[d["county_id"].isin(fips_keep)]
        if d.empty:
            continue
        for fips, g in d.groupby("county_id"):
            ev, thr = detect_events(g)
            ev["Time"] = to_local_date(ev["start"], OUTAGE_TZ) if len(ev) else pd.Series(dtype="datetime64[ns]")
            per_day = ev.groupby("Time").agg(duration_raw_hours=("duration", "sum"),
                                             n_outage_events=("duration", "size"),
                                             customers_out_peak=("peak", "max"))
            rec_days = to_local_date(g["time"], OUTAGE_TZ)
            per_day = per_day.join(rec_days.value_counts().rename("outage_records"), how="outer")
            per_day["county_id"] = fips
            daily.append(per_day.rename_axis("Time").reset_index())
            reporting[fips] = set(rec_days.unique())
            spans[fips] = (rec_days.min(), rec_days.max())
            missing[fips] = county_missing_days(g["time"])
            summary.append(dict(
                county_id=fips, county=g["county"].iloc[0], records=len(g),
                zero_records=int((g["sum"] == 0).sum()), first=rec_days.min().date(),
                last=rec_days.max().date(), p85_customers_threshold=thr, events=len(ev),
                event_hours_median=ev["duration"].median() if len(ev) else np.nan,
                event_hours_p90=ev["duration"].quantile(0.9) if len(ev) else np.nan,
                max_total_customers=pd.to_numeric(g["total_customers"], errors="coerce").max(),
                county_gap_days_flagged=len(missing[fips])))
            h = g.set_index("time")["sum"].resample("h").max()
            hourly.append(pd.DataFrame({"county_id": fips, "time": h.index, "out": h.to_numpy()}))
        print(f"  outage file {i}/{len(files)}: {os.path.basename(f)}  {len(d):,} records", flush=True)
    return (pd.concat(daily, ignore_index=True), pd.DataFrame(summary), reporting, spans,
            missing, pd.concat(hourly, ignore_index=True))


# -----------------------------------------------------------------------------
# 3. Coverage
# -----------------------------------------------------------------------------
def valid_days(reporting, spans, missing):
    """Days with usable outage data per county (all other days are flagged, not dropped)."""
    all_days = pd.date_range(min(s[0] for s in spans.values()), max(s[1] for s in spans.values()))
    covered = pd.DataFrame({f: (all_days >= s[0]) & (all_days <= s[1]) for f, s in spans.items()},
                           index=all_days)
    reported = pd.DataFrame({f: all_days.isin(list(r)) for f, r in reporting.items()}, index=all_days)
    share = (reported & covered).sum(axis=1) / covered.sum(axis=1).replace(0, np.nan)
    statewide_missing = share.index[share < MIN_STATEWIDE_REPORTING_SHARE]
    valid = {}
    for f, (a, b) in spans.items():
        own = pd.DatetimeIndex(sorted(reporting[f]))
        sys_gap = statewide_missing.difference(own)          # county silent on a system-outage day
        valid[f] = pd.date_range(a, b).difference(missing[f]).difference(sys_gap)
    return valid, share.rename("reporting_share"), statewide_missing


# -----------------------------------------------------------------------------
# 4. Diagnostics
# -----------------------------------------------------------------------------
def time_alignment(hourly_wx, hourly_out, diag_dir):
    """
    Correlation between hourly weather and hourly customers out (log scale) at
    lags from -24 h to +48 h, per county, then averaged. Both series have their
    mean hour-of-day cycle removed first, otherwise the shared diurnal cycle
    (afternoon storms, daytime activity) creates a spurious 24 h periodicity.
    A peak near 0 h means both sources share a time base; the decay describes
    the outage response lag.
    """
    def deseason(v, hours):
        m = pd.Series(v).groupby(hours).transform("mean").to_numpy()
        return v - m
    rows = []
    for fips, w in hourly_wx.groupby("county_id"):
        o = hourly_out[hourly_out["county_id"] == fips]
        if o.empty:
            continue
        w = w.set_index("time").sort_index()
        idx = pd.date_range(w.index.min(), w.index.max(), freq="h")
        hours = idx.hour
        out = deseason(np.log1p(o.set_index("time")["out"].reindex(idx).fillna(0).to_numpy()), hours)
        for v in ALIGN_VARS:
            if v not in w.columns:
                continue
            x = w[v].reindex(idx).to_numpy(float)
            x = x - pd.Series(x).groupby(hours).transform("mean").to_numpy()
            for lag in ALIGN_LAGS:
                # lag > 0: compare outage at t with weather at t - lag
                if lag >= 0:
                    a, b = x[:len(x) - lag], out[lag:]
                else:
                    a, b = x[-lag:], out[:len(out) + lag]
                ok = ~np.isnan(a)
                r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 and b[ok].std() > 0 else np.nan
                rows.append(dict(county_id=fips, variable=v, lag_h=lag, r=r))
    al = pd.DataFrame(rows)
    if al.empty:
        return al
    al.to_csv(os.path.join(diag_dir, "time_alignment_by_county.csv"), index=False)
    mean = al.groupby(["variable", "lag_h"])["r"].agg(["mean", "median"]).reset_index()
    mean.to_csv(os.path.join(diag_dir, "time_alignment.csv"), index=False)
    peaks = al.loc[al.groupby(["county_id", "variable"])["r"].idxmax()]
    print("  time alignment (lag of maximum correlation, hours; median over counties): " +
          ", ".join(f"{v} {peaks.loc[peaks['variable'] == v, 'lag_h'].median():+.0f} h" for v in ALIGN_VARS))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        for v, st in zip(ALIGN_VARS, ["-", "--"]):
            m = mean[mean["variable"] == v]
            ax.plot(m["lag_h"], m["mean"], st, color="#1f3b73", lw=1.6, label=v)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("lag (h): outage record time minus weather time")
        ax.set_ylabel("mean correlation across counties")
        ax.set_title("Time alignment of CONUS404 weather and EAGLE-I outages", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(diag_dir, f"time_alignment.{ext}"), dpi=300)
        plt.close(fig)
    except Exception as e:                       # plotting is optional here
        print(f"  (alignment plot skipped: {e})")
    return mean


# -----------------------------------------------------------------------------
# 5. Static attributes
# -----------------------------------------------------------------------------
def add_static(df):
    """Merge one row of static county attributes onto every county-day."""
    if not STATIC_FILE or not os.path.exists(STATIC_FILE):
        print("  ! static attribute file not found - skipped")
        return df
    st = pd.read_csv(STATIC_FILE)
    st["county_id"] = st["county_id"].astype(str).str.zfill(5)   # 12001 -> "12001", as in the dataset
    st = st.drop(columns=["county_name"])                        # already in the dataset
    return df.merge(st, on="county_id", how="left", validate="m:1")


# -----------------------------------------------------------------------------
# 6. Labels
# -----------------------------------------------------------------------------
def label(base, weather):
    """Weather-related outage classes. Thresholds: percentile of each filter
    variable over the county's outage days; exceedance on day t or t-1..t-lag
    (looked up in the full weather table, so the day before the first record
    also counts)."""
    missing = [c for c in FILTER_VARS if c not in base.columns]
    if missing:
        raise KeyError(f"filter variables not in data: {missing}")
    avail = base[base["outage_data_available"] == 1]
    src = avail[avail["duration_raw_hours"] > 0] if FILTER_BASIS == "outage_days" else avail
    thr = src.groupby("ID")[FILTER_VARS].quantile(FILTER_PERCENTILE)
    w = weather[["ID", "Time"] + FILTER_VARS].copy()
    t = thr.reindex(w["ID"]).set_axis(w.index)
    ex = np.zeros(len(w), bool)
    for v in FILTER_VARS:
        ex |= np.where(t[v] == 0, w[v] > 0, w[v] > t[v])
    exceed = pd.Series(ex, index=pd.MultiIndex.from_frame(w[["ID", "Time"]]))
    flag = np.zeros(len(base), bool)
    for lag in range(FILTER_LAG_DAYS + 1):
        idx = pd.MultiIndex.from_arrays([base["ID"], base["Time"] - pd.Timedelta(days=lag)])
        flag |= exceed.reindex(idx, fill_value=False).to_numpy()
    out = base.copy()
    out["extreme_weather"] = flag.astype(int)
    h = out["duration_raw_hours"].to_numpy()
    out["duration_hours"] = np.where(flag, h, 0.0)
    d = out["duration_hours"].to_numpy()
    out["duration"] = np.select([d <= 0, d < MINOR_UPPER, d < MAJOR_LOWER], [0, 1, 2], default=3)
    return out, thr


def suffix():
    return f"_conus404_4class_p{int(round(FILTER_PERCENTILE * 100))}"


def write_labels(base, weather):
    labeled, thr = label(base, weather)
    model = labeled.drop(columns=[c for c in OUTCOME_DIAGNOSTIC_COLS if c in labeled.columns])
    s = suffix()
    p_all = os.path.join(OUT_DIR, f"all_counties_classified_stat_all_days{s}.csv")
    p_out = os.path.join(OUT_DIR, f"all_counties_classified_stat_outage_days{s}.csv")
    model.to_csv(p_all, index=False, date_format="%Y-%m-%d")
    model[model["duration_raw_hours"] > 0].to_csv(p_out, index=False, date_format="%Y-%m-%d")
    thr.to_csv(os.path.join(OUT_DIR, "diagnostics", f"filter_thresholds{s}.csv"))

    counts = pd.crosstab(model["Time"].dt.year, model["duration"]).rename(columns=CLASS_NAMES)
    counts["county_days"] = counts.sum(axis=1)
    config = dict(filter_vars=FILTER_VARS, percentile=FILTER_PERCENTILE, basis=FILTER_BASIS,
                  lag_days=FILTER_LAG_DAYS, minor_upper_h=MINOR_UPPER, major_lower_h=MAJOR_LOWER,
                  outage_quantile=OUTAGE_QUANTILE, event_gap_h=EVENT_GAP_HOURS,
                  min_event_h=MIN_EVENT_HOURS, day_tz=DAY_TZ)
    text = ["Weather-related outage dataset (CONUS404 + EAGLE-I)",
            json.dumps(config, indent=2), "", "Class counts by year:", counts.to_string(), "",
            "Totals: " + ", ".join(f"{CLASS_NAMES[k]}={int((model['duration'] == k).sum()):,}"
                                   for k in CLASS_NAMES)]
    with open(os.path.join(OUT_DIR, f"dataset_summary{s}.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n" + "\n".join(text[3:]))
    print(f"\n  -> {p_all}\n  -> {p_out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    t0 = time.time()
    diag = os.path.join(OUT_DIR, "diagnostics")
    os.makedirs(diag, exist_ok=True)
    wx_cache = os.path.join(OUT_DIR, "weather_daily_conus404.parquet")
    base_path = os.path.join(OUT_DIR, "daily_base_all_days.csv")

    if args.stage == "label":
        print("Relabeling from cached base file")
        base = pd.read_csv(base_path, parse_dates=["Time"])
        weather = pd.read_parquet(wx_cache)
        write_labels(base, weather)
        return

    files = sorted(glob.glob(args.weather_glob))
    if not files:
        raise FileNotFoundError(f"no parquet files match {args.weather_glob}")
    print(f"1. Weather: {len(files)} file(s)")
    weather, hourly_wx, names = aggregate_weather(files, args.max_counties)

    ofiles = sorted(glob.glob(os.path.join(args.outage_dir, "*.csv")))
    if not ofiles:
        raise FileNotFoundError(f"no CSV files in {args.outage_dir}")
    print(f"\n2. Outages: {len(ofiles)} file(s)")
    fips_keep = set(weather["county_id"]) if args.max_counties else None
    daily_out, summary, reporting, spans, missing, hourly_out = process_outages(ofiles, fips_keep)

    print("\n3. Coverage")
    valid, share, statewide_missing = valid_days(reporting, spans, missing)
    share.to_csv(os.path.join(diag, "statewide_reporting_share.csv"))
    pd.Series(statewide_missing.date, name="Time").to_csv(os.path.join(diag, "statewide_missing_days.csv"), index=False)
    summary["statewide_days_flagged"] = [len(pd.date_range(*spans[f]).intersection(statewide_missing))
                                         for f in summary["county_id"]]
    print(f"  statewide system-outage days (flagged where the county is silent): {len(statewide_missing)}; county gap days flagged: "
          f"{int(summary['county_gap_days_flagged'].sum()):,}")

    print("\n4. Diagnostics: time alignment")
    time_alignment(hourly_wx, hourly_out, diag)

    print("\n5. Assembling daily base table (every county, every day)")
    first, last = weather["Time"].min(), weather["Time"].max()
    o_first = min(sp[0] for sp in spans.values())
    o_last = max(sp[1] for sp in spans.values())
    first, last = max(first, o_first), min(last, o_last)
    if START_DATE:
        first = max(first, pd.Timestamp(START_DATE))
    if END_DATE:
        last = min(last, pd.Timestamp(END_DATE))
    period = pd.date_range(first, last, freq="D")
    counties = sorted(set(weather["county_id"]) | set(spans))
    no_outage = sorted(set(weather["county_id"]) - set(spans))
    no_weather = sorted(set(spans) - set(weather["county_id"]))
    if no_outage:
        print(f"  ! counties with weather but no outage file (all days flagged): {no_outage}")
    if no_weather:
        print(f"  ! counties with outage data but no weather (weather NaN): {no_weather}")
    grid = pd.DataFrame(pd.MultiIndex.from_product([counties, period], names=["county_id", "Time"])
                        .to_frame(index=False))
    base = grid.merge(weather.drop(columns=["county_name"]), on=["county_id", "Time"], how="left")
    base = base.merge(daily_out, on=["county_id", "Time"], how="left")
    avail = pd.concat([pd.DataFrame({"county_id": f, "Time": d}) for f, d in valid.items()], ignore_index=True)
    avail["outage_data_available"] = 1
    base = base.merge(avail, on=["county_id", "Time"], how="left")
    base["outage_data_available"] = base["outage_data_available"].fillna(0).astype(int)
    base[["duration_raw_hours", "n_outage_events", "outage_records"]] = \
        base[["duration_raw_hours", "n_outage_events", "outage_records"]].fillna(0)
    miss_wx = int(base["weather_hours"].isna().sum())
    print(f"  period {first.date()} to {last.date()} ({len(period):,} days) x {len(counties)} counties; "
          f"county-days without weather: {miss_wx}")
    print(f"  county-days without usable outage data (kept, flagged 0): "
          f"{int((base['outage_data_available'] == 0).sum()):,} of {len(base):,}")

    names_all = {**{f: n for f, n in names.items()},
                 **summary.set_index("county_id")["county"].to_dict()}
    base["county_name"] = base["county_id"].map(names_all)
    canon = base["county_name"].map(lambda n: NAME_LOOKUP.get(norm_name(n)))
    unmapped = sorted(base.loc[canon.isna(), "county_name"].dropna().unique())
    if unmapped:
        print(f"  ! counties not in COUNTY_IDS (dropped): {unmapped}")
    base["ID"] = canon.map(COUNTY_IDS)
    base = base.dropna(subset=["ID"])
    base = add_static(base)
    first_cols = ["Time", "ID", "county_id", "county_name", "outage_data_available"]
    base = base[first_cols + [c for c in base.columns if c not in first_cols]]
    base = (base.assign(_n=base["ID"].str.split("_").str[1].astype(int))
                .sort_values(["_n", "Time"]).drop(columns="_n").reset_index(drop=True))

    # completeness check: every county has every day exactly once
    per = base.groupby("ID")["Time"].agg(["size", "nunique", "min", "max"])
    ok = (per["size"] == len(period)) & (per["nunique"] == len(period))
    if not ok.all():
        raise RuntimeError(f"incomplete daily coverage for: {list(per.index[~ok])}")
    print(f"  completeness check passed: {base['ID'].nunique()} counties x {len(period):,} days "
          f"= {len(base):,} rows, no day missing")

    weather["ID"] = weather["county_name"].map(lambda n: COUNTY_IDS.get(NAME_LOOKUP.get(norm_name(n))))
    weather.to_parquet(wx_cache, index=False)
    base.to_csv(base_path, index=False, date_format="%Y-%m-%d")

    cy = base.groupby(["ID", base["Time"].dt.year]).size().unstack(fill_value=0)
    cy.to_csv(os.path.join(diag, "days_by_county_year.csv"))
    summary.to_csv(os.path.join(diag, "outage_county_summary.csv"), index=False)
    cy_av = base.groupby(["ID", base["Time"].dt.year])["outage_data_available"].mean().unstack()
    cy_av.to_csv(os.path.join(diag, "outage_data_available_share_by_county_year.csv"))
    print(f"  base: {len(base):,} county-days, {base['ID'].nunique()} counties, "
          f"outage days {int((base['duration_raw_hours'] > 0).sum()):,}\n  -> {base_path}")

    print("\n6. Labels")
    write_labels(base, weather)
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["all", "label"], default="all",
                   help="'all' rebuilds the dataset; 'label' only relabels the cached base table")
    p.add_argument("--weather-glob", default=WEATHER_GLOB,
                   help="glob matching hourly CONUS404 county parquet files")
    p.add_argument("--outage-dir", default=OUTAGE_DIR,
                   help="directory containing one EAGLE-I CSV per county")
    p.add_argument("--static-file", default=STATIC_FILE,
                   help="county-level static attribute CSV; pass an empty string to skip")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="directory for processed datasets and diagnostics")
    p.add_argument("--start-date", default=START_DATE, help="optional first local day, YYYY-MM-DD")
    p.add_argument("--end-date", default=END_DATE, help="optional last local day, YYYY-MM-DD")
    p.add_argument("--max-counties", type=int, default=None,
                   help="quick smoke test using the first N counties")
    a = p.parse_args()

    OUT_DIR = a.out_dir
    STATIC_FILE = a.static_file
    START_DATE = a.start_date
    END_DATE = a.end_date
    main(a)
