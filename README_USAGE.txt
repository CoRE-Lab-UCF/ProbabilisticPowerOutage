POWER-OUTAGE FREQUENCY MODEL: CODE GUIDE
========================================

PURPOSE
-------
This folder is a GitHub-ready cleanup of the analysis scripts used for the
Florida weather-related power-outage frequency study.

The recommended workflow is:

  raw EAGLE-I + CONUS404
        |
        v
  1. build_conus404_outage_dataset.py
        |
        +--> daily_base_all_days.csv
        +--> all_counties_classified_stat_all_days_conus404_4class_p80.csv
        +--> diagnostics/
        |
        v
  2. threshold_justification.py          [method justification / sensitivity]
        |
        v
  3. loyo_frequency_model.py             [run each feature set]
        |
        +--> results/<feature_set>_partial2024/
        |
        +--> 4. evaluate_frequency.py     [selected full model]
        +--> 5. compare_feature_sets.py   [across feature sets]
        +--> 6. compare_spatial_temporal.py [optional decomposition]
        |
        v
  7. make_publication_figures.py         [final manuscript-style figures]
        |
        v
  8. make_supp_figs.py                   [standalone S1/S2 regeneration]

The scripts in legacy/ are retained only for provenance and are not needed for
the final figure workflow.


0. ENVIRONMENT
--------------
Recommended Python: 3.10 or newer.

Install dependencies:

  pip install -r requirements.txt

Required Python packages:
  numpy
  pandas
  matplotlib
  lightgbm
  pyarrow

Raw data are not included. See data/README.txt for the expected folder layout.

The supplied full processed CSV was 186.1 MiB, so it should not
be committed directly to ordinary GitHub history. The included sample CSV is
only for inspecting the schema.


1. scripts/build_conus404_outage_dataset.py
-------------------------------------------
ROLE
  Builds the daily county dataset from hourly CONUS404 weather and county-level
  EAGLE-I outage files.

MAIN PROCESSING
  - Converts source timestamps from UTC to America/New_York calendar days.
  - Aggregates hourly CONUS404 variables to daily county statistics.
  - Detects EAGLE-I outage events above the county-specific 85th percentile.
  - Declusters outage records using a 4 h gap.
  - Drops detected outage events shorter than 1 h.
  - Keeps every county-day and flags missing/unusable outage observations with
    outage_data_available = 0.
  - Merges static county attributes.
  - Applies the final weather-related filter and 4-class outage labels.

DEFAULT INPUT LAYOUT
  data/raw/eagle_i/*.csv
  data/raw/conus404/*.parquet
  data/raw/conus404/county_static.csv

EXAMPLE
  python scripts/build_conus404_outage_dataset.py \
      --outage-dir data/raw/eagle_i \
      --weather-glob "data/raw/conus404/*.parquet" \
      --static-file data/raw/conus404/county_static.csv \
      --out-dir data/processed

QUICK TEST
  python scripts/build_conus404_outage_dataset.py \
      --outage-dir data/raw/eagle_i \
      --weather-glob "data/raw/conus404/*.parquet" \
      --static-file data/raw/conus404/county_static.csv \
      --out-dir data/processed \
      --max-counties 5

IMPORTANT OUTPUTS
  data/processed/daily_base_all_days.csv
      Daily weather + raw outage duration + static attributes, before the final
      weather-related class labels. Preferred input to threshold_justification.py.

  data/processed/all_counties_classified_stat_all_days_conus404_4class_p80.csv
      Final modeling dataset. This is the input to loyo_frequency_model.py.

  data/processed/all_counties_classified_stat_outage_days_conus404_4class_p80.csv
      Outage-day subset.

  data/processed/weather_daily_conus404.parquet
      Cached daily weather table.

  data/processed/diagnostics/
      Includes time alignment, reporting coverage, threshold tables, and
      per-county summaries.

RELABEL ONLY
  After changing the filter/class constants in the configuration section:

  python scripts/build_conus404_outage_dataset.py \
      --stage label \
      --out-dir data/processed

This reuses daily_base_all_days.csv and weather_daily_conus404.parquet.


2. scripts/threshold_justification.py
------------------------------------
ROLE
  Produces the sensitivity/justification analysis used to select:
  - weather variables,
  - percentile threshold,
  - same-day / preceding-day window,
  - minor/moderate/major duration thresholds.

INPUT
  data/processed/daily_base_all_days.csv

EXAMPLE
  python scripts/threshold_justification.py \
      --base data/processed/daily_base_all_days.csv \
      --out results/threshold_justification

OUTPUT
  results/threshold_justification/
      Fig1--Fig9 diagnostic/justification figures
      FigS1 / FigS2 supporting figures when diagnostics are available
      tables/*.csv
      justification_summary.md

NOTE
  The final build script already contains the selected manuscript settings.
  This script documents why those settings were chosen. If the selected
  threshold changes, update the constants in the build script and rerun its
  --stage label step.


3. scripts/loyo_frequency_model.py
----------------------------------
ROLE
  Runs the 4-class LightGBM leave-one-year-out frequency model.

IMPORTANT
  This release uses the later 10-feature-set LOYO implementation, not the older
  four-set base/lead/storm/full version. The 10 sets are required by the feature
  comparison and final publication-figure scripts.

FEATURE SETS
  minimal
  base
  static
  antecedent
  forward
  forward_plus
  relative
  forward_static
  antecedent_static
  full

INPUT
  data/processed/all_counties_classified_stat_all_days_conus404_4class_p80.csv

DEFAULT BEHAVIOR
  - Trains on observed county-days only (outage_data_available = 1).
  - Predicts probabilities for every county-day in the held-out year.
  - Uses LightGBM multiclass log loss with no class resampling/weights.
  - Selects hyperparameters/number of trees using inner groups of whole years.
  - Sums daily class probabilities to obtain expected outage frequencies.
  - Allows partial 2024 as a held-out test year and stores results with the
    suffix _partial2024.

RUN ONE FEATURE SET
  python scripts/loyo_frequency_model.py \
      --features full \
      --data data/processed/all_counties_classified_stat_all_days_conus404_4class_p80.csv

RUN ALL TEN FEATURE SETS
  Run the command above once for each feature set. For example in a shell:

  for fs in minimal base static antecedent forward forward_plus relative \
            forward_static antecedent_static full; do
      python scripts/loyo_frequency_model.py --features "$fs"
  done

HPC / ARRAY MODE
  List jobs:
    python scripts/loyo_frequency_model.py --list-jobs

  Run one held-out year:
    python scripts/loyo_frequency_model.py --features full --test-year 2018

  Merge completed folds:
    python scripts/loyo_frequency_model.py --features full --merge

OUTPUT PER FEATURE SET
  results/<feature_set>_partial2024/
      folds/
      daily_predictions.csv
      yearly_series.csv
      yearly_counts.csv
      county_year_series.csv
      county_year_counts.csv
      frequency_metrics.csv
      fold_diagnostics.csv
      feature_importance.csv
      observed_vs_predicted_by_year.png


4. scripts/evaluate_frequency.py
-------------------------------
ROLE
  Performs the detailed evaluation of one completed model result folder,
  normally results/full_partial2024.

INPUT
  daily_predictions.csv and related files from loyo_frequency_model.py.

EXAMPLE
  python scripts/evaluate_frequency.py \
      --results results/full_partial2024

MAIN ANALYSES
  - complete water-year KGE, r, alpha, beta, and baseline-relative skill
  - county bootstrap confidence intervals
  - water-year jackknife
  - county-level spatial and anomaly skill
  - probability calibration / reliability
  - monthly seasonality
  - standard DJF/MAM/JJA/SON variability
  - hurricane-season / off-season regimes
  - tropical-cyclone storm windows

OUTPUT
  results/full_partial2024/evaluation_allyears/
      statewide_metrics.csv
      bootstrap_ci.csv
      headline_table.csv
      water_year_jackknife.csv
      water_year_jackknife_summary.csv
      county_level_metrics.csv
      per_county_metrics.csv
      calibration.csv
      calibration_summary.csv
      seasonal/
      diagnostic figures

The figures from this script are analysis/evaluation outputs. Use
make_publication_figures.py for the final manuscript styling.


5. scripts/compare_feature_sets.py
---------------------------------
ROLE
  Compares the ten feature-set model runs using their frequency_metrics.csv and
  yearly_series.csv files.

EXAMPLE
  python scripts/compare_feature_sets.py \
      --results-root results \
      --suffix _partial2024

OUTPUT
  results/comparison_partial2024/
      feature_set_comparison.csv
      feature_set_summary.csv
      Fig_feature_sets_partial2024.png
      Fig_feature_sets_partial2024.pdf

This script is useful for tables and feature-set diagnostics. The final
publication-style trade-off figure is generated by make_publication_figures.py.


6. scripts/compare_spatial_temporal.py
-------------------------------------
ROLE
  Quantifies temporal skill, spatial skill, county-year anomaly skill, and the
  decomposition of county-year MSE into bias, spatial, temporal, and interaction
  components.

EXAMPLE
  python scripts/compare_spatial_temporal.py \
      --results-root results \
      --suffix _partial2024

OUTPUT
  results/comparison/
      spatial_temporal_metrics.csv
      error_decomposition.csv
      Fig_spatial_temporal_tradeoff.png
      Fig_spatial_temporal_tradeoff.pdf

This analysis is optional for reproducing the final main figures because
make_publication_figures.py calculates the final trade-off directly.


7. scripts/make_publication_figures.py
-------------------------------------
ROLE
  Final publication-style figure generator. This is the recommended script for
  manuscript figures.

DEPENDENCY
  scripts/figure_config.py
  Keep both files in the same directory.

INPUTS
  --results-root
      Parent folder containing all ten feature-set result directories.

  --model-results
      Selected model result folder, normally results/full_partial2024.

  --boundaries
      Optional Florida county GeoJSON. Required only for the county difference
      map. If omitted, that map is skipped.

EXAMPLE
  python scripts/make_publication_figures.py \
      --results-root results \
      --model-results results/full_partial2024 \
      --boundaries data/boundaries/florida_counties.geojson

DEFAULT OUTPUT
  results/publication_figures/
      figures/   publication PNG + EPS
      data/      CSV tables used to draw the figures

FIGURE GROUPS
  calibration
  annual
  seasonality
  seasons
  storms
  spatial
  tradeoff

Generate only selected groups:
  python scripts/make_publication_figures.py \
      --results-root results \
      --model-results results/full_partial2024 \
      --only calibration annual seasonality

MONTHLY SEASONALITY
  Default:
    --seasonality-mode mean

  This first sums statewide outage county-days separately for each calendar
  month within each year, then averages those month-specific totals across the
  available years. This is the manuscript-preferred representation.

  Alternatives:
    --seasonality-mode totals
    --seasonality-mode rate

IMPORTANT FINAL FIGURES
  Fig_calibration_reliability
  Fig_annual_observed_vs_predicted_water_year
  Fig_seasonal_cycle_mean
  standard-season figures for any and major outages
  Fig_tropical_cyclones_and_regimes
  Fig_spatial_county_totals
  Fig_spatial_difference_maps
  Fig_temporal_spatial_tradeoff


8. scripts/figure_config.py
---------------------------
ROLE
  Shared plotting constants and helper functions used by
  make_publication_figures.py.

DO NOT RUN DIRECTLY.

It defines publication dimensions, colors, fonts, panel labels, axis cleanup,
and PNG/EPS saving.


9. scripts/make_supp_figs.py
----------------------------
ROLE
  Standalone regeneration of the time-alignment and data-coverage supplementary
  figures.

INPUTS
  data/processed/diagnostics/time_alignment.csv
  data/processed/diagnostics/time_alignment_by_county.csv
  and either the final modeling dataset or the coverage diagnostic table.

EXAMPLE
  python scripts/make_supp_figs.py \
      --diagnostics data/processed/diagnostics \
      --data data/processed/all_counties_classified_stat_all_days_conus404_4class_p80.csv \
      --out results/supplementary_figures

OUTPUT
  FigS1_time_alignment.png/.pdf
  FigS2_data_coverage.png/.pdf


LEGACY DIRECTORY
----------------
legacy/make_temporal_spatial_tradeoff.py
legacy/plot_annual_and_spatial.py
legacy/publication_style.py

These are retained for provenance only. They duplicate functionality now
consolidated in make_publication_figures.py and are not part of the recommended
reproduction order.


RECOMMENDED REPRODUCTION ORDER
------------------------------
A. Rebuild data
   python scripts/build_conus404_outage_dataset.py ...

B. Reproduce threshold justification
   python scripts/threshold_justification.py ...

C. Run all ten LOYO feature sets
   python scripts/loyo_frequency_model.py --features <set>

D. Evaluate selected full model
   python scripts/evaluate_frequency.py --results results/full_partial2024

E. Compare feature sets
   python scripts/compare_feature_sets.py --results-root results --suffix _partial2024

F. Optional spatial-temporal decomposition
   python scripts/compare_spatial_temporal.py --results-root results --suffix _partial2024

G. Generate final manuscript figures
   python scripts/make_publication_figures.py ...

H. Generate standalone supplementary alignment/coverage figures if needed
   python scripts/make_supp_figs.py ...


WHAT IS NOT INCLUDED
--------------------
1. Raw EAGLE-I CSV files.
2. Raw CONUS404 parquet files.
3. Florida county boundary GeoJSON.
4. The full 186.1 MiB processed modeling CSV.
5. Cluster/Slurm submission scripts.

These are data/environment dependencies rather than Python-code dependencies.

See:
  data/README.txt
  DATASET_SCHEMA.txt
  legacy/README.txt
