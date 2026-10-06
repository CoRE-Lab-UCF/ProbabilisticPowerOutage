# Weather-related power-outage frequency model

This repository contains the data-building, LightGBM leave-one-year-out
modeling, evaluation, feature-comparison, and publication-figure scripts used
for the Florida weather-related outage-frequency study.

Start with **`README_USAGE.txt`** for the complete run order, expected inputs,
and outputs.

## Main workflow

1. `scripts/build_conus404_outage_dataset.py`
2. `scripts/threshold_justification.py`
3. `scripts/loyo_frequency_model.py`
4. `scripts/evaluate_frequency.py`
5. `scripts/compare_feature_sets.py`
6. `scripts/compare_spatial_temporal.py` (optional diagnostic)
7. `scripts/make_publication_figures.py`
8. `scripts/make_supp_figs.py` (optional supplementary regeneration)

`make_publication_figures.py` is the final manuscript-style plotting workflow.
The older plotting scripts are kept under `legacy/` only for provenance.

The raw data and full processed modeling CSV are not bundled in this code
package. See `data/README.txt`.
