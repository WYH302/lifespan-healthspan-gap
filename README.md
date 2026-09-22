# The Global Gap Between Lifespan and Healthspan

Public source-code snapshot for the lifespan–healthspan gap project.

## Scope and reproducibility status

This repository contains the two non-backup Python scripts available in the local project at publication time. It is a partial source snapshot, not the complete analysis or reproducibility bundle. Manuscripts, editor sessions, personal correspondence, and backup variants are not included.

- `scripts/13_age60_module.py`: WHO age-60 LE/HALE analysis, conversion summaries, and figures.
- `scripts/14_combined_main_figure.py`: composite figure from previously generated summary tables.

## Real-world Robustness & Edge Deployment

For this population analysis, robustness concerns source revisions, missing country-years, weighting choices and uncertainty estimates, not lighting or image occlusion. The missing upstream panel and summary tables prevent rerunning those checks from this snapshot alone.

There is no trained neural model or edge-inference target in the two released scripts. Runtime and memory can be recorded once a frozen input set is supplied. Do not attach model-weight or GPU-demo badges to an analysis that does not use them.

## Dependencies

Install Python and run `python -m pip install -r requirements.txt`. Dependency versions are not pinned because the original environment was not available; compatibility has not been validated.

## Required inputs (not included)

For the age-60 analysis, provide `data/processed/panel_master.csv` with columns `iso3`, `year`, `population_thousands`, and `old_age_share`. The script retrieves WHO GHO indicators `WHOSIS_000015` and `WHOSIS_000007` or reuses its cached raw CSV files. The upstream panel-building code and source data were not present in the available local snapshot.

For the composite figure, provide these files under `outputs/tables/`:

- `table_absolute_relative_summary_changes.csv`
- `table_uncertainty_population_weighted_global_trend.csv`
- `table_uncertainty_baseline_le_tertile_conversion_2000_2019.csv`
- `table_uncertainty_baseline_sdi_tertile_conversion_2000_2019.csv`

The upstream code that generates these four tables is not included. Script 13 does not generate them. Input column names and expected categories are specified in each script. The composite figure retains original fixed annotations and axis limits, which must be reviewed when using different data.

## Usage after supplying inputs

```sh
python scripts/13_age60_module.py
python scripts/14_combined_main_figure.py
```

Outputs are written under `outputs/tables/` and `outputs/figures/`. Add `--refresh` to script 13 to re-download WHO data; live data may differ from the original study snapshot.

## Validation

Both scripts passed Python syntax parsing before publication. End-to-end analysis and numerical reproduction were not run because required inputs are absent. Publication does not establish reproducibility of the manuscript results.

## License

No additional reuse license is granted in this snapshot. Public repository visibility does not itself grant an open-source license. Third-party data remain subject to their providers' terms.
