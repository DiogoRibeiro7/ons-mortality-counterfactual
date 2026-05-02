# ONS Mortality Counterfactual

Python project that downloads the ONS *Monthly figures on deaths registered by area of usual residence* workbooks, extracts the England & Wales national series, and fits a pre-pandemic counterfactual to estimate excess mortality from March 2020 onwards.

The repository ships with two paths:

- **No-MySQL path (recommended).** A single CLI command (`ons-mortality run`) downloads every annual ONS workbook, parses both the legacy wide-format and the 2023+ long-format layouts, writes a tidy CSV, fits the counterfactual, and saves the chart.
- **MySQL path.** The original ingestion pipeline that stores raw source-file metadata and a normalized `monthly_deaths` table in MySQL via Docker. Useful when the project's data needs to be queried alongside other datasets.

![England & Wales counterfactual chart](figures/england_wales_counterfactual.png)

## What this repository does

1. Discovers ONS annual mortality workbooks from the public dataset page.
2. Downloads them with on-disk caching and polite throttling (the ONS site rate-limits aggressive clients).
3. Extracts the England & Wales national monthly series. Two layouts are supported:
   - 2006-2022 — wide format with one column per month and a "pyramid" admin hierarchy.
   - 2023+ — long format with columns ``Month``, ``Code``, ``Geography``, ``Number of deaths``.
4. When a final edition supersedes a provisional one, the more authoritative figure wins.
5. Fits a Bayesian-style linear regression (linear trend + annual Fourier seasonality, conjugate Gaussian posterior) on data strictly before March 2020.
6. Projects the no-pandemic trajectory forward and computes excess deaths against the counterfactual median.
7. Optionally loads everything into MySQL for downstream querying.

## Project structure

```text
ons-mortality-counterfactual/
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
├── Makefile
├── sql/
│   └── schema.sql
├── src/
│   └── ons_mortality/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── counterfactual.py
│       ├── database.py
│       ├── fetch.py
│       ├── ons.py
│       └── parser.py
├── scripts/
│   └── make_counterfactual_plot.py
├── notebooks/
│   └── 01_counterfactual_mortality.ipynb
├── tests/
│   ├── test_counterfactual.py
│   ├── test_fetch.py
│   ├── test_ons.py
│   └── test_parser.py
├── data/
│   ├── raw/        # cached ONS workbooks
│   └── processed/  # tidy CSV exports
└── figures/        # rendered charts
```

## Requirements

- Python 3.10+
- One of:
  - Poetry (recommended), or
  - `pip` with `requirements.txt`
- Docker + Docker Compose (only for the MySQL path)
- Internet access to download the ONS Excel files

## Quick start (no MySQL)

```bash
poetry install   # or: pip install -r requirements.txt && pip install -e .

poetry run ons-mortality run --start-year 2006 --end-year 2024
```

This single command:

- downloads ~19 annual workbooks into `data/raw/` (cached on subsequent runs),
- extracts the England & Wales monthly series into `data/processed/england_wales_monthly_deaths.csv`,
- fits the counterfactual,
- saves the chart to `figures/england_wales_counterfactual.png`.

If you want the data without the chart, use:

```bash
poetry run ons-mortality fetch --start-year 2006 --end-year 2024
```

If you already have the CSV and just want to refit / replot:

```bash
poetry run ons-mortality plot \
    --input data/processed/england_wales_monthly_deaths.csv \
    --output figures/england_wales_counterfactual.png \
    --fourier-order 3 --interval-mass 0.94
```

## Quick start (with MySQL)

```bash
cp .env.example .env
docker compose up -d
poetry install

poetry run ons-mortality init-db
poetry run ons-mortality discover --start-year 2006 --end-year 2024
poetry run ons-mortality ingest   --start-year 2006 --end-year 2024
poetry run ons-mortality export-national \
    --output data/processed/england_wales_monthly_deaths.csv
poetry run ons-mortality plot
```

## Notebooks

The repository ships three notebooks, each producing reproducible figures from the cached CSVs.

**[`01_counterfactual_mortality.ipynb`](notebooks/01_counterfactual_mortality.ipynb)** — the headline analysis.

1. Data load + shape checks
2. Seasonality and slow-trend decomposition
3. Year × month heatmap
4. Counterfactual fit
5. Monthly + cumulative excess deaths
6. Residual diagnostics
7. Sensitivity to the Fourier order
8. Demographic decomposition: how much of the pre-pandemic trend is population growth vs ageing vs per-person risk

Cumulative excess (2020-2024): **~245 000**.

**[`02_regional_excess.ipynb`](notebooks/02_regional_excess.ipynb)** — sub-national breakdown.

Refits the same model independently for the 9 English regions plus Wales, and ranks them by per-capita pandemic excess. The absolute and per-capita rankings disagree sharply: London tops the absolute table but is near the bottom per capita, while the North West tops the per-capita table at 5.48 excess deaths per 1 000.

![Per-capita pandemic excess by region](figures/regional_per_capita_excess.png)

Run `ons-mortality fetch-regional` first to build the input CSV.

**[`03_negative_binomial_glm.ipynb`](notebooks/03_negative_binomial_glm.ipynb)** — count model with population offset.

Refits the counterfactual as a negative binomial GLM with `log(population)` as an offset, so the model directly estimates per-capita monthly risk. Cumulative excess under this model: **~282 000** — about 15% higher than the Gaussian linear model in notebook 01. Both numbers are defensible; the gap is the difference between projecting absolute deaths at their historical slope vs holding per-capita risk near constant. Requires `statsmodels`.

**[`04_age_standardized_rate.ipynb`](notebooks/04_age_standardized_rate.ipynb)** — age-standardised mortality rate (ASMR).

The rigorous version of section 9 in notebook 01. Demonstrates the ESP-2013 weighting on a worked example, then loads the published ONS ASMR series for E&W (2006-2023), refits the counterfactual on the rate, and translates back to absolute deaths. Pre-pandemic ASMR was falling at **~1.4% per year**; the implied 2020-2023 cumulative excess is **~281 000 deaths**, agreeing closely with notebook 03's NB GLM and disagreeing with notebook 01's Gaussian-linear by ~15%. The takeaway: simple absolute-count models systematically understate excess because they absorb demographic uplift into their trend.

**[`05_weekly_mortality.ipynb`](notebooks/05_weekly_mortality.ipynb)** — weekly resolution.

Refits the counterfactual on the ONS weekly registrations dataset (2010-2024, 782 weeks). Pinpoints the **first-wave peak at 22 351 deaths in the week ending 17 April 2020** — roughly 2× the counterfactual for that week. Surfaces the **Christmas/New Year reporting artefact** that monthly aggregates hide (week 1 sits ~12% below trend, week 2 ~12% above, due to bank-holiday registration delays). Cumulative weekly excess (clipped) comes in ~36k below the monthly figure because clipping discards more deficit weeks at higher resolution; signed excess agrees within a few percent.

Run `ons-mortality fetch-weekly` first to build the input CSV.

**[`06_cause_decomposition.ipynb`](notebooks/06_cause_decomposition.ipynb)** — COVID-19 vs non-COVID excess.

Decomposes the year-by-year excess into direct COVID-19 deaths and non-COVID residual. Two distinct regimes: in **2020–2021 COVID-19 over-explains the excess** (lockdowns suppressed flu, road accidents, and other typical killers, so non-COVID mortality fell). In **2022 the split is roughly 80/20**. By **2023–2024 COVID-19 explains only ~30%** of excess; the residual ~20 000 deaths/year is attributable elsewhere (delayed treatment, NHS pressure, post-acute pandemic effects). Cumulative split: ~204k COVID-19 + ~42k non-COVID = ~246k total.

![Annual excess deaths: COVID-19 vs non-COVID component](figures/covid_vs_non_covid_excess.png)

**[`07_live_forecast.ipynb`](notebooks/07_live_forecast.ipynb)** — forward projection.

Two projections share one design matrix: the **counterfactual** trained on pre-2020 data only, and the **forecast** trained on all observed months. Both project 24 months past the latest ONS edition. The gap between them implies a **structural \"new normal\" elevation of ~35–40 thousand deaths per year** through 2026 — close to the recent post-pandemic excess running rate. Includes an "is the latest month inside the forecast 94% interval?" check that's the natural anchor for monitoring fresh ONS publications.

![Live forecast and counterfactual](figures/live_forecast.png)

**[`08_age_band_excess.ipynb`](notebooks/08_age_band_excess.ipynb)** — per-age-band counterfactual.

Refits the same counterfactual independently for seven canonical age bands (Under 1, 1-14, 15-44, 45-64, 65-74, 75-84, 85+) using the ONS weekly age breakdown. Cumulative 2020-2024 excess by band: **75-84 (~94k) > 85+ (~72k) > 45-64 (~40k) > 65-74 (~26k) > 15-44 (~10k)**, summing to ~243k (matches notebook 01 within ~1%). The more interesting finding is *temporal*: 85+ excess collapses ~90% from 2020 (~28k) to 2024 (~3k), while 45-64 stays stubbornly elevated. By 2024 the residual excess is concentrated in middle age, not the very elderly — the same regime change visible in notebook 06's COVID/non-COVID split, now seen through age. A 2024 reversion check on the 85+ band (observed/counterfactual ratio ≈ 1.00) shows the elderly story is *reversion to baseline*, not displacement undershoot.

Run `ons-mortality fetch-weekly-ages` first to build the input CSV.

![Pandemic excess by age band](figures/age_stratified_excess.png)

**[`09_sex_age_excess.ipynb`](notebooks/09_sex_age_excess.ipynb)** — sex × age decomposition.

Refits the counterfactual independently for every (sex, age band) pair using ONS's Persons / Males / Females weekly blocks. Cumulative 2020-2024 totals: **Male ~130k, Female ~114k** — a ~14% absolute gap. The skew is *concentrated in 45-64* (M ~26k vs F ~14k; ratio 1.86×) — exactly the band that notebook 08 flagged as the persistent post-2022 residual. Relative-to-counterfactual excess confirms men lead by ~3pp in 45-64 (14.4% vs 11.5%); women lead modestly in 15-44 (15.8% vs 12.8%) and 75-84 (13.4% vs 12.1%). The acute pandemic gap closes by 2024 (annual totals within ~7%) because elderly bands revert; **the male middle-age residual does not**.

Run `ons-mortality fetch-weekly-sex-ages` first to build the input CSV.

![Pandemic excess by sex and age band](figures/sex_age_excess.png)

**[`10_backtest_calibration.ipynb`](notebooks/10_backtest_calibration.ipynb)** — does the model actually work?

Tests the model itself rather than applying it to a new slice. Holds out 2017-2019 (36 months) and refits on 2006-2016 across all four stratifications used elsewhere in the repo: **32 independent fits, scored on the same holdout**. Headline finding: **calibration improves with finer slicing**. National-level nominal 94% intervals achieve only ~89% empirical coverage; the per-(sex × band) fits used in nb 09 cluster at 94-100% (mean 97%). Bias is uniformly positive at the national/regional level — the linear trend lags actual upward drift in the series by 1-3% — implying nb 01's headline ~245k could be over-stated by ~10-15%, which is the gap that nb 03's negative-binomial GLM and nb 04's ASMR analysis already partially close. The Fourier-order sweep on the holdout suggests order 2 would be marginally better calibrated than the default order 3, with no MAE cost. **Net verdict**: nb 08 and nb 09 conclusions stand without modification; nb 01 / nb 02 should be read with credible intervals widened ~5pp.

![Backtest calibration across 32 fits](figures/backtest_calibration.png)

**[`11_trend_specification.ipynb`](notebooks/11_trend_specification.ipynb)** — does a different trend term close the bias gap?

Direct response to nb 10's bias finding. The counterfactual model now supports three trend forms via `CounterfactualConfig(trend_spec=...)`: **`linear`** (existing default), **`log_linear`** (multiplicative — the demographer's standard for mortality), and **`quadratic`** (additive curvature). Compares all three on the same data. Headline: **none of them eliminates the bias cleanly**. log-linear improves coverage marginally (89% → 92%) but barely changes the headline (~243k → ~252k, ~+4%) because E&W mortality drift is too gentle for log/linear to differ meaningfully. Quadratic also improves coverage but **moves the headline by ~17%** (~243k → ~286k) — driven almost entirely by the 85+ band, where pre-pandemic deceleration causes quadratic to project a flat counterfactual and read the post-2020 deviation as much larger. The 45-64 band moves the *opposite* way under quadratic (40k → 25k), which suggests nb 09's "persistent middle-age male residual" finding is partly trend-spec-dependent. **Verdict**: keep linear as the default but report the headline as a range — *~245-285k cumulative excess (2020-2024), depending on how the pre-pandemic trend is extrapolated*. nb 03 (NB GLM, ~282k) and nb 04 (ASMR, ~281k) both fall near the upper end of this range and become the more theoretically defensible point estimates if a single number is required.

![Trend specification comparison](figures/trend_specification.png)

## Continuous refresh

[`.github/workflows/refresh-live-forecast.yml`](.github/workflows/refresh-live-forecast.yml) re-runs the live forecast on a monthly schedule (the 25th, 02:00 UTC — a day after ONS typically publishes the new edition). The workflow:

1. Sets up Python 3.12 and installs the package.
2. Runs `ons-mortality fetch` to pull the latest ONS monthly mortality data.
3. Re-executes notebook 01 (headline counterfactual) and notebook 07 (live forecast) in place.
4. Commits the refreshed CSV, figures, and notebook outputs back to `main` if anything changed.

Trigger it manually from the Actions tab via *Run workflow* — useful for one-off refreshes or testing.

## Database tables

The `init-db` and `ingest` commands populate two tables:

- **`ons_source_files`** — one record per downloaded workbook (`edition_year`, `source_url`, `local_filename`, `file_sha256`, `downloaded_at`).
- **`monthly_deaths`** — normalized rows (`source_file_id`, `edition_year`, `month_date`, `area_code`, `area_name`, `geography_type`, `deaths`, `is_final`).

Schema is defined in [`sql/schema.sql`](sql/schema.sql).

## Example SQL query (MySQL path only)

```sql
SELECT
    month_date,
    area_code,
    area_name,
    SUM(deaths) AS observed_deaths
FROM monthly_deaths
WHERE area_code = 'K04000001'
   OR LOWER(area_name) IN ('england and wales', 'england & wales')
GROUP BY month_date, area_code, area_name
ORDER BY month_date;
```

## Model

The counterfactual model is a Bayesian-style linear regression with:

- linear trend,
- annual Fourier seasonality (3 harmonics by default),
- posterior sampling from a conjugate Gaussian approximation,
- predictive uncertainty (parameter draws + observation-level Gaussian noise),
- a central credible interval whose mass is configurable (94% by default).

It is fit only on data before March 2020 and projected forward. Read it as a transparent baseline; a more formal version would use a negative-binomial likelihood, an explicit population offset, and a state-space trend.

## Important notes

The ONS workbook layout has changed several times since 2006. The fetch path detects the layout per sheet and falls back from wide to long extraction. Provisional editions are routinely revised, so the cached files have their SHA-256 recorded by the ingest path; rerun `fetch` with `--overwrite` to force a refresh.

## Development

```bash
poetry run pytest -q
poetry run ruff check .
poetry run mypy src
```

## License

ONS data is subject to the [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) unless otherwise stated. The Python code is provided as a portfolio example.
