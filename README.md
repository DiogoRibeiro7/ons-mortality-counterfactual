# ONS Mortality Counterfactual

Python project to download ONS monthly mortality files, store the cleaned records in a local MySQL database, and reproduce a monthly mortality counterfactual chart similar to the reference figure.

The project focuses on England and Wales monthly all-cause deaths. It keeps raw source-file metadata, downloads the annual ONS Excel workbooks, parses them into a long monthly table, and then fits a Bayesian-style counterfactual model on pre-pandemic data.

## What this repository does

1. Starts a local MySQL database with Docker.
2. Downloads ONS annual Excel files for monthly deaths registered by area of usual residence.
3. Stores the raw source-file metadata and SHA-256 hash.
4. Parses the Excel workbooks into a normalized MySQL table.
5. Exports a national England and Wales monthly series.
6. Fits a pre-2020 counterfactual model.
7. Saves a plot with observed deaths, counterfactual median, 94% interval, and excess deaths.

## Project structure

```text
ons-mortality-counterfactual/
├── docker-compose.yml
├── pyproject.toml
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
│       ├── ons.py
│       └── parser.py
├── scripts/
│   └── make_counterfactual_plot.py
├── notebooks/
│   └── 01_counterfactual_mortality.ipynb
├── tests/
│   ├── test_counterfactual.py
│   └── test_parser.py
├── data/
│   ├── raw/
│   └── processed/
└── figures/
```

## Requirements

- Python 3.10+
- Poetry
- Docker and Docker Compose
- Internet access to download the ONS Excel files

## Quick start

```bash
cp .env.example .env

docker compose up -d
poetry install

poetry run ons-mortality init-db
poetry run ons-mortality discover --start-year 2006 --end-year 2022
poetry run ons-mortality ingest --start-year 2006 --end-year 2022
poetry run ons-mortality export-national --output data/processed/england_wales_monthly_deaths.csv
poetry run python scripts/make_counterfactual_plot.py
```

The final chart is saved to:

```text
figures/england_wales_counterfactual.png
```

## Database tables

### `ons_source_files`

Stores one record per downloaded workbook.

Important columns:

- `edition_year`
- `source_url`
- `local_filename`
- `file_sha256`
- `downloaded_at`

### `monthly_deaths`

Stores normalized monthly death counts.

Important columns:

- `source_file_id`
- `edition_year`
- `month_date`
- `area_code`
- `area_name`
- `geography_type`
- `deaths`
- `is_final`

## Example SQL query

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
- annual Fourier seasonality,
- posterior sampling from a conjugate Gaussian approximation,
- predictive uncertainty,
- 94% interval using the 3rd and 97th percentiles.

The model is fit only on data before March 2020 and then projected forward.

This is intended as a transparent baseline. A more formal version could later use PyMC or Stan with a negative binomial likelihood, latent trend, and explicit population offsets.

## Important notes

The ONS workbook layout has changed over time. The parser is deliberately defensive and scans every sheet for month-like columns. It handles both `.xls` and `.xlsx` workbooks.

The ingestion keeps raw files under `data/raw/` and stores a hash in MySQL. This matters because the latest ONS yearly file can be revised during the year.

## Development

Run tests:

```bash
poetry run pytest -q
```

Run linting:

```bash
poetry run ruff check .
poetry run mypy src
```

## License

This repository is a code scaffold. ONS data is subject to the Open Government Licence unless otherwise stated on the ONS page.
