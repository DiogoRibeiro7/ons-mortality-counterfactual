"""England & Wales mid-year population estimates.

The data file is small enough to ship with the repository. Values come
from ONS *Population estimates for the UK, England and Wales, Scotland
and Northern Ireland* (mid-year series MYE2). Numbers are rounded to
the nearest thousand and reflect post-2021-census revisions where
applicable.

To refresh: replace ``data/processed/england_wales_population.csv``
with a more recent extract from the ONS publication
https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland
keeping the same column names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_POPULATION_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "england_wales_population.csv"
)

DEFAULT_REGIONAL_POPULATION_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "england_wales_regional_population.csv"
)

DEFAULT_ASMR_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "england_wales_asmr.csv"
)

DEFAULT_COVID_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "england_wales_covid_deaths.csv"
)

REQUIRED_COLUMNS = {"year", "population_total", "pop_65_plus", "pop_75_plus"}
REQUIRED_REGIONAL_COLUMNS = {
    "year",
    "region_code",
    "region_name",
    "population_total",
}
REQUIRED_ASMR_COLUMNS = {"year", "asmr_persons_per_100k"}
REQUIRED_COVID_COLUMNS = {"year", "deaths_involving_covid"}


def load_population(path: Path | None = None) -> pd.DataFrame:
    """
    Load England & Wales mid-year population estimates.

    Returns a DataFrame indexed by year (integer) with one column per
    age aggregate. Useful for computing crude or age-standardised
    mortality rates against the deaths series.

    Parameters
    ----------
    path:
        Optional override for the CSV location. Defaults to the bundled
        file shipped with the repository.

    Returns
    -------
    pd.DataFrame
        Columns: ``population_total``, ``pop_65_plus``, ``pop_75_plus``,
        plus derived ``share_65_plus`` and ``share_75_plus``.
    """
    csv_path = path or DEFAULT_POPULATION_CSV
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Population CSV is missing columns: {sorted(missing)}"
        )

    df = df.set_index("year").sort_index()
    df["share_65_plus"] = df["pop_65_plus"] / df["population_total"]
    df["share_75_plus"] = df["pop_75_plus"] / df["population_total"]
    return df


def load_regional_population(path: Path | None = None) -> pd.DataFrame:
    """
    Load England & Wales mid-year population estimates by region.

    Returns a long DataFrame with one row per (year, region). Useful for
    computing crude regional death rates against the regional deaths series.

    Parameters
    ----------
    path:
        Optional override for the CSV location. Defaults to the bundled
        regional file shipped with the repository.

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, ``region_code``, ``region_name``,
        ``population_total``.
    """
    csv_path = path or DEFAULT_REGIONAL_POPULATION_CSV
    df = pd.read_csv(csv_path)

    missing = REQUIRED_REGIONAL_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Regional population CSV is missing columns: {sorted(missing)}"
        )

    return df.sort_values(["region_code", "year"]).reset_index(drop=True)


def load_asmr(path: Path | None = None) -> pd.DataFrame:
    """
    Load the published England & Wales age-standardised mortality rate series.

    Source: ONS *Death registrations summary tables - England and Wales*, with
    rates standardised to the 2013 European Standard Population (ESP-2013).
    The bundled CSV is the persons (both-sexes) series for 2006-2023; refresh
    by replacing the file with a more recent extract.

    Parameters
    ----------
    path:
        Optional override for the CSV location.

    Returns
    -------
    pd.DataFrame
        Year-indexed frame with one column ``asmr_persons_per_100k``.
    """
    csv_path = path or DEFAULT_ASMR_CSV
    df = pd.read_csv(csv_path)

    missing = REQUIRED_ASMR_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"ASMR CSV is missing columns: {sorted(missing)}")

    return df.set_index("year").sort_index()


def load_covid_deaths(path: Path | None = None) -> pd.DataFrame:
    """
    Load published England & Wales annual deaths involving COVID-19.

    Source: ONS *Deaths registered involving the coronavirus (COVID-19),
    England and Wales*. The bundled CSV is rounded to the nearest hundred
    and based on registrations published through the dataset's most recent
    edition. The 2024 figure is provisional and likely to be revised
    upward as late registrations come in.

    Refresh by replacing
    ``data/processed/england_wales_covid_deaths.csv`` with a more recent
    extract; keep the ``year`` and ``deaths_involving_covid`` column names.
    """
    csv_path = path or DEFAULT_COVID_CSV
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COVID_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"COVID deaths CSV is missing columns: {sorted(missing)}"
        )

    return df.set_index("year").sort_index()
