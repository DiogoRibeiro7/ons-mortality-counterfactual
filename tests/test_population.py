"""Tests for the bundled England & Wales population loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ons_mortality.population import (
    DEFAULT_ASMR_CSV,
    DEFAULT_COVID_CSV,
    DEFAULT_POPULATION_CSV,
    DEFAULT_REGIONAL_POPULATION_CSV,
    REQUIRED_COLUMNS,
    load_asmr,
    load_covid_deaths,
    load_population,
    load_regional_population,
)


def test_default_population_csv_ships_with_repo() -> None:
    """The bundled file is part of the repository, not generated."""
    assert DEFAULT_POPULATION_CSV.exists()


def test_load_population_returns_year_indexed_frame() -> None:
    """Year is the index and the required aggregates are present."""
    df = load_population()

    assert df.index.name == "year"
    assert df.index.is_monotonic_increasing
    assert REQUIRED_COLUMNS.issubset({"year", *df.columns})
    assert "share_65_plus" in df.columns
    assert "share_75_plus" in df.columns


def test_share_65_plus_is_within_plausible_range() -> None:
    """E&W has been ~15-20% over 65 in the years we ship."""
    df = load_population()
    assert df["share_65_plus"].between(0.14, 0.22).all()
    assert df["share_75_plus"].between(0.06, 0.12).all()


def test_population_grows_monotonically_excluding_2021_revision() -> None:
    """ONS MYE rises year-on-year except where a census triggers a rebase."""
    df = load_population()
    diffs = df["population_total"].diff().dropna()
    # Allow at most one negative step (the 2021 census-driven revision).
    assert (diffs < 0).sum() <= 1


def test_load_population_raises_on_missing_columns(tmp_path: Path) -> None:
    """The loader surfaces a clear error when the schema is wrong."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"year": [2020], "wrong": [1]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_population(path=bad)


def test_default_regional_population_csv_ships_with_repo() -> None:
    """The bundled regional file is part of the repository, not generated."""
    assert DEFAULT_REGIONAL_POPULATION_CSV.exists()


def test_load_regional_population_covers_all_ten_regions() -> None:
    """9 English regions + Wales for every shipped year."""
    df = load_regional_population()

    assert df["region_code"].nunique() == 10
    assert "W92000004" in set(df["region_code"])
    # Each year should have all 10 regions.
    counts = df.groupby("year")["region_code"].nunique()
    assert (counts == 10).all()


def test_regional_population_sums_close_to_national() -> None:
    """Sum across regions should be within a few percent of the national total."""
    regional = load_regional_population()
    national = load_population()

    summed = regional.groupby("year")["population_total"].sum()
    overlap_years = summed.index.intersection(national.index)

    diffs = (
        (summed.loc[overlap_years] - national.loc[overlap_years, "population_total"])
        / national.loc[overlap_years, "population_total"]
    ).abs()

    assert (diffs < 0.01).all(), f"max relative diff: {diffs.max():.4f}"


def test_load_regional_population_raises_on_missing_columns(
    tmp_path: Path,
) -> None:
    """Schema validation surfaces a clear error."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"year": [2020], "region_code": ["X"]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_regional_population(path=bad)


def test_default_asmr_csv_ships_with_repo() -> None:
    """The bundled ASMR file is part of the repository."""
    assert DEFAULT_ASMR_CSV.exists()


def test_load_asmr_returns_year_indexed_frame() -> None:
    """ASMR loader returns a year-indexed frame with the expected column."""
    df = load_asmr()

    assert df.index.name == "year"
    assert df.index.is_monotonic_increasing
    assert "asmr_persons_per_100k" in df.columns


def test_asmr_values_are_in_plausible_range() -> None:
    """ASMR for E&W has been ~900-1300 per 100k for the years we ship."""
    df = load_asmr()
    assert df["asmr_persons_per_100k"].between(800, 1400).all()


def test_asmr_shows_pre_pandemic_decline_then_2020_spike() -> None:
    """The series should drop 2006-2019 and spike up in 2020."""
    df = load_asmr()
    assert df.loc[2019, "asmr_persons_per_100k"] < df.loc[2006, "asmr_persons_per_100k"]
    assert df.loc[2020, "asmr_persons_per_100k"] > df.loc[2019, "asmr_persons_per_100k"]


def test_load_asmr_raises_on_missing_columns(tmp_path: Path) -> None:
    """Schema validation surfaces a clear error."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"year": [2020], "wrong": [1]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_asmr(path=bad)


def test_default_covid_csv_ships_with_repo() -> None:
    """The bundled COVID-19 deaths file is part of the repository."""
    assert DEFAULT_COVID_CSV.exists()


def test_load_covid_deaths_returns_year_indexed_frame() -> None:
    """COVID deaths loader returns a year-indexed frame with the expected column."""
    df = load_covid_deaths()
    assert df.index.name == "year"
    assert df.index.is_monotonic_increasing
    assert "deaths_involving_covid" in df.columns


def test_covid_deaths_peak_is_2020_or_2021() -> None:
    """The pandemic peak should be in 2020 or 2021."""
    df = load_covid_deaths()
    peak_year = int(df["deaths_involving_covid"].idxmax())
    assert peak_year in (2020, 2021)


def test_covid_deaths_decline_after_2021() -> None:
    """Vaccination + Omicron-then-mild waves: the trend after 2021 declines."""
    df = load_covid_deaths()
    assert (
        df.loc[2024, "deaths_involving_covid"]
        < df.loc[2021, "deaths_involving_covid"]
    )


def test_load_covid_deaths_raises_on_missing_columns(tmp_path: Path) -> None:
    """Schema validation surfaces a clear error."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"year": [2020], "wrong": [1]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_covid_deaths(path=bad)
