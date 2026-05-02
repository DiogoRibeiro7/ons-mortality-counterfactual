"""Tests for the counterfactual model and its input validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ons_mortality.counterfactual import (
    CounterfactualConfig,
    build_design_matrix,
    fit_counterfactual,
    validate_counterfactual_input,
)


def test_fit_counterfactual_returns_expected_columns() -> None:
    """End-to-end: model fits and outputs the canonical columns."""
    dates = pd.date_range("2015-01-01", "2021-12-01", freq="MS")
    month = dates.month.to_numpy()
    trend = np.arange(len(dates)) * 10.0
    seasonal = 1_000.0 * np.cos(2.0 * np.pi * month / 12.0)
    deaths = 40_000.0 + trend + seasonal

    # Add a small pandemic-like shock after onset.
    deaths = deaths + np.where(dates >= pd.Timestamp("2020-03-01"), 2_000.0, 0.0)

    df = pd.DataFrame({"month_date": dates, "observed_deaths": deaths})
    config = CounterfactualConfig(n_posterior_samples=200, random_seed=1)

    result = fit_counterfactual(df, config=config)

    expected_columns = {
        "month_date",
        "observed_deaths",
        "counterfactual_median",
        "counterfactual_lower",
        "counterfactual_upper",
        "excess_deaths",
    }
    assert expected_columns.issubset(result.columns)
    assert len(result) == len(df)
    pre_pandemic = result["month_date"] < "2020-03-01"
    assert result.loc[pre_pandemic, "excess_deaths"].eq(0).all()


def test_validate_counterfactual_input_rejects_missing_columns() -> None:
    """Validation must surface the schema contract clearly."""
    bad = pd.DataFrame({"month_date": ["2020-01-01"], "deaths": [1]})
    with pytest.raises(ValueError, match="observed_deaths"):
        validate_counterfactual_input(bad)


def test_validate_counterfactual_input_rejects_negative_values() -> None:
    """Death counts cannot be negative."""
    bad = pd.DataFrame(
        {
            "month_date": pd.to_datetime(["2019-01-01", "2019-02-01"]),
            "observed_deaths": [40_000, -1],
        }
    )
    with pytest.raises(ValueError, match="negative"):
        validate_counterfactual_input(bad)


def test_validate_counterfactual_input_dedupes_and_sorts() -> None:
    """Duplicate months keep the last value and rows come back sorted."""
    df = pd.DataFrame(
        {
            "month_date": pd.to_datetime(
                ["2019-02-01", "2019-01-01", "2019-01-01"]
            ),
            "observed_deaths": [44_000, 50_000, 51_000],
        }
    )
    out = validate_counterfactual_input(df)

    assert list(out["month_date"]) == [
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2019-02-01"),
    ]
    assert out.loc[0, "observed_deaths"] == 51_000


def test_build_design_matrix_shape_grows_with_fourier_order() -> None:
    """Design matrix has 1 (intercept) + 1 (trend) + 2*K seasonal columns."""
    dates = pd.date_range("2018-01-01", periods=24, freq="MS")

    matrix_one = build_design_matrix(dates, fourier_order=1)
    matrix_three = build_design_matrix(dates, fourier_order=3)

    assert matrix_one.shape == (24, 4)
    assert matrix_three.shape == (24, 8)


def test_build_design_matrix_rejects_zero_order() -> None:
    """fourier_order < 1 isn't a meaningful seasonality term."""
    dates = pd.date_range("2018-01-01", periods=12, freq="MS")
    with pytest.raises(ValueError, match="fourier_order"):
        build_design_matrix(dates, fourier_order=0)


def test_fit_counterfactual_excess_is_nonnegative_post_onset() -> None:
    """Post-pandemic excess is clipped at zero (deficits collapse to 0)."""
    dates = pd.date_range("2015-01-01", "2021-06-01", freq="MS")
    month = dates.month.to_numpy()
    seasonal = 1_000.0 * np.cos(2.0 * np.pi * month / 12.0)
    deaths = 40_000.0 + np.arange(len(dates)) * 5.0 + seasonal

    df = pd.DataFrame({"month_date": dates, "observed_deaths": deaths})
    result = fit_counterfactual(
        df,
        config=CounterfactualConfig(n_posterior_samples=200, random_seed=1),
    )

    assert (result["excess_deaths"] >= 0).all()
