from __future__ import annotations

import numpy as np
import pandas as pd

from ons_mortality.counterfactual import CounterfactualConfig, fit_counterfactual


def test_fit_counterfactual_returns_expected_columns() -> None:
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
    assert result.loc[result["month_date"] < "2020-03-01", "excess_deaths"].eq(0).all()
