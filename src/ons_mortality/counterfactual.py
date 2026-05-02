"""Counterfactual mortality model and plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CounterfactualConfig:
    """Configuration for the mortality counterfactual model."""

    pandemic_onset: str = "2020-03-01"
    n_posterior_samples: int = 8_000
    random_seed: int = 42
    fourier_order: int = 3
    interval_mass: float = 0.94


def validate_counterfactual_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize the counterfactual input data.

    Parameters
    ----------
    df:
        Input data with `month_date` and `observed_deaths` columns.

    Returns
    -------
    pd.DataFrame
        Cleaned and sorted data.
    """
    required = {"month_date", "observed_deaths"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = df.loc[:, ["month_date", "observed_deaths"]].copy()
    clean["month_date"] = pd.to_datetime(clean["month_date"])
    clean["observed_deaths"] = pd.to_numeric(clean["observed_deaths"], errors="raise")

    if clean["observed_deaths"].isna().any():
        raise ValueError("observed_deaths contains missing values.")

    if (clean["observed_deaths"] < 0).any():
        raise ValueError("observed_deaths cannot contain negative values.")

    clean = clean.sort_values("month_date").drop_duplicates("month_date", keep="last")
    return clean.reset_index(drop=True)


def build_design_matrix(dates: pd.Series | pd.DatetimeIndex, fourier_order: int = 3) -> np.ndarray:
    """
    Build a trend-plus-seasonality design matrix.

    Parameters
    ----------
    dates:
        Monthly dates.
    fourier_order:
        Number of annual Fourier harmonics.

    Returns
    -------
    np.ndarray
        Regression design matrix.
    """
    if fourier_order < 1:
        raise ValueError("fourier_order must be at least 1.")

    resolved_dates = pd.DatetimeIndex(pd.to_datetime(dates))
    n_dates = len(resolved_dates)

    if n_dates == 0:
        raise ValueError("dates cannot be empty.")

    trend = np.arange(n_dates, dtype=float)
    month = resolved_dates.month.astype(float)

    seasonal_terms: list[np.ndarray] = []
    for harmonic in range(1, fourier_order + 1):
        seasonal_terms.append(np.sin(2.0 * np.pi * harmonic * month / 12.0))
        seasonal_terms.append(np.cos(2.0 * np.pi * harmonic * month / 12.0))

    return np.column_stack([np.ones(n_dates), trend, *seasonal_terms])


def fit_counterfactual(
    df: pd.DataFrame,
    config: CounterfactualConfig | None = None,
) -> pd.DataFrame:
    """
    Fit a Bayesian-style counterfactual model on pre-pandemic data.

    The model uses a conjugate Gaussian linear-regression approximation. It is
    intentionally transparent and fast. The output contains predictive draws
    summarized by a median and a central interval.

    Parameters
    ----------
    df:
        DataFrame with `month_date` and `observed_deaths`.
    config:
        Optional model configuration.

    Returns
    -------
    pd.DataFrame
        Input data plus counterfactual median, interval, and excess deaths.
    """
    cfg = config or CounterfactualConfig()
    clean = validate_counterfactual_input(df)

    if cfg.n_posterior_samples < 100:
        raise ValueError("n_posterior_samples should be at least 100.")

    if not 0.0 < cfg.interval_mass < 1.0:
        raise ValueError("interval_mass must be between 0 and 1.")

    pandemic_onset = pd.Timestamp(cfg.pandemic_onset)
    train_mask = clean["month_date"] < pandemic_onset

    if train_mask.sum() < 24:
        raise ValueError("At least 24 pre-pandemic months are required.")

    X_train = build_design_matrix(clean.loc[train_mask, "month_date"], cfg.fourier_order)
    y_train = clean.loc[train_mask, "observed_deaths"].to_numpy(dtype=float)
    X_all = build_design_matrix(clean["month_date"], cfg.fourier_order)

    n_obs, n_params = X_train.shape
    if n_obs <= n_params:
        raise ValueError("Not enough observations for the requested Fourier order.")

    rng = np.random.default_rng(cfg.random_seed)

    xtx_inv = np.linalg.pinv(X_train.T @ X_train)
    beta_hat = xtx_inv @ X_train.T @ y_train

    residuals = y_train - X_train @ beta_hat
    degrees_of_freedom = n_obs - n_params
    sigma2_hat = float((residuals @ residuals) / degrees_of_freedom)

    if sigma2_hat <= 0.0:
        raise ValueError("Residual variance is not positive; check input data.")

    sigma2_samples = (
        degrees_of_freedom
        * sigma2_hat
        / rng.chisquare(df=degrees_of_freedom, size=cfg.n_posterior_samples)
    )

    predictions = np.empty((cfg.n_posterior_samples, len(clean)), dtype=float)

    for idx, sigma2 in enumerate(sigma2_samples):
        beta_sample = rng.multivariate_normal(mean=beta_hat, cov=sigma2 * xtx_inv)
        mean_prediction = X_all @ beta_sample

        # Predictive uncertainty includes observation-level residual variation.
        predictions[idx, :] = mean_prediction + rng.normal(
            loc=0.0,
            scale=np.sqrt(sigma2),
            size=len(clean),
        )

    alpha = 1.0 - cfg.interval_mass
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    output = clean.copy()
    output["counterfactual_median"] = np.quantile(predictions, 0.50, axis=0)
    output["counterfactual_lower"] = np.quantile(predictions, lower_q, axis=0)
    output["counterfactual_upper"] = np.quantile(predictions, upper_q, axis=0)
    output["excess_deaths"] = np.where(
        output["month_date"] >= pandemic_onset,
        np.maximum(output["observed_deaths"] - output["counterfactual_median"], 0.0),
        0.0,
    )

    return output


def plot_counterfactual(
    result: pd.DataFrame,
    output_path: Path | None = None,
    title_prefix: str = "England & Wales monthly mortality",
    config: CounterfactualConfig | None = None,
    observed_label: str = "observed monthly deaths (England & Wales)",
) -> None:
    """
    Plot observed deaths against the estimated no-pandemic counterfactual.

    Parameters
    ----------
    result:
        Output from `fit_counterfactual`.
    output_path:
        Optional path where the figure should be saved.
    title_prefix:
        First line of the chart title.
    config:
        Model configuration used for the pandemic onset date.
    observed_label:
        Legend label used for the observed-deaths line.
    """
    cfg = config or CounterfactualConfig()
    required = {
        "month_date",
        "observed_deaths",
        "counterfactual_median",
        "counterfactual_lower",
        "counterfactual_upper",
        "excess_deaths",
    }
    missing = required.difference(result.columns)

    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")

    plot_df = result.copy()
    plot_df["month_date"] = pd.to_datetime(plot_df["month_date"])

    pandemic_onset = pd.Timestamp(cfg.pandemic_onset)
    post_mask = plot_df["month_date"] >= pandemic_onset
    cumulative_excess = float(plot_df.loc[post_mask, "excess_deaths"].sum())

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7))

    interval_pct = int(round(cfg.interval_mass * 100))
    ax.fill_between(
        plot_df["month_date"],
        plot_df["counterfactual_lower"],
        plot_df["counterfactual_upper"],
        alpha=0.25,
        label=f"counterfactual {interval_pct}% interval  (no pandemic)",
    )

    ax.plot(
        plot_df["month_date"],
        plot_df["counterfactual_median"],
        linewidth=2.2,
        label="counterfactual median",
    )

    ax.plot(
        plot_df["month_date"],
        plot_df["observed_deaths"],
        color="black",
        linewidth=2.0,
        label=observed_label,
    )

    ax.fill_between(
        plot_df.loc[post_mask, "month_date"],
        plot_df.loc[post_mask, "counterfactual_median"],
        plot_df.loc[post_mask, "observed_deaths"],
        where=(
            plot_df.loc[post_mask, "observed_deaths"]
            > plot_df.loc[post_mask, "counterfactual_median"]
        ),
        alpha=0.30,
        label=f"excess deaths (≈ {cumulative_excess / 1000:.0f}k cumulative)",
    )

    ax.axvline(pandemic_onset, color="gray", linestyle="--", linewidth=1.2)
    ax.text(
        pandemic_onset + pd.DateOffset(months=2),
        ax.get_ylim()[1] * 0.96,
        "pandemic onset",
        fontsize=12,
        color="dimgray",
    )

    ax.set_title(
        f"{title_prefix} vs. the world without the pandemic\n"
        "Bayesian counterfactual fit on pre-2020 data, projected forward",
        loc="left",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_ylabel("monthly all-cause deaths", fontsize=13)
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=12)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    plt.close(fig)
