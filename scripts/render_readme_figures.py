"""Render the four headline figures embedded in the README.

Each figure is a self-contained PNG saved under ``figures/``. The script
is intentionally independent of the notebooks so the GitHub Action can
regenerate the rendered README assets without re-executing the heavier
notebook narratives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ons_mortality.counterfactual import (
    CounterfactualConfig,
    fit_counterfactual,
    plot_counterfactual,
)
from ons_mortality.population import load_covid_deaths, load_regional_population

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEATHS_CSV = DATA_DIR / "england_wales_monthly_deaths.csv"
REGIONAL_CSV = DATA_DIR / "england_wales_regional_monthly_deaths.csv"

PANDEMIC_ONSET = pd.Timestamp("2020-03-01")
DPI = 150

# ---------------------------------------------------------------------------
# Figure 1 — Headline national counterfactual (notebook 01)
# ---------------------------------------------------------------------------


def render_headline_counterfactual() -> None:
    """Re-render the national counterfactual chart with the standard styling."""
    deaths = pd.read_csv(DEATHS_CSV, parse_dates=["month_date"])
    config = CounterfactualConfig(
        pandemic_onset=str(PANDEMIC_ONSET.date()),
        n_posterior_samples=4_000,
        fourier_order=3,
        interval_mass=0.94,
        random_seed=42,
    )
    result = fit_counterfactual(deaths[["month_date", "observed_deaths"]], config=config)
    plot_counterfactual(
        result=result,
        output_path=FIGURES_DIR / "england_wales_counterfactual.png",
        config=config,
    )


# ---------------------------------------------------------------------------
# Figure 2 — Per-capita regional excess (notebook 02)
# ---------------------------------------------------------------------------


def render_regional_per_capita() -> None:
    """Bar chart of cumulative per-capita pandemic excess across regions."""
    regional = pd.read_csv(REGIONAL_CSV, parse_dates=["month_date"])
    pop_2019 = (
        load_regional_population()
        .query("year == 2019")
        .set_index("region_code")["population_total"]
    )

    config = CounterfactualConfig(
        pandemic_onset=str(PANDEMIC_ONSET.date()),
        n_posterior_samples=2_000,
        fourier_order=3,
        random_seed=42,
    )

    excess_by_region: dict[str, tuple[str, float]] = {}
    for code in regional["region_code"].unique():
        region_rows = regional[regional["region_code"] == code]
        region_name = region_rows["region_name"].iloc[0]
        series = region_rows[["month_date", "observed_deaths"]].reset_index(drop=True)
        fit = fit_counterfactual(series, config=config)
        post = fit[fit["month_date"] >= PANDEMIC_ONSET]
        cumulative = float(post["excess_deaths"].sum())
        per_1000 = cumulative / pop_2019.loc[code] * 1_000
        excess_by_region[code] = (region_name, per_1000)

    sorted_regions = sorted(excess_by_region.items(), key=lambda kv: kv[1][1])
    names = [name for _, (name, _) in sorted_regions]
    values = [value for _, (_, value) in sorted_regions]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["C3" if v == max(values) else "C0" if v == min(values) else "0.4" for v in values]
    ax.barh(names, values, color=colors)
    for name, value in zip(names, values, strict=True):
        ax.text(value + 0.05, name, f"{value:.2f}", va="center", fontsize=9)
    ax.set_xlabel("cumulative excess per 1 000 of 2019 population")
    ax.set_title(
        "Per-capita pandemic excess by region (2020-2024)",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regional_per_capita_excess.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — COVID vs non-COVID excess (notebook 06)
# ---------------------------------------------------------------------------


def render_covid_decomposition() -> None:
    """Stacked-bar of annual excess split into COVID-19 and non-COVID."""
    deaths = pd.read_csv(DEATHS_CSV, parse_dates=["month_date"])
    config = CounterfactualConfig(
        pandemic_onset=str(PANDEMIC_ONSET.date()),
        n_posterior_samples=4_000,
        fourier_order=3,
        random_seed=42,
    )
    fit = fit_counterfactual(deaths[["month_date", "observed_deaths"]], config=config)
    fit["year"] = fit["month_date"].dt.year
    fit["excess_signed"] = fit["observed_deaths"] - fit["counterfactual_median"]

    post = fit[fit["month_date"] >= PANDEMIC_ONSET]
    annual = post.groupby("year").agg(
        total_signed=("excess_signed", "sum"),
    )

    covid = load_covid_deaths()
    table = annual.join(covid)
    table["non_covid_signed"] = table["total_signed"] - table["deaths_involving_covid"]

    years = table.index.values
    covid_bar = table["deaths_involving_covid"].values
    non_covid_bar = table["non_covid_signed"].values
    total_outline = table["total_signed"].values

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(years, covid_bar, color="C3", label="deaths involving COVID-19")
    ax.bar(
        years,
        non_covid_bar,
        bottom=np.where(non_covid_bar >= 0, covid_bar, 0),
        color="C0",
        label="non-COVID excess (signed)",
    )
    ax.plot(
        years,
        total_outline,
        "k_",
        markersize=40,
        markeredgewidth=2.5,
        label="total signed excess",
        zorder=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Annual excess deaths: COVID-19 vs non-COVID component",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_ylabel("deaths")
    ax.set_xticks(years)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "covid_vs_non_covid_excess.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Live forecast (notebook 07)
# ---------------------------------------------------------------------------


def _design_matrix(dates: pd.DatetimeIndex, fourier_order: int) -> np.ndarray:
    n = len(dates)
    trend = np.arange(n, dtype=float)
    month = dates.month.astype(float).to_numpy()
    cols = [np.ones(n), trend]
    for k in range(1, fourier_order + 1):
        cols.append(np.sin(2 * np.pi * k * month / 12.0))
        cols.append(np.cos(2 * np.pi * k * month / 12.0))
    return np.column_stack(cols)


def _fit_predict(
    dates: pd.DatetimeIndex,
    y_train: np.ndarray,
    train_mask: np.ndarray,
    fourier_order: int = 3,
    n_draws: int = 4_000,
    seed: int = 42,
) -> pd.DataFrame:
    x_all = _design_matrix(dates, fourier_order)
    x_train = x_all[train_mask]
    beta = np.linalg.solve(x_train.T @ x_train, x_train.T @ y_train)
    sigma2 = float(np.sum((y_train - x_train @ beta) ** 2) / (len(y_train) - x_train.shape[1]))

    rng = np.random.default_rng(seed)
    cov = sigma2 * np.linalg.inv(x_train.T @ x_train)
    beta_samples = rng.multivariate_normal(beta, cov, size=n_draws)
    means = x_all @ beta_samples.T
    pred = means + rng.normal(0, np.sqrt(sigma2), size=means.shape)

    return pd.DataFrame(
        {
            "date": dates,
            "median": np.quantile(pred, 0.50, axis=1),
            "lower": np.quantile(pred, 0.03, axis=1),
            "upper": np.quantile(pred, 0.97, axis=1),
        }
    )


def render_live_forecast(projection_months: int = 24) -> None:
    """Counterfactual + forecast both projected past the last observed month."""
    deaths = (
        pd.read_csv(DEATHS_CSV, parse_dates=["month_date"])
        .sort_values("month_date")
        .reset_index(drop=True)
    )
    last_obs = deaths["month_date"].max()
    horizon = pd.date_range(
        start=deaths["month_date"].min(),
        end=last_obs + pd.DateOffset(months=projection_months),
        freq="MS",
    )

    cf_train_mask = horizon < PANDEMIC_ONSET
    cf_y = deaths.loc[deaths["month_date"] < PANDEMIC_ONSET, "observed_deaths"].to_numpy(
        dtype=float
    )
    cf_fit = _fit_predict(horizon, cf_y, cf_train_mask)

    fc_train_mask = horizon <= last_obs
    fc_y = deaths["observed_deaths"].to_numpy(dtype=float)
    fc_fit = _fit_predict(horizon, fc_y, fc_train_mask)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(
        cf_fit["date"], cf_fit["lower"], cf_fit["upper"],
        color="C0", alpha=0.15, label="counterfactual 94% interval",
    )
    ax.fill_between(
        fc_fit["date"], fc_fit["lower"], fc_fit["upper"],
        color="C3", alpha=0.15, label="forecast 94% interval",
    )
    ax.plot(cf_fit["date"], cf_fit["median"], color="C0", linewidth=2.0, label="counterfactual")
    ax.plot(fc_fit["date"], fc_fit["median"], color="C3", linewidth=2.0, label="forecast")
    ax.plot(
        deaths["month_date"], deaths["observed_deaths"],
        color="black", linewidth=1.4, label="observed",
    )
    ax.axvline(PANDEMIC_ONSET, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(last_obs, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title(
        f"Live forecast and counterfactual (projection through {horizon[-1]:%Y-%m})",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax.set_ylabel("monthly deaths")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "live_forecast.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    """Render all four README figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    render_headline_counterfactual()
    print("Saved figures/england_wales_counterfactual.png")
    render_regional_per_capita()
    print("Saved figures/regional_per_capita_excess.png")
    render_covid_decomposition()
    print("Saved figures/covid_vs_non_covid_excess.png")
    render_live_forecast()
    print("Saved figures/live_forecast.png")


if __name__ == "__main__":
    main()
