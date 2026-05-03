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
    TREND_SPECS,
    CounterfactualConfig,
    fit_counterfactual,
    plot_counterfactual,
)
from ons_mortality.fetch import AGE_BAND_ORDER
from ons_mortality.population import load_covid_deaths, load_regional_population

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEATHS_CSV = DATA_DIR / "england_wales_monthly_deaths.csv"
REGIONAL_CSV = DATA_DIR / "england_wales_regional_monthly_deaths.csv"
AGE_CSV = DATA_DIR / "england_wales_weekly_age_deaths.csv"
SEX_AGE_CSV = DATA_DIR / "england_wales_weekly_sex_age_deaths.csv"
CAUSE_CSV = DATA_DIR / "england_wales_cause_by_sex_age.csv"

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
# Figure 5 — Age-stratified excess (notebook 08)
# ---------------------------------------------------------------------------


def render_age_stratified_excess() -> None:
    """Two-panel chart: cumulative absolute excess + annual stack by band."""
    if not AGE_CSV.exists():
        print(f"Skipping age figure — {AGE_CSV.name} not found.")
        return

    ages = pd.read_csv(AGE_CSV, parse_dates=["week_ending"])
    ages["month_date"] = ages["week_ending"].values.astype("datetime64[M]")
    monthly = (
        ages.groupby(["age_band", "month_date"])["observed_deaths"]
            .sum()
            .reset_index()
    )
    monthly["month_date"] = pd.to_datetime(monthly["month_date"])

    config = CounterfactualConfig(
        pandemic_onset=str(PANDEMIC_ONSET.date()),
        n_posterior_samples=2_000,
        fourier_order=3,
        random_seed=42,
    )

    fits: list[pd.DataFrame] = []
    for band in AGE_BAND_ORDER:
        series = (
            monthly[monthly["age_band"] == band]
            [["month_date", "observed_deaths"]]
            .reset_index(drop=True)
        )
        fit = fit_counterfactual(series, config=config)
        fit["age_band"] = band
        fits.append(fit)

    all_fits = pd.concat(fits, ignore_index=True)
    post = all_fits[all_fits["month_date"] >= PANDEMIC_ONSET].copy()
    post["year"] = post["month_date"].dt.year

    cum_abs = (
        post.groupby("age_band")["excess_deaths"].sum().reindex(AGE_BAND_ORDER)
    )
    annual = (
        post.groupby(["age_band", "year"])["excess_deaths"]
            .sum().unstack("year").reindex(AGE_BAND_ORDER)
    )

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.0, 1.4]}
    )
    cmap = plt.get_cmap("plasma_r")
    colors = [
        cmap(i / (len(AGE_BAND_ORDER) - 1))
        for i in range(len(AGE_BAND_ORDER))
    ]

    ax_l.barh(list(AGE_BAND_ORDER), cum_abs.values, color="C3")
    for band, value in zip(AGE_BAND_ORDER, cum_abs.values, strict=True):
        ax_l.text(
            value + max(cum_abs) * 0.01, band,
            f"{value/1000:.1f}k", va="center", fontsize=10,
        )
    ax_l.invert_yaxis()
    ax_l.set_xlabel("cumulative excess deaths (2020-2024)")
    ax_l.set_title(
        "Where the excess landed",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax_l.grid(alpha=0.25, axis="x")

    years = list(annual.columns)
    bottoms = np.zeros(len(years), dtype=float)
    for band, color in zip(AGE_BAND_ORDER, colors, strict=True):
        values = annual.loc[band].values
        ax_r.bar(
            years, values, bottom=bottoms, color=color, label=band,
            edgecolor="white", linewidth=0.4,
        )
        bottoms = bottoms + values
    ax_r.set_xticks(years)
    ax_r.set_ylabel("annual excess deaths")
    ax_r.set_title(
        "How the age signature shifted",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax_r.grid(alpha=0.25, axis="y")
    handles, labels = ax_r.get_legend_handles_labels()
    ax_r.legend(
        handles[::-1], labels[::-1],
        loc="upper right", ncol=2, fontsize=9, title="age band",
    )

    fig.suptitle(
        "Pandemic excess by age band: 75-84 dominates absolutely; "
        "middle-age excess persists post-2022",
        fontsize=15, fontweight="bold", x=0.0, ha="left",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "age_stratified_excess.png",
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Sex × age excess (notebook 09)
# ---------------------------------------------------------------------------


def render_sex_age_excess() -> None:
    """Two-panel chart: absolute and relative excess by (sex, age band)."""
    if not SEX_AGE_CSV.exists():
        print(f"Skipping sex-age figure — {SEX_AGE_CSV.name} not found.")
        return

    raw = pd.read_csv(SEX_AGE_CSV, parse_dates=["week_ending"])
    raw["month_date"] = raw["week_ending"].values.astype("datetime64[M]")
    monthly = (
        raw.groupby(["sex", "age_band", "month_date"])["observed_deaths"]
            .sum()
            .reset_index()
    )
    monthly["month_date"] = pd.to_datetime(monthly["month_date"])

    config = CounterfactualConfig(
        pandemic_onset=str(PANDEMIC_ONSET.date()),
        n_posterior_samples=2_000,
        fourier_order=3,
        random_seed=42,
    )

    fits: list[pd.DataFrame] = []
    for sex in ("Male", "Female"):
        for band in AGE_BAND_ORDER:
            series = (
                monthly[(monthly["sex"] == sex) & (monthly["age_band"] == band)]
                [["month_date", "observed_deaths"]]
                .reset_index(drop=True)
            )
            fit = fit_counterfactual(series, config=config)
            fit["sex"] = sex
            fit["age_band"] = band
            fits.append(fit)

    all_fits = pd.concat(fits, ignore_index=True)
    post = all_fits[all_fits["month_date"] >= PANDEMIC_ONSET].copy()

    cum_excess = (
        post.groupby(["sex", "age_band"])["excess_deaths"].sum()
            .unstack("sex")[["Male", "Female"]].reindex(AGE_BAND_ORDER)
    )
    cum_cf = (
        post.groupby(["sex", "age_band"])["counterfactual_median"].sum()
            .unstack("sex")[["Male", "Female"]].reindex(AGE_BAND_ORDER)
    )
    rel_excess = cum_excess / cum_cf * 100

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    bands = list(AGE_BAND_ORDER)
    y = np.arange(len(bands))
    width = 0.4
    m_color = "#1f77b4"
    f_color = "#d62728"

    ax_l.barh(y - width / 2, cum_excess["Male"].values, width,
              color=m_color, label="Male")
    ax_l.barh(y + width / 2, cum_excess["Female"].values, width,
              color=f_color, label="Female")
    abs_max = max(cum_excess.values.max(), 1)
    for i, band in enumerate(bands):
        ax_l.text(
            cum_excess.loc[band, "Male"] + abs_max * 0.01, i - width / 2,
            f"{cum_excess.loc[band, 'Male']/1000:.1f}k",
            va="center", fontsize=9, color=m_color,
        )
        ax_l.text(
            cum_excess.loc[band, "Female"] + abs_max * 0.01, i + width / 2,
            f"{cum_excess.loc[band, 'Female']/1000:.1f}k",
            va="center", fontsize=9, color=f_color,
        )
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(bands)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("cumulative excess deaths (2020-2024)")
    ax_l.set_title(
        "Absolute excess by (sex, band)",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax_l.legend(loc="lower right")
    ax_l.grid(alpha=0.25, axis="x")

    ax_r.barh(y - width / 2, rel_excess["Male"].values, width,
              color=m_color, label="Male")
    ax_r.barh(y + width / 2, rel_excess["Female"].values, width,
              color=f_color, label="Female")
    for i, band in enumerate(bands):
        ax_r.text(
            rel_excess.loc[band, "Male"] + 0.2, i - width / 2,
            f"{rel_excess.loc[band, 'Male']:.1f}%",
            va="center", fontsize=9, color=m_color,
        )
        ax_r.text(
            rel_excess.loc[band, "Female"] + 0.2, i + width / 2,
            f"{rel_excess.loc[band, 'Female']:.1f}%",
            va="center", fontsize=9, color=f_color,
        )
    ax_r.set_xlabel("cumulative excess as % of counterfactual")
    ax_r.set_title(
        "Relative excess by (sex, band)",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax_r.grid(alpha=0.25, axis="x")

    fig.suptitle(
        "Male excess concentrates in working-age (45-64); "
        "female excess relatively higher in 75-84 and 15-44",
        fontsize=15, fontweight="bold", x=0.0, ha="left",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "sex_age_excess.png",
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Backtest calibration (notebook 10)
# ---------------------------------------------------------------------------


def render_backtest_calibration() -> None:
    """Strip plot of empirical coverage on a 2017-2019 holdout, by stratification."""
    train_end = pd.Timestamp("2017-01-01")
    test_end = pd.Timestamp("2020-01-01")

    def backtest(series: pd.DataFrame) -> float:
        sub = series[series["month_date"] < test_end][
            ["month_date", "observed_deaths"]
        ]
        config = CounterfactualConfig(
            pandemic_onset=str(train_end.date()),
            n_posterior_samples=2_000,
            fourier_order=3,
            interval_mass=0.94,
            random_seed=42,
        )
        fit = fit_counterfactual(sub, config=config)
        holdout = fit[
            (fit["month_date"] >= train_end) & (fit["month_date"] < test_end)
        ]
        within = (
            (holdout["observed_deaths"] >= holdout["counterfactual_lower"])
            & (holdout["observed_deaths"] <= holdout["counterfactual_upper"])
        )
        return float(within.mean())

    coverages: dict[str, list[float]] = {
        "National (1)": [],
        "Regional (10)": [],
        "Age band (7)": [],
        "Sex × age (14)": [],
    }

    if DEATHS_CSV.exists():
        national = pd.read_csv(DEATHS_CSV, parse_dates=["month_date"])
        coverages["National (1)"].append(backtest(national))

    if REGIONAL_CSV.exists():
        regional = pd.read_csv(REGIONAL_CSV, parse_dates=["month_date"])
        for code in regional["region_code"].unique():
            series = (
                regional[regional["region_code"] == code]
                [["month_date", "observed_deaths"]]
                .reset_index(drop=True)
            )
            coverages["Regional (10)"].append(backtest(series))

    if AGE_CSV.exists():
        ages = pd.read_csv(AGE_CSV, parse_dates=["week_ending"])
        ages["month_date"] = ages["week_ending"].values.astype("datetime64[M]")
        ages_monthly = (
            ages.groupby(["age_band", "month_date"])["observed_deaths"]
            .sum().reset_index()
        )
        ages_monthly["month_date"] = pd.to_datetime(ages_monthly["month_date"])
        for band in AGE_BAND_ORDER:
            series = (
                ages_monthly[ages_monthly["age_band"] == band]
                [["month_date", "observed_deaths"]]
                .reset_index(drop=True)
            )
            coverages["Age band (7)"].append(backtest(series))

    if SEX_AGE_CSV.exists():
        sa = pd.read_csv(SEX_AGE_CSV, parse_dates=["week_ending"])
        sa["month_date"] = sa["week_ending"].values.astype("datetime64[M]")
        sa_monthly = (
            sa.groupby(["sex", "age_band", "month_date"])["observed_deaths"]
            .sum().reset_index()
        )
        sa_monthly["month_date"] = pd.to_datetime(sa_monthly["month_date"])
        for sex in ("Male", "Female"):
            for band in AGE_BAND_ORDER:
                series = (
                    sa_monthly[
                        (sa_monthly["sex"] == sex)
                        & (sa_monthly["age_band"] == band)
                    ][["month_date", "observed_deaths"]]
                    .reset_index(drop=True)
                )
                coverages["Sex × age (14)"].append(backtest(series))

    if not any(coverages.values()):
        print("Skipping backtest figure — no input CSVs found.")
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    strats = list(coverages.keys())
    colors = ["C0", "C1", "C2", "C3"]
    rng = np.random.RandomState(7)
    for i, strat in enumerate(strats):
        vals = coverages[strat]
        if not vals:
            continue
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(
            vals, np.full(len(vals), i) + jitter,
            s=80, color=colors[i], alpha=0.75,
            edgecolor="white", linewidth=0.7,
        )
    ax.axvline(
        0.94, color="black", linestyle="--", linewidth=1.2,
        label="nominal 94%",
    )
    ax.set_yticks(range(len(strats)))
    ax.set_yticklabels(
        [f"{s}\nn={len(coverages[s])}" for s in strats]
    )
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.02)
    ax.set_xlabel("achieved coverage on 2017-2019 holdout")
    ax.set_title(
        "Backtest calibration: aggregated series under-cover; "
        "per-(sex × band) fits are well-calibrated",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "backtest_calibration.png",
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — Trend specification (notebook 11)
# ---------------------------------------------------------------------------


def render_trend_specification() -> None:
    """Two-panel chart: national projections + per-band excess by trend spec."""
    if not DEATHS_CSV.exists() or not AGE_CSV.exists():
        print("Skipping trend-spec figure — input CSVs not found.")
        return

    spec_labels = {
        "linear": "linear (default)",
        "log_linear": "log-linear",
        "quadratic": "quadratic",
    }
    spec_colors = {"linear": "C0", "log_linear": "C2", "quadratic": "C3"}

    national = pd.read_csv(DEATHS_CSV, parse_dates=["month_date"])
    ages = pd.read_csv(AGE_CSV, parse_dates=["week_ending"])
    ages["month_date"] = ages["week_ending"].values.astype("datetime64[M]")
    ages_monthly = (
        ages.groupby(["age_band", "month_date"])["observed_deaths"]
            .sum().reset_index()
    )
    ages_monthly["month_date"] = pd.to_datetime(ages_monthly["month_date"])

    national_fits: dict[str, pd.DataFrame] = {}
    band_excess: dict[str, dict[str, float]] = {s: {} for s in TREND_SPECS}
    for spec in TREND_SPECS:
        config = CounterfactualConfig(
            pandemic_onset=str(PANDEMIC_ONSET.date()),
            n_posterior_samples=2_000,
            fourier_order=3,
            interval_mass=0.94,
            random_seed=42,
            trend_spec=spec,
        )
        national_fits[spec] = fit_counterfactual(
            national[["month_date", "observed_deaths"]], config=config,
        )
        for band in AGE_BAND_ORDER:
            series = (
                ages_monthly[ages_monthly["age_band"] == band]
                [["month_date", "observed_deaths"]].reset_index(drop=True)
            )
            fit = fit_counterfactual(series, config=config)
            post = fit[fit["month_date"] >= PANDEMIC_ONSET]
            band_excess[spec][band] = float(post["excess_deaths"].sum())

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05]},
    )

    for spec in TREND_SPECS:
        fit = national_fits[spec]
        ax_l.fill_between(
            fit["month_date"], fit["counterfactual_lower"],
            fit["counterfactual_upper"],
            alpha=0.08, color=spec_colors[spec],
        )
        ax_l.plot(
            fit["month_date"], fit["counterfactual_median"],
            color=spec_colors[spec], linewidth=2.0, label=spec_labels[spec],
        )
    ax_l.plot(
        national["month_date"], national["observed_deaths"],
        color="black", linewidth=1.0, alpha=0.7, label="observed",
    )
    ax_l.axvline(PANDEMIC_ONSET, color="gray", linestyle="--", linewidth=1.0)
    ax_l.set_xlim(pd.Timestamp("2015-01-01"), national["month_date"].max())
    ax_l.set_title(
        "National counterfactual by trend spec",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax_l.set_ylabel("monthly deaths")
    ax_l.legend(loc="upper left", fontsize=10)
    ax_l.grid(alpha=0.25)

    bands = list(AGE_BAND_ORDER)
    y_pos = np.arange(len(bands))
    width = 0.27
    for i, spec in enumerate(TREND_SPECS):
        values = [band_excess[spec][b] for b in bands]
        offset = (i - 1) * width
        ax_r.barh(
            y_pos + offset, values, width,
            color=spec_colors[spec], alpha=0.85, label=spec_labels[spec],
        )
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(bands)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("cumulative excess deaths (2020-2024)")
    ax_r.set_title(
        "Per-band excess by trend spec",
        loc="left", fontweight="bold", fontsize=14,
    )
    ax_r.grid(alpha=0.25, axis="x")
    ax_r.legend(loc="lower right", fontsize=10)

    totals_str = " / ".join(
        f"{spec_labels[s]}: {sum(band_excess[s].values()) / 1000:.0f}k"
        for s in TREND_SPECS
    )
    fig.suptitle(
        f"Trend specification matters: cumulative excess is {totals_str}",
        fontsize=15, fontweight="bold", x=0.0, ha="left",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "trend_specification.png",
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — Cause decomposition for Male 45-64 (notebook 12)
# ---------------------------------------------------------------------------


def render_cause_decomposition() -> None:
    """Two-panel chart: M 45-64 stacked excess by year + M-vs-F by group."""
    if not CAUSE_CSV.exists():
        print(f"Skipping cause figure — {CAUSE_CSV.name} not found.")
        return

    raw = pd.read_csv(CAUSE_CSV)
    all_causes_code = "A00 to R99 U00 to Y89"
    deaths = raw[
        (raw["place"] == "All places")
        & (raw["icd_chapter_code"] != all_causes_code)
    ].copy()

    pre_years = list(range(2015, 2020))
    post_years = list(range(2020, 2025))

    def fit(sex: str, age: str) -> pd.DataFrame:
        sub = deaths[(deaths["sex"] == sex) & (deaths["age_band"] == age)]
        pivot = (
            sub.pivot_table(
                index="icd_chapter_name", columns="year",
                values="deaths", fill_value=0,
            ).astype(int)
        )
        rows: list[dict] = []
        x_pre = np.array([y - pre_years[0] for y in pre_years], dtype=float)
        for chapter in pivot.index:
            if pivot.loc[chapter].sum() == 0:
                continue
            y_train = np.array(
                [pivot.loc[chapter, y] for y in pre_years], dtype=float,
            )
            if y_train.std() == 0:
                slope, intercept = 0.0, float(y_train.mean())
            else:
                slope, intercept = np.polyfit(x_pre, y_train, 1)
            for year in post_years:
                projected = intercept + slope * (year - pre_years[0])
                observed = int(pivot.loc[chapter, year])
                rows.append({
                    "chapter": chapter, "year": year,
                    "excess": observed - projected,
                })
        return pd.DataFrame(rows)

    m_df = fit("Male", "45-64")
    f_df = fit("Female", "45-64")

    groups = [
        ("COVID-19 (U codes)",
         ["Chapter 22 - Codes for special purposes"], "#d62728"),
        ("Circulatory (heart, stroke)",
         ["Chapter 9 - Diseases of the circulatory system"], "#1f77b4"),
        ("Digestive (incl. alcohol-related)",
         ["Chapter 11 - Diseases of the digestive system"], "#2ca02c"),
        ("Neoplasms (cancer)",
         ["Chapter 2 - Neoplasms"], "#9467bd"),
        ("External (suicide, drugs, accidents)",
         ["Chapter 20 - External causes of morbidity and mortality"], "#ff7f0e"),
    ]

    def grouped(df: pd.DataFrame, year: int) -> dict[str, float]:
        out: dict[str, float] = {}
        accounted: set[str] = set()
        for label, chaps, _ in groups:
            v = df[
                (df["year"] == year) & (df["chapter"].isin(chaps))
            ]["excess"].sum()
            out[label] = float(v)
            accounted |= set(chaps)
        out["Other"] = float(
            df[(df["year"] == year) & (~df["chapter"].isin(accounted))]
            ["excess"].sum()
        )
        return out

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    labels_with_other = [g[0] for g in groups] + ["Other"]
    colors_with_other = [g[2] for g in groups] + ["#999999"]

    bottoms_pos = np.zeros(len(post_years))
    bottoms_neg = np.zeros(len(post_years))
    seen_labels: set[str] = set()
    for label, color in zip(labels_with_other, colors_with_other, strict=True):
        vals = np.array(
            [grouped(m_df, yr)[label] for yr in post_years], dtype=float,
        )
        for j, v in enumerate(vals):
            kw = {"label": label} if label not in seen_labels else {}
            if v >= 0:
                ax_l.bar(post_years[j], v, bottom=bottoms_pos[j], color=color,
                         edgecolor="white", linewidth=0.4, **kw)
                bottoms_pos[j] += v
            else:
                ax_l.bar(post_years[j], v, bottom=bottoms_neg[j], color=color,
                         edgecolor="white", linewidth=0.4, **kw)
                bottoms_neg[j] += v
            seen_labels.add(label)
    ax_l.axhline(0, color="black", linewidth=0.7)
    ax_l.set_xticks(post_years)
    ax_l.set_ylabel("annual excess deaths (vs pre-pandemic linear projection)")
    ax_l.set_title(
        "Male 45-64: COVID dominates 2020-21,\nCVD persists post-2022",
        loc="left", fontweight="bold", fontsize=13,
    )
    ax_l.legend(loc="upper right", fontsize=8, ncol=1)
    ax_l.grid(alpha=0.25, axis="y")

    group_labels = [g[0] for g in groups]
    m_totals = []
    f_totals = []
    for _, chaps, _ in groups:
        m_totals.append(float(m_df[m_df["chapter"].isin(chaps)]["excess"].sum()))
        f_totals.append(float(f_df[f_df["chapter"].isin(chaps)]["excess"].sum()))

    y_pos = np.arange(len(group_labels))
    width = 0.4
    ax_r.barh(y_pos - width / 2, m_totals, width,
              color="#1f77b4", label="Male", alpha=0.85)
    ax_r.barh(y_pos + width / 2, f_totals, width,
              color="#d62728", label="Female", alpha=0.85)
    ax_r.axvline(0, color="black", linewidth=0.7)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(group_labels, fontsize=10)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("cumulative excess deaths 2020-2024")
    ax_r.set_title(
        "45-64 cumulative excess by cause group: Male vs Female",
        loc="left", fontweight="bold", fontsize=13,
    )
    ax_r.grid(alpha=0.25, axis="x")
    ax_r.legend(loc="lower right")

    fig.suptitle(
        "The 45-64 male residual is cardiovascular + alcohol-related, "
        "not drugs/suicide",
        fontsize=15, fontweight="bold", x=0.0, ha="left",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cause_decomposition.png",
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    """Render all nine README figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    render_headline_counterfactual()
    print("Saved figures/england_wales_counterfactual.png")
    render_regional_per_capita()
    print("Saved figures/regional_per_capita_excess.png")
    render_covid_decomposition()
    print("Saved figures/covid_vs_non_covid_excess.png")
    render_live_forecast()
    print("Saved figures/live_forecast.png")
    render_age_stratified_excess()
    print("Saved figures/age_stratified_excess.png")
    render_sex_age_excess()
    print("Saved figures/sex_age_excess.png")
    render_backtest_calibration()
    print("Saved figures/backtest_calibration.png")
    render_trend_specification()
    print("Saved figures/trend_specification.png")
    render_cause_decomposition()
    print("Saved figures/cause_decomposition.png")


if __name__ == "__main__":
    main()
