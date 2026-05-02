"""Generate the England and Wales mortality counterfactual chart.

Kept for backwards compatibility with the README. Prefer:

    poetry run ons-mortality plot

which exposes the same flags as a proper CLI subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ons_mortality.counterfactual import (
    CounterfactualConfig,
    fit_counterfactual,
    plot_counterfactual,
)
from ons_mortality.database import (
    create_mysql_engine,
    read_national_monthly_deaths,
)

INPUT_CSV = Path("data/processed/england_wales_monthly_deaths.csv")
OUTPUT_PNG = Path("figures/england_wales_counterfactual.png")


def load_input_data() -> pd.DataFrame:
    """
    Load mortality data from CSV, falling back to MySQL when needed.

    Returns
    -------
    pd.DataFrame
        Monthly national deaths for England and Wales.
    """
    if INPUT_CSV.exists():
        return pd.read_csv(INPUT_CSV)

    engine = create_mysql_engine()
    return read_national_monthly_deaths(engine)


def main() -> None:
    """Run the counterfactual analysis and save the chart."""
    df = load_input_data()
    config = CounterfactualConfig(pandemic_onset="2020-03-01")
    result = fit_counterfactual(df, config=config)
    plot_counterfactual(result=result, output_path=OUTPUT_PNG, config=config)
    print(f"Saved chart to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
