"""Command-line interface for ONS mortality ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ons_mortality.config import load_path_config
from ons_mortality.counterfactual import (
    CounterfactualConfig,
    fit_counterfactual,
    plot_counterfactual,
)
from ons_mortality.database import (
    create_mysql_engine,
    init_database,
    insert_records_ignore,
    read_national_monthly_deaths,
)
from ons_mortality.fetch import (
    fetch_national_monthly_deaths,
    fetch_regional_monthly_deaths,
    fetch_weekly_national_deaths,
)
from ons_mortality.ons import (
    discover_ons_files,
    iter_downloaded_files,
    register_source_file,
)
from ons_mortality.parser import normalise_workbook


def _add_year_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common start/end year arguments to a parser."""
    parser.add_argument("--start-year", type=int, default=2006, help="First ONS edition year.")
    parser.add_argument("--end-year", type=int, default=2022, help="Last ONS edition year.")


def handle_init_db(args: argparse.Namespace) -> None:
    """Initialize the MySQL schema."""
    engine = create_mysql_engine()
    init_database(engine=engine, schema_path=Path(args.schema))
    print(f"Initialized database using schema: {args.schema}")


def handle_discover(args: argparse.Namespace) -> None:
    """Discover available ONS files."""
    files = discover_ons_files(start_year=args.start_year, end_year=args.end_year)

    if not files:
        print("No files discovered.")
        return

    for file in files:
        status = "final" if file.is_final else "provisional"
        print(f"{file.edition_year} [{status}] {file.filename} -> {file.url}")


def handle_ingest(args: argparse.Namespace) -> None:
    """Download, parse, and insert ONS workbooks into MySQL."""
    engine = create_mysql_engine()
    paths = load_path_config()

    files = discover_ons_files(start_year=args.start_year, end_year=args.end_year)
    if not files:
        raise RuntimeError("No ONS workbooks were discovered.")

    total_inserted = 0

    for file, local_path in iter_downloaded_files(
        files,
        output_dir=paths.raw_dir,
        overwrite=args.overwrite,
    ):
        source_file_id = register_source_file(engine=engine, file=file, local_path=local_path)
        parsed = normalise_workbook(
            path=local_path,
            source_file_id=source_file_id,
            edition_year=file.edition_year,
            is_final=file.is_final,
        )

        if parsed.empty:
            print(f"No usable rows found in {local_path.name}")
            continue

        records = parsed.to_dict(orient="records")
        insert_records_ignore(engine=engine, table_name="monthly_deaths", records=records)
        total_inserted += len(records)
        print(f"Parsed {len(records):,} rows from {local_path.name}")

    print(f"Finished ingestion. Parsed rows before duplicate filtering: {total_inserted:,}")


def handle_export_national(args: argparse.Namespace) -> None:
    """Export England and Wales national monthly deaths to CSV."""
    engine = create_mysql_engine()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_national_monthly_deaths(engine)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df):,} rows to {output_path}")


def handle_plot(args: argparse.Namespace) -> None:
    """Fit and save a counterfactual chart from a CSV or MySQL export."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.exists():
        df = pd.read_csv(input_path)
    else:
        engine = create_mysql_engine()
        df = read_national_monthly_deaths(engine)

    config = CounterfactualConfig(
        pandemic_onset=args.pandemic_onset,
        n_posterior_samples=args.samples,
        random_seed=args.seed,
        fourier_order=args.fourier_order,
        interval_mass=args.interval_mass,
    )

    result = fit_counterfactual(df, config=config)
    plot_counterfactual(result=result, output_path=output_path, config=config)
    print(f"Saved chart to {output_path}")


def handle_fetch(args: argparse.Namespace) -> None:
    """Download ONS workbooks and write the E&W national series to CSV."""
    paths = load_path_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_dir
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_national_monthly_deaths(
        raw_dir=raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )
    df.to_csv(csv_path, index=False)

    span = f"{df['month_date'].min():%Y-%m} -> {df['month_date'].max():%Y-%m}"
    print(f"Wrote {len(df):,} rows ({span}) to {csv_path}")


def handle_fetch_weekly(args: argparse.Namespace) -> None:
    """Download ONS weekly workbooks and write the E&W weekly series to CSV."""
    paths = load_path_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_dir / "weekly"
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_weekly_national_deaths(
        raw_dir=raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )
    df.to_csv(csv_path, index=False)

    span = f"{df['week_ending'].min():%Y-%m-%d} -> {df['week_ending'].max():%Y-%m-%d}"
    print(f"Wrote {len(df):,} weekly rows ({span}) to {csv_path}")


def handle_fetch_regional(args: argparse.Namespace) -> None:
    """Download ONS workbooks and write the per-region monthly series to CSV."""
    paths = load_path_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_dir
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_regional_monthly_deaths(
        raw_dir=raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )
    df.to_csv(csv_path, index=False)

    span = f"{df['month_date'].min():%Y-%m} -> {df['month_date'].max():%Y-%m}"
    n_regions = df['region_code'].nunique()
    print(
        f"Wrote {len(df):,} rows across {n_regions} regions ({span}) "
        f"to {csv_path}"
    )


def handle_run(args: argparse.Namespace) -> None:
    """Fetch, fit, and plot in one go (no MySQL)."""
    paths = load_path_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_dir
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_national_monthly_deaths(
        raw_dir=raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df):,} rows to {csv_path}")

    config = CounterfactualConfig(
        pandemic_onset=args.pandemic_onset,
        n_posterior_samples=args.samples,
        random_seed=args.seed,
        fourier_order=args.fourier_order,
        interval_mass=args.interval_mass,
    )
    result = fit_counterfactual(df, config=config)
    plot_counterfactual(result=result, output_path=output_path, config=config)
    print(f"Saved chart to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="ons-mortality",
        description="Download ONS monthly mortality data into MySQL and plot counterfactuals.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize the MySQL schema.")
    init_parser.add_argument("--schema", default="sql/schema.sql", help="Path to schema.sql.")
    init_parser.set_defaults(func=handle_init_db)

    discover_parser = subparsers.add_parser("discover", help="Discover ONS workbooks.")
    _add_year_arguments(discover_parser)
    discover_parser.set_defaults(func=handle_discover)

    ingest_parser = subparsers.add_parser("ingest", help="Download and ingest ONS workbooks.")
    _add_year_arguments(ingest_parser)
    ingest_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even when a local copy already exists.",
    )
    ingest_parser.set_defaults(func=handle_ingest)

    export_parser = subparsers.add_parser(
        "export-national",
        help="Export England and Wales national monthly deaths from MySQL to CSV.",
    )
    export_parser.add_argument(
        "--output",
        default="data/processed/england_wales_monthly_deaths.csv",
        help="Output CSV path.",
    )
    export_parser.set_defaults(func=handle_export_national)

    plot_parser = subparsers.add_parser("plot", help="Fit and save the counterfactual plot.")
    plot_parser.add_argument(
        "--input",
        default="data/processed/england_wales_monthly_deaths.csv",
        help="Input CSV path. If missing, data is read from MySQL.",
    )
    plot_parser.add_argument(
        "--output",
        default="figures/england_wales_counterfactual.png",
        help="Output PNG path.",
    )
    plot_parser.add_argument("--pandemic-onset", default="2020-03-01")
    plot_parser.add_argument("--samples", type=int, default=8_000)
    plot_parser.add_argument("--seed", type=int, default=42)
    plot_parser.add_argument(
        "--fourier-order",
        type=int,
        default=3,
        help="Number of annual Fourier harmonics in the seasonality term.",
    )
    plot_parser.add_argument(
        "--interval-mass",
        type=float,
        default=0.94,
        help="Central credible-interval mass (0 < x < 1).",
    )
    plot_parser.set_defaults(func=handle_plot)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help=(
            "Download ONS workbooks and write the England & Wales monthly "
            "series to CSV. No MySQL required."
        ),
    )
    _add_year_arguments(fetch_parser)
    fetch_parser.add_argument(
        "--output",
        default="data/processed/england_wales_monthly_deaths.csv",
        help="Output CSV path.",
    )
    fetch_parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override the raw download cache directory.",
    )
    fetch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even when cached.",
    )
    fetch_parser.set_defaults(func=handle_fetch)

    fetch_regional_parser = subparsers.add_parser(
        "fetch-regional",
        help=(
            "Download ONS workbooks and write per-region (9 English regions "
            "+ Wales) monthly deaths to CSV."
        ),
    )
    _add_year_arguments(fetch_regional_parser)
    fetch_regional_parser.add_argument(
        "--output",
        default="data/processed/england_wales_regional_monthly_deaths.csv",
        help="Output CSV path.",
    )
    fetch_regional_parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override the raw download cache directory.",
    )
    fetch_regional_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even when cached.",
    )
    fetch_regional_parser.set_defaults(func=handle_fetch_regional)

    fetch_weekly_parser = subparsers.add_parser(
        "fetch-weekly",
        help=(
            "Download ONS weekly workbooks and write the England & Wales "
            "weekly deaths series to CSV. No MySQL required."
        ),
    )
    fetch_weekly_parser.add_argument(
        "--start-year", type=int, default=2010,
        help="First weekly edition year (default: 2010).",
    )
    fetch_weekly_parser.add_argument(
        "--end-year", type=int, default=2024,
        help="Last weekly edition year (default: 2024).",
    )
    fetch_weekly_parser.add_argument(
        "--output",
        default="data/processed/england_wales_weekly_deaths.csv",
        help="Output CSV path.",
    )
    fetch_weekly_parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override the raw download cache directory (defaults to data/raw/weekly).",
    )
    fetch_weekly_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even when cached.",
    )
    fetch_weekly_parser.set_defaults(func=handle_fetch_weekly)

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Fetch ONS data, fit the counterfactual, and save the chart in "
            "one step (no MySQL)."
        ),
    )
    _add_year_arguments(run_parser)
    run_parser.add_argument(
        "--csv",
        default="data/processed/england_wales_monthly_deaths.csv",
        help="Where to write the intermediate CSV.",
    )
    run_parser.add_argument(
        "--output",
        default="figures/england_wales_counterfactual.png",
        help="Output PNG path.",
    )
    run_parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override the raw download cache directory.",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even when cached.",
    )
    run_parser.add_argument("--pandemic-onset", default="2020-03-01")
    run_parser.add_argument("--samples", type=int, default=8_000)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--fourier-order", type=int, default=3)
    run_parser.add_argument("--interval-mass", type=float, default=0.94)
    run_parser.set_defaults(func=handle_run)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
