"""Database access utilities for ONS mortality ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from ons_mortality.config import DatabaseConfig, load_database_config


def create_mysql_engine(config: DatabaseConfig | None = None) -> Engine:
    """
    Create a SQLAlchemy MySQL engine.

    Parameters
    ----------
    config:
        Optional database configuration. If omitted, configuration is read from
        environment variables.

    Returns
    -------
    Engine
        SQLAlchemy engine with connection pre-ping enabled.
    """
    resolved_config = config or load_database_config()
    return create_engine(resolved_config.sqlalchemy_url, pool_pre_ping=True, future=True)


def init_database(engine: Engine, schema_path: Path) -> None:
    """
    Execute the SQL schema file against the configured MySQL database.

    Parameters
    ----------
    engine:
        SQLAlchemy engine.
    schema_path:
        Path to the schema SQL file.
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql_text = schema_path.read_text(encoding="utf-8")

    # SQLAlchemy does not support multi-statements by default with all drivers.
    # Splitting is enough here because the schema does not contain stored routines.
    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def insert_records_ignore(engine: Engine, table_name: str, records: list[dict[str, Any]]) -> None:
    """
    Insert records into a MySQL table using INSERT IGNORE.

    Parameters
    ----------
    engine:
        SQLAlchemy engine.
    table_name:
        Destination table name. Only known project tables are accepted.
    records:
        Row dictionaries to insert.
    """
    allowed_tables = {"monthly_deaths"}
    if table_name not in allowed_tables:
        raise ValueError(f"Unsupported destination table: {table_name}")

    if not records:
        return

    columns = list(records[0].keys())
    column_sql = ", ".join(f"`{column}`" for column in columns)
    value_sql = ", ".join(f":{column}" for column in columns)

    with engine.begin() as conn:
        conn.execute(
            text(f"INSERT IGNORE INTO `{table_name}` ({column_sql}) VALUES ({value_sql})"),
            records,
        )


def read_national_monthly_deaths(engine: Engine) -> pd.DataFrame:
    """
    Read England and Wales national monthly deaths from MySQL.

    The ONS national geography code is usually K04000001. The area-name fallback
    is kept because older workbooks may not expose the same code consistently.

    Parameters
    ----------
    engine:
        SQLAlchemy engine.

    Returns
    -------
    pd.DataFrame
        DataFrame with `month_date` and `observed_deaths`.
    """
    query = text(
        """
        SELECT
            month_date,
            SUM(deaths) AS observed_deaths
        FROM monthly_deaths
        WHERE area_code = 'K04000001'
           OR LOWER(area_name) IN ('england and wales', 'england & wales')
        GROUP BY month_date
        ORDER BY month_date
        """
    )

    df = pd.read_sql_query(query, con=engine)

    if df.empty:
        raise RuntimeError(
            "No England and Wales national rows were found. "
            "Check the parsed area names with: "
            "SELECT DISTINCT area_code, area_name FROM monthly_deaths ORDER BY area_name;"
        )

    df["month_date"] = pd.to_datetime(df["month_date"])
    df["observed_deaths"] = pd.to_numeric(df["observed_deaths"], errors="raise").astype(int)
    return df
