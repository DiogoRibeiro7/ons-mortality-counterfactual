"""Parsing utilities for ONS monthly deaths Excel workbooks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

MONTH_ALIASES: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

AREA_NAME_CANDIDATES = {
    "area name",
    "areaname",
    "area",
    "name",
    "geography",
    "local authority",
    "local authority name",
    "county",
    "region",
}

AREA_CODE_CANDIDATES = {
    "area code",
    "areacode",
    "code",
    "geography code",
    "geographycode",
}


def normalise_text(value: object) -> str:
    """
    Convert a cell value into a normalized string.

    Parameters
    ----------
    value:
        Any cell value.

    Returns
    -------
    str
        Lowercase normalized text with compact whitespace.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def month_number_from_header(value: object) -> int | None:
    """
    Infer the month number from an Excel column header.

    Parameters
    ----------
    value:
        Header value.

    Returns
    -------
    int | None
        Month number in 1..12, or None if the header is not month-like.
    """
    text = normalise_text(value)

    if not text:
        return None

    # Keep the first alphabetic token. This handles headers such as "January(p)".
    token_match = re.search(r"[a-z]+", text)
    if token_match is None:
        return None

    token = token_match.group(0)
    return MONTH_ALIASES.get(token)


def read_workbook_sheets(path: Path) -> Iterable[tuple[str, pd.DataFrame]]:
    """
    Read all non-empty sheets from an ONS workbook.

    Parameters
    ----------
    path:
        Workbook path.

    Yields
    ------
    tuple[str, pd.DataFrame]
        Sheet name and raw sheet data.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        engine = "openpyxl"
    elif suffix == ".xls":
        engine = "xlrd"
    else:
        raise ValueError(f"Unsupported workbook extension: {suffix}")

    sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)

    for sheet_name, sheet_df in sheets.items():
        non_empty = sheet_df.dropna(how="all").dropna(axis=1, how="all")
        if not non_empty.empty:
            yield str(sheet_name), non_empty


def find_header_row(sheet_df: pd.DataFrame) -> int | None:
    """
    Find the row containing month-like headers.

    Parameters
    ----------
    sheet_df:
        Raw sheet data.

    Returns
    -------
    int | None
        Header row index, or None if no suitable row is found.
    """
    for idx, row in sheet_df.iterrows():
        month_count = sum(month_number_from_header(value) is not None for value in row.tolist())
        if month_count >= 3:
            return int(idx)

    return None


def _deduplicate_columns(columns: list[str]) -> list[str]:
    """Make DataFrame columns unique while preserving readable names."""
    seen: dict[str, int] = {}
    output: list[str] = []

    for column in columns:
        base = column.strip() if column.strip() else "unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1

        if count == 0:
            output.append(base)
        else:
            output.append(f"{base}_{count}")

    return output


def _find_candidate_column(columns: Iterable[str], candidates: set[str]) -> str | None:
    """Find the first column whose normalized name matches known candidates."""
    for column in columns:
        if normalise_text(column) in candidates:
            return column

    return None


def _choose_area_name_column(data: pd.DataFrame, month_columns: list[str]) -> str:
    """
    Choose the area-name column from a normalized raw sheet.

    If there is no explicit area-name header, use the first non-month column.
    """
    explicit = _find_candidate_column(data.columns, AREA_NAME_CANDIDATES)
    if explicit is not None:
        return explicit

    non_month_columns = [column for column in data.columns if column not in month_columns]
    if not non_month_columns:
        raise ValueError("Could not infer an area-name column.")

    return non_month_columns[0]


def normalise_ons_sheet(
    sheet_name: str,
    sheet_df: pd.DataFrame,
    source_file_id: int,
    edition_year: int,
    is_final: bool,
) -> pd.DataFrame:
    """
    Convert one ONS workbook sheet into long monthly death rows.

    Parameters
    ----------
    sheet_name:
        Name of the Excel sheet. Stored as `geography_type`.
    sheet_df:
        Raw sheet content with no assumed header.
    source_file_id:
        Database ID of the source workbook.
    edition_year:
        ONS edition year.
    is_final:
        Whether the ONS page marks the edition as final.

    Returns
    -------
    pd.DataFrame
        Normalized monthly death rows.
    """
    if edition_year < 1900:
        raise ValueError("edition_year is not valid.")

    df = sheet_df.dropna(how="all").dropna(axis=1, how="all").copy()
    if df.empty:
        return pd.DataFrame()

    header_index = find_header_row(df)
    if header_index is None:
        return pd.DataFrame()

    raw_header = ["" if pd.isna(value) else str(value).strip() for value in df.loc[header_index]]
    columns = _deduplicate_columns(raw_header)

    data = df.loc[header_index + 1 :].copy()
    data.columns = columns
    data = data.dropna(how="all")

    if data.empty:
        return pd.DataFrame()

    month_columns = [
        column
        for column in data.columns
        if month_number_from_header(column) is not None
    ]
    if len(month_columns) < 3:
        return pd.DataFrame()

    area_name_column = _choose_area_name_column(data, month_columns)
    area_code_column = _find_candidate_column(data.columns, AREA_CODE_CANDIDATES)

    id_columns = [area_name_column]
    if area_code_column is not None and area_code_column != area_name_column:
        id_columns.insert(0, area_code_column)

    long_df = data.melt(
        id_vars=id_columns,
        value_vars=month_columns,
        var_name="month_name",
        value_name="deaths",
    )

    long_df = long_df.rename(columns={area_name_column: "area_name"})

    if area_code_column is not None and area_code_column != area_name_column:
        long_df = long_df.rename(columns={area_code_column: "area_code"})
    else:
        long_df["area_code"] = None

    long_df["month_number"] = long_df["month_name"].map(month_number_from_header)
    long_df["deaths"] = pd.to_numeric(long_df["deaths"], errors="coerce")
    long_df["area_name"] = long_df["area_name"].astype(str).str.strip()
    long_df["area_code"] = long_df["area_code"].where(pd.notna(long_df["area_code"]), None)

    clean_df = long_df.dropna(subset=["month_number", "area_name", "deaths"]).copy()
    clean_df = clean_df[clean_df["area_name"].astype(str).str.strip().ne("")]

    if clean_df.empty:
        return pd.DataFrame()

    clean_df["month_date"] = pd.to_datetime(
        {
            "year": edition_year,
            "month": clean_df["month_number"].astype(int),
            "day": 1,
        },
        errors="raise",
    ).dt.date

    clean_df["source_file_id"] = int(source_file_id)
    clean_df["edition_year"] = int(edition_year)
    clean_df["geography_type"] = str(sheet_name)
    clean_df["is_final"] = bool(is_final)
    clean_df["deaths"] = clean_df["deaths"].round().astype(int)

    output = clean_df[
        [
            "source_file_id",
            "edition_year",
            "month_date",
            "area_code",
            "area_name",
            "geography_type",
            "deaths",
            "is_final",
        ]
    ].copy()

    # Remove obvious non-data rows that sometimes sit below the table.
    output = output[~output["area_name"].str.lower().str.startswith("source")]
    output = output[~output["area_name"].str.lower().str.startswith("notes")]

    return output.reset_index(drop=True)


def normalise_workbook(
    path: Path,
    source_file_id: int,
    edition_year: int,
    is_final: bool,
) -> pd.DataFrame:
    """
    Parse all usable sheets from one ONS workbook.

    Parameters
    ----------
    path:
        Local workbook path.
    source_file_id:
        Database ID of the source workbook.
    edition_year:
        ONS edition year.
    is_final:
        Whether the ONS page marks the edition as final.

    Returns
    -------
    pd.DataFrame
        Concatenated normalized monthly death rows.
    """
    frames: list[pd.DataFrame] = []

    for sheet_name, sheet_df in read_workbook_sheets(path):
        parsed = normalise_ons_sheet(
            sheet_name=sheet_name,
            sheet_df=sheet_df,
            source_file_id=source_file_id,
            edition_year=edition_year,
            is_final=is_final,
        )
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        return pd.DataFrame(
            columns=[
                "source_file_id",
                "edition_year",
                "month_date",
                "area_code",
                "area_name",
                "geography_type",
                "deaths",
                "is_final",
            ]
        )

    return pd.concat(frames, ignore_index=True)
