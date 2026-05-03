"""High-level ONS download + parse helpers that don't require MySQL.

The ingestion pipeline in :mod:`ons_mortality.cli` (``ingest`` subcommand)
loads everything into MySQL. This module is a lightweight alternative:
discover the workbooks, download them with on-disk caching, parse each one
in memory, and return a single tidy national series for England & Wales.

The ONS workbook layout has changed several times since 2006 (early
editions put each admin level in its own column; later ones use Area code
+ Area name headers). Rather than make the generic parser handle every
variant, we use a focused extractor here that scans every non-month cell
of every sheet for the England & Wales marker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ons_mortality.ons import (
    ONS_WEEKLY_DATASET_URL,
    ONSFile,
    discover_ons_files,
    download_file,
    iter_downloaded_files,
)
from ons_mortality.parser import (
    find_header_row,
    month_number_from_header,
    read_workbook_sheets,
)

ENGLAND_AND_WALES_CODE = "K04000001"
ENGLAND_AND_WALES_NAMES = {
    "england and wales",
    "england & wales",
    "england, wales",
}

# 9 English regions + Wales. Code is the canonical ONS GSS code; names are
# the spellings we accept from the workbooks (lowercase, stripped).
REGION_REGISTRY: dict[str, tuple[str, frozenset[str]]] = {
    "E12000001": ("North East", frozenset({"north east"})),
    "E12000002": ("North West", frozenset({"north west"})),
    "E12000003": (
        "Yorkshire and The Humber",
        frozenset({
            "yorkshire and the humber",
            "yorkshire and humber",
            "yorkshire & the humber",
            "yorkshire and the humber 2",
        }),
    ),
    "E12000004": ("East Midlands", frozenset({"east midlands"})),
    "E12000005": ("West Midlands", frozenset({"west midlands"})),
    "E12000006": (
        "East of England",
        frozenset({"east", "east of england", "eastern"}),
    ),
    "E12000007": ("London", frozenset({"london"})),
    "E12000008": ("South East", frozenset({"south east"})),
    "E12000009": ("South West", frozenset({"south west"})),
    "W92000004": ("Wales", frozenset({"wales"})),
}

REGION_NAME_TO_CODE: dict[str, str] = {
    name: code
    for code, (_, aliases) in REGION_REGISTRY.items()
    for name in aliases
}


def _looks_like_england_and_wales(value: object) -> bool:
    """Return True if a single cell value identifies the E&W national row."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text.upper() == ENGLAND_AND_WALES_CODE:
        return True
    return text.lower() in ENGLAND_AND_WALES_NAMES


def _extract_england_wales_from_sheet(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Pull (month, deaths) pairs for the E&W national total from one sheet.

    The function locates the month-name header row, then returns the first
    data row that contains "England and Wales" (or `K04000001`) in any
    non-month column.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index = find_header_row(df)
    if header_index is None:
        return []

    header_row = df.loc[header_index].tolist()
    month_columns: list[tuple[int, int]] = []
    for column_idx, header_value in enumerate(header_row):
        month = month_number_from_header(header_value)
        if month is not None:
            month_columns.append((column_idx, month))

    if len(month_columns) < 3:
        return []

    month_column_idx = {column_idx for column_idx, _ in month_columns}

    # Map positional index back to the actual column label so we can use .loc
    column_labels = list(df.columns)

    for _, row in df.loc[header_index + 1:].iterrows():
        is_ew = any(
            _looks_like_england_and_wales(row.iloc[column_idx])
            for column_idx in range(len(row))
            if column_idx not in month_column_idx
        )
        if not is_ew:
            continue

        out: list[tuple[pd.Timestamp, int]] = []
        for column_idx, month in month_columns:
            label = column_labels[column_idx]
            value = row[label]
            deaths = pd.to_numeric(value, errors="coerce")
            if pd.isna(deaths):
                continue
            month_date = pd.Timestamp(year=edition_year, month=month, day=1)
            out.append((month_date, int(round(float(deaths)))))
        return out

    return []


def _extract_from_long_format(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Pull (month, deaths) pairs from a 2023+ long-format ONS sheet.

    These workbooks store one row per (month, geography). The header row
    has columns like ``Month``, ``Code``, ``Geography``, ``Number of
    deaths``. England & Wales is represented by either ``K04000001``
    ("England, Wales and non-residents") or by summing the country-level
    rows for England (``E92000001``) and Wales (``W92000004``).
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        has_month = "month" in normalized
        has_deaths = any("number of deaths" in v for v in normalized)
        if has_month and has_deaths:
            header_index = int(idx)
            break

    if header_index is None:
        return []

    headers = [str(v).strip() for v in df.loc[header_index].tolist()]
    body = df.loc[header_index + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all")
    if body.empty:
        return []

    body.columns = [str(c).strip() for c in body.columns]
    name_lookup = {c.lower(): c for c in body.columns}

    month_col = name_lookup.get("month")
    code_col = name_lookup.get("code")
    deaths_col = next(
        (c for k, c in name_lookup.items() if "number of deaths" in k),
        None,
    )
    if month_col is None or deaths_col is None:
        return []

    if code_col is not None:
        codes = body[code_col].astype(str).str.strip().str.upper()
        ew_mask = codes == "K04000001"
        if not ew_mask.any():
            ew_mask = codes.isin({"E92000001", "W92000004"})
    else:
        geo_col = name_lookup.get("geography")
        if geo_col is None:
            return []
        geos = body[geo_col].astype(str).str.strip().str.lower()
        ew_mask = geos.isin({"england and wales", "england & wales"})
        if not ew_mask.any():
            ew_mask = geos.isin({"england", "wales"})

    selected = body.loc[ew_mask, [month_col, deaths_col]].copy()
    if selected.empty:
        return []

    selected[deaths_col] = pd.to_numeric(
        selected[deaths_col], errors="coerce"
    )
    selected = selected.dropna(subset=[deaths_col])
    selected["month_number"] = selected[month_col].map(month_number_from_header)
    selected = selected.dropna(subset=["month_number"])

    monthly = (
        selected.groupby("month_number", as_index=False)[deaths_col]
        .sum()
        .sort_values("month_number")
    )

    out: list[tuple[pd.Timestamp, int]] = []
    for _, row in monthly.iterrows():
        month_int = int(row["month_number"])
        deaths_val = int(round(float(row[deaths_col])))
        out.append((pd.Timestamp(year=edition_year, month=month_int, day=1), deaths_val))
    return out


def extract_england_wales_national(
    path: Path,
    edition_year: int,
) -> pd.DataFrame:
    """Return the E&W national monthly series for one workbook."""
    rows: list[tuple[pd.Timestamp, int]] = []
    for _, sheet_df in read_workbook_sheets(path):
        rows = _extract_england_wales_from_sheet(sheet_df, edition_year)
        if rows:
            break
        rows = _extract_from_long_format(sheet_df, edition_year)
        if rows:
            break

    if not rows:
        return pd.DataFrame(columns=["month_date", "observed_deaths"])

    return pd.DataFrame(rows, columns=["month_date", "observed_deaths"])


def fetch_national_monthly_deaths(
    raw_dir: Path,
    start_year: int = 2006,
    end_year: int | None = None,
    overwrite: bool = False,
    files: list[ONSFile] | None = None,
) -> pd.DataFrame:
    """
    Download every ONS annual workbook in range and return one tidy series.

    Workbooks are cached under ``raw_dir`` so repeated runs only re-read
    files from disk. The result is a deduplicated England & Wales monthly
    deaths frame, sorted by month, with provenance columns kept.

    Parameters
    ----------
    raw_dir:
        Directory used to cache downloaded workbooks.
    start_year, end_year:
        Inclusive bounds passed to :func:`discover_ons_files`.
    overwrite:
        Force re-download even when a cached copy exists.
    files:
        Optional pre-discovered list (useful for testing). When given,
        ``start_year``/``end_year`` are ignored.

    Returns
    -------
    pd.DataFrame
        Columns: ``month_date``, ``observed_deaths``, ``edition_year``,
        ``is_final``, ``source_filename``.
    """
    if files is None:
        files = discover_ons_files(start_year=start_year, end_year=end_year)
    if not files:
        raise RuntimeError(
            "No ONS workbooks were discovered for the requested year range."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ons_file, local_path in iter_downloaded_files(
        files,
        output_dir=raw_dir,
        overwrite=overwrite,
    ):
        national = extract_england_wales_national(
            path=local_path,
            edition_year=ons_file.edition_year,
        )
        if national.empty:
            continue

        national = national.assign(
            edition_year=ons_file.edition_year,
            is_final=ons_file.is_final,
            source_filename=local_path.name,
        )
        frames.append(national)

    if not frames:
        raise RuntimeError(
            "Workbooks were downloaded but no England & Wales rows were "
            "extracted. The ONS sheet layout may have changed."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["month_date"] = pd.to_datetime(combined["month_date"])
    combined["observed_deaths"] = (
        pd.to_numeric(combined["observed_deaths"], errors="raise").astype(int)
    )

    # When the same month appears in more than one workbook (e.g. a final
    # edition supersedes a provisional one), keep the most authoritative row:
    # final beats provisional, then the latest edition wins.
    combined["_is_final_int"] = combined["is_final"].astype(int)
    combined = combined.sort_values(
        ["month_date", "_is_final_int", "edition_year"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["month_date"], keep="first")

    output = combined[
        [
            "month_date",
            "observed_deaths",
            "edition_year",
            "is_final",
            "source_filename",
        ]
    ].sort_values("month_date").reset_index(drop=True)

    return output


def _match_region(value: object) -> str | None:
    """Return the canonical region GSS code for a cell value, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    upper = text.upper()
    if upper in REGION_REGISTRY:
        return upper
    return REGION_NAME_TO_CODE.get(text.lower())


def _extract_regions_wide_format(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, int]]:
    """
    Pull (month, region_code, deaths) tuples from a wide-format sheet.

    For each data row we scan every non-month cell looking for any of the
    canonical region names or codes. The first match wins, so we don't
    double-count regions that mention several admin labels.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index = find_header_row(df)
    if header_index is None:
        return []

    header_row = df.loc[header_index].tolist()
    month_columns: list[tuple[int, int]] = []
    for column_idx, header_value in enumerate(header_row):
        month = month_number_from_header(header_value)
        if month is not None:
            month_columns.append((column_idx, month))

    if len(month_columns) < 3:
        return []

    month_column_idx = {column_idx for column_idx, _ in month_columns}
    column_labels = list(df.columns)

    seen_codes: set[str] = set()
    out: list[tuple[pd.Timestamp, str, int]] = []

    for _, row in df.loc[header_index + 1:].iterrows():
        matched_code: str | None = None
        for column_idx in range(len(row)):
            if column_idx in month_column_idx:
                continue
            code = _match_region(row.iloc[column_idx])
            if code is not None:
                matched_code = code
                break

        if matched_code is None or matched_code in seen_codes:
            continue
        seen_codes.add(matched_code)

        for column_idx, month in month_columns:
            label = column_labels[column_idx]
            value = row[label]
            deaths = pd.to_numeric(value, errors="coerce")
            if pd.isna(deaths):
                continue
            month_date = pd.Timestamp(year=edition_year, month=month, day=1)
            out.append((month_date, matched_code, int(round(float(deaths)))))

    return out


def _extract_regions_long_format(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, int]]:
    """Pull (month, region_code, deaths) tuples from a 2023+ long-format sheet."""
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        has_month = "month" in normalized
        has_deaths = any("number of deaths" in v for v in normalized)
        if has_month and has_deaths:
            header_index = int(idx)
            break

    if header_index is None:
        return []

    headers = [str(v).strip() for v in df.loc[header_index].tolist()]
    body = df.loc[header_index + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all")
    if body.empty:
        return []

    body.columns = [str(c).strip() for c in body.columns]
    name_lookup = {c.lower(): c for c in body.columns}

    month_col = name_lookup.get("month")
    code_col = name_lookup.get("code")
    geo_col = name_lookup.get("geography")
    deaths_col = next(
        (c for k, c in name_lookup.items() if "number of deaths" in k),
        None,
    )
    if month_col is None or deaths_col is None:
        return []

    if code_col is not None:
        body["__region_code__"] = (
            body[code_col].astype(str).str.strip().str.upper()
            .map(lambda c: c if c in REGION_REGISTRY else None)
        )
    elif geo_col is not None:
        body["__region_code__"] = (
            body[geo_col].astype(str).str.strip().str.lower()
            .map(REGION_NAME_TO_CODE)
        )
    else:
        return []

    region_rows = body.dropna(subset=["__region_code__"]).copy()
    if region_rows.empty:
        return []

    region_rows[deaths_col] = pd.to_numeric(
        region_rows[deaths_col], errors="coerce"
    )
    region_rows = region_rows.dropna(subset=[deaths_col])
    region_rows["month_number"] = region_rows[month_col].map(
        month_number_from_header
    )
    region_rows = region_rows.dropna(subset=["month_number"])

    grouped = (
        region_rows.groupby(["__region_code__", "month_number"], as_index=False)[
            deaths_col
        ]
        .sum()
        .sort_values(["__region_code__", "month_number"])
    )

    out: list[tuple[pd.Timestamp, str, int]] = []
    for _, row in grouped.iterrows():
        month_int = int(row["month_number"])
        deaths_val = int(round(float(row[deaths_col])))
        out.append(
            (
                pd.Timestamp(year=edition_year, month=month_int, day=1),
                str(row["__region_code__"]),
                deaths_val,
            )
        )
    return out


def extract_regional_monthly_deaths(
    path: Path,
    edition_year: int,
) -> pd.DataFrame:
    """Return the per-region monthly series for one workbook."""
    rows: list[tuple[pd.Timestamp, str, int]] = []
    for _, sheet_df in read_workbook_sheets(path):
        rows = _extract_regions_wide_format(sheet_df, edition_year)
        if rows:
            break
        rows = _extract_regions_long_format(sheet_df, edition_year)
        if rows:
            break

    if not rows:
        return pd.DataFrame(columns=["month_date", "region_code", "observed_deaths"])

    df = pd.DataFrame(rows, columns=["month_date", "region_code", "observed_deaths"])
    df["region_name"] = df["region_code"].map(
        lambda code: REGION_REGISTRY[code][0]
    )
    return df[["month_date", "region_code", "region_name", "observed_deaths"]]


def fetch_regional_monthly_deaths(
    raw_dir: Path,
    start_year: int = 2006,
    end_year: int | None = None,
    overwrite: bool = False,
    files: list[ONSFile] | None = None,
) -> pd.DataFrame:
    """
    Download every ONS annual workbook in range and return per-region rows.

    Result is a long DataFrame with one row per (month, region). Provenance
    columns are kept; deduplication prefers final over provisional editions
    and the most recent edition year on ties.
    """
    if files is None:
        files = discover_ons_files(start_year=start_year, end_year=end_year)
    if not files:
        raise RuntimeError(
            "No ONS workbooks were discovered for the requested year range."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ons_file, local_path in iter_downloaded_files(
        files,
        output_dir=raw_dir,
        overwrite=overwrite,
    ):
        regional = extract_regional_monthly_deaths(
            path=local_path,
            edition_year=ons_file.edition_year,
        )
        if regional.empty:
            continue

        regional = regional.assign(
            edition_year=ons_file.edition_year,
            is_final=ons_file.is_final,
            source_filename=local_path.name,
        )
        frames.append(regional)

    if not frames:
        raise RuntimeError(
            "Workbooks were downloaded but no regional rows were extracted. "
            "The ONS sheet layout may have changed."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["month_date"] = pd.to_datetime(combined["month_date"])
    combined["observed_deaths"] = (
        pd.to_numeric(combined["observed_deaths"], errors="raise").astype(int)
    )

    combined["_is_final_int"] = combined["is_final"].astype(int)
    combined = combined.sort_values(
        ["region_code", "month_date", "_is_final_int", "edition_year"],
        ascending=[True, True, False, False],
    ).drop_duplicates(subset=["region_code", "month_date"], keep="first")

    output = combined[
        [
            "month_date",
            "region_code",
            "region_name",
            "observed_deaths",
            "edition_year",
            "is_final",
            "source_filename",
        ]
    ].sort_values(["region_code", "month_date"]).reset_index(drop=True)

    return output


# ---------------------------------------------------------------------------
# Weekly mortality (different ONS dataset, different layout)
# ---------------------------------------------------------------------------

WEEKLY_TOTAL_LABEL_PATTERNS = (
    "total deaths, all ages",
    "total deaths all ages",
)


def _extract_weekly_wide_format(
    sheet_df: pd.DataFrame,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Pull (week_ending, deaths) tuples from the 2010-2022 wide layout.

    Layout: weeks across columns. One specific row labelled "Total deaths,
    all ages" carries the E&W weekly count; a "Week ended" row above it
    carries the dates.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    week_end_row_idx: int | None = None
    total_row_idx: int | None = None

    for idx, row in df.iterrows():
        first = row.iloc[0]
        if not isinstance(first, str):
            continue
        text = first.strip().lower()
        if text == "week ended" and week_end_row_idx is None:
            week_end_row_idx = int(idx)
        if any(p in text for p in WEEKLY_TOTAL_LABEL_PATTERNS):
            total_row_idx = int(idx)
            break

    if week_end_row_idx is None or total_row_idx is None:
        return []

    week_end_row = df.loc[week_end_row_idx]
    total_row = df.loc[total_row_idx]

    out: list[tuple[pd.Timestamp, int]] = []
    for column_idx in range(len(week_end_row)):
        date_value = week_end_row.iloc[column_idx]
        deaths_value = total_row.iloc[column_idx]
        if pd.isna(date_value) or pd.isna(deaths_value):
            continue
        try:
            week_end = pd.Timestamp(date_value)
        except (ValueError, TypeError):
            continue
        deaths = pd.to_numeric(deaths_value, errors="coerce")
        if pd.isna(deaths):
            continue
        out.append((week_end, int(round(float(deaths)))))

    return out


def _extract_weekly_long_format(
    sheet_df: pd.DataFrame,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Pull (week_ending, deaths) tuples from the 2023+ long-format Table_1.

    Each row is one (week, area, sex, age, IMD, place) cell. We filter to
    Area = "England, Wales and non-residents" / "England and Wales" and the
    "All people / All ages / All groups / All places" totals.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        if (
            "week number" in normalized
            and any("week ending" in v or "week ended" in v for v in normalized)
            and any("number of deaths" in v for v in normalized)
        ):
            header_index = int(idx)
            break

    if header_index is None:
        return []

    headers = [str(v).strip() for v in df.loc[header_index].tolist()]
    body = df.loc[header_index + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all")
    if body.empty:
        return []

    body.columns = [str(c).strip() for c in body.columns]
    name_lookup = {c.lower(): c for c in body.columns}

    week_end_col = next(
        (c for k, c in name_lookup.items() if "week ending" in k or "week ended" in k),
        None,
    )
    deaths_col = next(
        (c for k, c in name_lookup.items() if "number of deaths" in k),
        None,
    )
    if week_end_col is None or deaths_col is None:
        return []

    area_col = name_lookup.get("area of usual residence") or name_lookup.get("area")
    sex_col = name_lookup.get("sex")
    age_col = next(
        (c for k, c in name_lookup.items() if k.startswith("age group")),
        None,
    )
    imd_col = next(
        (c for k, c in name_lookup.items() if "imd" in k),
        None,
    )
    place_col = next(
        (c for k, c in name_lookup.items() if "place of occurrence" in k),
        None,
    )

    if area_col is None:
        return []

    ew_aliases = {
        "england, wales and non-residents",
        "england and wales",
        "england & wales",
    }
    mask = body[area_col].astype(str).str.strip().str.lower().isin(ew_aliases)

    if sex_col is not None:
        mask &= body[sex_col].astype(str).str.strip().str.lower() == "all people"
    if age_col is not None:
        mask &= body[age_col].astype(str).str.strip().str.lower() == "all ages"
    if imd_col is not None:
        mask &= body[imd_col].astype(str).str.strip().str.lower() == "all groups"
    if place_col is not None:
        mask &= body[place_col].astype(str).str.strip().str.lower() == "all places"

    selected = body.loc[mask, [week_end_col, deaths_col]].copy()
    if selected.empty:
        return []

    selected[deaths_col] = pd.to_numeric(selected[deaths_col], errors="coerce")
    selected = selected.dropna(subset=[deaths_col])

    out: list[tuple[pd.Timestamp, int]] = []
    for _, row in selected.iterrows():
        try:
            week_end = pd.Timestamp(row[week_end_col])
        except (ValueError, TypeError):
            continue
        out.append((week_end, int(round(float(row[deaths_col])))))
    return out


def _extract_weekly_hybrid_format(
    sheet_df: pd.DataFrame,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Pull (week_ending, deaths) tuples from the 2022/2023 hybrid layout.

    The header has named columns ``Week ended`` / ``Week ending`` and
    ``Total deaths England and Wales``; the same E&W label appears twice
    (current and prior year side-by-side). We take the first occurrence,
    which is the current edition year.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    week_end_col_idx: int | None = None
    deaths_col_idx: int | None = None

    for idx, row in df.iterrows():
        values = [str(v).strip() for v in row.tolist()]
        we_match: int | None = None
        ew_match: int | None = None
        for ci, value in enumerate(values):
            lowered = value.lower()
            if we_match is None and ("week ended" in lowered or "week ending" in lowered):
                we_match = ci
            if (
                ew_match is None
                and "total deaths" in lowered
                and "england and wales" in lowered
                and "five-year" not in lowered
            ):
                ew_match = ci
        if we_match is not None and ew_match is not None:
            header_index = int(idx)
            week_end_col_idx = we_match
            deaths_col_idx = ew_match
            break

    if header_index is None or week_end_col_idx is None or deaths_col_idx is None:
        return []

    out: list[tuple[pd.Timestamp, int]] = []
    for _, row in df.loc[header_index + 1:].iterrows():
        date_value = row.iloc[week_end_col_idx]
        deaths_value = row.iloc[deaths_col_idx]
        if pd.isna(date_value) or pd.isna(deaths_value):
            continue
        try:
            week_end = pd.Timestamp(date_value)
        except (ValueError, TypeError):
            continue
        deaths = pd.to_numeric(deaths_value, errors="coerce")
        if pd.isna(deaths):
            continue
        out.append((week_end, int(round(float(deaths)))))

    return out


def extract_weekly_national_deaths(path: Path) -> pd.DataFrame:
    """Return the E&W weekly series for one workbook (any supported layout)."""
    rows: list[tuple[pd.Timestamp, int]] = []
    for _, sheet_df in read_workbook_sheets(path):
        rows = _extract_weekly_wide_format(sheet_df)
        if rows:
            break
        rows = _extract_weekly_hybrid_format(sheet_df)
        if rows:
            break
        rows = _extract_weekly_long_format(sheet_df)
        if rows:
            break

    if not rows:
        return pd.DataFrame(columns=["week_ending", "observed_deaths"])

    return (
        pd.DataFrame(rows, columns=["week_ending", "observed_deaths"])
        .drop_duplicates(subset=["week_ending"])
        .sort_values("week_ending")
        .reset_index(drop=True)
    )


def fetch_weekly_national_deaths(
    raw_dir: Path,
    start_year: int = 2010,
    end_year: int | None = None,
    overwrite: bool = False,
    files: list[ONSFile] | None = None,
) -> pd.DataFrame:
    """
    Download every ONS weekly workbook in range and return one tidy series.

    Workbooks are cached under ``raw_dir`` so repeated runs only re-read
    files from disk. The result is a deduplicated England & Wales weekly
    deaths frame, sorted by week ending date.
    """
    if files is None:
        files = discover_ons_files(
            start_year=start_year,
            end_year=end_year,
            dataset_url=ONS_WEEKLY_DATASET_URL,
        )
    if not files:
        raise RuntimeError(
            "No ONS weekly workbooks were discovered for the requested range."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ons_file, local_path in iter_downloaded_files(
        files,
        output_dir=raw_dir,
        overwrite=overwrite,
    ):
        weekly = extract_weekly_national_deaths(path=local_path)
        if weekly.empty:
            continue

        weekly = weekly.assign(
            edition_year=ons_file.edition_year,
            is_final=ons_file.is_final,
            source_filename=local_path.name,
        )
        frames.append(weekly)

    if not frames:
        raise RuntimeError(
            "Workbooks were downloaded but no E&W weekly rows were extracted."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["week_ending"] = pd.to_datetime(combined["week_ending"])
    combined["observed_deaths"] = (
        pd.to_numeric(combined["observed_deaths"], errors="raise").astype(int)
    )

    # When the same week appears in multiple workbooks (year boundaries) keep
    # the more authoritative figure: final beats provisional, then latest year.
    combined["_is_final_int"] = combined["is_final"].astype(int)
    combined = combined.sort_values(
        ["week_ending", "_is_final_int", "edition_year"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["week_ending"], keep="first")

    output = combined[
        [
            "week_ending",
            "observed_deaths",
            "edition_year",
            "is_final",
            "source_filename",
        ]
    ].sort_values("week_ending").reset_index(drop=True)

    return output


# ---------------------------------------------------------------------------
# Weekly mortality by age band
# ---------------------------------------------------------------------------

# Seven canonical age bands. Chosen to match the coarse 2010-2019 ONS layout
# so older editions don't lose any signal under aggregation; fine 5-year
# bands from 2020+ collapse into these seven.
AGE_BAND_ORDER = (
    "Under 1",
    "1-14",
    "15-44",
    "45-64",
    "65-74",
    "75-84",
    "85+",
)

# Maps every age-label spelling we've seen across the 2010-2024 weekly
# editions to the canonical band. Lower-cased; whitespace-stripped at lookup.
_AGE_BAND_ALIASES: dict[str, str] = {
    # Under 1
    "<1": "Under 1",
    "under 1": "Under 1",
    "under 1 year": "Under 1",
    # 1-14
    "01-14": "1-14",
    "1-14": "1-14",
    "01-04": "1-14",
    "1-4": "1-14",
    "1 to 4": "1-14",
    "05-09": "1-14",
    "5-9": "1-14",
    "5 to 9": "1-14",
    "10-14": "1-14",
    "10 to 14": "1-14",
    # 15-44
    "15-44": "15-44",
    "15-19": "15-44",
    "15 to 19": "15-44",
    "20-24": "15-44",
    "20 to 24": "15-44",
    "25-29": "15-44",
    "25 to 29": "15-44",
    "30-34": "15-44",
    "30 to 34": "15-44",
    "35-39": "15-44",
    "35 to 39": "15-44",
    "40-44": "15-44",
    "40 to 44": "15-44",
    # 45-64
    "45-64": "45-64",
    "45-49": "45-64",
    "45 to 49": "45-64",
    "50-54": "45-64",
    "50 to 54": "45-64",
    "55-59": "45-64",
    "55 to 59": "45-64",
    "60-64": "45-64",
    "60 to 64": "45-64",
    # 65-74
    "65-74": "65-74",
    "65-69": "65-74",
    "65 to 69": "65-74",
    "70-74": "65-74",
    "70 to 74": "65-74",
    # 75-84
    "75-84": "75-84",
    "75-79": "75-84",
    "75 to 79": "75-84",
    "80-84": "75-84",
    "80 to 84": "75-84",
    # 85+
    "85+": "85+",
    "85-89": "85+",
    "85 to 89": "85+",
    "90+": "85+",
    "90 and over": "85+",
}


def _canonical_age_band(label: object) -> str | None:
    """Map a raw age-band label to one of the seven canonical bands."""
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    return _AGE_BAND_ALIASES.get(key)


def _is_persons_header(value: object) -> bool:
    """Return True for the 'Persons'/'People' totals header (any footnote suffix)."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return False
    head = text.split()[0]
    return head in {"persons", "people"}


def _is_sex_split_header(value: object) -> bool:
    """Return True for the 'Males'/'Females' headers that end the persons block."""
    if not isinstance(value, str):
        return False
    head = value.strip().lower().split()
    return bool(head) and head[0] in {"males", "females"}


def _extract_weekly_ages_block_format(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, int]]:
    """
    Pull (week_ending, age_band, deaths) tuples from the 2010-2021 block layout.

    Layout: a "Week ended" row in column A carries the week-ending dates
    across columns; a "Persons"/"People" header introduces the all-sex age
    block, followed by a "Deaths by age group" row and one row per band.
    The persons block lived in column A through 2018 and moved to column B
    in 2019; bands were coarse (7 labels) until 2019 and fine 5-year (19
    labels) from 2020. We detect the column from the persons header and
    sum any number of fine bands into the seven canonical bands; iteration
    stops at the first "Males"/"Females" header (start of the next block).
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty:
        return []

    week_end_row_idx: int | None = None
    persons_row_idx: int | None = None
    label_col: int = 0

    for idx in range(len(df)):
        first = df.iloc[idx, 0]
        if (
            week_end_row_idx is None
            and isinstance(first, str)
            and first.strip().lower() == "week ended"
        ):
            week_end_row_idx = idx
        if persons_row_idx is None:
            if _is_persons_header(first):
                persons_row_idx = idx
                label_col = 0
            elif df.shape[1] > 1 and _is_persons_header(df.iloc[idx, 1]):
                persons_row_idx = idx
                label_col = 1

    if week_end_row_idx is None or persons_row_idx is None:
        return []

    week_end_row = df.iloc[week_end_row_idx]
    week_ends: list[tuple[int, pd.Timestamp]] = []
    for column_idx in range(len(week_end_row)):
        date_value = week_end_row.iloc[column_idx]
        if pd.isna(date_value):
            continue
        try:
            week_end = pd.Timestamp(date_value)
        except (ValueError, TypeError):
            continue
        if pd.isna(week_end):
            continue
        week_ends.append((column_idx, week_end))

    if not week_ends:
        return []

    band_totals: dict[tuple[pd.Timestamp, str], int] = {}

    for idx in range(persons_row_idx + 1, len(df)):
        row = df.iloc[idx]
        label = row.iloc[label_col]
        if _is_sex_split_header(label):
            break
        band = _canonical_age_band(label)
        if band is None:
            continue
        for column_idx, week_end in week_ends:
            value = row.iloc[column_idx]
            deaths = pd.to_numeric(value, errors="coerce")
            if pd.isna(deaths):
                continue
            key = (week_end, band)
            band_totals[key] = band_totals.get(key, 0) + int(round(float(deaths)))

    return [(week_end, band, total) for (week_end, band), total in band_totals.items()]


def _extract_weekly_ages_fine_cols(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, int]]:
    """
    Pull (week_ending, age_band, deaths) tuples from the 2022 hybrid layout.

    The persons block on sheet "2" has a clean header row whose columns are
    age bands: ``Week number``, ``Week ending``, ``All ages``, ``<1``,
    ``01-04``, ..., ``90+``. One row per week. The same sheet stacks
    Persons / Males / Females tables vertically, so we stop at the second
    occurrence of any week-ending date (start of the Males block) and sum
    the fine bands into the seven canonical bands.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        if (
            "week ending" in normalized
            and any(_canonical_age_band(v) for v in row.tolist())
        ):
            header_index = int(idx)
            break

    if header_index is None:
        return []

    headers = [str(v).strip() for v in df.loc[header_index].tolist()]
    week_end_col_idx: int | None = None
    age_columns: list[tuple[int, str]] = []
    for column_idx, header in enumerate(headers):
        if header.lower() == "week ending":
            week_end_col_idx = column_idx
            continue
        band = _canonical_age_band(header)
        if band is not None:
            age_columns.append((column_idx, band))

    if week_end_col_idx is None or not age_columns:
        return []

    band_totals: dict[tuple[pd.Timestamp, str], int] = {}
    seen_weeks: set[pd.Timestamp] = set()

    for _, row in df.loc[header_index + 1:].iterrows():
        date_value = row.iloc[week_end_col_idx]
        if pd.isna(date_value):
            continue
        try:
            week_end = pd.Timestamp(date_value)
        except (ValueError, TypeError):
            continue
        if pd.isna(week_end):
            continue
        # The next table (Males) repeats the same dates; stop before it.
        if week_end in seen_weeks:
            break
        seen_weeks.add(week_end)
        for column_idx, band in age_columns:
            value = row.iloc[column_idx]
            deaths = pd.to_numeric(value, errors="coerce")
            if pd.isna(deaths):
                continue
            key = (week_end, band)
            band_totals[key] = band_totals.get(key, 0) + int(round(float(deaths)))

    return [(week_end, band, total) for (week_end, band), total in band_totals.items()]


def _extract_weekly_ages_long(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, int]]:
    """
    Pull (week_ending, age_band, deaths) tuples from the 2023+ long format.

    Each row is one (week, area, sex, age, IMD, place) cell. We filter to
    the all-residents E&W area, all sexes, all IMD groups, all places, and
    keep only age bands (excluding "All ages"). Fine 5-year bands collapse
    into the seven canonical bands.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        if (
            "week number" in normalized
            and any("week ending" in v or "week ended" in v for v in normalized)
            and any("number of deaths" in v for v in normalized)
            and any(v.startswith("age group") for v in normalized)
        ):
            header_index = int(idx)
            break

    if header_index is None:
        return []

    headers = [str(v).strip() for v in df.loc[header_index].tolist()]
    body = df.loc[header_index + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all")
    if body.empty:
        return []

    body.columns = [str(c).strip() for c in body.columns]
    name_lookup = {c.lower(): c for c in body.columns}

    week_end_col = next(
        (c for k, c in name_lookup.items() if "week ending" in k or "week ended" in k),
        None,
    )
    deaths_col = next(
        (c for k, c in name_lookup.items() if "number of deaths" in k),
        None,
    )
    age_col = next(
        (c for k, c in name_lookup.items() if k.startswith("age group")),
        None,
    )
    area_col = name_lookup.get("area of usual residence") or name_lookup.get("area")
    sex_col = name_lookup.get("sex")
    imd_col = next((c for k, c in name_lookup.items() if "imd" in k), None)
    place_col = next(
        (c for k, c in name_lookup.items() if "place of occurrence" in k),
        None,
    )

    if week_end_col is None or deaths_col is None or age_col is None or area_col is None:
        return []

    ew_aliases = {
        "england, wales and non-residents",
        "england and wales",
        "england & wales",
    }
    mask = body[area_col].astype(str).str.strip().str.lower().isin(ew_aliases)
    if sex_col is not None:
        mask &= body[sex_col].astype(str).str.strip().str.lower() == "all people"
    if imd_col is not None:
        mask &= body[imd_col].astype(str).str.strip().str.lower() == "all groups"
    if place_col is not None:
        mask &= body[place_col].astype(str).str.strip().str.lower() == "all places"
    # Drop the "All ages" total — we want the per-band breakdown.
    mask &= body[age_col].astype(str).str.strip().str.lower() != "all ages"

    selected = body.loc[mask, [week_end_col, age_col, deaths_col]].copy()
    if selected.empty:
        return []

    selected[deaths_col] = pd.to_numeric(selected[deaths_col], errors="coerce")
    selected = selected.dropna(subset=[deaths_col])
    selected["__band__"] = selected[age_col].map(_canonical_age_band)
    selected = selected.dropna(subset=["__band__"])

    band_totals: dict[tuple[pd.Timestamp, str], int] = {}
    for _, row in selected.iterrows():
        try:
            week_end = pd.Timestamp(row[week_end_col])
        except (ValueError, TypeError):
            continue
        if pd.isna(week_end):
            continue
        band = str(row["__band__"])
        key = (week_end, band)
        band_totals[key] = band_totals.get(key, 0) + int(round(float(row[deaths_col])))

    return [(week_end, band, total) for (week_end, band), total in band_totals.items()]


def extract_weekly_age_deaths(path: Path, edition_year: int) -> pd.DataFrame:
    """Return the E&W weekly per-age-band series for one workbook."""
    rows: list[tuple[pd.Timestamp, str, int]] = []
    for _, sheet_df in read_workbook_sheets(path):
        rows = _extract_weekly_ages_block_format(sheet_df, edition_year)
        if rows:
            break
        rows = _extract_weekly_ages_fine_cols(sheet_df, edition_year)
        if rows:
            break
        rows = _extract_weekly_ages_long(sheet_df, edition_year)
        if rows:
            break

    if not rows:
        return pd.DataFrame(
            columns=["week_ending", "age_band", "observed_deaths"]
        )

    df = pd.DataFrame(rows, columns=["week_ending", "age_band", "observed_deaths"])
    return (
        df.groupby(["week_ending", "age_band"], as_index=False)["observed_deaths"]
        .sum()
        .sort_values(["week_ending", "age_band"])
        .reset_index(drop=True)
    )


def fetch_weekly_age_deaths(
    raw_dir: Path,
    start_year: int = 2010,
    end_year: int | None = None,
    overwrite: bool = False,
    files: list[ONSFile] | None = None,
) -> pd.DataFrame:
    """
    Download every ONS weekly workbook in range and return per-age-band rows.

    Result is a long DataFrame with one row per (week, age_band) using the
    seven canonical bands. Provenance columns are kept; deduplication
    prefers final over provisional editions and the most recent edition
    year on ties.
    """
    if files is None:
        files = discover_ons_files(
            start_year=start_year,
            end_year=end_year,
            dataset_url=ONS_WEEKLY_DATASET_URL,
        )
    if not files:
        raise RuntimeError(
            "No ONS weekly workbooks were discovered for the requested range."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ons_file, local_path in iter_downloaded_files(
        files,
        output_dir=raw_dir,
        overwrite=overwrite,
    ):
        ages = extract_weekly_age_deaths(
            path=local_path,
            edition_year=ons_file.edition_year,
        )
        if ages.empty:
            continue

        ages = ages.assign(
            edition_year=ons_file.edition_year,
            is_final=ons_file.is_final,
            source_filename=local_path.name,
        )
        frames.append(ages)

    if not frames:
        raise RuntimeError(
            "Workbooks were downloaded but no per-age-band rows were extracted. "
            "The ONS weekly sheet layout may have changed."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["week_ending"] = pd.to_datetime(combined["week_ending"])
    combined["observed_deaths"] = (
        pd.to_numeric(combined["observed_deaths"], errors="raise").astype(int)
    )

    combined["_is_final_int"] = combined["is_final"].astype(int)
    combined = combined.sort_values(
        ["age_band", "week_ending", "_is_final_int", "edition_year"],
        ascending=[True, True, False, False],
    ).drop_duplicates(subset=["age_band", "week_ending"], keep="first")

    output = combined[
        [
            "week_ending",
            "age_band",
            "observed_deaths",
            "edition_year",
            "is_final",
            "source_filename",
        ]
    ].sort_values(["age_band", "week_ending"]).reset_index(drop=True)

    return output


# ---------------------------------------------------------------------------
# Weekly mortality by sex × age band
# ---------------------------------------------------------------------------

# Canonical sex labels — match the 2024+ ONS long-format spelling.
SEX_ORDER = ("All people", "Male", "Female")


def _classify_sex_header(value: object) -> str | None:
    """Map a sex-block header label to its canonical sex.

    The 2010-2021 block headers say "Persons 4" / "Males 6" / "Females 5"
    (with a footnote suffix); the 2022-2023 sheet "2" titles say "Table 2a:
    ... people" / "Table 2b: ... males" / "Table 2c: ... females". We pick
    the first matching keyword regardless of position in the string.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    # "females" before "males" because "males" is a substring of "females".
    if "females" in text or "female" in text:
        return "Female"
    if "males" in text or "male" in text:
        return "Male"
    if "persons" in text or "people" in text:
        return "All people"
    return None


def _extract_weekly_sex_ages_block_format(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, str, int]]:
    """
    Pull (week_ending, sex, age_band, deaths) tuples from the 2010-2021 layout.

    Same shape as :func:`_extract_weekly_ages_block_format`, but extracts
    all three sex blocks (Persons → "All people", Males → "Male", Females
    → "Female") rather than only the persons total. Each block runs from
    its sex header until the next sex header (or end of sheet).
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty:
        return []

    week_end_row_idx: int | None = None
    sex_blocks: list[tuple[int, str, int]] = []

    for idx in range(len(df)):
        first = df.iloc[idx, 0]
        if (
            week_end_row_idx is None
            and isinstance(first, str)
            and first.strip().lower() == "week ended"
        ):
            week_end_row_idx = idx
        sex = _classify_sex_header(first)
        if sex is not None:
            sex_blocks.append((idx, sex, 0))
            continue
        if df.shape[1] > 1:
            sex = _classify_sex_header(df.iloc[idx, 1])
            if sex is not None:
                sex_blocks.append((idx, sex, 1))

    if week_end_row_idx is None or not sex_blocks:
        return []

    week_end_row = df.iloc[week_end_row_idx]
    week_ends: list[tuple[int, pd.Timestamp]] = []
    for column_idx in range(len(week_end_row)):
        date_value = week_end_row.iloc[column_idx]
        if pd.isna(date_value):
            continue
        try:
            week_end = pd.Timestamp(date_value)
        except (ValueError, TypeError):
            continue
        if pd.isna(week_end):
            continue
        week_ends.append((column_idx, week_end))

    if not week_ends:
        return []

    band_totals: dict[tuple[pd.Timestamp, str, str], int] = {}

    sex_blocks.sort(key=lambda item: item[0])
    for block_idx, (start_idx, sex, label_col) in enumerate(sex_blocks):
        end_idx = (
            sex_blocks[block_idx + 1][0]
            if block_idx + 1 < len(sex_blocks)
            else len(df)
        )
        for row_idx in range(start_idx + 1, end_idx):
            row = df.iloc[row_idx]
            band = _canonical_age_band(row.iloc[label_col])
            if band is None:
                continue
            for column_idx, week_end in week_ends:
                value = row.iloc[column_idx]
                deaths = pd.to_numeric(value, errors="coerce")
                if pd.isna(deaths):
                    continue
                key = (week_end, sex, band)
                band_totals[key] = (
                    band_totals.get(key, 0) + int(round(float(deaths)))
                )

    return [
        (week_end, sex, band, total)
        for (week_end, sex, band), total in band_totals.items()
    ]


def _extract_weekly_sex_ages_fine_cols(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, str, int]]:
    """
    Pull (week_ending, sex, age_band, deaths) tuples from the 2022-2023 layout.

    Sheet "2" stacks three tables vertically — Table 2a (people), 2b
    (males), 2c (females) — each with its own ``Week ending`` /
    ``All ages`` / fine-band header row. We find every header row, scan a
    handful of rows above each for the sex marker (e.g. "Table 2b: ...
    males"), and treat the rows up to the next header as that sex's block.
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_indices: list[int] = []
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        if (
            "week ending" in normalized
            and any(_canonical_age_band(v) for v in row.tolist())
        ):
            header_indices.append(int(idx))

    if not header_indices:
        return []

    # Sequentialize the indices so we can take ranges.
    df_indices = list(df.index)
    band_totals: dict[tuple[pd.Timestamp, str, str], int] = {}

    for h_idx, header_index in enumerate(header_indices):
        # Look up to five rows above the header for a sex marker.
        sex: str | None = None
        for back in range(1, 6):
            scan_idx = header_index - back
            if scan_idx < df_indices[0]:
                break
            if scan_idx not in df.index:
                continue
            for value in df.loc[scan_idx].tolist():
                sex = _classify_sex_header(value)
                if sex is not None:
                    break
            if sex is not None:
                break
        if sex is None:
            continue

        headers = [str(v).strip() for v in df.loc[header_index].tolist()]
        week_end_col_idx: int | None = None
        age_columns: list[tuple[int, str]] = []
        for column_idx, header in enumerate(headers):
            if header.lower() == "week ending":
                week_end_col_idx = column_idx
                continue
            band = _canonical_age_band(header)
            if band is not None:
                age_columns.append((column_idx, band))

        if week_end_col_idx is None or not age_columns:
            continue

        end_index = (
            header_indices[h_idx + 1]
            if h_idx + 1 < len(header_indices)
            else df_indices[-1] + 1
        )

        for _, row in df.loc[header_index + 1:end_index - 1].iterrows():
            date_value = row.iloc[week_end_col_idx]
            if pd.isna(date_value):
                continue
            try:
                week_end = pd.Timestamp(date_value)
            except (ValueError, TypeError):
                continue
            if pd.isna(week_end):
                continue
            for column_idx, band in age_columns:
                value = row.iloc[column_idx]
                deaths = pd.to_numeric(value, errors="coerce")
                if pd.isna(deaths):
                    continue
                key = (week_end, sex, band)
                band_totals[key] = (
                    band_totals.get(key, 0) + int(round(float(deaths)))
                )

    return [
        (week_end, sex, band, total)
        for (week_end, sex, band), total in band_totals.items()
    ]


def _extract_weekly_sex_ages_long(
    sheet_df: pd.DataFrame,
    edition_year: int,
) -> list[tuple[pd.Timestamp, str, str, int]]:
    """
    Pull (week_ending, sex, age_band, deaths) tuples from the 2024+ layout.

    Each row is one (week, area, sex, age, IMD, place) cell. We filter to
    the all-residents E&W area, all IMD groups, all places, and keep all
    three sex values (All people / Male / Female) and any band row that
    isn't "All ages". Fine 5-year bands collapse into the seven canonical
    bands per (week, sex).
    """
    df = sheet_df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    header_index: int | None = None
    for idx, row in df.iterrows():
        normalized = [str(v).strip().lower() for v in row.tolist()]
        if (
            "week number" in normalized
            and any("week ending" in v or "week ended" in v for v in normalized)
            and any("number of deaths" in v for v in normalized)
            and any(v.startswith("age group") for v in normalized)
            and "sex" in normalized
        ):
            header_index = int(idx)
            break

    if header_index is None:
        return []

    headers = [str(v).strip() for v in df.loc[header_index].tolist()]
    body = df.loc[header_index + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all")
    if body.empty:
        return []

    body.columns = [str(c).strip() for c in body.columns]
    name_lookup = {c.lower(): c for c in body.columns}

    week_end_col = next(
        (c for k, c in name_lookup.items() if "week ending" in k or "week ended" in k),
        None,
    )
    deaths_col = next(
        (c for k, c in name_lookup.items() if "number of deaths" in k),
        None,
    )
    age_col = next(
        (c for k, c in name_lookup.items() if k.startswith("age group")),
        None,
    )
    area_col = name_lookup.get("area of usual residence") or name_lookup.get("area")
    sex_col = name_lookup.get("sex")
    imd_col = next((c for k, c in name_lookup.items() if "imd" in k), None)
    place_col = next(
        (c for k, c in name_lookup.items() if "place of occurrence" in k),
        None,
    )

    if (
        week_end_col is None
        or deaths_col is None
        or age_col is None
        or area_col is None
        or sex_col is None
    ):
        return []

    ew_aliases = {
        "england, wales and non-residents",
        "england and wales",
        "england & wales",
    }
    mask = body[area_col].astype(str).str.strip().str.lower().isin(ew_aliases)
    if imd_col is not None:
        mask &= body[imd_col].astype(str).str.strip().str.lower() == "all groups"
    if place_col is not None:
        mask &= body[place_col].astype(str).str.strip().str.lower() == "all places"
    mask &= body[age_col].astype(str).str.strip().str.lower() != "all ages"

    sex_aliases = {
        "all people": "All people",
        "male": "Male",
        "female": "Female",
    }
    sex_normalized = body[sex_col].astype(str).str.strip().str.lower()
    mask &= sex_normalized.isin(sex_aliases.keys())

    selected = body.loc[mask, [week_end_col, sex_col, age_col, deaths_col]].copy()
    if selected.empty:
        return []

    selected[deaths_col] = pd.to_numeric(selected[deaths_col], errors="coerce")
    selected = selected.dropna(subset=[deaths_col])
    selected["__sex__"] = (
        selected[sex_col].astype(str).str.strip().str.lower().map(sex_aliases)
    )
    selected["__band__"] = selected[age_col].map(_canonical_age_band)
    selected = selected.dropna(subset=["__sex__", "__band__"])

    band_totals: dict[tuple[pd.Timestamp, str, str], int] = {}
    for _, row in selected.iterrows():
        try:
            week_end = pd.Timestamp(row[week_end_col])
        except (ValueError, TypeError):
            continue
        if pd.isna(week_end):
            continue
        sex = str(row["__sex__"])
        band = str(row["__band__"])
        key = (week_end, sex, band)
        band_totals[key] = (
            band_totals.get(key, 0) + int(round(float(row[deaths_col])))
        )

    return [
        (week_end, sex, band, total)
        for (week_end, sex, band), total in band_totals.items()
    ]


def extract_weekly_sex_age_deaths(path: Path, edition_year: int) -> pd.DataFrame:
    """Return the E&W weekly per-(sex × age) series for one workbook."""
    rows: list[tuple[pd.Timestamp, str, str, int]] = []
    for _, sheet_df in read_workbook_sheets(path):
        rows = _extract_weekly_sex_ages_block_format(sheet_df, edition_year)
        if rows:
            break
        rows = _extract_weekly_sex_ages_fine_cols(sheet_df, edition_year)
        if rows:
            break
        rows = _extract_weekly_sex_ages_long(sheet_df, edition_year)
        if rows:
            break

    if not rows:
        return pd.DataFrame(
            columns=["week_ending", "sex", "age_band", "observed_deaths"]
        )

    df = pd.DataFrame(
        rows, columns=["week_ending", "sex", "age_band", "observed_deaths"]
    )
    return (
        df.groupby(["week_ending", "sex", "age_band"], as_index=False)
        ["observed_deaths"].sum()
        .sort_values(["week_ending", "sex", "age_band"])
        .reset_index(drop=True)
    )


def fetch_weekly_sex_age_deaths(
    raw_dir: Path,
    start_year: int = 2010,
    end_year: int | None = None,
    overwrite: bool = False,
    files: list[ONSFile] | None = None,
) -> pd.DataFrame:
    """
    Download every ONS weekly workbook in range and return per-(sex × band) rows.

    Result is a long DataFrame with one row per (week, sex, age_band) using
    the seven canonical age bands and three canonical sex labels
    (All people / Male / Female). Provenance columns are kept; deduplication
    prefers final over provisional editions, then the most recent year.
    """
    if files is None:
        files = discover_ons_files(
            start_year=start_year,
            end_year=end_year,
            dataset_url=ONS_WEEKLY_DATASET_URL,
        )
    if not files:
        raise RuntimeError(
            "No ONS weekly workbooks were discovered for the requested range."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ons_file, local_path in iter_downloaded_files(
        files,
        output_dir=raw_dir,
        overwrite=overwrite,
    ):
        rows = extract_weekly_sex_age_deaths(
            path=local_path,
            edition_year=ons_file.edition_year,
        )
        if rows.empty:
            continue

        rows = rows.assign(
            edition_year=ons_file.edition_year,
            is_final=ons_file.is_final,
            source_filename=local_path.name,
        )
        frames.append(rows)

    if not frames:
        raise RuntimeError(
            "Workbooks were downloaded but no per-(sex × age) rows were extracted. "
            "The ONS weekly sheet layout may have changed."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["week_ending"] = pd.to_datetime(combined["week_ending"])
    combined["observed_deaths"] = (
        pd.to_numeric(combined["observed_deaths"], errors="raise").astype(int)
    )

    combined["_is_final_int"] = combined["is_final"].astype(int)
    combined = combined.sort_values(
        ["sex", "age_band", "week_ending", "_is_final_int", "edition_year"],
        ascending=[True, True, True, False, False],
    ).drop_duplicates(subset=["sex", "age_band", "week_ending"], keep="first")

    output = combined[
        [
            "week_ending",
            "sex",
            "age_band",
            "observed_deaths",
            "edition_year",
            "is_final",
            "source_filename",
        ]
    ].sort_values(["sex", "age_band", "week_ending"]).reset_index(drop=True)

    return output


# ---------------------------------------------------------------------------
# Annual deaths by cause × sex × age (ONS Series DR reference table)
# ---------------------------------------------------------------------------

# The 2024 edition of the ONS "Deaths registered in England and Wales: Series
# DR" reference tables ships a single Table_5 covering 2015-2024 with
# ICD-10 chapter × sex × age × place breakdowns. Earlier annual editions
# only contain the reporting year, so we point at the 2024 file directly.
SERIES_DR_2024_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "birthsdeathsandmarriages/deaths/datasets/"
    "deathsregisteredinenglandandwalesseriesdrreferencetables/"
    "2024/annualdeaths2024.xlsx"
)
SERIES_DR_2024_FILENAME = "annualdeaths2024.xlsx"

# Map ONS Series DR age labels to our canonical age bands. The Series DR
# table reports adult bands at the same boundaries as our weekly canonical
# bands, plus several finer breakdowns under age 5 that don't align cleanly
# with our "Under 1" / "1-14" — we drop those rows downstream.
SERIES_DR_AGE_MAP: dict[str, str] = {
    "15 to 44": "15-44",
    "45 to 64": "45-64",
    "65 to 74": "65-74",
    "75 to 84": "75-84",
    "85 years and over": "85+",
}


def fetch_cause_by_sex_age(
    raw_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download the ONS Series DR reference table and return a tidy long DataFrame.

    The 2024 edition of the dataset (the file ``annualdeaths2024.xlsx``) is
    the only one that ships a multi-year cause × sex × age × place table —
    earlier annual editions report only their reporting year. We download
    that single file (cached under ``raw_dir``), parse Table_5, and emit a
    long DataFrame restricted to the five adult age bands we model
    elsewhere in the repo.

    Returns
    -------
    pd.DataFrame
        Columns: ``year`` (2015-2024), ``sex`` (All people / Male / Female),
        ``age_band`` (15-44, 45-64, 65-74, 75-84, 85+), ``icd_chapter_code``,
        ``icd_chapter_name``, ``place`` (10 categories incl. "All places"),
        ``deaths``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / SERIES_DR_2024_FILENAME
    file = ONSFile(
        edition_year=2024,
        url=SERIES_DR_2024_URL,
        filename=SERIES_DR_2024_FILENAME,
        is_final=True,
    )
    # download_file prefixes "edition_year_" to the filename; keep parity by
    # passing a directory the helper writes into and then reading from there.
    download_file(file=file, output_dir=raw_dir, overwrite=overwrite)
    actual_path = raw_dir / f"2024_{SERIES_DR_2024_FILENAME}"
    if not actual_path.exists():
        # Some environments pre-cache directly under the canonical name;
        # fall back to that if present.
        actual_path = local_path

    df = pd.read_excel(actual_path, sheet_name="Table_5", skiprows=4)
    df = df.rename(columns={
        "Year of registration": "year",
        "Sex": "sex",
        "Age group": "age_band_raw",
        "ICD-10 chapter codes": "icd_chapter_code",
        "ICD-10 chapter name": "icd_chapter_name",
        "Place of death": "place",
        "Number of deaths": "deaths",
    })

    df["age_band"] = df["age_band_raw"].map(SERIES_DR_AGE_MAP)
    df = df.dropna(subset=["age_band"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0).astype(int)
    # ONS embeds a U+201A (SINGLE LOW-9 QUOTATION MARK) as a separator
    # inside multi-range chapter labels — collapse it (and any U+FFFD
    # replacement character that occasionally appears) into a single space.
    junk_chars = "‚�"
    for col in ("icd_chapter_code", "icd_chapter_name"):
        cleaned = df[col].astype(str)
        for ch in junk_chars:
            cleaned = cleaned.str.replace(ch, " ", regex=False)
        df[col] = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()

    return df[
        [
            "year", "sex", "age_band", "icd_chapter_code",
            "icd_chapter_name", "place", "deaths",
        ]
    ].sort_values(
        ["year", "sex", "age_band", "icd_chapter_code", "place"]
    ).reset_index(drop=True)
