"""Tests for the ONS workbook parser."""

from __future__ import annotations

import pandas as pd

from ons_mortality.parser import month_number_from_header, normalise_ons_sheet


def test_month_number_from_header_handles_full_and_abbreviated_names() -> None:
    """Header parser must recognize abbreviations and provisional markers."""
    assert month_number_from_header("January") == 1
    assert month_number_from_header("Jan") == 1
    assert month_number_from_header("September(p)") == 9
    assert month_number_from_header("Total") is None


def test_normalise_ons_sheet_extracts_long_monthly_rows() -> None:
    """A typical 12-month sheet melts into one row per area-month."""
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    england_wales_values = [
        50_000, 44_000, 48_000, 45_000, 43_000, 42_000,
        41_000, 40_000, 42_000, 46_000, 47_000, 49_000,
    ]
    england_values = [
        47_000, 41_000, 45_000, 42_000, 40_000, 39_000,
        38_000, 37_000, 39_000, 43_000, 44_000, 46_000,
    ]

    raw = pd.DataFrame(
        [
            [None, None, *([None] * 12)],
            ["Area code", "Area name", *months],
            ["K04000001", "England and Wales", *england_wales_values],
            ["E92000001", "England", *england_values],
        ]
    )

    parsed = normalise_ons_sheet(
        sheet_name="Table 1",
        sheet_df=raw,
        source_file_id=10,
        edition_year=2020,
        is_final=True,
    )

    assert len(parsed) == 24
    assert set(parsed["area_name"]) == {"England and Wales", "England"}
    assert parsed["source_file_id"].eq(10).all()
    assert parsed["edition_year"].eq(2020).all()
    jan_ew = parsed[
        (parsed["area_name"] == "England and Wales")
        & (parsed["month_date"].astype(str) == "2020-01-01")
    ]
    assert jan_ew["deaths"].iloc[0] == 50_000


def test_normalise_ons_sheet_handles_provisional_month_markers() -> None:
    """Headers like 'September(p)' must still be recognized as months."""
    months_with_marker = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September(p)",
        "October(p)",
        "November",
        "December",
    ]
    values = [
        50_000, 44_000, 48_000, 45_000, 43_000, 42_000,
        41_000, 40_000, 42_000, 46_000, 47_000, 49_000,
    ]

    raw = pd.DataFrame(
        [
            ["Area code", "Area name", *months_with_marker],
            ["K04000001", "England and Wales", *values],
        ]
    )

    parsed = normalise_ons_sheet(
        sheet_name="Table 1",
        sheet_df=raw,
        source_file_id=11,
        edition_year=2021,
        is_final=False,
    )

    assert len(parsed) == 12
    assert parsed["is_final"].eq(False).all()
