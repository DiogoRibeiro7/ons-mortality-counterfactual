"""Tests for the no-MySQL ONS fetch path.

The extractor needs to handle two distinct ONS layouts:

1. **Wide format (2006-2022)** - one row per geography, one column per
   month. The country-level row may sit in a column other than the first
   non-month column, so we must scan every non-month cell for the
   England & Wales marker rather than relying on header heuristics.
2. **Long format (2023+)** - one row per (geography, month). Header has
   ``Month`` / ``Code`` / ``Geography`` / ``Number of deaths`` columns.
"""

from __future__ import annotations

import pandas as pd

from ons_mortality.fetch import (
    REGION_REGISTRY,
    _extract_england_wales_from_sheet,
    _extract_from_long_format,
    _extract_regions_long_format,
    _extract_regions_wide_format,
    _extract_weekly_hybrid_format,
    _extract_weekly_long_format,
    _extract_weekly_wide_format,
    _looks_like_england_and_wales,
    _match_region,
)


def test_looks_like_england_and_wales_matches_code_and_names() -> None:
    """The marker check accepts both the K-code and several name spellings."""
    assert _looks_like_england_and_wales("K04000001")
    assert _looks_like_england_and_wales(" k04000001 ")
    assert _looks_like_england_and_wales("England and Wales")
    assert _looks_like_england_and_wales("ENGLAND AND WALES")
    assert _looks_like_england_and_wales("England & Wales")
    assert not _looks_like_england_and_wales("England")
    assert not _looks_like_england_and_wales("")
    assert not _looks_like_england_and_wales(123)


def test_extract_wide_format_pyramid_layout() -> None:
    """Old-style sheet: area names are split across multiple columns."""
    months = ["Jan-06", "Feb-06", "Mar-06", "Apr-06", "May-06", "Jun-06",
              "Jul-06", "Aug-06", "Sep-06", "Oct-06", "Nov-06", "Dec-06"]
    sheet = pd.DataFrame(
        [
            ["Monthly figures on deaths", *([None] * 14), "England and Wales"],
            ["Area Codes", None, None, None, *months],
            [None, "TOTAL REGISTRATIONS 1", None, None, *list(range(49000, 49012))],
            [None, "ENGLAND AND WALES", None, None, *list(range(49000, 49012))],
            ["921", "ENGLAND", None, None, *list(range(46000, 46012))],
            ["00EH", None, "Darlington UA", None, *list(range(100, 112))],
        ]
    )

    rows = _extract_england_wales_from_sheet(sheet, edition_year=2006)

    assert len(rows) == 12
    assert rows[0] == (pd.Timestamp("2006-01-01"), 49000)
    assert rows[-1] == (pd.Timestamp("2006-12-01"), 49011)


def test_extract_wide_format_uses_k_code_when_name_is_missing() -> None:
    """The K04000001 code alone should be enough to identify the row."""
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    sheet = pd.DataFrame(
        [
            ["Code", "Name", *months],
            ["K04000001", "Total", *list(range(50000, 50012))],
            ["E92000001", "England", *list(range(46000, 46012))],
        ]
    )

    rows = _extract_england_wales_from_sheet(sheet, edition_year=2024)

    assert len(rows) == 12
    assert rows[3] == (pd.Timestamp("2024-04-01"), 50003)


def test_extract_long_format_with_k_code() -> None:
    """The 2023+ layout uses one row per (month, geography)."""
    sheet = pd.DataFrame(
        [
            ["Month", "Code", "Geography type", "Geography", "Number of deaths"],
            ["January", "K04000001", "Country", "ENGLAND, WALES AND NON-RESIDENTS", 58_000],
            ["January", "E92000001", "Country", "ENGLAND", 55_000],
            ["February", "K04000001", "Country", "ENGLAND, WALES AND NON-RESIDENTS", 47_000],
            ["February", "E92000001", "Country", "ENGLAND", 44_000],
        ]
    )

    rows = _extract_from_long_format(sheet, edition_year=2024)

    assert len(rows) == 2
    assert rows[0] == (pd.Timestamp("2024-01-01"), 58_000)
    assert rows[1] == (pd.Timestamp("2024-02-01"), 47_000)


def test_extract_weekly_wide_format_2010_2022_layout() -> None:
    """The 2010-2021 weekly layout has weeks across columns."""
    week_dates = pd.to_datetime([f"2019-01-{d:02d}" for d in (4, 11, 18, 25)])
    sheet = pd.DataFrame(
        [
            ["Week number", None, 1, 2, 3, 4],
            ["Week ended", None, *week_dates.tolist()],
            ["Total deaths, all ages", None, 10955, 12609, 11860, 11740],
            ["Five-year average", None, 12273, 13670, 13056, 12486],
        ]
    )

    rows = _extract_weekly_wide_format(sheet)

    assert len(rows) == 4
    assert rows[0] == (pd.Timestamp("2019-01-04"), 10955)
    assert rows[-1] == (pd.Timestamp("2019-01-25"), 11740)


def test_extract_weekly_hybrid_format_2022_2023_layout() -> None:
    """The 2022/2023 weekly layout has named E&W columns."""
    sheet = pd.DataFrame(
        [
            ["Sheet 1", None, None, None, None, None],
            ["[note 1]", None, None, None, None, None],
            [
                "Week number", "Week ended",
                "Total deaths England and Wales",
                "Total deaths England and Wales",
                "Five-year average, England and Wales",
                "Total deaths England (2022)",
            ],
            [1, pd.Timestamp("2022-01-07"), 12262, 17748, 13298, 11470],
            [2, pd.Timestamp("2022-01-14"), 13311, 18038, 14182, 12399],
        ]
    )

    rows = _extract_weekly_hybrid_format(sheet)

    assert len(rows) == 2
    assert rows[0] == (pd.Timestamp("2022-01-07"), 12262)
    # The first occurrence wins — the second 'Total deaths E&W' (prior year) ignored.
    assert rows[1] == (pd.Timestamp("2022-01-14"), 13311)


def test_extract_weekly_long_format_2024_layout() -> None:
    """The 2024+ weekly layout is fully long-format with filterable columns."""
    sheet = pd.DataFrame(
        [
            [
                "Week number", "Week ending", "Area of usual residence",
                "Sex", "Age group (years)", "IMD quantile group",
                "Place of occurrence", "Number of deaths",
            ],
            [
                1, pd.Timestamp("2024-01-05"), "England, Wales and non-residents",
                "All people", "All ages", "All groups", "All places", 13000,
            ],
            [
                1, pd.Timestamp("2024-01-05"), "England",
                "All people", "All ages", "All groups", "All places", 12200,
            ],
            [
                1, pd.Timestamp("2024-01-05"), "England, Wales and non-residents",
                "Male", "All ages", "All groups", "All places", 6500,
            ],
            [
                2, pd.Timestamp("2024-01-12"), "England, Wales and non-residents",
                "All people", "All ages", "All groups", "All places", 14500,
            ],
        ]
    )

    rows = _extract_weekly_long_format(sheet)

    # Only the two "All people / All ages / All groups / All places" rows match.
    assert len(rows) == 2
    assert rows[0] == (pd.Timestamp("2024-01-05"), 13000)
    assert rows[1] == (pd.Timestamp("2024-01-12"), 14500)


def test_extract_long_format_falls_back_to_england_plus_wales() -> None:
    """If K04000001 is absent, England + Wales country totals are summed."""
    sheet = pd.DataFrame(
        [
            ["Month", "Code", "Geography", "Number of deaths"],
            ["January", "E92000001", "ENGLAND", 50_000],
            ["January", "W92000004", "WALES", 3_000],
            ["February", "E92000001", "ENGLAND", 40_000],
            ["February", "W92000004", "WALES", 2_500],
        ]
    )

    rows = _extract_from_long_format(sheet, edition_year=2024)

    assert len(rows) == 2
    assert rows[0] == (pd.Timestamp("2024-01-01"), 53_000)
    assert rows[1] == (pd.Timestamp("2024-02-01"), 42_500)


def test_extract_returns_empty_for_unrecognized_sheet() -> None:
    """Unfamiliar sheets must return [] rather than raising."""
    sheet = pd.DataFrame([["a", "b"], [1, 2]])
    assert _extract_england_wales_from_sheet(sheet, edition_year=2020) == []
    assert _extract_from_long_format(sheet, edition_year=2020) == []


def test_match_region_accepts_codes_and_aliases() -> None:
    """Both GSS codes and the canonical name spellings should match."""
    assert _match_region("E12000007") == "E12000007"
    assert _match_region(" e12000007 ") == "E12000007"
    assert _match_region("LONDON") == "E12000007"
    assert _match_region("london") == "E12000007"
    assert _match_region("Yorkshire and Humber") == "E12000003"
    assert _match_region("WALES") == "W92000004"
    assert _match_region("Darlington UA") is None
    assert _match_region("") is None
    assert _match_region(123) is None


def test_region_registry_covers_all_english_regions_plus_wales() -> None:
    """Sanity: 9 English regions + Wales = 10 entries."""
    assert len(REGION_REGISTRY) == 10
    assert "W92000004" in REGION_REGISTRY
    english_codes = [c for c in REGION_REGISTRY if c.startswith("E12")]
    assert len(english_codes) == 9


def test_extract_regions_wide_format_pyramid_layout() -> None:
    """The 2006-2022 pyramid layout: regions in col 1, LAs in col 2."""
    months = ["Jan-15", "Feb-15", "Mar-15", "Apr-15", "May-15", "Jun-15",
              "Jul-15", "Aug-15", "Sep-15", "Oct-15", "Nov-15", "Dec-15"]
    sheet = pd.DataFrame(
        [
            ["Area Codes", None, None, None, *months],
            [None, "ENGLAND AND WALES", None, None, *list(range(50000, 50012))],
            ["A", "NORTH EAST", None, None, *list(range(2500, 2512))],
            ["00EH", None, "Darlington UA", None, *list(range(100, 112))],
            ["B", "NORTH WEST", None, None, *list(range(7000, 7012))],
            ["G", "LONDON", None, None, *list(range(7500, 7512))],
            ["W", "WALES", None, None, *list(range(2900, 2912))],
        ]
    )

    rows = _extract_regions_wide_format(sheet, edition_year=2015)

    codes = {code for _, code, _ in rows}
    assert codes == {"E12000001", "E12000002", "E12000007", "W92000004"}

    ne_jan = next(d for ts, c, d in rows if c == "E12000001" and ts.month == 1)
    assert ne_jan == 2500
    london_dec = next(d for ts, c, d in rows if c == "E12000007" and ts.month == 12)
    assert london_dec == 7511


def test_extract_regions_long_format_filters_to_region_rows() -> None:
    """The 2023+ long format flags regions explicitly via Geography type."""
    sheet = pd.DataFrame(
        [
            ["Month", "Code", "Geography type", "Geography", "Number of deaths"],
            ["January", "K04000001", "Country", "ENGLAND, WALES AND NON-RESIDENTS", 58_000],
            ["January", "E92000001", "Country", "ENGLAND", 55_000],
            ["January", "E12000001", "Region", "NORTH EAST", 3_000],
            ["January", "E12000007", "Region", "LONDON", 4_500],
            ["January", "W92000004", "Country", "WALES", 3_100],
            ["February", "E12000001", "Region", "NORTH EAST", 2_700],
            ["January", "E06000005", "Local Authority", "Darlington", 130],
        ]
    )

    rows = _extract_regions_long_format(sheet, edition_year=2024)

    codes = {code for _, code, _ in rows}
    # Only regions + Wales — Country/LA rows must be excluded.
    assert codes == {"E12000001", "E12000007", "W92000004"}
    london_jan = next(
        d for ts, c, d in rows if c == "E12000007" and ts.month == 1
    )
    assert london_jan == 4_500
