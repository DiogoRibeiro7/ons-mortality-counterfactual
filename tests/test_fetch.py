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
    AGE_BAND_ORDER,
    REGION_REGISTRY,
    SEX_ORDER,
    _canonical_age_band,
    _classify_sex_header,
    _extract_england_wales_from_sheet,
    _extract_from_long_format,
    _extract_regions_long_format,
    _extract_regions_wide_format,
    _extract_weekly_ages_block_format,
    _extract_weekly_ages_fine_cols,
    _extract_weekly_ages_long,
    _extract_weekly_hybrid_format,
    _extract_weekly_long_format,
    _extract_weekly_sex_ages_block_format,
    _extract_weekly_sex_ages_fine_cols,
    _extract_weekly_sex_ages_long,
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


def test_canonical_age_band_collapses_fine_into_seven() -> None:
    """The seven canonical bands cover every label spelling we extract."""
    assert _canonical_age_band("<1") == "Under 1"
    assert _canonical_age_band("Under 1 year") == "Under 1"
    assert _canonical_age_band("Under 1") == "Under 1"
    # Fine 5-year bands collapse upward.
    assert _canonical_age_band("1-4") == "1-14"
    assert _canonical_age_band("01-04") == "1-14"
    assert _canonical_age_band("5-9") == "1-14"
    assert _canonical_age_band("10 to 14") == "1-14"
    assert _canonical_age_band("01-14") == "1-14"
    assert _canonical_age_band("20-24") == "15-44"
    assert _canonical_age_band("40-44") == "15-44"
    assert _canonical_age_band("60-64") == "45-64"
    assert _canonical_age_band("70-74") == "65-74"
    assert _canonical_age_band("75-79") == "75-84"
    assert _canonical_age_band("85-89") == "85+"
    assert _canonical_age_band("90+") == "85+"
    assert _canonical_age_band("90 and over") == "85+"
    # Non-matching labels return None and are silently dropped upstream.
    assert _canonical_age_band("All ages") is None
    assert _canonical_age_band("") is None
    assert _canonical_age_band(123) is None
    assert set(AGE_BAND_ORDER) == {
        "Under 1", "1-14", "15-44", "45-64", "65-74", "75-84", "85+",
    }


def test_block_format_coarse_labels_in_column_a() -> None:
    """2010-2018 layout: 7 coarse age labels in column A under 'Persons'."""
    weeks = [pd.Timestamp(f"2015-01-{day:02d}") for day in (4, 11, 18, 25)]
    sheet = pd.DataFrame(
        [
            ["Week ended", None, *weeks],
            ["Total deaths, all ages", None, 11000, 11500, 11200, 10800],
            ["Persons 4", None, None, None, None, None],
            ["Deaths by age group", None, None, None, None, None],
            ["Under 1 year", None, 50, 55, 60, 45],
            ["01-14", None, 20, 25, 18, 22],
            ["15-44", None, 200, 220, 210, 215],
            ["45-64", None, 1500, 1600, 1550, 1450],
            ["65-74", None, 1800, 1900, 1850, 1750],
            ["75-84", None, 3000, 3100, 3050, 2900],
            ["85+", None, 4430, 4600, 4462, 4418],
            ["Males 4", None, None, None, None, None],
            ["Deaths by age group", None, None, None, None, None],
            ["Under 1 year", None, 27, 30, 32, 25],
        ]
    )

    rows = _extract_weekly_ages_block_format(sheet, edition_year=2015)

    bands = sorted({band for _, band, _ in rows})
    assert bands == sorted(AGE_BAND_ORDER)
    assert len(rows) == 7 * 4

    # Stop at "Males" — the persons "85+" line is the last row counted.
    week1_85plus = next(d for ts, b, d in rows if ts == weeks[0] and b == "85+")
    assert week1_85plus == 4430


def test_block_format_fine_labels_in_column_b_are_summed() -> None:
    """2020-2021 layout: fine 5-year labels in column B sum into 7 bands."""
    weeks = [pd.Timestamp("2020-04-03"), pd.Timestamp("2020-04-10")]
    sheet = pd.DataFrame(
        [
            ["Week ended", None, *weeks],
            [None, "Persons 6", None, None],
            [None, "Deaths by age group", None, None],
            [None, "<1", 51, 49],
            [None, "1-4", 8, 7],
            [None, "5-9", 6, 5],
            [None, "10-14", 7, 8],
            [None, "15-19", 30, 35],
            [None, "20-24", 40, 45],
            [None, "25-29", 50, 55],
            [None, "30-34", 60, 65],
            [None, "35-39", 48, 52],
            [None, "40-44", 60, 70],
            [None, "45-49", 200, 220],
            [None, "50-54", 350, 370],
            [None, "55-59", 500, 530],
            [None, "60-64", 810, 850],
            [None, "65-69", 1200, 1250],
            [None, "70-74", 1534, 1580],
            [None, "75-79", 2000, 2100],
            [None, "80-84", 3005, 3150],
            [None, "85-89", 3500, 3700],
            [None, "90+", 2928, 3100],
            [None, "Males 6", None, None],
            [None, "Deaths by age group", None, None],
            [None, "<1", 26, 25],
        ]
    )

    rows = _extract_weekly_ages_block_format(sheet, edition_year=2020)

    bands = sorted({band for _, band, _ in rows})
    assert bands == sorted(AGE_BAND_ORDER)

    # 1-4 + 5-9 + 10-14 = 8 + 6 + 7 = 21 for week 1.
    week1_1_14 = next(d for ts, b, d in rows if ts == weeks[0] and b == "1-14")
    assert week1_1_14 == 21
    # 85-89 + 90+ = 3500 + 2928 = 6428 for week 1.
    week1_85plus = next(d for ts, b, d in rows if ts == weeks[0] and b == "85+")
    assert week1_85plus == 6428


def test_fine_cols_format_stops_before_males_table() -> None:
    """The 2022/2023 hybrid layout stacks Persons / Males / Females tables."""
    persons_header = ["Week number", "Week ending", "All ages", "<1", "1-4", "5-9",
                      "10-14", "15-44", "45-64", "65-74", "75-84", "85-89", "90+"]
    persons_row = [1, pd.Timestamp("2022-01-07"), 12262,
                   49, 5, 3, 6, 600, 2900, 3940, 7124, 4900, 4970]
    # Males block starts a few rows below — must not be summed in.
    males_row = [1, pd.Timestamp("2022-01-07"), 6131,
                 27, 3, 2, 3, 320, 1500, 2000, 3500, 2500, 2400]

    sheet = pd.DataFrame(
        [
            ["Sheet 2: Persons table"] + [None] * 12,
            persons_header,
            persons_row,
            ["Sheet 2: Males table"] + [None] * 12,
            persons_header,
            males_row,
        ]
    )

    rows = _extract_weekly_ages_fine_cols(sheet, edition_year=2022)

    bands = {band for _, band, _ in rows}
    assert bands == set(AGE_BAND_ORDER)
    # 1-4 + 5-9 + 10-14 = 5 + 3 + 6 = 14.
    week1_1_14 = next(d for ts, b, d in rows if b == "1-14")
    assert week1_1_14 == 14
    # 85-89 + 90+ = 4900 + 4970 = 9870 — Males not added.
    week1_85plus = next(d for ts, b, d in rows if b == "85+")
    assert week1_85plus == 9870


def test_long_format_excludes_all_ages_and_subgroup_totals() -> None:
    """The 2023+ long layout filters to the all-totals slice and band rows."""
    sheet = pd.DataFrame(
        [
            ["Week number", "Week ending", "Area of usual residence",
             "Sex", "Age group (years)", "IMD quantile group",
             "Place of occurrence", "Number of deaths"],
            # Selected (all totals × age bands):
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "Under 1", "All groups", "All places", 29],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "1 to 4", "All groups", "All places", 5],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "5 to 9", "All groups", "All places", 6],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "85 to 89", "All groups", "All places", 1314],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "90 and over", "All groups", "All places", 1633],
            # Filtered out — All ages total double-counts the age breakdown.
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "All ages", "All groups", "All places", 7151],
            # Filtered out — sub-area row (England-only).
            [1, pd.Timestamp("2024-12-27"), "England",
             "All people", "Under 1", "All groups", "All places", 27],
            # Filtered out — Male slice would double-count Persons.
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "Male", "Under 1", "All groups", "All places", 17],
            # Filtered out — sub-place slice.
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "Under 1", "All groups", "Hospital", 18],
        ]
    )

    rows = _extract_weekly_ages_long(sheet, edition_year=2024)

    by_band = {band: deaths for _, band, deaths in rows}
    assert by_band["Under 1"] == 29
    # 1 to 4 + 5 to 9 = 11 (only 2 of the 1-14 sub-bands shown).
    assert by_band["1-14"] == 11
    # 85 to 89 + 90 and over = 1314 + 1633 = 2947.
    assert by_band["85+"] == 2947


def test_classify_sex_header_handles_block_and_table_titles() -> None:
    """Each ONS sex marker spelling should map to the canonical label."""
    assert _classify_sex_header("Persons 4") == "All people"
    assert _classify_sex_header("People 2") == "All people"
    assert _classify_sex_header("Males 6") == "Male"
    assert _classify_sex_header("Females 5") == "Female"
    # Long-format / 2024+ singular spellings.
    assert _classify_sex_header("Male") == "Male"
    assert _classify_sex_header("Female") == "Female"
    assert _classify_sex_header("All people") == "All people"
    # Real 2022 sheet "2" titles.
    assert _classify_sex_header("Table 2a: ... by age group, people") == "All people"
    assert _classify_sex_header("Table 2b: ... by age group, males") == "Male"
    assert _classify_sex_header("Table 2c: ... by age group, females") == "Female"
    # "Females" must beat "males" (substring trap) and "Female" beats "Male".
    assert _classify_sex_header("Female cohort") == "Female"
    # No match.
    assert _classify_sex_header("Deaths by age group") is None
    assert _classify_sex_header("") is None
    assert _classify_sex_header(123) is None
    # Sanity: SEX_ORDER lists the three canonical labels.
    assert set(SEX_ORDER) == {"All people", "Male", "Female"}


def test_block_format_sex_extraction_splits_three_blocks() -> None:
    """Persons / Males / Females blocks each yield their own (sex, band) rows."""
    weeks = [pd.Timestamp("2015-01-04"), pd.Timestamp("2015-01-11")]
    sheet = pd.DataFrame(
        [
            ["Week ended", None, *weeks],
            ["Persons 4", None, None, None],
            ["Deaths by age group", None, None, None],
            ["Under 1 year", None, 50, 55],
            ["01-14", None, 20, 25],
            ["15-44", None, 200, 220],
            ["45-64", None, 1500, 1600],
            ["65-74", None, 1800, 1900],
            ["75-84", None, 3000, 3100],
            ["85+", None, 4430, 4600],
            ["Males 4", None, None, None],
            ["Deaths by age group", None, None, None],
            ["Under 1 year", None, 27, 30],
            ["01-14", None, 12, 14],
            ["15-44", None, 130, 145],
            ["45-64", None, 900, 960],
            ["65-74", None, 1010, 1090],
            ["75-84", None, 1500, 1550],
            ["85+", None, 1750, 1820],
            ["Females 4", None, None, None],
            ["Deaths by age group", None, None, None],
            ["Under 1 year", None, 23, 25],
            ["01-14", None, 8, 11],
            ["15-44", None, 70, 75],
            ["45-64", None, 600, 640],
            ["65-74", None, 790, 810],
            ["75-84", None, 1500, 1550],
            ["85+", None, 2680, 2780],
        ]
    )

    rows = _extract_weekly_sex_ages_block_format(sheet, edition_year=2015)
    table = pd.DataFrame(
        rows, columns=["week_ending", "sex", "age_band", "deaths"]
    )

    # 3 sex × 7 bands × 2 weeks = 42 rows.
    assert len(table) == 42
    assert set(table["sex"]) == {"All people", "Male", "Female"}
    assert set(table["age_band"]) == set(AGE_BAND_ORDER)

    # Male + Female == All people for every (week, band) pair.
    pivot = table.pivot_table(
        index=["week_ending", "age_band"],
        columns="sex", values="deaths", aggfunc="sum",
    )
    assert (pivot["Male"] + pivot["Female"] == pivot["All people"]).all()

    # Spot-check: 2015-01-04, Male 45-64 = 900 (single fine band) and the
    # Persons 85+ matches the literal value (no aggregation needed).
    male_45_64 = next(
        d for w, s, b, d in rows
        if w == weeks[0] and s == "Male" and b == "45-64"
    )
    assert male_45_64 == 900
    persons_85 = next(
        d for w, s, b, d in rows
        if w == weeks[0] and s == "All people" and b == "85+"
    )
    assert persons_85 == 4430


def test_fine_cols_sex_extraction_picks_up_table_titles() -> None:
    """Three stacked tables (2a/2b/2c) each tagged via the title above the header."""
    persons_header = ["Week number", "Week ending", "All ages", "<1",
                      "15-44", "85-89", "90+"]
    persons_row = [1, pd.Timestamp("2022-01-07"), 12000, 49, 600, 4900, 4970]
    males_row = [1, pd.Timestamp("2022-01-07"), 6100, 28, 350, 2400, 2360]
    females_row = [1, pd.Timestamp("2022-01-07"), 5900, 21, 250, 2500, 2610]

    sheet = pd.DataFrame(
        [
            ["Table 2a: ... people, registered 2022"] + [None] * 6,
            persons_header,
            persons_row,
            ["Table 2b: ... males, registered 2022"] + [None] * 6,
            persons_header,
            males_row,
            ["Table 2c: ... females, registered 2022"] + [None] * 6,
            persons_header,
            females_row,
        ]
    )

    rows = _extract_weekly_sex_ages_fine_cols(sheet, edition_year=2022)
    table = pd.DataFrame(
        rows, columns=["week_ending", "sex", "age_band", "deaths"]
    )

    # 3 sex × 5 distinct canonical bands (Under 1, 15-44, 85+ from <1+15-44+85-89+90+)
    # × 1 week. <1 → Under 1; 15-44 maps directly; 85-89 + 90+ → 85+.
    assert set(table["sex"]) == {"All people", "Male", "Female"}
    by_sex = {
        s: dict(zip(g["age_band"], g["deaths"], strict=True))
        for s, g in table.groupby("sex")
    }
    # Persons 85+ = 4900 + 4970 = 9870.
    assert by_sex["All people"]["85+"] == 9870
    # Males 85+ = 2400 + 2360 = 4760.
    assert by_sex["Male"]["85+"] == 4760
    # Females Under 1 = 21.
    assert by_sex["Female"]["Under 1"] == 21
    # Identity holds: M + F = All people for every band that appears.
    for band in by_sex["All people"]:
        assert (
            by_sex["Male"].get(band, 0) + by_sex["Female"].get(band, 0)
            == by_sex["All people"][band]
        )


def test_long_format_sex_extraction_keeps_three_sex_slices() -> None:
    """The 2024+ long layout: keep All people / Male / Female and drop sub-area rows."""
    sheet = pd.DataFrame(
        [
            ["Week number", "Week ending", "Area of usual residence",
             "Sex", "Age group (years)", "IMD quantile group",
             "Place of occurrence", "Number of deaths"],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "Under 1", "All groups", "All places", 29],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "Male", "Under 1", "All groups", "All places", 17],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "Female", "Under 1", "All groups", "All places", 12],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "85 to 89", "All groups", "All places", 1314],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "Male", "85 to 89", "All groups", "All places", 580],
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "Female", "85 to 89", "All groups", "All places", 734],
            # Filtered out — All ages would double-count.
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "All people", "All ages", "All groups", "All places", 7151],
            # Filtered out — sub-area row.
            [1, pd.Timestamp("2024-12-27"), "England",
             "Male", "Under 1", "All groups", "All places", 16],
            # Filtered out — sub-place slice.
            [1, pd.Timestamp("2024-12-27"), "England, Wales and non-residents",
             "Female", "Under 1", "All groups", "Hospital", 9],
        ]
    )

    rows = _extract_weekly_sex_ages_long(sheet, edition_year=2024)
    by = {(s, b): d for _, s, b, d in rows}

    assert by[("All people", "Under 1")] == 29
    assert by[("Male", "Under 1")] == 17
    assert by[("Female", "Under 1")] == 12
    # 85 to 89 → 85+ canonical band.
    assert by[("All people", "85+")] == 1314
    assert by[("Male", "85+")] == 580
    assert by[("Female", "85+")] == 734
    # Identity: M + F = All people.
    assert by[("Male", "Under 1")] + by[("Female", "Under 1")] == by[("All people", "Under 1")]
    assert by[("Male", "85+")] + by[("Female", "85+")] == by[("All people", "85+")]
