from __future__ import annotations

import pandas as pd

from ons_mortality.parser import month_number_from_header, normalise_ons_sheet


def test_month_number_from_header_handles_full_and_abbreviated_names() -> None:
    assert month_number_from_header("January") == 1
    assert month_number_from_header("Jan") == 1
    assert month_number_from_header("September(p)") == 9
    assert month_number_from_header("Total") is None


def test_normalise_ons_sheet_extracts_long_monthly_rows() -> None:
    raw = pd.DataFrame(
        [
            [None, None, None, None],
            ["Area code", "Area name", "January", "February"],
            ["K04000001", "England and Wales", 50_000, 44_000],
            ["E92000001", "England", 47_000, 41_000],
        ]
    )

    parsed = normalise_ons_sheet(
        sheet_name="Table 1",
        sheet_df=raw,
        source_file_id=10,
        edition_year=2020,
        is_final=True,
    )

    assert len(parsed) == 4
    assert set(parsed["area_name"]) == {"England and Wales", "England"}
    assert set(parsed["deaths"]) == {50_000, 44_000, 47_000, 41_000}
    assert parsed["source_file_id"].eq(10).all()
    assert parsed["edition_year"].eq(2020).all()
