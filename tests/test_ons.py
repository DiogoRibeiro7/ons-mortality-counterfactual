"""Tests for ONS discovery / download helpers that don't need network access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ons_mortality.ons import _extract_filename, _is_workbook_link, sha256_file


def test_extract_filename_uses_uri_query_parameter() -> None:
    """ONS exposes files as `/file?uri=/path/to/file.xlsx`."""
    url = (
        "https://www.ons.gov.uk/file?uri=/peoplepopulation/"
        "deaths/2020edition.xlsx"
    )
    assert _extract_filename(url) == "2020edition.xlsx"


def test_extract_filename_decodes_percent_escapes() -> None:
    url = "https://www.ons.gov.uk/file?uri=/data/2020%20edition.xlsx"
    assert _extract_filename(url) == "2020 edition.xlsx"


def test_extract_filename_falls_back_to_path() -> None:
    url = "https://example.test/data/monthly.xls"
    assert _extract_filename(url) == "monthly.xls"


def test_extract_filename_raises_when_unparseable() -> None:
    with pytest.raises(ValueError):
        _extract_filename("https://example.test/")


def test_is_workbook_link_recognizes_common_shapes() -> None:
    assert _is_workbook_link("/some/file.xlsx")
    assert _is_workbook_link("/some/file.xls")
    assert _is_workbook_link("/file?uri=/path/to/foo.xlsx")
    assert not _is_workbook_link("/peoplepopulation/dataset")


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hello ons"
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "does-not-exist.bin")
