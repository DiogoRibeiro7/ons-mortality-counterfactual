"""ONS dataset discovery, download, and source-file registration."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from sqlalchemy import Engine, text

ONS_DATASET_URL = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/"
    "birthsdeathsandmarriages/deaths/datasets/"
    "monthlyfiguresondeathsregisteredbyareaofusualresidence"
)

ONS_WEEKLY_DATASET_URL = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/"
    "birthsdeathsandmarriages/deaths/datasets/"
    "weeklyprovisionalfiguresondeathsregisteredinenglandandwales"
)

# ONS occasionally rejects requests sent with the default `python-requests` UA.
ONS_HTTP_HEADERS = {
    "User-Agent": (
        "ons-mortality-counterfactual/0.1 "
        "(+https://github.com/diogoribeiro/ons-mortality-counterfactual)"
    ),
    "Accept": "*/*",
}


@dataclass(frozen=True)
class ONSFile:
    """Metadata for a downloadable ONS annual mortality workbook."""

    edition_year: int
    url: str
    filename: str
    is_final: bool


def _extract_filename(url: str) -> str:
    """
    Extract a useful filename from an ONS file URL.

    ONS frequently exposes files as `/file?uri=/path/to/file.xlsx`, so the
    filename is usually inside the `uri` query parameter.
    """
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)

    if "uri" in query and query["uri"]:
        return Path(unquote(query["uri"][0])).name

    candidate = Path(parsed_url.path).name
    if candidate:
        return candidate

    raise ValueError(f"Could not extract filename from URL: {url}")


def _is_workbook_link(href: str) -> bool:
    """Return True if an HTML href points to an ONS Excel workbook."""
    lowered = href.lower()
    return ".xls" in lowered or ".xlsx" in lowered or "/file?uri=" in lowered


def _find_workbook_link_for_heading(heading: Tag) -> str | None:
    """
    Locate the first ONS Excel link associated with an edition heading.

    The ONS dataset page uses a collapsible widget where the heading and the
    download button live in sibling divs, so we walk up from the heading to
    successively larger ancestors until we find one that contains a workbook
    link inside its subtree.
    """
    node: Tag | None = heading
    for _ in range(6):
        if node is None:
            return None
        for link in node.find_all("a", href=True):
            href = str(link.get("href"))
            if _is_workbook_link(href):
                return href
        parent = node.parent
        node = parent if isinstance(parent, Tag) else None
    return None


def discover_ons_files(
    start_year: int = 2006,
    end_year: int | None = None,
    dataset_url: str = ONS_DATASET_URL,
) -> list[ONSFile]:
    """
    Discover ONS annual Excel workbooks from the public dataset page.

    Parameters
    ----------
    start_year:
        First edition year to keep.
    end_year:
        Last edition year to keep. If omitted, all years after `start_year` are kept.
    dataset_url:
        ONS dataset landing page.

    Returns
    -------
    list[ONSFile]
        Sorted list of discovered workbooks.
    """
    if start_year < 1900:
        raise ValueError("start_year is not valid.")

    if end_year is not None and end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")

    response = requests.get(dataset_url, headers=ONS_HTTP_HEADERS, timeout=45)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    files: list[ONSFile] = []

    heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6")
    edition_re = re.compile(r"(?P<year>20\d{2}|19\d{2})\s+edition", re.IGNORECASE)

    for heading in soup.find_all(heading_tags):
        if not isinstance(heading, Tag):
            continue

        heading_text = heading.get_text(" ", strip=True)
        year_match = edition_re.search(heading_text)

        if year_match is None:
            continue

        year = int(year_match.group("year"))
        is_final = "final" in heading_text.lower()

        if year < start_year:
            continue

        if end_year is not None and year > end_year:
            continue

        # The ONS page wraps each edition in a collapsible block where the
        # heading sits in one child div and the download link in another.
        # Walk up to the nearest container that also holds the link.
        workbook_link = _find_workbook_link_for_heading(heading)
        if workbook_link is None:
            continue

        absolute_url = urljoin(dataset_url, workbook_link)
        files.append(
            ONSFile(
                edition_year=year,
                url=absolute_url,
                filename=_extract_filename(absolute_url),
                is_final=is_final,
            )
        )

    unique_by_year: dict[int, ONSFile] = {}
    for file in files:
        unique_by_year[file.edition_year] = file

    return sorted(unique_by_year.values(), key=lambda item: item.edition_year)


def download_file(
    file: ONSFile,
    output_dir: Path,
    overwrite: bool = False,
    max_retries: int = 6,
    initial_backoff: float = 15.0,
    request_delay: float = 3.0,
) -> Path:
    """
    Download an ONS workbook to disk.

    The ONS site rate-limits aggressive clients with HTTP 429. This helper
    sleeps briefly between requests and applies exponential backoff (honoring
    a `Retry-After` header when present) so a full historical fetch can run
    unattended.

    Parameters
    ----------
    file:
        ONS file metadata.
    output_dir:
        Directory where the workbook should be stored.
    overwrite:
        Whether to overwrite an existing local copy.
    max_retries:
        Number of additional attempts after the first one fails.
    initial_backoff:
        Seconds to wait before the second attempt; doubles each retry.
    request_delay:
        Polite pause before issuing a fresh download.

    Returns
    -------
    Path
        Local workbook path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / f"{file.edition_year}_{file.filename}"

    if local_path.exists() and not overwrite:
        return local_path

    if request_delay > 0:
        time.sleep(request_delay)

    backoff = initial_backoff
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with requests.get(
                file.url,
                headers=ONS_HTTP_HEADERS,
                stream=True,
                timeout=90,
            ) as response:
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else backoff
                    time.sleep(wait)
                    backoff *= 2
                    continue
                response.raise_for_status()
                with local_path.open("wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_handle.write(chunk)
            return local_path
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in (429, 500, 502, 503, 504):
                raise
            time.sleep(backoff)
            backoff *= 2
        except requests.ConnectionError as exc:
            last_exc = exc
            time.sleep(backoff)
            backoff *= 2

        if attempt == max_retries:
            break

    raise RuntimeError(
        f"Failed to download {file.url} after {max_retries + 1} attempts"
    ) from last_exc


def sha256_file(path: Path) -> str:
    """
    Compute a SHA-256 digest for a local file.

    Parameters
    ----------
    path:
        Local file path.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def register_source_file(engine: Engine, file: ONSFile, local_path: Path) -> int:
    """
    Register a downloaded workbook in `ons_source_files`.

    Parameters
    ----------
    engine:
        SQLAlchemy engine.
    file:
        ONS file metadata.
    local_path:
        Local workbook path.

    Returns
    -------
    int
        Source-file database ID.
    """
    file_hash = sha256_file(local_path)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT IGNORE INTO ons_source_files (
                    edition_year,
                    source_url,
                    local_filename,
                    file_sha256
                )
                VALUES (
                    :edition_year,
                    :source_url,
                    :local_filename,
                    :file_sha256
                )
                """
            ),
            {
                "edition_year": file.edition_year,
                "source_url": file.url,
                "local_filename": local_path.name,
                "file_sha256": file_hash,
            },
        )

        source_id = conn.execute(
            text("SELECT id FROM ons_source_files WHERE file_sha256 = :file_sha256"),
            {"file_sha256": file_hash},
        ).scalar_one()

    return int(source_id)


def iter_downloaded_files(
    files: Iterable[ONSFile],
    output_dir: Path,
    overwrite: bool = False,
) -> Iterable[tuple[ONSFile, Path]]:
    """
    Download several workbooks and yield their local paths.

    Parameters
    ----------
    files:
        Workbooks to download.
    output_dir:
        Raw data directory.
    overwrite:
        Whether to overwrite existing files.
    """
    for file in files:
        yield file, download_file(file=file, output_dir=output_dir, overwrite=overwrite)
