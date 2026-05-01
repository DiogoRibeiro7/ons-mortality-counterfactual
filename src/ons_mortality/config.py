"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the local MySQL database."""

    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def sqlalchemy_url(self) -> str:
        """Build a SQLAlchemy connection URL for PyMySQL."""
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )


@dataclass(frozen=True)
class PathConfig:
    """Local file-system paths used by the project."""

    raw_dir: Path
    processed_dir: Path
    figures_dir: Path


def load_database_config() -> DatabaseConfig:
    """
    Load MySQL configuration from environment variables.

    Returns
    -------
    DatabaseConfig
        Validated connection configuration.
    """
    load_dotenv()

    port_raw = os.environ.get("MYSQL_PORT", "3306")

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError("MYSQL_PORT must be an integer.") from exc

    return DatabaseConfig(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=port,
        database=os.environ.get("MYSQL_DATABASE", "ons_mortality"),
        user=os.environ.get("MYSQL_USER", "ons_user"),
        password=os.environ.get("MYSQL_PASSWORD", "ons_password"),
    )


def load_path_config() -> PathConfig:
    """
    Load local directory configuration from environment variables.

    Returns
    -------
    PathConfig
        Project directory paths.
    """
    load_dotenv()

    return PathConfig(
        raw_dir=Path(os.environ.get("ONS_RAW_DIR", "data/raw")),
        processed_dir=Path(os.environ.get("ONS_PROCESSED_DIR", "data/processed")),
        figures_dir=Path(os.environ.get("FIGURES_DIR", "figures")),
    )
