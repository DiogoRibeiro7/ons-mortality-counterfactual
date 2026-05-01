CREATE DATABASE IF NOT EXISTS ons_mortality
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE ons_mortality;

CREATE TABLE IF NOT EXISTS ons_source_files (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    edition_year INT NOT NULL,
    source_url TEXT NOT NULL,
    local_filename VARCHAR(512) NOT NULL,
    file_sha256 CHAR(64) NOT NULL,
    downloaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_source_file_sha256 (file_sha256),
    KEY idx_source_file_edition_year (edition_year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS monthly_deaths (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    source_file_id BIGINT UNSIGNED NOT NULL,
    edition_year INT NOT NULL,
    month_date DATE NOT NULL,
    area_code VARCHAR(64) NULL,
    area_name VARCHAR(255) NOT NULL,
    geography_type VARCHAR(255) NULL,
    deaths INT NOT NULL,
    is_final BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_monthly_deaths_source_file
        FOREIGN KEY (source_file_id)
        REFERENCES ons_source_files(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    UNIQUE KEY uq_monthly_deaths_source_area_month (
        source_file_id,
        month_date,
        area_code,
        area_name,
        geography_type
    ),
    KEY idx_monthly_deaths_month_date (month_date),
    KEY idx_monthly_deaths_area_code (area_code),
    KEY idx_monthly_deaths_area_name (area_name),
    KEY idx_monthly_deaths_edition_year (edition_year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
