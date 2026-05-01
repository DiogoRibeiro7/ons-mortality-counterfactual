.PHONY: up down install init-db discover ingest export plot test lint typecheck

up:
	docker compose up -d

down:
	docker compose down

install:
	poetry install

init-db:
	poetry run ons-mortality init-db

discover:
	poetry run ons-mortality discover --start-year 2006 --end-year 2022

ingest:
	poetry run ons-mortality ingest --start-year 2006 --end-year 2022

export:
	poetry run ons-mortality export-national --output data/processed/england_wales_monthly_deaths.csv

plot:
	poetry run python scripts/make_counterfactual_plot.py

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src
