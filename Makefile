.ONESHELL:
SHELL := /bin/bash

.SILENT:

DEFAULT_GOAL := help

.PHONY: help
help: 
	awk 'BEGIN {FS = ":.*?## "} /^[%a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install:
	poetry config virtualenvs.in-project true --local
	poetry env use 3.12
	poetry install --with dev,test
	poetry run pre-commit install

.PHONY: install-all
install-all:
	poetry config virtualenvs.in-project true --local
	poetry env use 3.12
	poetry install --with dev,test
	poetry run pre-commit install

.PHONY: lock
lock: 
	poetry lock

.PHONY: checks
checks: style-check static-check

.PHONY: style-check
style-check: 
	poetry run ruff check evaluate tests

.PHONY: static-check
static-check: 
	poetry run mypy evaluate tests

.PHONY: restyle
restyle: 
	poetry run ruff format .
	poetry run ruff check --fix .

fix-style:
	poetry run ruff check --fix

.PHONY: requirements
requirements: 
	poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev,test

.PHONY: tests
tests: 
	PYTHONPATH=. poetry run pytest -s tests

.PHONY: run
run: 
	PYTHONPATH=. poetry run python -m evaluate.typer_cli

.PHONY: build
build: 
	poetry build --format wheel --clean --output dist
	