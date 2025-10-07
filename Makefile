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
	poetry install --with dev

.PHONY: install-all
install-all: ## Create poetry environment and install all dependencies.
	poetry config virtualenvs.in-project true --local
	poetry env use 3.12.3
	poetry install --all-extras

.PHONY: lock
lock: 
	poetry lock

.PHONY: checks
checks: style-check static-check

.PHONY: style-check
style-check: ## Run style checks.
	printf "Style Checking with Ruff\n"
	poetry run ruff check

.PHONY: static-check
static-check: ## Run strict typing checks.
	printf "Static Checking with Mypy\n"
	poetry run mypy .

.PHONY: restyle
restyle: ## Reformat code with ruff.
	poetry run ruff format .
	poetry run ruff check --fix .

.PHONY: requirements
requirements: 
	poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev

.PHONY: tests
tests: 
	PYTHONPATH=. poetry run pytest -s tests

.PHONY: build
build: 
	poetry build --format wheel --clean --output dist
	