.ONESHELL:
SHELL := /bin/bash

.SILENT:

DEFAULT_GOAL := help

.PHONY: help
help: ## Show available make targets
	awk 'BEGIN {FS = ":.*?## "} /^[%a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Create poetry env and install all dependencies
	poetry config virtualenvs.in-project true --local
	poetry env use 3.11
	poetry install
	poetry run pre-commit install

.PHONY: install-all
install-all: ## Create poetry env and install all dependencies (including extras)
	poetry config virtualenvs.in-project true --local
	poetry env use 3.11
	poetry install --all-extras
	poetry run pre-commit install

.PHONY: lock
lock: ## Update poetry lock file
	poetry lock

.PHONY: checks
checks: style-check static-check ## Run all checks

.PHONY: style-check
style-check: ## Run style checks (ruff)
	printf "Style Checking with Ruff\n"
	poetry run ruff check

.PHONY: static-check
static-check: ## Run strict typing checks (mypy)
	printf "Static Checking with Mypy\n"
	poetry run mypy .

.PHONY: restyle
restyle: ## Reformat code with ruff
	poetry run ruff format .
	poetry run ruff check --fix .

.PHONY: requirements
requirements: ## Generate requirements.txt from poetry (excluding dev & test groups)
	poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev,test

.PHONY: tests
tests: ## Run tests
	PYTHONPATH=. poetry run pytest -s

.PHONY: run
run: ## Run evaluation CLI (dev)
	PYTHONPATH=. poetry run python -m evaluate.typer_cli

.PHONY: build
build: ## Build the project wheel
	poetry build --format wheel --clean --output dist