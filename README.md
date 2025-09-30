make install
make checks
make tests
make run
make requirements
make build
make lock
make restyle
make style-check
make static-check

poetry config virtualenvs.in-project true --local
poetry env use 3.11
poetry install --with dev,test
poetry run pre-commit install
poetry run ruff check
poetry run mypy .
PYTHONPATH=. poetry run pytest -s
PYTHONPATH=. poetry run python -m evaluate.typer_cli
poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev,test
poetry build --format wheel --clean --output dist
