# Development
format-all:
	@isort . \
		--skip setup.py \
		--skip .venv \
		--skip build \
		--skip dist \
		--skip __pycache__ \
		--skip docs \
		--skip static \
		--skip .conda
	@black . \
		--exclude setup.py \
		--exclude .venv \
		--exclude build \
		--exclude dist \
		--exclude __pycache__ \
		--exclude docs \
		--exclude static \
		--exclude .conda

install-all:
	poetry install --all-extras --all-groups

update-all:
	poetry update
	poetry export --without-hashes -f requirements.txt --output requirements.txt
	poetry export --without-hashes --all-extras --all-groups -f requirements.txt --output requirements-all.txt

mkdocs:
	mkdocs serve

pytest:
	python -m pytest --cov=languru --cov-config=.coveragerc --cov-report=xml:coverage.xml

# Scripts
build-example-datasets:
	PYTHONPATH=. python scripts/build_example_datasets.py
