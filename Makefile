# Development
format-all:
	isort . --skip setup.py && black --exclude setup.py .

install-all:
	poetry install -E all --with dev

update-all:
	poetry update && \
		poetry export --without-hashes -E all -f requirements.txt --output requirements.txt && \
		poetry export --without-hashes -E all --with dev -f requirements.txt --output requirements-all.txt

mkdocs:
	mkdocs serve

pytest:
	python -m pytest --cov=languru --cov-config=.coveragerc --cov-report=xml:coverage.xml

# Server
run-server:
	fastapi run duckdb_vss_api.py

run-server-dev:
	fastapi dev duckdb_vss_api.py
