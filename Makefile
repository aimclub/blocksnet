SOURCE_DIR = blocksnet

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

venv: #then source .venv/bin/activate
	python3 -m venv .venv

install:
	pip install .

install-dev:
	pip install -e '.[dev]'

install-docs:
	pip install -e '.[docs]'

tests:
	pytest tests

tests-cov:
	pytest tests --cov