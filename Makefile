SOURCE_DIR = blocksnet

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

install:
	pip install .

venv: #then source .venv/bin/activate
	python -m venv .venv

install-dev:
	pip install -e '.[dev]'

install-docs:
	pip install -e '.[docs]'

build:
	python3.10 -m build .

clean:
	rm -rf ./build ./dist ./blocksnet.egg-info

update-pypi: clean build
	python3 -m twine upload dist/*

update-test-pypi: clean build
	python3 -m twine upload --repository testpypi dist/*

test:
	pytest tests

test-cov:
	pytest tests --cov
