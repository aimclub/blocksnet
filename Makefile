SOURCE_DIR = blocksnet

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

install:
	pip install .

install-dev:
	python3 -m pip install -e '.[dev]' --config-settings editable_mode=strict

build:
	python3 -m build .

clean:
	rm -rf ./build ./dist ./blocksnet.egg-info

udpate-pypi: clean build
	python3 -m twine upload dist/*

install-from-build:
	python3 -m wheel install dist/blocksnet-*.whl
