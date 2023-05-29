SOURCE_DIR = masterplan_tools

lint:
	python -m pylint ${SOURCE_DIR}

format:
	python -m black ${SOURCE_DIR}

install:
	python -m pip install .

install-dev:
	python -m pip install -e . --config-settings editable_mode=strict

build:
	python -m build .

clean:
	rm -rf ./build ./dist ./masterplan_tools.egg-info

udpate-pypi: clean build
	python -m twine upload  dist/*

install-from-build:
	python -m wheel install dist/masterplan_tools-*.whl
