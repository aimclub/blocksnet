SOURCE_DIR = masterplan_tools

lint:
	python3 -m pylint ${SOURCE_DIR}

format:
	python3 -m black ${SOURCE_DIR}

install:
	python3 -m pip install .

install-dev:
	python3 -m pip install -e '.[dev]' --config-settings editable_mode=strict

build:
	python3 -m build .

clean:
	rm -rf ./build ./dist ./masterplan_tools.egg-info

udpate-pypi: clean build
	python3 -m twine upload dist/*

install-from-build:
	python3 -m wheel install dist/masterplan_tools-*.whl
