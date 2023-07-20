SOURCE_DIR = masterplan_tools

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

install:
	pip install .

install-dev:
	pip install -e . --config-settings editable_mode=strict

build:
	python3 -m build .

clean:
	rm -rf ./build ./dist ./masterplan_tools.egg-info

udpate-pypi: clean build
	python3 -m twine upload dist/*

install-from-build:
	python3 -m wheel install dist/masterplan_tools-*.whl
