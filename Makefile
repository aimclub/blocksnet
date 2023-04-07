FILES = get_blocks/*

lint:
	python -m pylint ${FILES}

format:
	python -m black ${FILES}
