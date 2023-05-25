FILES = masterplan_tools/*

lint:
	python -m pylint ${FILES}

format:
	python -m black ${FILES}

