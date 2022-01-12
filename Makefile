install:
	pip install -r requirements.txt

install-dev: install
	pip install -r dev.requirements.txt
	pre-commit install

lint:
	python -m pylint src
	python -m flake8 src
