# Makefile

.PHONY: help
help:
	@echo "Commands:"
	@echo "install            : installs requirements in existing environment"
	@echo "env                : set up/activate the virtual environment"
	@echo "devenv             : set up/activate the virtual environment with dev packages"
	@echo "style              : runs style formatting"
	@echo "clean              : cleans all unecessary files"
	@echo "docs               : serve generated documentation"

# Installation
.PHONY: install
install:
	python -m pip install -U pip
	python -m pip install .

# Installation
.PHONY: env
install:
	python -m pip install -U pip
	python -m pip install poetry
	python -m poetry shell
	python -m poetry install --no-dev

# Installation
.PHONY: devenv
install:
	python -m pip install -U pip
	python -m pip install poetry
	python -m poetry shell
	python -m poetry install
	pre-commit install

# Styling
.PHONY: style
style:
	black .
	flake8
	isort .

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

# Documentation
.PHONY: docs
docs:
	python -m mkdocs serve
