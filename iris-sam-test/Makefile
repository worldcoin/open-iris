.PHONY: report-coverage html-coverage build-docs format-code check-code check-ruff check-docstring

all: report-coverage html-coverage build-docs format-code check-code check-ruff check-docstring

report-coverage:
	coverage run -m pytest
	coverage report --ignore-errors --show-missing --skip-empty

html-coverage:
	coverage run -m pytest
	coverage html -i
	open ./htmlcov/index.html

build-docs:
	cd docs/source/_code_subpages/; rm -f *.rst
	cd docs; sphinx-apidoc -o ./source/_code_subpages ../src/iris; make clean html

format-code:
	isort src/iris tests
	black src/iris tests docs

check-code: check-ruff check-docstrings

check-ruff:
	@printf "Running check-ruff\n"
	ruff check src/iris scripts tests setup.py
	@printf "\n"

check-docstrings:
	pydocstyle --explain src/iris scripts setup.py
