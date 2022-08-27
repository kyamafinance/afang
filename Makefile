VENV := venv
PIP = $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python3

ARGS ?=

all: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt requirements-no-deps.txt
	python3 -m venv $(VENV)
	@# uninstall packages that were installed without dependencies to prevent conflicts.
	./$(PIP) uninstall -y -r requirements-no-deps.txt
	./$(PIP) install -r requirements.txt
	@# install packages without dependencies.
	./$(PIP) install --no-deps -r requirements-no-deps.txt

setup: $(VENV)/bin/activate

run: $(VENV)/bin/activate
	./$(PYTHON) -m afang $(ARGS)

test: $(VENV)/bin/activate
	./$(PYTHON) -m pytest --cov=afang/ tests/

clean:
	rm -rf $(VENV)/
	find . -type f -name '*.pyc' -delete

.PHONY: all setup run test clean
