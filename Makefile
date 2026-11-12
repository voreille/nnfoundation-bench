#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = nnfoundation-bench
TASK ?= kits_liver
DEVICE ?= gpu

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Run tests
.PHONY: test
test:
	uv run pytest

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## List available classification tasks
.PHONY: tasks
tasks:
	uv run nnfb-data list

## Download raw data for TASK (default: kits_liver)
.PHONY: download
download:
	uv run nnfb-data download $(TASK)

## Build the processed dataset for TASK (ROI segmentation, 1mm resampling, 192^3 crop, splits)
.PHONY: data
data: download
	uv run nnfb-data prepare $(TASK) --device $(DEVICE)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
