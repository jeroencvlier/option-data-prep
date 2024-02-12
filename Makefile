VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PROJECT_ROOT := $(shell pwd)
VENV_EXISTS := $(shell test -d $(VENV) && echo 1 || echo 0)
.PHONY: clean setup install test find-option-chain-data



find-option-chain-data:
	@echo "\n  ##################################################"
	@echo "    Searching for 'option_chain_data' folder..."
	@current_dir=$$(pwd); \
	home_dir=$$(echo ~); \
	while [ "$$current_dir" != "$$home_dir" ] && [ -z "$$option_chain_data_dir" ]; do \
		found=$$(find $$current_dir -type d -name "option_chain_data" -print -quit); \
		if [ -n "$$found" ]; then \
			option_chain_data_dir=$$found; \
			break; \
		else \
			echo "    > Folder not found in $$current_dir"; \
			current_dir=$$(dirname $$current_dir); \
		fi; \
	done; \
	if [ -n "$$option_chain_data_dir" ]; then \
		echo "    > Folder found at $$option_chain_data_dir"; \
		echo "OPTION_DATA_LOCAL_ROOT=$$option_chain_data_dir" >> settings.ini; \
	else \
		echo "    !!!! Folder not found !!!!"; \
	fi;	
	@echo "  ##################################################\n"

init-config:
	@echo "Setting up the configuration file..."
	@if [ -f settings.ini ]; then \
		echo "Existing settings.ini found. Cleaning up..."; \
		rm -f settings.ini; \
	fi
	@echo "[paths]" > settings.ini
	@echo "${VENV}_root=$(PROJECT_ROOT)" >> settings.ini
	@$(MAKE) find-option-chain-data
	@echo "settings.ini created with default paths. Please update the paths according to your environment."


setup:
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists. Skipping creation."; \
	fi


install:
	@echo "Installing dependencies..."
	@echo "Python version: $(shell $(PYTHON) --version)"
	@$(PIP) install --upgrade pip 
	@$(PIP) install --upgrade pytest 
	@$(PIP) install --upgrade ipykernel
	@$(PIP) install --upgrade pipreqs 
	@$(PIP) install --upgrade python-dotenv
	@$(PIP) install -r requirements.txt -qq
	@$(PIP) install -e .
	@echo "Dependencies installed."

dev: setup install init-config
	@echo "To activate the virtual environment, run:"
	@echo ">>>   source $(VENV)/bin/activate   <<<"

test: 
	@echo "Running tests..."
	@$(PYTHON) -m pytest -v

clean: 
	@echo "Cleaning up..."
	@if [ -d "$(VENV)" ]; then rm -rf $(VENV); fi
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@if [ -d "src.egg-info" ]; then rm -rf src.egg-info; fi
	@if [ -d ".pytest_cache" ]; then rm -rf .pytest_cache; fi
	@echo "Clean up complete."

reset: clean
	@sleep 2
	@$(MAKE) setup
	@sleep 2
	@$(MAKE) install 
	@$(MAKE) init-config
	
	@echo "Reset complete."

help:
	@echo "Available commands:"
	@echo "  setup    - Set up the virtual environment"
	@echo "  install  - Install the project and dependencies"
	@echo "  dev      - Set up the development environment"
	@echo "  test     - Run tests"
	@echo "  train    - Train model"
	@echo "  clean    - Clean up the project"
	@echo "  help     - Show this message"