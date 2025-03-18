.PHONY: install help


# Generate environment
env:
	poetry run python -m scripts.simulate_env

# Install dependencies
install:
	poetry install

# Runs 'help' by default, if no target is specified
.DEFAULT_GOAL := help
# Help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  install         - Install dependencies"
	@echo "  help            - Display this help message"
	@echo ""
