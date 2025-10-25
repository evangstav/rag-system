.PHONY: help test-rag test-rag-force test-rag-history test-rag-compare test-rag-clean

# Default target
help:
	@echo "RAG System - Make Commands"
	@echo "=========================="
	@echo ""
	@echo "Testing:"
	@echo "  make test-rag           - Run end-to-end RAG evaluation test"
	@echo "  make test-rag-force     - Force re-ingestion and run test"
	@echo "  make test-rag-history   - Show recent test run history"
	@echo "  make test-rag-compare   - Compare two test runs (usage: make test-rag-compare RUNS='4 5')"
	@echo "  make test-rag-clean     - Delete test collection from Qdrant"
	@echo ""
	@echo "Examples:"
	@echo "  make test-rag                    # Run test with default PDF"
	@echo "  make test-rag PDF=tests/data/custom.pdf  # Use custom PDF"
	@echo "  make test-rag-compare RUNS='10 11'       # Compare runs 10 and 11"
	@echo ""

# Configuration
PDF ?= tests/data/How to Train Guide.pdf
SUITE ?= tests/data/my_suite.json
K ?= 5
PYTHON ?= python3

# Run end-to-end RAG evaluation test
test-rag:
	@echo "Running end-to-end RAG evaluation test..."
	@cd $(shell pwd) && $(PYTHON) scripts/e2e_rag_test.py \
		--pdf "$(PDF)" \
		--suite "$(SUITE)" \
		--k $(K)

# Force re-ingestion and run test
test-rag-force:
	@echo "Running test with forced re-ingestion..."
	@cd $(shell pwd) && $(PYTHON) scripts/e2e_rag_test.py \
		--pdf "$(PDF)" \
		--suite "$(SUITE)" \
		--k $(K) \
		--force-ingest

# Show recent test run history
test-rag-history:
	@$(PYTHON) scripts/e2e_rag_test.py --history

# Compare two test runs
test-rag-compare:
ifndef RUNS
	@echo "Error: Please specify RUNS='id1 id2'"
	@echo "Example: make test-rag-compare RUNS='4 5'"
	@exit 1
endif
	@$(PYTHON) scripts/e2e_rag_test.py --compare $(RUNS)

# Clean test collection
test-rag-clean:
	@echo "Warning: This will delete the test collection from Qdrant"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) scripts/e2e_rag_test.py --clean; \
	else \
		echo "Cancelled."; \
	fi
