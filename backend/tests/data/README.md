# Test Data Directory

This directory contains test data files for RAG evaluation.

## Required Files

### PDF Document
- **Name:** `How to Train Guide.pdf` (or your own PDF)
- **Purpose:** Test document for ingestion and evaluation
- **Location:** Place your test PDF in this directory

### Test Suite
- **File:** `my_suite.json`
- **Purpose:** Contains evaluation queries and expected results
- **Status:** ✅ Already present

## Setup

1. **Add your test PDF:**
   ```bash
   cp /path/to/your/document.pdf backend/tests/data/
   ```

2. **Run the test:**
   ```bash
   cd backend
   make test-rag
   ```

   Or with a custom PDF:
   ```bash
   make test-rag PDF=tests/data/your-document.pdf
   ```

## Notes

- PDFs are gitignored by default to avoid large file commits
- The test system uses a dedicated Qdrant collection (`test_rag_evaluation`)
- Results are stored in `tests/results/evaluation_history.db` (also gitignored)
