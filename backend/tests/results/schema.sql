-- Evaluation History Database Schema
-- Tracks all RAG evaluation test runs for comparison and analysis

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    git_hash TEXT,
    git_dirty BOOLEAN DEFAULT 0,

    -- PDF metadata
    pdf_filename TEXT NOT NULL,
    pdf_path TEXT NOT NULL,
    pdf_hash TEXT NOT NULL,

    -- Collection info
    collection_name TEXT NOT NULL,
    num_chunks INTEGER,

    -- Configuration (stored as JSON)
    config TEXT NOT NULL,  -- {chunk_size, chunk_overlap, model, etc.}

    -- Evaluation results (stored as JSON)
    aggregate_metrics TEXT NOT NULL,  -- {precision_at_k, recall_at_k, ...}
    breakdowns TEXT,  -- {by_query_type: {...}, by_difficulty: {...}}
    query_results TEXT,  -- Full per-query results

    -- Performance
    total_latency_ms REAL,
    avg_latency_ms REAL,

    -- Notes
    notes TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluation_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_git_hash ON evaluation_runs(git_hash);
CREATE INDEX IF NOT EXISTS idx_pdf_hash ON evaluation_runs(pdf_hash);
