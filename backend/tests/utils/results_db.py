"""
SQLite database for tracking RAG evaluation results over time.

Provides an interface to save, query, and compare evaluation runs.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class EvaluationResultsDB:
    """Manage evaluation results in SQLite database."""

    def __init__(self, db_path: str | Path | None = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
                    Defaults to tests/results/evaluation_history.db
        """
        if db_path is None:
            # Default to tests/results directory
            db_path = Path(__file__).parent.parent / "results" / "evaluation_history.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        schema_path = self.db_path.parent / "schema.sql"

        if schema_path.exists():
            with open(schema_path) as f:
                self.conn.executescript(f.read())
        else:
            # Fallback inline schema
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    git_hash TEXT,
                    git_dirty BOOLEAN DEFAULT 0,
                    pdf_filename TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    pdf_hash TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    num_chunks INTEGER,
                    config TEXT NOT NULL,
                    aggregate_metrics TEXT NOT NULL,
                    breakdowns TEXT,
                    query_results TEXT,
                    total_latency_ms REAL,
                    avg_latency_ms REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluation_runs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_git_hash ON evaluation_runs(git_hash);
                CREATE INDEX IF NOT EXISTS idx_pdf_hash ON evaluation_runs(pdf_hash);
            """)

        self.conn.commit()

    def save_run(
        self,
        pdf_filename: str,
        pdf_path: str,
        pdf_hash: str,
        collection_name: str,
        num_chunks: int,
        config: Dict[str, Any],
        aggregate_metrics: Dict[str, float],
        breakdowns: Dict[str, Any],
        query_results: List[Dict[str, Any]],
        total_latency_ms: float,
        git_hash: Optional[str] = None,
        git_dirty: bool = False,
        notes: Optional[str] = None,
    ) -> int:
        """
        Save an evaluation run to the database.

        Args:
            pdf_filename: Name of the PDF file
            pdf_path: Full path to PDF
            pdf_hash: Hash of PDF content
            collection_name: Qdrant collection name
            num_chunks: Number of chunks created
            config: Configuration dict (chunk_size, model, etc.)
            aggregate_metrics: Aggregate evaluation metrics
            breakdowns: Breakdown by query type and difficulty
            query_results: Full per-query results
            total_latency_ms: Total evaluation time
            git_hash: Current git commit hash
            git_dirty: Whether git working directory has uncommitted changes
            notes: Optional notes about this run

        Returns:
            ID of the inserted run
        """
        timestamp = datetime.utcnow().isoformat()
        avg_latency_ms = aggregate_metrics.get("avg_latency_ms", 0.0)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO evaluation_runs (
                timestamp, git_hash, git_dirty,
                pdf_filename, pdf_path, pdf_hash,
                collection_name, num_chunks,
                config, aggregate_metrics, breakdowns, query_results,
                total_latency_ms, avg_latency_ms, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                git_hash,
                git_dirty,
                pdf_filename,
                pdf_path,
                pdf_hash,
                collection_name,
                num_chunks,
                json.dumps(config),
                json.dumps(aggregate_metrics),
                json.dumps(breakdowns),
                json.dumps(query_results),
                total_latency_ms,
                avg_latency_ms,
                notes,
            ),
        )

        self.conn.commit()
        return cursor.lastrowid

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)
        )
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row)
        return None

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Get the most recent evaluation run."""
        cursor = self.conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row)
        return None

    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent runs."""
        cursor = self.conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_runs_by_git_hash(self, git_hash: str) -> List[Dict[str, Any]]:
        """Get all runs for a specific git commit."""
        cursor = self.conn.execute(
            "SELECT * FROM evaluation_runs WHERE git_hash = ? ORDER BY id DESC",
            (git_hash,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def compare_runs(self, run_id_1: int, run_id_2: int) -> Dict[str, Any]:
        """
        Compare two evaluation runs.

        Args:
            run_id_1: First run ID (baseline)
            run_id_2: Second run ID (comparison)

        Returns:
            Dict with comparison results
        """
        run1 = self.get_run(run_id_1)
        run2 = self.get_run(run_id_2)

        if not run1 or not run2:
            raise ValueError(f"Run {run_id_1} or {run_id_2} not found")

        metrics1 = json.loads(run1["aggregate_metrics"])
        metrics2 = json.loads(run2["aggregate_metrics"])

        diffs = {}
        for key in metrics1:
            if key in metrics2:
                diff = metrics2[key] - metrics1[key]
                pct_change = (diff / metrics1[key] * 100) if metrics1[key] != 0 else 0
                diffs[key] = {
                    "baseline": metrics1[key],
                    "current": metrics2[key],
                    "diff": diff,
                    "pct_change": pct_change,
                }

        return {
            "run1": run1,
            "run2": run2,
            "metric_diffs": diffs,
        }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary with parsed JSON."""
        result = dict(row)

        # Parse JSON fields
        if result.get("config"):
            result["config"] = json.loads(result["config"])
        if result.get("aggregate_metrics"):
            result["aggregate_metrics"] = json.loads(result["aggregate_metrics"])
        if result.get("breakdowns"):
            result["breakdowns"] = json.loads(result["breakdowns"])
        if result.get("query_results"):
            result["query_results"] = json.loads(result["query_results"])

        return result

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
