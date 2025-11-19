"""
MCP Analytics Server - Query analytics, performance metrics, and insights.
"""
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    class FastMCP:
        def __init__(self, name):
            self.name = name
            print(f"Warning: MCP package not available. {name} server will not function.")

        def tool(self):
            def decorator(func):
                return func
            return decorator

        def run(self, transport="stdio"):
            print(f"Warning: Cannot run {self.name} - MCP package not installed")
            pass

    MCP_AVAILABLE = False

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter
import statistics

mcp = FastMCP("Analytics")

# Storage path
STORAGE_PATH = Path("mcp_data/analytics")
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
DB_PATH = STORAGE_PATH / "analytics.db"


def _init_db():
    """Initialize analytics database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Query logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT,
            query TEXT NOT NULL,
            collection_name TEXT,
            response_time_ms INTEGER,
            num_documents_retrieved INTEGER,
            num_sources_used INTEGER,
            grounding_check_passed BOOLEAN,
            web_search_used BOOLEAN,
            answer_length INTEGER,
            user_feedback TEXT,
            metadata TEXT
        )
    ''')

    # Performance metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            metric_value REAL NOT NULL,
            collection_name TEXT,
            metadata TEXT
        )
    ''')

    # Popular queries table (for caching and optimization)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS popular_queries (
            query_hash TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            last_seen TEXT NOT NULL,
            avg_response_time REAL
        )
    ''')

    # Create indices
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON query_logs(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON query_logs(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection ON query_logs(collection_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_type ON performance_metrics(metric_type)')

    conn.commit()
    conn.close()


_init_db()


@mcp.tool()
async def log_query(
    query: str,
    session_id: str = None,
    collection_name: str = None,
    response_time_ms: int = None,
    num_documents_retrieved: int = None,
    num_sources_used: int = None,
    grounding_check_passed: bool = None,
    web_search_used: bool = False,
    answer_length: int = None,
    metadata: dict = None
) -> dict:
    """
    Log a RAG query for analytics.

    Args:
        query: The user's query
        session_id: Session identifier
        collection_name: Collection queried
        response_time_ms: Response time in milliseconds
        num_documents_retrieved: Number of documents retrieved
        num_sources_used: Number of sources used in answer
        grounding_check_passed: Whether grounding check passed
        web_search_used: Whether web search was used
        answer_length: Length of generated answer
        metadata: Additional metadata

    Returns:
        dict with success status and query ID
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO query_logs (
                timestamp, session_id, query, collection_name,
                response_time_ms, num_documents_retrieved, num_sources_used,
                grounding_check_passed, web_search_used, answer_length, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, session_id, query, collection_name,
            response_time_ms, num_documents_retrieved, num_sources_used,
            grounding_check_passed, web_search_used, answer_length,
            json.dumps(metadata) if metadata else None
        ))

        query_id = cursor.lastrowid

        # Update popular queries
        import hashlib
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()

        cursor.execute('SELECT count, avg_response_time FROM popular_queries WHERE query_hash = ?', (query_hash,))
        result = cursor.fetchone()

        if result:
            count, avg_rt = result
            new_count = count + 1
            new_avg_rt = ((avg_rt * count) + response_time_ms) / new_count if response_time_ms else avg_rt

            cursor.execute('''
                UPDATE popular_queries
                SET count = ?, last_seen = ?, avg_response_time = ?
                WHERE query_hash = ?
            ''', (new_count, timestamp, new_avg_rt, query_hash))
        else:
            cursor.execute('''
                INSERT INTO popular_queries (query_hash, query, count, last_seen, avg_response_time)
                VALUES (?, ?, 1, ?, ?)
            ''', (query_hash, query, timestamp, response_time_ms or 0))

        conn.commit()
        conn.close()

        return {"success": True, "query_id": query_id, "timestamp": timestamp}

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_query_analytics(
    time_period: str = "24h",
    collection_name: str = None,
    session_id: str = None
) -> dict:
    """
    Get analytics for queries over a time period.

    Args:
        time_period: Time period (e.g., "1h", "24h", "7d", "30d")
        collection_name: Filter by collection
        session_id: Filter by session

    Returns:
        dict with analytics data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Parse time period
        period_map = {"1h": 1/24, "24h": 1, "7d": 7, "30d": 30}
        days = period_map.get(time_period, 1)
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Build query
        where_clauses = ["timestamp >= ?"]
        params = [cutoff]

        if collection_name:
            where_clauses.append("collection_name = ?")
            params.append(collection_name)

        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)

        where_clause = " AND ".join(where_clauses)

        # Get metrics
        cursor.execute(f'''
            SELECT
                COUNT(*) as total_queries,
                AVG(response_time_ms) as avg_response_time,
                AVG(num_documents_retrieved) as avg_docs_retrieved,
                AVG(num_sources_used) as avg_sources_used,
                SUM(CASE WHEN web_search_used = 1 THEN 1 ELSE 0 END) as web_searches,
                SUM(CASE WHEN grounding_check_passed = 1 THEN 1 ELSE 0 END) as grounding_passed,
                AVG(answer_length) as avg_answer_length
            FROM query_logs
            WHERE {where_clause}
        ''', params)

        row = cursor.fetchone()

        analytics = {
            "time_period": time_period,
            "total_queries": row[0] or 0,
            "avg_response_time_ms": round(row[1], 2) if row[1] else 0,
            "avg_documents_retrieved": round(row[2], 2) if row[2] else 0,
            "avg_sources_used": round(row[3], 2) if row[3] else 0,
            "web_searches": row[4] or 0,
            "grounding_checks_passed": row[5] or 0,
            "avg_answer_length": round(row[6], 2) if row[6] else 0
        }

        # Get top queries
        cursor.execute(f'''
            SELECT query, COUNT(*) as count
            FROM query_logs
            WHERE {where_clause}
            GROUP BY query
            ORDER BY count DESC
            LIMIT 10
        ''', params)

        analytics["top_queries"] = [
            {"query": row[0], "count": row[1]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return analytics

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_popular_queries(limit: int = 10) -> dict:
    """
    Get most popular queries for caching optimization.

    Args:
        limit: Number of queries to return

    Returns:
        dict with popular queries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT query, count, last_seen, avg_response_time
            FROM popular_queries
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,))

        queries = [
            {
                "query": row[0],
                "count": row[1],
                "last_seen": row[2],
                "avg_response_time_ms": row[3]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return {"popular_queries": queries, "count": len(queries)}

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def log_performance_metric(
    metric_type: str,
    metric_value: float,
    collection_name: str = None,
    metadata: dict = None
) -> dict:
    """
    Log a performance metric.

    Args:
        metric_type: Type of metric (e.g., "ingestion_time", "indexing_time", "cache_hit_rate")
        metric_value: Numeric value
        collection_name: Associated collection
        metadata: Additional metadata

    Returns:
        dict with success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO performance_metrics (timestamp, metric_type, metric_value, collection_name, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, metric_type, metric_value, collection_name, json.dumps(metadata) if metadata else None))

        conn.commit()
        conn.close()

        return {"success": True, "timestamp": timestamp}

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_performance_trends(
    metric_type: str,
    time_period: str = "24h"
) -> dict:
    """
    Get performance trends over time.

    Args:
        metric_type: Type of metric to analyze
        time_period: Time period (e.g., "1h", "24h", "7d")

    Returns:
        dict with trend data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Parse time period
        period_map = {"1h": 1/24, "24h": 1, "7d": 7, "30d": 30}
        days = period_map.get(time_period, 1)
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT metric_value, timestamp
            FROM performance_metrics
            WHERE metric_type = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        ''', (metric_type, cutoff))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"error": "No data found for this metric"}

        values = [row[0] for row in rows]
        timestamps = [row[1] for row in rows]

        return {
            "metric_type": metric_type,
            "time_period": time_period,
            "data_points": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "recent_trend": "improving" if values[-1] < statistics.mean(values) else "declining",
            "timestamps": timestamps[-10:],  # Last 10 timestamps
            "values": values[-10:]  # Last 10 values
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    if MCP_AVAILABLE:
        mcp.run()
    else:
        print("MCP package not installed. Please install with: pip install mcp")
