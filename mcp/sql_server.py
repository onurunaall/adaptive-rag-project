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
  
import sqlite3
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re

mcp = FastMCP("SQLRAG")

# Database connection pool
db_connections = {}

# Allowed SQL operations for safety
ALLOWED_SQL_KEYWORDS = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 
                        'ON', 'AND', 'OR', 'ORDER', 'BY', 'GROUP', 'HAVING', 
                        'LIMIT', 'OFFSET', 'AS', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN'}

def validate_table_name(cursor, table_name: str) -> bool:
    """Validate that a table name exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
        (table_name,)
    )
    return cursor.fetchone() is not None

def validate_column_names(cursor, table_name: str, column_names: List[str]) -> bool:
    """Validate that column names exist in the specified table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    valid_columns = {col[1] for col in cursor.fetchall()}
    return all(col in valid_columns for col in column_names)

@mcp.tool()
async def query_structured_data(
    query: str, 
    db_path: str,
    max_rows: int = 100,
    validate_query: bool = True
) -> dict:
    """
    Execute SQL query with security validation.
    
    Args:
        query: SQL query to execute
        db_path: Path to SQLite database
        max_rows: Maximum number of rows to return
        validate_query: Whether to validate query for safety
    
    Returns:
        Query results with column names
    """
    try:
        # Basic SQL injection prevention
        if validate_query:
            query_upper = query.upper().strip()
            
            # Only allow SELECT queries
            if not query_upper.startswith('SELECT'):
                return {
                    "success": False,
                    "error": "Only SELECT queries are allowed"
                }
            
            # Check for dangerous keywords
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'EXEC', 'EXECUTE']
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return {
                        "success": False,
                        "error": f"Query contains forbidden keyword: {keyword}"
                    }
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Execute query with LIMIT to prevent excessive data retrieval
        if 'LIMIT' not in query.upper():
            query = f"{query} LIMIT {max_rows}"
        
        cursor.execute(query)
        
        # Fetch results
        rows = cursor.fetchall()
        
        if rows:
            # Convert to list of dicts
            results = [dict(row) for row in rows]
            columns = list(results[0].keys()) if results else []
        else:
            results = []
            columns = []
        
        conn.close()
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "columns": columns,
            "row_count": len(results),
            "truncated": len(results) == max_rows
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": f"SQL Error: {str(e)}",
            "query": query
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def get_database_schema(db_path: str) -> dict:
    """
    Get complete database schema for better query generation.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Database schema information
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {"tables": {}}
        
        for table in tables:
            # Validate table name (already from sqlite_master, so safe)
            # Get table info using parameterized query where possible
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get row count safely
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = [row[1] for row in cursor.fetchall()]
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            foreign_keys = [
                {"column": row[3], "ref_table": row[2], "ref_column": row[4]}
                for row in cursor.fetchall()
            ]
            
            schema["tables"][table] = {
                "columns": [
                    {
                        "name": col[1],
                        "type": col[2],
                        "nullable": not col[3],
                        "default": col[4],
                        "primary_key": bool(col[5])
                    }
                    for col in columns
                ],
                "row_count": row_count,
                "indexes": indexes,
                "foreign_keys": foreign_keys
            }
        
        conn.close()
        
        return {
            "success": True,
            "database": db_path,
            "schema": schema,
            "table_count": len(tables)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def get_table_sample(
    db_path: str,
    table_name: str,
    sample_size: int = 5
) -> dict:
    """
    Get sample rows from a table with validation.
    
    Args:
        db_path: Path to SQLite database
        table_name: Name of the table
        sample_size: Number of sample rows
    
    Returns:
        Sample data from the table
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Validate table name
        if not validate_table_name(cursor, table_name):
            conn.close()
            return {
                "success": False,
                "error": f"Table '{table_name}' does not exist"
            }
        
        # Safe query since table name is validated
        query = f"SELECT * FROM {table_name} LIMIT ?"
        cursor.execute(query, (sample_size,))
        
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Convert to list of dicts
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        conn.close()
        
        return {
            "success": True,
            "table": table_name,
            "sample_data": results,
            "columns": columns
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def natural_language_to_sql(
    question: str,
    db_path: str,
    context: Optional[str] = None,
    use_llm: bool = False,
    llm_model: Optional[str] = None
) -> dict:
    """
    Convert natural language to SQL with improved pattern matching.
    For production, integrate with LLM for better conversion.
    
    Args:
        question: Natural language question
        db_path: Path to database
        context: Additional context for query generation
        use_llm: Whether to use LLM for conversion (requires setup)
        llm_model: LLM model to use
    
    Returns:
        Generated SQL query and explanation
    """
    try:
        # Get schema first
        schema_result = await get_database_schema(db_path)
        
        if not schema_result["success"]:
            return schema_result
        
        schema = schema_result["schema"]
        
        if not schema["tables"]:
            return {
                "success": False,
                "error": "No tables found in database"
            }
        
        question_lower = question.lower()
        
        # Find relevant table based on question keywords
        relevant_table = None
        relevant_score = 0
        
        for table_name in schema["tables"].keys():
            table_lower = table_name.lower()
            # Score based on matching words
            score = 0
            if table_lower in question_lower:
                score += 10
            
            # Check column names
            for col in schema["tables"][table_name]["columns"]:
                if col["name"].lower() in question_lower:
                    score += 5
            
            if score > relevant_score:
                relevant_score = score
                relevant_table = table_name
        
        if not relevant_table:
            relevant_table = list(schema["tables"].keys())[0]
        
        # Extract conditions and operations
        columns = schema["tables"][relevant_table]["columns"]
        column_names = [col["name"] for col in columns]
        
        # Build SQL based on patterns
        sql = None
        explanation = None
        
        # COUNT patterns
        if any(word in question_lower for word in ["count", "how many", "number of", "total"]):
            if "group by" in question_lower or "per" in question_lower or "by" in question_lower:
                # Try to find grouping column
                group_col = None
                for col in column_names:
                    if col.lower() in question_lower and col.lower() not in ["id", "created", "updated"]:
                        group_col = col
                        break
                
                if group_col:
                    sql = f"SELECT {group_col}, COUNT(*) as count FROM {relevant_table} GROUP BY {group_col}"
                    explanation = f"Counting rows grouped by {group_col}"
                else:
                    sql = f"SELECT COUNT(*) as total_count FROM {relevant_table}"
                    explanation = f"Counting total rows in {relevant_table}"
            else:
                sql = f"SELECT COUNT(*) as total_count FROM {relevant_table}"
                explanation = f"Counting total rows in {relevant_table}"
        
        # AVERAGE/SUM/MAX/MIN patterns
        elif any(word in question_lower for word in ["average", "avg", "mean"]):
            numeric_cols = [col["name"] for col in columns if "INT" in col["type"] or "REAL" in col["type"]]
            if numeric_cols:
                col = numeric_cols[0]
                sql = f"SELECT AVG({col}) as average_{col} FROM {relevant_table}"
                explanation = f"Calculating average of {col}"
        
        elif any(word in question_lower for word in ["sum", "total"]):
            numeric_cols = [col["name"] for col in columns if "INT" in col["type"] or "REAL" in col["type"]]
            if numeric_cols:
                col = numeric_cols[0]
                sql = f"SELECT SUM({col}) as total_{col} FROM {relevant_table}"
                explanation = f"Calculating sum of {col}"
        
        elif any(word in question_lower for word in ["maximum", "max", "highest", "largest"]):
            numeric_cols = [col["name"] for col in columns if "INT" in col["type"] or "REAL" in col["type"]]
            if numeric_cols:
                col = numeric_cols[0]
                sql = f"SELECT MAX({col}) as max_{col}, * FROM {relevant_table} ORDER BY {col} DESC LIMIT 1"
                explanation = f"Finding maximum {col} value"
        
        elif any(word in question_lower for word in ["minimum", "min", "lowest", "smallest"]):
            numeric_cols = [col["name"] for col in columns if "INT" in col["type"] or "REAL" in col["type"]]
            if numeric_cols:
                col = numeric_cols[0]
                sql = f"SELECT MIN({col}) as min_{col}, * FROM {relevant_table} ORDER BY {col} ASC LIMIT 1"
                explanation = f"Finding minimum {col} value"
        
        # TIME-based patterns
        elif any(word in question_lower for word in ["latest", "recent", "newest", "last"]):
            date_cols = [col["name"] for col in columns 
                        if any(t in col["name"].lower() for t in ["date", "time", "created", "updated", "timestamp"])]
            if date_cols:
                date_col = date_cols[0]
                sql = f"SELECT * FROM {relevant_table} ORDER BY {date_col} DESC LIMIT 10"
                explanation = f"Getting latest records ordered by {date_col}"
            else:
                sql = f"SELECT * FROM {relevant_table} LIMIT 10"
                explanation = f"Getting sample records from {relevant_table}"
        
        elif any(word in question_lower for word in ["oldest", "earliest", "first"]):
            date_cols = [col["name"] for col in columns 
                        if any(t in col["name"].lower() for t in ["date", "time", "created", "updated", "timestamp"])]
            if date_cols:
                date_col = date_cols[0]
                sql = f"SELECT * FROM {relevant_table} ORDER BY {date_col} ASC LIMIT 10"
                explanation = f"Getting oldest records ordered by {date_col}"
        
        # FILTER patterns
        elif "where" in question_lower or "filter" in question_lower or "only" in question_lower:
            # Try to extract filter conditions
            sql = f"SELECT * FROM {relevant_table} WHERE 1=1 LIMIT 20"
            explanation = f"Filtered query on {relevant_table} (add specific conditions)"
        
        # Default: SELECT all with limit
        else:
            sql = f"SELECT * FROM {relevant_table} LIMIT 20"
            explanation = f"Getting sample data from {relevant_table}"
        
        return {
            "success": True,
            "question": question,
            "generated_sql": sql,
            "explanation": explanation,
            "table_used": relevant_table,
            "available_columns": column_names,
            "note": "For production use, integrate with LLM for better SQL generation"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def create_rag_context_from_db(
    db_path: str,
    tables: List[str] = None,
    include_schema: bool = True,
    include_sample: bool = True,
    sample_size: int = 3
) -> dict:
    """
    Create comprehensive context from database for RAG.
    
    Args:
        db_path: Path to database
        tables: Specific tables to include (None for all)
        include_schema: Whether to include schema information
        include_sample: Whether to include sample data
        sample_size: Number of sample rows per table
    
    Returns:
        Structured context for RAG ingestion
    """
    try:
        context_data = {
            "database": db_path,
            "timestamp": datetime.now().isoformat(),
            "tables": {}
        }
        
        # Get schema
        schema_result = await get_database_schema(db_path)
        if not schema_result["success"]:
            return schema_result
        
        schema = schema_result["schema"]
        
        # Filter tables if specified
        tables_to_process = tables if tables else list(schema["tables"].keys())
        
        # Validate requested tables exist
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for table_name in tables_to_process:
            if not validate_table_name(cursor, table_name):
                conn.close()
                return {
                    "success": False,
                    "error": f"Table '{table_name}' does not exist"
                }
        
        conn.close()
        
        for table_name in tables_to_process:
            if table_name not in schema["tables"]:
                continue
            
            table_context = {
                "name": table_name,
                "row_count": schema["tables"][table_name]["row_count"]
            }
            
            if include_schema:
                table_context["schema"] = schema["tables"][table_name]
            
            if include_sample and schema["tables"][table_name]["row_count"] > 0:
                sample_result = await get_table_sample(db_path, table_name, sample_size)
                if sample_result["success"]:
                    table_context["sample_data"] = sample_result["sample_data"]
            
            context_data["tables"][table_name] = table_context
        
        # Generate text summary for RAG
        text_summary = f"Database: {db_path}\n"
        text_summary += f"Tables: {len(context_data['tables'])}\n\n"
        
        for table_name, table_info in context_data["tables"].items():
            text_summary += f"Table: {table_name}\n"
            text_summary += f"Rows: {table_info['row_count']}\n"
            
            if "schema" in table_info:
                text_summary += "Columns:\n"
                for col in table_info["schema"]["columns"]:
                    pk = " (PRIMARY KEY)" if col.get("primary_key") else ""
                    text_summary += f"  - {col['name']} ({col['type']}){pk}\n"
                
                if table_info["schema"].get("foreign_keys"):
                    text_summary += "Foreign Keys:\n"
                    for fk in table_info["schema"]["foreign_keys"]:
                        text_summary += f"  - {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}\n"
            
            if "sample_data" in table_info and table_info["sample_data"]:
                text_summary += f"Sample data (first row): {table_info['sample_data'][0]}\n"
            
            text_summary += "\n"
        
        return {
            "success": True,
            "context_data": context_data,
            "text_summary": text_summary,
            "table_count": len(context_data["tables"])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def execute_batch_queries(
    queries: List[str],
    db_path: str,
    stop_on_error: bool = False
) -> dict:
    """
    Execute multiple SQL queries in sequence with validation.
    
    Args:
        queries: List of SQL queries
        db_path: Path to database
        stop_on_error: Whether to stop execution on first error
    
    Returns:
        Results of all queries
    """
    try:
        results = []
        
        for i, query in enumerate(queries):
            result = await query_structured_data(query, db_path, validate_query=True)
            result["query_index"] = i
            results.append(result)
            
            if not result["success"] and stop_on_error:
                break
        
        success_count = sum(1 for r in results if r.get("success"))
        
        return {
            "success": True,
            "total_queries": len(queries),
            "successful_queries": success_count,
            "failed_queries": len(queries) - success_count,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if MCP_AVAILABLE:
            mcp.run(transport="stdio")
    else:
        print("MCP package not available. Server cannot run.")
        print("Install with: pip install mcp>=1.6.0")
