import pytest
import sqlite3
from mcp.sql_server import _sanitize_identifier, _quote_identifier


def test_sanitize_identifier_valid():
    """Test valid identifiers pass sanitization."""
    assert _sanitize_identifier("users") == "users"
    assert _sanitize_identifier("user_accounts") == "user_accounts"
    assert _sanitize_identifier("_private") == "_private"
    assert _sanitize_identifier("Table123") == "Table123"


def test_sanitize_identifier_invalid():
    """Test invalid identifiers are rejected."""
    with pytest.raises(ValueError):
        _sanitize_identifier("users; DROP TABLE users--")
    
    with pytest.raises(ValueError):
        _sanitize_identifier("users OR 1=1")
    
    with pytest.raises(ValueError):
        _sanitize_identifier("users'--")
    
    with pytest.raises(ValueError):
        _sanitize_identifier("123invalid")  # Can't start with number


def test_quote_identifier():
    """Test identifier quoting escapes properly."""
    assert _quote_identifier("users") == '"users"'
    assert _quote_identifier('user"table') == '"user""table"'  # Escaped quote
    assert _quote_identifier("user's") == '"user\'s"'  # Single quote OK


@pytest.mark.asyncio
async def test_query_validation_prevents_injection(tmp_path):
    """Test that malicious queries are rejected."""
    from mcp.sql_server import query_structured_data
    
    # Create temp database FILE (not :memory:)
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO test VALUES (1, 'test')")
    conn.commit()
    conn.close()
    
    malicious_queries = [
        "SELECT * FROM test; DROP TABLE test;",  # Query stacking
        "SELECT * FROM test WHERE 1=1--",  # Comment
        "SELECT * FROM test /* comment */ WHERE 1=1",  # Block comment
        "DROP TABLE test",  # Not a SELECT
        "DELETE FROM test",  # Not a SELECT
        "UPDATE test SET name='hacked'",  # Not a SELECT
    ]
    
    for query in malicious_queries:
        result = await query_structured_data(query, str(db_path), validate_query=True)
        assert not result["success"], f"Query should be rejected: {query}"
        assert "error" in result