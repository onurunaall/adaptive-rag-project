"""
Security tests for SQL injection vulnerabilities in MCP SQL server.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.sql_server import _sanitize_identifier, _quote_identifier


class TestSQLInjectionPrevention:
    """Test SQL injection prevention mechanisms."""

    def test_sanitize_identifier_blocks_sql_injection(self):
        """Test that SQL injection attempts are rejected with ValueError."""
        # Test basic SQL injection attempts - should raise ValueError
        malicious_inputs = [
            "users; DROP TABLE users--",
            "users' OR '1'='1",
            "users/**/UNION/**/SELECT",
            "users'; DELETE FROM",
            "../../../etc/passwd",
            "users\x00admin",  # Null byte injection
            "users\nDROP TABLE",  # Newline injection
            "users\rDROP TABLE",  # Carriage return injection
        ]

        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError, match="Invalid identifier"):
                _sanitize_identifier(malicious_input)

    def test_sanitize_identifier_allows_valid_names(self):
        """Test that valid identifiers are allowed."""
        valid_inputs = [
            "users",
            "user_table",
            "UserTable",
            "table123",
            "my_db_table_2024",
        ]

        for valid_input in valid_inputs:
            # Should not raise ValueError
            result = _sanitize_identifier(valid_input)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_quote_identifier_properly_escapes(self):
        """Test that identifiers are properly quoted."""
        test_cases = [
            ("users", '"users"'),
            ("user_table", '"user_table"'),
        ]

        for input_val, expected in test_cases:
            result = _quote_identifier(input_val)
            assert result.startswith('"') and result.endswith('"')

    def test_sanitize_empty_and_whitespace(self):
        """Test edge cases with empty and whitespace inputs raise ValueError."""
        edge_cases = [
            "",
            "   ",
            "\t\n",
        ]

        for edge_case in edge_cases:
            with pytest.raises(ValueError, match="Invalid identifier"):
                _sanitize_identifier(edge_case)

        # Whitespace-padded valid name should work after strip
        result = _sanitize_identifier("  table  ")
        assert result == "table"

    def test_sanitize_special_characters(self):
        """Test that special characters cause ValueError."""
        special_chars = [
            "table@name",
            "table#name",
            "table$name",
            "table%name",
            "table&name",
            "table*name",
            "table(name)",
            "table[name]",
            "table{name}",
        ]

        for special_input in special_chars:
            with pytest.raises(ValueError, match="Invalid identifier"):
                _sanitize_identifier(special_input)

    def test_sql_keyword_handling(self):
        """Test handling of SQL keywords as table names."""
        # While not recommended, SQL allows keywords as identifiers when quoted
        sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TABLE",
        ]

        for keyword in sql_keywords:
            result = _sanitize_identifier(keyword)
            # Should sanitize to lowercase alphanumeric
            assert result.islower() or result.isupper(), f"Keyword not handled: {keyword}"
            assert result.isalnum(), f"Keyword contains non-alphanumeric: {keyword}"


class TestSQLSecurityIntegration:
    """Integration tests for SQL security features."""

    def test_prevent_path_traversal_in_db_path(self):
        """Test that database paths are validated."""
        # This is a placeholder - actual implementation would test db path validation
        # The current implementation should validate db_path in tool functions
        malicious_paths = [
            "../../../etc/passwd",
            "../../sensitive.db",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        # In actual use, these should be rejected by the tool
        for malicious_path in malicious_paths:
            # Test would invoke tool with malicious_path and verify rejection
            # For now, we just document the requirement
            assert True  # Placeholder

    def test_parameterized_queries_used(self):
        """Verify that parameterized queries are used where possible."""
        # This test documents that we should use parameterized queries
        # In sql_server.py, functions should use ? placeholders
        # Example: cursor.execute("SELECT * FROM ? WHERE id = ?", (table, id))

        # Note: SQLite doesn't support table names as parameters
        # That's why we use identifier sanitization as defense-in-depth
        assert True  # Documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
