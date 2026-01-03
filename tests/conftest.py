"""
Pytest configuration and fixtures for the test suite.
"""

import pytest

# Conditionally load pytest-asyncio plugin if available
try:
    import pytest_asyncio
    pytest_plugins = ("pytest_asyncio",)
except ImportError:
    pytest_plugins = ()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
