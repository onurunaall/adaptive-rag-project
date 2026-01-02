"""
Pytest configuration and fixtures for the test suite.
"""

import pytest


# Configure pytest-asyncio mode
pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
