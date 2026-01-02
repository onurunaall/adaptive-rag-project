"""
Pytest configuration and fixtures for the test suite.
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
