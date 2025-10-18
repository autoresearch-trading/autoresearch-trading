"""Test configuration and fixtures."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add tests directory to Python path so fixtures can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


@pytest.fixture(scope="session", autouse=True)
def mock_questdb():
    """Mock QuestDB connection for tests that don't need real database."""
    if os.getenv("SKIP_QUESTDB_TESTS", "false").lower() == "true":
        with patch("psycopg.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            yield mock_conn
    else:
        yield None


@pytest.fixture
def questdb_available():
    """Check if QuestDB is available for integration tests."""
    try:
        import psycopg

        conn_string = "host=localhost port=8812 user=admin password=quest dbname=qdb"
        with psycopg.connect(conn_string, connect_timeout=5):
            return True
    except Exception:
        return False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "questdb: mark test as requiring QuestDB")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip QuestDB tests when database is not available."""
    questdb_available = True
    try:
        import psycopg

        conn_string = "host=localhost port=8812 user=admin password=quest dbname=qdb"
        with psycopg.connect(conn_string, connect_timeout=5):
            pass
    except Exception:
        questdb_available = False

    for item in items:
        if "questdb" in item.keywords and not questdb_available:
            item.add_marker(pytest.mark.skip(reason="QuestDB not available"))
