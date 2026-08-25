"""Pytest fixtures for evaluation suite."""

import pytest
from agent.agent_loop import SupportAgent
from agent.ingest import build_index
from agent.session import get_session_manager


@pytest.fixture(scope="session")
def vector_index():
    """Ensure ChromaDB index is built for test execution."""
    return build_index(force_rebuild=False)


@pytest.fixture(scope="session")
def agent(vector_index):
    """Instantiate support agent for evaluation."""
    return SupportAgent()


@pytest.fixture
def session_manager():
    """Get session manager for state inspection."""
    return get_session_manager()
