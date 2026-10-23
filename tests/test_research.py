import pytest
import time
from src.tools import is_safe_url, read_web_page
from src.models import ResearchState, ResearchBudget, Source


def test_ssrf_protection_blocks_malicious_urls():
    # Loopback / Localhost
    safe, err = is_safe_url("http://localhost:8080/secret")
    assert not safe
    assert "localhost/loopback" in err

    safe, err = is_safe_url("http://127.0.0.1:5000")
    assert not safe

    # Cloud metadata service (AWS/GCP/Azure link-local)
    safe, err = is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert not safe
    assert "blocked" in err

    # Private internal subnet
    safe, err = is_safe_url("http://192.168.1.100/router")
    assert not safe

    safe, err = is_safe_url("http://10.0.0.1/db")
    assert not safe

    # Invalid protocol
    safe, err = is_safe_url("file:///etc/passwd")
    assert not safe
    assert "Blocked protocol" in err

    # Valid public URL
    safe, err = is_safe_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
    assert safe
    assert err is None


def test_source_tracking_and_deduplication():
    state = ResearchState(question="Test Question")

    # Add first source
    s1 = state.add_source(
        url="https://example.com/rag-guide",
        title="RAG Guide",
        snippet="Intro to RAG"
    )
    assert s1.id == "S1"
    assert len(state.sources) == 1

    # Add second different source
    s2 = state.add_source(
        url="https://example.com/graphrag",
        title="GraphRAG Paper",
        snippet="Intro to GraphRAG"
    )
    assert s2.id == "S2"
    assert len(state.sources) == 2

    # Add duplicate URL (should return existing source without incrementing ID)
    s1_dup = state.add_source(
        url="https://example.com/rag-guide",
        title="RAG Guide Updated",
        snippet="Duplicate snippet"
    )
    assert s1_dup.id == "S1"
    assert len(state.sources) == 2


def test_budget_boundaries_enforcement():
    budget = ResearchBudget(
        max_tool_calls=3,
        max_execution_time_seconds=2.0,
        max_total_tokens=1000
    )
    state = ResearchState(question="Budget Test", budget=budget)

    # Initially within budget
    exceeded, _ = state.is_budget_exceeded()
    assert not exceeded

    # Exceed tool calls
    state.tool_calls_count = 3
    exceeded, reason = state.is_budget_exceeded()
    assert exceeded
    assert "Max tool calls limit reached" in reason

    # Reset tool calls and test timeout
    state.tool_calls_count = 0
    state.start_time = time.time() - 3.0  # simulate 3 seconds elapsed
    exceeded, reason = state.is_budget_exceeded()
    assert exceeded
    assert "timeout exceeded" in reason
