import os
import pytest
from src.tools import (
    calculator,
    get_current_time,
    save_research_note,
    create_default_registry,
    RESEARCH_NOTES_FILE
)
from src.agent import AgentState, AgentStep


def test_calculator_valid_expressions():
    assert "470" in calculator("(25 * 14) + 120")
    assert "256" in calculator("2 ** 8")
    assert "25.0" in calculator("100 / 4")


def test_calculator_division_by_zero():
    result = calculator("10 / 0")
    assert "Division by zero" in result


def test_calculator_blocks_malicious_code():
    # Attempting to execute unauthorized code should fail safely
    result = calculator("__import__('os').system('echo hacked')")
    assert "Error evaluating expression" in result


def test_get_current_time():
    time_str = get_current_time()
    assert "Current Date & Time" in time_str


def test_save_research_note():
    topic = "Test Unit Note"
    content = "This is a test research note created by pytest."
    result = save_research_note(topic, content)
    assert "Success" in result
    assert os.path.exists(RESEARCH_NOTES_FILE)

    with open(RESEARCH_NOTES_FILE, "r", encoding="utf-8") as f:
        saved_content = f.read()
    assert topic in saved_content
    assert content in saved_content


def test_tool_registry_execution_and_validation():
    registry = create_default_registry()

    # Valid execution
    obs = registry.execute("calculator", {"expression": "10 + 5"})
    assert "15" in obs

    # Non-existent tool
    obs_missing = registry.execute("non_existent_tool", {})
    assert "Error: Tool 'non_existent_tool' does not exist" in obs_missing

    # Invalid arguments (missing required parameter)
    obs_bad_args = registry.execute("calculator", {})
    assert "Argument Error" in obs_bad_args


def test_agent_state_cycle_detection():
    state = AgentState(objective="Investigate GraphRAG")

    # Step 1: Call web_search
    state.record_step(AgentStep(
        step_number=1,
        thought=None,
        tool_name="web_search",
        tool_arguments={"query": "GraphRAG vs RAG"},
        observation="Some results..."
    ))
    assert not state.is_cycle_detected("web_search", {"query": "GraphRAG vs RAG"})

    # Step 2: Call web_search with exact same query again
    state.record_step(AgentStep(
        step_number=2,
        thought=None,
        tool_name="web_search",
        tool_arguments={"query": "GraphRAG vs RAG"},
        observation="Same results..."
    ))

    # Now the detector should flag that we are repeating ourselves
    assert state.is_cycle_detected("web_search", {"query": "GraphRAG vs RAG"})
