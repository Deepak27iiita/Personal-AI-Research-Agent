import os
import json
import pytest
from src.memory import MemoryStore, ResearchMemory, PrivacyGuard, MemoryFilter
from src.models import ResearchState, ResearchPlan, Source, CitationCheck, VerificationResult


@pytest.fixture
def memory_store(tmp_path):
    """Provides a fresh isolated SQLite memory store for each test."""
    db_file = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=str(db_file))
    yield store
    store.close()


# ---------------------------------------------------------------------------
# 1. Privacy Guard Tests
# ---------------------------------------------------------------------------
def test_privacy_guard_redacts_credentials():
    dirty_text = (
        "Found API key: AQ.DummyFakeKeyForTestingPurposeOnly123456789 and "
        "OpenAI key sk-abcdef1234567890abcdef1234567890. "
        "Contact researcher at admin@example.com with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    )
    clean = PrivacyGuard.scrub(dirty_text)

    assert "AQ.DummyFakeKeyForTestingPurposeOnly123456789" not in clean
    assert "sk-abcdef1234567890abcdef1234567890" not in clean
    assert "admin@example.com" not in clean
    assert "[REDACTED_API_KEY]" in clean
    assert "[REDACTED_EMAIL]" in clean
    assert "Bearer [REDACTED_TOKEN]" in clean


def test_privacy_guard_scrub_findings():
    findings = [
        "Agentic RAG uses routing agents to choose retrieval pathways.",
        "Internal test token: token='ghp_fake1234567890secretkeyhere'"
    ]
    scrubbed = PrivacyGuard.scrub_findings(findings)
    assert len(scrubbed) == 2
    assert "token=[REDACTED]" in scrubbed[1]


# ---------------------------------------------------------------------------
# 2. Memory Filter Tests (Policy Enforcement)
# ---------------------------------------------------------------------------
def test_memory_filter_rejects_empty_state():
    state = ResearchState(question="")
    mem = MemoryFilter.extract_clean_memory(state, session_id="test_sess")
    assert mem is None


def test_memory_filter_extracts_verified_claims():
    state = ResearchState(question="What is Agentic RAG?")
    state.final_report = (
        "# Research Report: Agentic RAG\n"
        "Agentic RAG introduces autonomous reasoning loops to standard information retrieval.\n"
        "## Verified Factual Findings\n"
        "- Routing agents dynamically evaluate user queries to pick retrieval strategies [S1].\n"
        "- Iterative query reformulation increases answer accuracy by 35% [S2].\n"
    )
    state.plan = ResearchPlan(
        sub_questions=["What is Agentic RAG?"],
        comparison_dimensions=["Latency", "Answer Quality"],
        initial_hypotheses=[]
    )
    s1 = state.add_source(url="https://arxiv.org/abs/agentic-rag", title="Agentic RAG Paper", snippet="Overview")
    s2 = state.add_source(url="https://langchain.com/agentic", title="LangChain Guide", snippet="Guide")

    state.verification = VerificationResult(
        total_claims_checked=2,
        verified_claims_count=2,
        fidelity_score=1.0,
        checks=[
            CitationCheck(claim="Routing agents dynamically evaluate user queries [S1]", source_id="S1", status="VERIFIED"),
            CitationCheck(claim="Phantom ungrounded claim [S99]", source_id="S99", status="UNSUPPORTED")
        ]
    )

    mem = MemoryFilter.extract_clean_memory(state, session_id="sess_123")
    assert mem is not None
    assert "Agentic Rag" in mem.topic or "Agentic RAG" in mem.topic
    assert mem.confidence_score == 1.0
    assert len(mem.sources) == 2
    # Only the VERIFIED check should have been included from checks
    assert any("Routing agents" in f for f in mem.key_findings)
    assert not any("Phantom ungrounded claim" in f for f in mem.key_findings)


# ---------------------------------------------------------------------------
# 3. Memory Store CRUD & Consolidation Tests
# ---------------------------------------------------------------------------
def test_memory_store_save_get_and_delete(memory_store):
    mem = ResearchMemory(
        id="mem_001",
        topic="Agentic RAG",
        summary="Agentic RAG utilizes LLM agents for dynamic query rewriting and routing.",
        key_findings=[
            "Supports multi-turn corrective retrieval.",
            "Increases token cost compared to naive RAG."
        ],
        sources=[{"id": "S1", "url": "https://example.com/agentic", "title": "Agentic RAG Overview"}],
        comparison_dimensions=["Latency", "Cost", "Correctness"],
        tags=["rag", "agentic", "routing"],
        confidence_score=0.95,
        session_id="session_1"
    )

    # Save
    saved_id = memory_store.save(mem)
    assert saved_id == "mem_001"

    # Get
    retrieved = memory_store.get("mem_001")
    assert retrieved is not None
    assert retrieved.topic == "Agentic RAG"
    assert len(retrieved.key_findings) == 2
    assert retrieved.confidence_score == 0.95

    # List all
    all_mems = memory_store.list_all()
    assert len(all_mems) == 1

    # Delete
    deleted = memory_store.delete("mem_001")
    assert deleted is True
    assert memory_store.get("mem_001") is None
    assert len(memory_store.list_all()) == 0


def test_memory_store_consolidates_existing_topic(memory_store):
    mem1 = ResearchMemory(
        id="mem_001",
        topic="GraphRAG",
        summary="Initial research on Microsoft GraphRAG.",
        key_findings=["Builds hierarchical knowledge graphs from text."],
        sources=[{"id": "S1", "url": "https://example.com/g1", "title": "GraphRAG 1"}],
        comparison_dimensions=["Index Cost"],
        tags=["graph", "rag"],
        confidence_score=0.80,
        session_id="session_1"
    )
    memory_store.save(mem1)

    # Second research session on the same topic with new findings
    mem2 = ResearchMemory(
        id="mem_002",
        topic="GraphRAG",
        summary="Extended research on GraphRAG community summaries.",
        key_findings=[
            "Builds hierarchical knowledge graphs from text.",  # duplicate claim
            "Performs global community answer synthesis with Leiden clustering."  # new claim
        ],
        sources=[{"id": "S2", "url": "https://example.com/g2", "title": "GraphRAG 2"}],
        comparison_dimensions=["Query Latency"],
        tags=["clustering"],
        confidence_score=0.95,
        session_id="session_2"
    )
    saved_id = memory_store.save(mem2)

    # Should update existing record without creating duplicate
    assert saved_id == "mem_001"
    consolidated = memory_store.get("mem_001")
    assert consolidated.access_count == 2
    assert len(consolidated.key_findings) == 2
    assert "global community answer synthesis" in consolidated.key_findings[1]
    assert len(consolidated.sources) == 2
    assert consolidated.confidence_score == 0.95


def test_memory_store_delete_by_topic_and_clear_all(memory_store):
    m1 = ResearchMemory(
        id="m1", topic="Agentic RAG", summary="S1", key_findings=["F1"],
        sources=[], comparison_dimensions=[], tags=["rag"], confidence_score=0.9, session_id="s1"
    )
    m2 = ResearchMemory(
        id="m2", topic="Vector Databases", summary="S2", key_findings=["F2"],
        sources=[], comparison_dimensions=[], tags=["database"], confidence_score=0.9, session_id="s2"
    )
    memory_store.save(m1)
    memory_store.save(m2)
    assert len(memory_store.list_all()) == 2

    # Delete by topic
    deleted_count = memory_store.delete_by_topic("Vector Databases")
    assert deleted_count == 1
    assert len(memory_store.list_all()) == 1

    # Clear all
    cleared = memory_store.clear_all()
    assert cleared == 1
    assert len(memory_store.list_all()) == 0


def test_memory_store_export_and_import_json(memory_store, tmp_path):
    m1 = ResearchMemory(
        id="m1", topic="Enterprise Search", summary="S1", key_findings=["F1"],
        sources=[{"id": "S1", "url": "https://test.com", "title": "T1"}],
        comparison_dimensions=["Recall"], tags=["search"], confidence_score=0.9, session_id="s1"
    )
    memory_store.save(m1)

    export_path = str(tmp_path / "memories_export.json")
    exported_count = memory_store.export_to_json(export_path)
    assert exported_count == 1
    assert os.path.exists(export_path)

    # Fresh store
    new_store_path = str(tmp_path / "imported_memory.db")
    new_store = MemoryStore(db_path=new_store_path)
    imported_count = new_store.import_from_json(export_path)
    assert imported_count == 1

    imported_mem = new_store.get("m1")
    assert imported_mem is not None
    assert imported_mem.topic == "Enterprise Search"
    new_store.close()


# ---------------------------------------------------------------------------
# 4. Cross-Session Continuity & Retrieval Relevance Tests
# ---------------------------------------------------------------------------
def test_cross_session_continuation_retrieval(memory_store):
    """
    Session 1: "I'm researching Agentic RAG."
    Session 2: "Continue my research and compare it with GraphRAG."
    """
    # Simulate Session 1: Storing Agentic RAG research
    session_1_mem = ResearchMemory(
        id="mem_session_1",
        topic="Agentic RAG",
        summary="Agentic RAG coordinates retrieval with reasoning loops and dynamic query decomposition.",
        key_findings=[
            "Uses iterative query reformulation when initial document retrieval is insufficient.",
            "Higher query latency due to multi-step LLM function calls."
        ],
        sources=[{"id": "S1", "url": "https://arxiv.org/abs/agentic-rag", "title": "Agentic RAG Foundations"}],
        comparison_dimensions=["Query Latency", "Accuracy", "Reasoning Depth"],
        tags=["agentic", "rag", "retrieval", "llm"],
        confidence_score=0.95,
        session_id="session_1"
    )
    memory_store.save(session_1_mem)

    # Simulate Session 2 query:
    session_2_query = "Continue my research and compare it with GraphRAG."
    matches = memory_store.retrieve(session_2_query, limit=2, min_score=0.35)

    assert len(matches) > 0
    top_mem, score = matches[0]
    assert top_mem.topic == "Agentic RAG"
    assert score >= 0.35
    assert "iterative query reformulation" in top_mem.key_findings[0]


def test_irrelevant_query_filtered_by_threshold(memory_store):
    # Store research on Quantum Computing
    mem = ResearchMemory(
        id="mem_qc",
        topic="Superconducting Qubits",
        summary="Detailed analysis of dilution refrigerators and transmon qubit coherence.",
        key_findings=["T1 relaxation times remain bounded by dielectric surface loss."],
        sources=[],
        comparison_dimensions=["Coherence Time"],
        tags=["quantum", "physics"],
        confidence_score=0.9,
        session_id="session_qc"
    )
    memory_store.save(mem)

    # Query for completely unrelated topic
    unrelated_query = "How to bake Neapolitan sourdough pizza at home with high hydration?"
    matches = memory_store.retrieve(unrelated_query, limit=2, min_score=0.40)

    # Should be filtered out by relevance score threshold
    assert len(matches) == 0
