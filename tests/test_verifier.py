"""
Unit tests for the CitationVerifier and Reflection/Self-Correction subsystem.
"""

import pytest
from src.models import Source, VerificationResult
from src.verifier import CitationVerifier


def test_extract_cited_claims():
    report = """
# Research Report

## 1. Executive Summary
Agentic RAG integrates query routing and tool calling into standard retrieval pipelines [S1].
Traditional RAG relies on static top-k vector indexing without autonomous loops [S2].
This sentence has no citation and should be ignored.
GraphRAG builds knowledge graphs for global summarization [S1][S3].
"""
    extracted = CitationVerifier.extract_cited_claims(report)
    assert len(extracted) == 4

    # Check that claims were extracted with their source IDs
    assert any(claim_id == "S1" for _, claim_id in extracted)
    assert any(claim_id == "S2" for _, claim_id in extracted)
    assert any(claim_id == "S3" for _, claim_id in extracted)


def test_detect_phantom_sources():
    sources = {
        "S1": Source(id="S1", url="https://example.com/s1", title="Source 1", snippet="Snippet 1"),
        "S2": Source(id="S2", url="https://example.com/s2", title="Source 2", snippet="Snippet 2")
    }

    cited_claims = [
        ("Claim citing valid source [S1]", "S1"),
        ("Claim citing phantom source [S3]", "S3"),
        ("Claim citing another phantom [S99]", "S99"),
        ("Claim citing valid source [S2]", "S2")
    ]

    phantoms = CitationVerifier.detect_phantom_sources(cited_claims, sources)
    assert phantoms == ["S3", "S99"]


def test_heuristic_single_claim_verification():
    verifier = CitationVerifier()
    source = Source(
        id="S1",
        url="https://example.com/benchmarks",
        title="RAG Latency Benchmarks",
        snippet="In enterprise evaluations, traditional RAG showed a median response latency of 800ms."
    )

    # 1. Grounded claim
    grounded_claim = "Traditional RAG demonstrated an enterprise median response latency around 800ms [S1]."
    check_grounded = verifier.verify_single_claim(grounded_claim, source, use_llm=False)
    assert check_grounded.status == "VERIFIED"
    assert check_grounded.evidence_quote is not None

    # 2. Hallucinated / ungrounded claim
    hallucinated_claim = "Quantum teleportation was proven to increase throughput by 9999% [S1]."
    check_hallucinated = verifier.verify_single_claim(hallucinated_claim, source, use_llm=False)
    assert check_hallucinated.status == "UNSUPPORTED"

    # 3. Phantom source claim
    check_phantom = verifier.verify_single_claim("Some claim [SX]", None, use_llm=False)
    assert check_phantom.status == "PHANTOM_SOURCE"


def test_audit_report_and_fidelity_score():
    verifier = CitationVerifier()
    sources = {
        "S1": Source(
            id="S1",
            url="https://example.com/rag",
            title="Agentic RAG Overview",
            snippet="Agentic RAG utilizes iterative loops and dynamic tool selection."
        )
    }

    report_text = """
## Findings
- Agentic RAG employs iterative loops and dynamic tool selection [S1].
- GraphRAG uses quantum neural teleportation across hyperspace [S99].
"""

    audit = verifier.audit_report(report_text, sources, use_llm=False)
    assert audit.total_claims_checked == 2
    assert audit.verified_claims_count == 1
    assert audit.unsupported_claims_count == 1
    assert audit.phantom_sources_found == ["S99"]
    assert audit.fidelity_score == 0.5
    assert audit.needs_refinement is True
    assert len(audit.critic_notes) > 0


def test_refine_report_graceful_fallback():
    verifier = CitationVerifier()
    sources = {
        "S1": Source(id="S1", url="https://example.com", title="Test", snippet="Valid snippet")
    }
    verification = VerificationResult(
        total_claims_checked=1,
        verified_claims_count=0,
        unsupported_claims_count=1,
        phantom_sources_found=["S2"],
        fidelity_score=0.0,
        critic_notes=["Critique: Phantom S2 detected"],
        needs_refinement=True
    )

    draft = "Draft report with [S2]."
    # Should run and return refined string without crashing
    refined = verifier.refine_report(draft, verification, sources)
    assert isinstance(refined, str)
    assert len(refined) > 0
