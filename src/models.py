"""
Data models for the Personal AI Research Agent.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class Source:
    """
    Represents an external information source with strict provenance.
    """
    id: str  # e.g., 'S1', 'S2'
    url: str
    title: str
    snippet: str
    full_text: Optional[str] = None
    fetched_at: float = field(default_factory=time.time)


@dataclass
class ResearchPlan:
    """
    The structured plan created before searching.
    """
    sub_questions: List[str]
    comparison_dimensions: List[str]
    initial_hypotheses: List[str]


@dataclass
class ResearchBudget:
    """
    Operational safety boundaries to prevent runaway execution.
    """
    max_tool_calls: int = 8
    max_execution_time_seconds: float = 90.0
    max_total_tokens: int = 30000


@dataclass
class ResearchStep:
    step_number: int
    action: str
    arguments: Dict[str, Any]
    observation_summary: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchState:
    """
    The living state of the entire research mission.
    """
    question: str
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    plan: Optional[ResearchPlan] = None
    sources: Dict[str, Source] = field(default_factory=dict)
    steps: List[ResearchStep] = field(default_factory=list)
    final_report: Optional[str] = None
    verification: Optional["VerificationResult"] = None
    
    # Telemetry and bounds tracking
    tool_calls_count: int = 0
    estimated_tokens_used: int = 0
    start_time: float = field(default_factory=time.time)
    completed: bool = False
    termination_reason: Optional[str] = None

    # Long-term memory integration
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    memory_id: Optional[str] = None

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def is_budget_exceeded(self) -> Tuple[bool, Optional[str]]:
        if self.tool_calls_count >= self.budget.max_tool_calls:
            return True, f"Max tool calls limit reached ({self.budget.max_tool_calls})"
        if self.elapsed_seconds() >= self.budget.max_execution_time_seconds:
            return True, f"Execution timeout exceeded ({self.budget.max_execution_time_seconds}s)"
        if self.estimated_tokens_used >= self.budget.max_total_tokens:
            return True, f"Estimated token budget exceeded ({self.budget.max_total_tokens})"
        return False, None

    def add_source(self, url: str, title: str, snippet: str, full_text: Optional[str] = None) -> Source:
        # Check if URL already tracked
        for s in self.sources.values():
            if s.url.strip().lower() == url.strip().lower():
                if full_text and not s.full_text:
                    s.full_text = full_text
                return s

        source_id = f"S{len(self.sources) + 1}"
        source = Source(
            id=source_id,
            url=url,
            title=title or "Untitled Web Resource",
            snippet=snippet,
            full_text=full_text
        )
        self.sources[source_id] = source
        return source


@dataclass
class CitationCheck:
    """
    Result of verifying a single cited claim against external source evidence.
    """
    claim: str
    source_id: str
    status: str  # 'VERIFIED', 'UNSUPPORTED', 'PHANTOM_SOURCE'
    evidence_quote: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class VerificationResult:
    """
    Comprehensive verification scorecard and reflection feedback for a synthesized report.
    """
    total_claims_checked: int = 0
    verified_claims_count: int = 0
    unsupported_claims_count: int = 0
    phantom_sources_found: List[str] = field(default_factory=list)
    fidelity_score: float = 0.0  # 0.0 to 1.0 (verified / total)
    checks: List[CitationCheck] = field(default_factory=list)
    critic_notes: List[str] = field(default_factory=list)
    needs_refinement: bool = False
    corrected_report: Optional[str] = None
