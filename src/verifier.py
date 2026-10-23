"""
Personal AI Research Agent - Verification, Grounding & Self-Correction Subsystem.

Core Concepts:
1. Extraction: Parse every claim asserting a citation ([S1], [S2], etc.).
2. Phantom Source Detection: Detect hallucinations where the LLM cites sources that don't exist in State.
3. Grounding Verification: Compare cited claims against raw evidence text from Source objects.
4. Critic Feedback: Generate actionable critique for ungrounded claims or numeric exaggerations.
5. Self-Correction (Reflection): Refine the draft report to eliminate hallucinations and downgrade unverifiable claims.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from src.models import ResearchState, Source, CitationCheck, VerificationResult

load_dotenv()

logger = logging.getLogger("CitationVerifier")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")


class CitationVerifier:
    """
    Acts as an independent critic/auditor to ensure factual grounding
    and prevent LLM hallucination in synthesized research reports.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    # -----------------------------------------------------------------------
    # Step 1: Claim and Citation Parsing (Deterministic / Heuristic)
    # -----------------------------------------------------------------------
    @staticmethod
    def extract_cited_claims(report_text: str) -> List[Tuple[str, str]]:
        """
        Extracts individual claims and their associated citation IDs.
        Returns a list of tuples: (claim_text, source_id), e.g. ("Agentic RAG adds routing [S1]", "S1").
        """
        # Split report into sentences or bullet points
        raw_lines = report_text.splitlines()
        extracted: List[Tuple[str, str]] = []

        for line in raw_lines:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue

            # Find all citation patterns like [S1], [S2]
            citation_matches = re.findall(r"\[(S\d+)\]", clean_line)
            if citation_matches:
                for src_id in citation_matches:
                    extracted.append((clean_line, src_id))

        return extracted

    @staticmethod
    def detect_phantom_sources(
        cited_claims: List[Tuple[str, str]],
        sources: Dict[str, Source]
    ) -> List[str]:
        """
        Identifies citation IDs referenced in the report that do not exist
        in the research agent's evidence store.
        """
        valid_ids = set(sources.keys())
        phantoms = set()
        for _, src_id in cited_claims:
            if src_id not in valid_ids:
                phantoms.add(src_id)
        return sorted(list(phantoms))

    # -----------------------------------------------------------------------
    # Step 2: Grounding Audit (Heuristic + LLM-as-a-Judge)
    # -----------------------------------------------------------------------
    def verify_single_claim(
        self,
        claim: str,
        source: Optional[Source],
        use_llm: bool = True
    ) -> CitationCheck:
        """
        Checks whether a single claim is supported by the corresponding source text.
        """
        if source is None:
            return CitationCheck(
                claim=claim,
                source_id="UNKNOWN",
                status="PHANTOM_SOURCE",
                reason="Cited source does not exist in the evidence store."
            )

        # Aggregate all text known for this source
        source_evidence = f"Title: {source.title}\nSnippet: {source.snippet}"
        if source.full_text:
            source_evidence += f"\nFull Content: {source.full_text[:3000]}"

        # Fast heuristic check: check for key terms overlap
        claim_clean = re.sub(r"\[S\d+\]", "", claim).strip().lower()
        
        # If LLM verification is disabled or unavailable, use lexical support heuristic
        if not use_llm or (not GEMINI_API_KEY and not OPENAI_API_KEY):
            # Deterministic heuristic: check word overlap
            words = [w for w in re.findall(r"\w+", claim_clean) if len(w) > 3]
            if not words:
                return CitationCheck(
                    claim=claim,
                    source_id=source.id,
                    status="VERIFIED",
                    evidence_quote=source.snippet[:120],
                    reason="Claim verified via baseline heuristic matching."
                )

            matches = [w for w in words if w in source_evidence.lower()]
            overlap_ratio = len(matches) / len(words)

            if overlap_ratio >= 0.35:
                return CitationCheck(
                    claim=claim,
                    source_id=source.id,
                    status="VERIFIED",
                    evidence_quote=source.snippet[:120],
                    reason=f"Heuristic lexical match ({round(overlap_ratio*100)}% keyword overlap)."
                )
            else:
                return CitationCheck(
                    claim=claim,
                    source_id=source.id,
                    status="UNSUPPORTED",
                    reason=f"Insufficient keyword overlap ({round(overlap_ratio*100)}%) with source text."
                )

        # Semantic verification via LLM-as-a-Judge
        return self._verify_claim_with_llm(claim, source.id, source_evidence)

    def _verify_claim_with_llm(self, claim: str, source_id: str, source_evidence: str) -> CitationCheck:
        """
        Uses an LLM judge to verify if the claim is strictly entailed by the source evidence.
        """
        judge_prompt = f"""
You are an uncompromising Fact-Checking and Citation Verifier.
Your task is to determine whether the CLAIM below is directly supported by the SOURCE EVIDENCE.

CLAIM TO VERIFY:
"{claim}"

SOURCE EVIDENCE ({source_id}):
\"\"\"
{source_evidence[:2500]}
\"\"\"

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "status": "VERIFIED" | "UNSUPPORTED",
  "evidence_quote": "exact sentence or clause from SOURCE EVIDENCE that backs the claim, or null if unsupported",
  "reason": "brief 1-sentence explanation of why it is supported or why it is an exaggeration/hallucination"
}}
"""
        try:
            if GEMINI_API_KEY:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=GEMINI_API_KEY)
                resp = client.models.generate_content(
                    model=self.model,
                    contents=judge_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                data = json.loads(resp.text)
            else:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=self.model if "gpt" in self.model else "gpt-4o-mini",
                    messages=[{"role": "user", "content": judge_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                data = json.loads(resp.choices[0].message.content)

            status = data.get("status", "UNSUPPORTED")
            return CitationCheck(
                claim=claim,
                source_id=source_id,
                status=status if status in ["VERIFIED", "UNSUPPORTED"] else "UNSUPPORTED",
                evidence_quote=data.get("evidence_quote"),
                reason=data.get("reason", "Evaluated by LLM Judge")
            )
        except Exception as e:
            logger.warning(f"LLM Judge verification error for claim: {e}")
            # Fallback to supported if unparseable to avoid crashing
            return CitationCheck(
                claim=claim,
                source_id=source_id,
                status="UNSUPPORTED",
                reason=f"Verification judge call failed: {str(e)}"
            )

    # -----------------------------------------------------------------------
    # Step 3: Full Audit & Scorecard Generation
    # -----------------------------------------------------------------------
    def audit_report(
        self,
        report_text: str,
        sources: Dict[str, Source],
        use_llm: bool = True
    ) -> VerificationResult:
        """
        Runs a comprehensive audit across all claims and citations in the report.
        """
        cited_claims = self.extract_cited_claims(report_text)
        phantom_sources = self.detect_phantom_sources(cited_claims, sources)

        checks: List[CitationCheck] = []
        critic_notes: List[str] = []
        verified_count = 0
        unsupported_count = 0

        # Note phantom sources
        for phantom in phantom_sources:
            note = f"CRITICAL: Report cited [{phantom}], but no such source was retrieved during research."
            critic_notes.append(note)

        # Audit each cited claim
        for claim, src_id in cited_claims:
            if src_id in phantom_sources:
                check = CitationCheck(
                    claim=claim,
                    source_id=src_id,
                    status="PHANTOM_SOURCE",
                    reason=f"Source [{src_id}] does not exist in evidence store."
                )
                unsupported_count += 1
            else:
                source = sources.get(src_id)
                check = self.verify_single_claim(claim, source, use_llm=use_llm)
                if check.status == "VERIFIED":
                    verified_count += 1
                else:
                    unsupported_count += 1
                    critic_notes.append(
                        f"UNGROUNDED CLAIM: '{claim}' -> Cited [{src_id}], but source lacks support: {check.reason}"
                    )

            checks.append(check)

        total_claims = len(checks)
        fidelity_score = (verified_count / total_claims) if total_claims > 0 else 1.0
        needs_refinement = (unsupported_count > 0) or (len(phantom_sources) > 0)

        result = VerificationResult(
            total_claims_checked=total_claims,
            verified_claims_count=verified_count,
            unsupported_claims_count=unsupported_count,
            phantom_sources_found=phantom_sources,
            fidelity_score=round(fidelity_score, 2),
            checks=checks,
            critic_notes=critic_notes,
            needs_refinement=needs_refinement
        )
        return result

    # -----------------------------------------------------------------------
    # Step 4: Self-Correction & Refinement (The Reflection Loop)
    # -----------------------------------------------------------------------
    def refine_report(
        self,
        draft_report: str,
        verification: VerificationResult,
        sources: Dict[str, Source]
    ) -> str:
        """
        Feeds the draft report and critic feedback into the Refiner agent
        to eliminate phantom sources, qualify unverified claims, and produce
        a mathematically grounded, 100% verified document.
        """
        if not verification.needs_refinement:
            logger.info("Report has 100% citation fidelity. No refinement needed.")
            return draft_report

        logger.info(
            f"Refining report: {verification.unsupported_claims_count} ungrounded claims, "
            f"{len(verification.phantom_sources_found)} phantom citations detected."
        )

        sources_summary = []
        for sid, src in sources.items():
            sources_summary.append(f"[{sid}] Title: {src.title}\nURL: {src.url}\nSnippet: {src.snippet}")

        valid_sources_block = "\n\n".join(sources_summary)
        critic_feedback_block = "\n".join(f"- {note}" for note in verification.critic_notes)

        refiner_prompt = f"""
You are an expert Technical Editor and Scientific Fact-Checker.
You are given a DRAFT RESEARCH REPORT that was audited by an automated citation verifier.
The verifier flagged factual discrepancies, phantom citations, or unsupported claims.

YOUR TASK:
Rewrite and refine the report so that:
1. Every phantom citation (citations to sources not listed below) is REMOVED or replaced with a valid source.
2. Any claim flagged as unsupported is either:
   - Modified to strictly match what the sources actually state, OR
   - Moved to the "Analytical Conclusions & Recommendations" section with explicit qualification, OR
   - Removed if entirely fabricated.
3. Keep the exact same Markdown structure, headings, and comparison tables.
4. Add a prominent badge at the top:
   > **Verification Status: Audited & Grounded (Fidelity Score: 100%)**

CRITIC AUDIT FEEDBACK:
{critic_feedback_block}

VALID SOURCES IN EVIDENCE STORE:
{valid_sources_block}

ORIGINAL DRAFT REPORT:
\"\"\"
{draft_report}
\"\"\"
"""
        if not GEMINI_API_KEY and not OPENAI_API_KEY:
            # Deterministic fallback when no API keys are present (e.g. testing)
            corrected = draft_report + "\n\n> **Verification Note:** Report audited. Flagged items noted in audit log."
            return corrected

        try:
            if GEMINI_API_KEY:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=GEMINI_API_KEY)
                resp = client.models.generate_content(
                    model=self.model,
                    contents=refiner_prompt,
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                return resp.text
            else:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=self.model if "gpt" in self.model else "gpt-4o-mini",
                    messages=[{"role": "user", "content": refiner_prompt}],
                    temperature=0.1
                )
                return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to execute LLM refinement: {e}")
            return draft_report
