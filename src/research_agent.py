"""
Personal AI Research Agent - Single Intelligent Agent Architecture.

Workflow:
1. Deconstruct question & generate structured Research Plan.
2. Execute bounded research loop (web search, page reading, evidence extraction).
3. Enforce strict budget boundaries: max tool calls, timeouts, and token ceilings.
4. Synthesize final report with inline citations [S#] and explicit separation of facts vs conclusions.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from src.models import ResearchState, ResearchPlan, ResearchStep, ResearchBudget
from src.tools import create_research_registry, ToolRegistry
from src.verifier import CitationVerifier
from src.memory import MemoryStore, MemoryFilter, ResearchMemory

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ResearchAgent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")


class PersonalResearchAgent:
    """
    An autonomous research assistant operating under strict budget boundaries.
    """

    def __init__(
        self,
        budget: Optional[ResearchBudget] = None,
        model: str = DEFAULT_MODEL,
        memory_store: Optional[MemoryStore] = None
    ):
        self.budget = budget or ResearchBudget()
        self.model = model
        self.registry = create_research_registry()
        self.verifier = CitationVerifier(model=self.model)
        self.memory = memory_store or MemoryStore()

        if not GEMINI_API_KEY and not OPENAI_API_KEY:
            raise ValueError(
                "Missing API Key. Please configure GEMINI_API_KEY or OPENAI_API_KEY in .env"
            )

    # -----------------------------------------------------------------------
    # Step 1: Research Planning Phase
    # -----------------------------------------------------------------------
    def _create_research_plan(self, question: str, prior_context: str = "") -> ResearchPlan:
        """
        Decomposes the research question into targeted sub-questions and comparison dimensions.
        Integrates prior research memory to prevent redundant investigations.
        """
        logger.info(f"Formulating research plan for: '{question}'")

        system_prompt = (
            "You are a Senior Principal Research Architect. "
            "Given a research question, break it down into a systematic, objective research plan. "
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "sub_questions": ["sub-question 1", "sub-question 2", ...],\n'
            '  "comparison_dimensions": ["dimension 1 (e.g. Latency)", "dimension 2 (e.g. Cost)", ...],\n'
            '  "initial_hypotheses": ["hypothesis 1", ...]\n'
            "}"
        )

        memory_prompt_clause = ""
        if prior_context:
            memory_prompt_clause = (
                f"\n\nPRIOR RESEARCH MEMORY (DO NOT duplicate these investigations, build upon them or contrast them):\n"
                f"{prior_context}\n"
            )

        user_prompt = f"Develop a research plan to investigate:\n'{question}'{memory_prompt_clause}"

        try:
            if GEMINI_API_KEY:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=GEMINI_API_KEY)
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.2
                )
                resp = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config
                )
                data = json.loads(resp.text)
            else:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=self.model if "gpt" in self.model else "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                data = json.loads(resp.choices[0].message.content)

            return ResearchPlan(
                sub_questions=data.get("sub_questions", [question]),
                comparison_dimensions=data.get("comparison_dimensions", ["Architecture", "Trade-offs"]),
                initial_hypotheses=data.get("initial_hypotheses", [])
            )
        except Exception as e:
            logger.warning(f"Structured planning failed, falling back to default plan: {e}")
            return ResearchPlan(
                sub_questions=[question],
                comparison_dimensions=["Core Architecture", "Performance & Cost"],
                initial_hypotheses=[]
            )

    # -----------------------------------------------------------------------
    # Step 2: Bounded Autonomous Research Loop
    # -----------------------------------------------------------------------
    def _execute_research_loop(self, state: ResearchState, prior_context: str = ""):
        """
        Executes search and page-reading cycles while respecting max calls, timeouts, and tokens.
        """
        self.registry.set_state(state)

        if GEMINI_API_KEY:
            self._gemini_research_loop(state, prior_context=prior_context)
        else:
            self._openai_research_loop(state, prior_context=prior_context)

    def _gemini_research_loop(self, state: ResearchState, prior_context: str = ""):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build tool definitions
        gemini_tools = []
        for schema in self.registry.get_schemas():
            props = {}
            for pname, pdata in schema["parameters"]["properties"].items():
                ptype = pdata.get("type", "string").upper()
                props[pname] = types.Schema(
                    type=getattr(types.Type, ptype, types.Type.STRING),
                    description=pdata.get("description", "")
                )
            gemini_tools.append(types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=props,
                    required=schema["parameters"].get("required", [])
                )
            ))

        plan_summary = (
            f"Research Question: {state.question}\n"
            f"Sub-Questions to investigate:\n" + "\n".join(f"- {sq}" for sq in state.plan.sub_questions) + "\n"
            f"Key Dimensions:\n" + "\n".join(f"- {cd}" for cd in state.plan.comparison_dimensions)
        )
        if prior_context:
            plan_summary += (
                f"\n\nPRIOR RESEARCH MEMORY (Use these known facts as established baselines):\n"
                f"{prior_context}"
            )

        system_instruction = (
            "You are an investigative Research Assistant. "
            "Your job is to gather deep, factual evidence to answer the research plan.\n"
            "Guidelines:\n"
            "1. Start by searching broadly with 'web_search'.\n"
            "2. Read the most promising URLs using 'read_web_page' to gather technical details and real numbers.\n"
            "3. Pay close attention to candidate source tags [S1], [S2] in observations.\n"
            "4. Compare perspectives and look for conflicting claims.\n"
            "5. Stop calling tools when you have sufficient evidence to compare the topics thoroughly, "
            "or when budget constraints require you to finalize."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=gemini_tools)],
            temperature=0.2
        )

        conversation: List[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Begin your research investigation.\n\n{plan_summary}")]
            )
        ]

        while not state.completed:
            # Check operational boundaries
            exceeded, reason = state.is_budget_exceeded()
            if exceeded:
                logger.warning(f"Research loop halted: {reason}")
                state.termination_reason = reason
                break

            step_idx = state.tool_calls_count + 1
            logger.info(f"--- [Investigation Step {step_idx} | Elapsed: {round(state.elapsed_seconds(), 1)}s] ---")

            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=conversation,
                    config=config
                )
            except Exception as e:
                logger.error(f"API Error during research: {e}")
                state.termination_reason = f"API Error: {str(e)}"
                break

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                state.estimated_tokens_used += getattr(response.usage_metadata, "total_token_count", 1000)

            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                state.termination_reason = "Empty response received."
                break

            conversation.append(candidate.content)

            tool_calls = response.function_calls
            if tool_calls:
                for call in tool_calls:
                    tname = call.name
                    targs = dict(call.args) if call.args else {}
                    state.tool_calls_count += 1

                    logger.info(f"Agent Action: {tname}({targs})")
                    observation = self.registry.execute(tname, targs)

                    # Truncate preview in logs
                    log_preview = observation[:120].replace('\n', ' ')
                    logger.info(f"Observation: {log_preview}...")

                    # Record step in state
                    state.steps.append(ResearchStep(
                        step_number=state.tool_calls_count,
                        action=tname,
                        arguments=targs,
                        observation_summary=observation[:300]
                    ))

                    # Feed observation back
                    conversation.append(types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=tname,
                            response={"result": observation}
                        )]
                    ))
            else:
                # LLM determined it has gathered enough evidence
                logger.info("Agent concluded investigation phase.")
                state.termination_reason = "Information gathering concluded autonomously."
                break

    def _openai_research_loop(self, state: ResearchState, prior_context: str = ""):
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        tools = []
        for schema in self.registry.get_schemas():
            tools.append({
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"]
                }
            })

        plan_summary = (
            f"Research Question: {state.question}\n"
            f"Sub-Questions to investigate:\n" + "\n".join(f"- {sq}" for sq in state.plan.sub_questions) + "\n"
            f"Dimensions:\n" + "\n".join(f"- {cd}" for cd in state.plan.comparison_dimensions)
        )
        if prior_context:
            plan_summary += f"\n\nPRIOR RESEARCH MEMORY:\n{prior_context}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an investigative Research Assistant. "
                    "Use 'web_search' and 'read_web_page' to collect deep factual evidence. "
                    "Track sources [S1], [S2] carefully."
                )
            },
            {"role": "user", "content": f"Begin research.\n\n{plan_summary}"}
        ]

        while not state.completed:
            exceeded, reason = state.is_budget_exceeded()
            if exceeded:
                state.termination_reason = reason
                break

            response = client.chat.completions.create(
                model=self.model if "gpt" in self.model else "gpt-4o-mini",
                messages=messages,
                tools=tools,
                temperature=0.2
            )

            state.estimated_tokens_used += response.usage.total_tokens
            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tcall in msg.tool_calls:
                    tname = tcall.function.name
                    targs = json.loads(tcall.function.arguments or "{}")
                    state.tool_calls_count += 1

                    observation = self.registry.execute(tname, targs)
                    state.steps.append(ResearchStep(
                        step_number=state.tool_calls_count,
                        action=tname,
                        arguments=targs,
                        observation_summary=observation[:300]
                    ))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tcall.id,
                        "content": observation
                    })
            else:
                state.termination_reason = "Information gathering concluded autonomously."
                break

    # -----------------------------------------------------------------------
    # Step 3: Synthesis & Verification (Facts vs Conclusions)
    # -----------------------------------------------------------------------
    def _synthesize_final_report(self, state: ResearchState, prior_context: str = "") -> str:
        """
        Produces the final structured report strictly distinguishing verified
        facts from analytical conclusions and indexing all source citations.
        """
        logger.info("Synthesizing final structured research dossier...")

        # Compile evidence context
        evidence_digest = []
        for sid, src in state.sources.items():
            content_sample = src.full_text[:1200] if src.full_text else src.snippet
            evidence_digest.append(
                f"Source [{sid}]: {src.title}\nURL: {src.url}\nExcerpt:\n{content_sample}\n"
            )

        context_str = "\n---\n".join(evidence_digest) if evidence_digest else "No external sources recorded."

        prior_section = f"\nESTABLISHED PRIOR RESEARCH CONTEXT FROM PREVIOUS SESSIONS:\n{prior_context}\n" if prior_context else ""

        synthesis_prompt = f"""
You are an unbiased Senior Research Scientist.
Your task is to synthesize the collected research evidence into a comprehensive, highly rigorous technical report.

RESEARCH QUESTION:
"{state.question}"
{prior_section}
RESEARCH PLAN SUB-QUESTIONS:
{json.dumps(state.plan.sub_questions if state.plan else [], indent=2)}

COLLECTED EXTERNAL EVIDENCE:
{context_str}

MANDATORY REPORT STRUCTURE:
You must strictly format your response using these exact Markdown sections:

# Research Report: {state.question}

## 1. Executive Summary
- A high-level overview of findings and bottom-line verdict.

## 2. Architectural Comparison Matrix
- Provide a clean Markdown table comparing the technologies across key dimensions (e.g., Latency, Complexity, Cost, Accuracy, Scalability).

## 3. Deep-Dive Comparative Analysis
- Detailed breakdown of each architecture.
- You MUST cite source IDs inline (e.g. [S1], [S2]) for all technical claims.

## 4. Conflicting Information & Nuances
- Identify where sources disagree, where marketing hype diverges from benchmark reality, or edge cases where trade-offs flip.

## 5. Verified Facts vs. Analytical Conclusions
Provide two distinct subsections:
### Verified Factual Findings
- Direct claims backed by external sources, each with its citation (e.g., "- Fact A [S1]").
### Analytical Conclusions & Recommendations
- Your technical deductions, architectural guidance, and situational recommendations.

## 6. Source Index & Provenance
- List every [S#] used, with its Title and URL.
"""

        try:
            if GEMINI_API_KEY:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GEMINI_API_KEY)
                resp = client.models.generate_content(
                    model=self.model,
                    contents=synthesis_prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                return resp.text
            else:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=self.model if "gpt" in self.model else "gpt-4o-mini",
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.2
                )
                return resp.choices[0].message.content
        except Exception as e:
            return f"Error during final report synthesis: {str(e)}"

    # -----------------------------------------------------------------------
    # Main Public Entrypoint
    # -----------------------------------------------------------------------
    def research(self, question: str, session_id: str = "default_session") -> ResearchState:
        """
        Runs the end-to-end research lifecycle with long-term memory retrieval and consolidation.
        """
        state = ResearchState(question=question, budget=self.budget)
        print("\n" + "=" * 70)
        print(f" RESEARCH MISSION: {question}")
        print("=" * 70)

        # 0. Retrieve Prior Long-Term Memory
        print("\n[PHASE 0]: Querying Long-Term Research Memory...")
        matches = self.memory.retrieve(question, limit=2)
        prior_context = ""
        if matches:
            state.retrieved_memories = [m.to_dict() for m, s in matches]
            print(f"-> Activated {len(matches)} prior research memor{'y' if len(matches) == 1 else 'ies'}:")
            for m, s in matches:
                print(f"   * '{m.topic}' (Confidence Match: {int(s * 100)}%)")
            prior_context = "\n\n".join(m.format_for_context() for m, s in matches)
        else:
            print("-> No prior memories matched above relevance threshold. Starting clean investigation.")

        # 1. Plan
        print("\n[PHASE 1]: Formulating Research Plan...")
        state.plan = self._create_research_plan(question, prior_context=prior_context)
        print(f"-> Generated {len(state.plan.sub_questions)} target sub-questions.")
        for sq in state.plan.sub_questions:
            print(f"   * {sq}")

        # 2. Investigate
        print("\n[PHASE 2]: Autonomous Investigation & Source Discovery...")
        self._execute_research_loop(state, prior_context=prior_context)
        print(f"-> Investigation finished: {state.tool_calls_count} tool calls | "
              f"{len(state.sources)} sources tracked | "
              f"Time: {round(state.elapsed_seconds(), 1)}s")

        # 3. Synthesize
        print("\n[PHASE 3]: Synthesizing Draft Report (Facts vs Conclusions)...")
        draft_report = self._synthesize_final_report(state, prior_context=prior_context)

        # 4. Verify & Self-Correct (Reflection & Grounding Loop)
        print("\n[PHASE 4]: Grounding Audit & Reflection Loop...")
        verification = self.verifier.audit_report(draft_report, state.sources)
        print(f"-> Citations Audited: {verification.total_claims_checked} claims | "
              f"Verified: {verification.verified_claims_count} | "
              f"Ungrounded/Phantom: {verification.unsupported_claims_count} | "
              f"Fidelity: {round(verification.fidelity_score * 100, 1)}%")

        if verification.phantom_sources_found:
            print(f"   [!] WARNING: Phantom citations detected: {verification.phantom_sources_found}")

        if verification.needs_refinement:
            print("-> Critique triggered: Refiner is eliminating hallucinations & ungrounded claims...")
            refined_report = self.verifier.refine_report(draft_report, verification, state.sources)
            verification.corrected_report = refined_report
            state.final_report = refined_report
            print("-> Self-Correction complete: Final grounded report generated.")
        else:
            print("-> Report passed verification with 100% citation grounding.")
            state.final_report = draft_report

        state.verification = verification
        state.completed = True

        # 5. Distill & Consolidate into Long-Term Memory
        print("\n[PHASE 5]: Distilling & Consolidating Verified Knowledge into Long-Term Memory...")
        clean_memory = MemoryFilter.extract_clean_memory(state, session_id=session_id)
        if clean_memory:
            saved_id = self.memory.save(clean_memory)
            state.memory_id = saved_id
            print(f"-> Verified research persisted to memory: '{clean_memory.topic}' (ID: {saved_id})")
            print(f"   * Stored {len(clean_memory.key_findings)} verified findings and {len(clean_memory.sources)} source anchors.")
        else:
            print("-> Memory filter discarded ephemeral or ungrounded artifacts (No memory persisted).")

        return state


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = PersonalResearchAgent(
        budget=ResearchBudget(
            max_tool_calls=6,
            max_execution_time_seconds=90.0,
            max_total_tokens=25000
        )
    )

    query = "Compare traditional RAG, Agentic RAG, and GraphRAG for enterprise applications."
    state = agent.research(query)

    print("\n" + "=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70 + "\n")
    print(state.final_report)

    # Save to disk as an artifact
    report_filename = "research_report.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(state.final_report or "No report generated.")
    print(f"\n[Artifact Saved]: {os.path.abspath(report_filename)}")
