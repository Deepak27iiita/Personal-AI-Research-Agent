"""
Manual Agent Execution Loop (ReAct State Machine)

This module implements the core agentic reasoning cycle from scratch:
1. Agent State & Working Memory management
2. Tool selection via native LLM Function Calling schemas
3. Tool argument validation and safe execution
4. Tool results returned as observations back to the LLM
5. Infinite loop detection and maximum step limits
6. Graceful error recovery
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from src.tools import ToolRegistry, create_default_registry

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AgentLoop")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")


# ---------------------------------------------------------------------------
# 1. Agent State
# ---------------------------------------------------------------------------
@dataclass
class AgentStep:
    step_number: int
    thought: Optional[str]
    tool_name: Optional[str]
    tool_arguments: Optional[Dict[str, Any]]
    observation: Optional[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """
    Represents the full working memory and trajectory of an agentic session.
    """
    objective: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_iterations: int = 0
    max_iterations: int = 6
    completed: bool = False
    start_time: float = field(default_factory=time.time)

    def record_step(self, step: AgentStep):
        self.steps.append(step)
        self.total_iterations += 1

    def is_cycle_detected(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Detects if the agent is 'thrashing' by repeating the exact same
        tool call with the exact same arguments multiple times.
        """
        call_signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        count = 0
        for s in self.steps:
            if s.tool_name and s.tool_arguments is not None:
                past_sig = f"{s.tool_name}:{json.dumps(s.tool_arguments, sort_keys=True)}"
                if past_sig == call_signature:
                    count += 1
        return count >= 2  # Already tried twice with no new result


# ---------------------------------------------------------------------------
# 2. ReAct Agent Implementation
# ---------------------------------------------------------------------------
class ResearchAgent:
    """
    A standalone ReAct agent that autonomously selects tools and solves queries.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 6
    ):
        self.registry = registry or create_default_registry()
        self.model = model
        self.max_iterations = max_iterations

        if not GEMINI_API_KEY and not OPENAI_API_KEY:
            raise ValueError(
                "Missing API Key. Set GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
            )

    def _convert_schemas_to_gemini(self) -> List[Any]:
        """Converts registry schemas into Gemini's expected types."""
        from google.genai import types

        gemini_tools = []
        for schema in self.registry.get_schemas():
            # Build property definitions
            properties = {}
            for prop_name, prop_data in schema["parameters"]["properties"].items():
                prop_type = prop_data.get("type", "string").upper()
                type_enum = getattr(types.Type, prop_type, types.Type.STRING)
                properties[prop_name] = types.Schema(
                    type=type_enum,
                    description=prop_data.get("description", "")
                )

            declaration = types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=schema["parameters"].get("required", [])
                )
            )
            gemini_tools.append(declaration)

        return [types.Tool(function_declarations=gemini_tools)]

    def _convert_schemas_to_openai(self) -> List[Dict[str, Any]]:
        """Converts registry schemas into OpenAI's function calling schema."""
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
        return tools

    def run(self, objective: str) -> AgentState:
        """
        Executes the autonomous ReAct agent loop until the objective is satisfied,
        the iteration limit is reached, or a loop cycle is caught.
        """
        state = AgentState(objective=objective, max_iterations=self.max_iterations)
        logger.info(f"Starting agent run for objective: '{objective}'")

        if GEMINI_API_KEY:
            return self._run_gemini_loop(state)
        else:
            return self._run_openai_loop(state)

    # -----------------------------------------------------------------------
    # Gemini ReAct Loop
    # -----------------------------------------------------------------------
    def _run_gemini_loop(self, state: AgentState) -> AgentState:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        tools = self._convert_schemas_to_gemini()

        system_instruction = (
            "You are an expert Research Assistant. You have access to tools. "
            "When asked a question, decide whether to use one or more tools to retrieve "
            "facts, calculate values, check the time, or record notes. "
            "Do NOT guess facts or dates; verify using tools when needed. "
            "When you have collected all required information, provide your comprehensive "
            "final answer directly to the user."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.1
        )

        # Multi-turn message history for Gemini
        conversation_contents: List[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=state.objective)]
            )
        ]

        while not state.completed:
            # 1. Protection: Max iterations guard
            if state.total_iterations >= state.max_iterations:
                logger.warning(f"Hit maximum iteration limit ({state.max_iterations}). Forcing termination.")
                state.final_answer = (
                    f"Agent reached the maximum iteration budget ({state.max_iterations} steps) "
                    f"before fully concluding. Observations collected: {len(state.steps)} steps."
                )
                state.completed = True
                break

            step_num = state.total_iterations + 1
            logger.info(f"--- [Step {step_num}/{state.max_iterations}] Calling LLM ---")

            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=conversation_contents,
                    config=config
                )
            except Exception as e:
                logger.error(f"LLM API Call Error: {e}")
                state.final_answer = f"Agent stopped due to API error: {str(e)}"
                state.completed = True
                break

            # Append the assistant's turn to conversation history
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                state.final_answer = "Error: Empty response from LLM."
                state.completed = True
                break

            conversation_contents.append(candidate.content)

            # Check if model requested a tool call (function call)
            tool_calls = response.function_calls

            if tool_calls:
                for call in tool_calls:
                    tool_name = call.name
                    tool_args = dict(call.args) if call.args else {}

                    logger.info(f"LLM Selected Tool: '{tool_name}' with args: {tool_args}")

                    # 2. Protection: Infinite loop cycle detection
                    if state.is_cycle_detected(tool_name, tool_args):
                        logger.warning(f"Detected repeated call loop for tool '{tool_name}'. Intervening.")
                        observation = (
                            f"System Warning: You have already executed '{tool_name}' with these exact "
                            f"arguments twice. Please stop repeating this call and either use a different "
                            f"query or synthesize your final answer."
                        )
                    else:
                        # 3. Tool Execution via Registry
                        observation = self.registry.execute(tool_name, tool_args)

                    logger.info(f"Tool Observation (preview): {observation[:120]}...")

                    # Record step in Agent State
                    step = AgentStep(
                        step_number=step_num,
                        thought=None,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        observation=observation
                    )
                    state.record_step(step)

                    # 4. Feed Observation back to the LLM
                    tool_response_part = types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation}
                    )
                    conversation_contents.append(
                        types.Content(role="user", parts=[tool_response_part])
                    )

            else:
                # No tool was requested: the LLM produced its final answer
                final_text = response.text or "Completed with no text."
                logger.info("LLM determined the task is complete. Final answer generated.")
                state.final_answer = final_text
                state.completed = True

        return state

    # -----------------------------------------------------------------------
    # OpenAI ReAct Loop
    # -----------------------------------------------------------------------
    def _run_openai_loop(self, state: AgentState) -> AgentState:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        tools = self._convert_schemas_to_openai()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Research Assistant with access to tools. "
                    "Use tools to fetch live facts, evaluate calculations, check time, or save notes. "
                    "Provide a thorough final answer once you have all the facts."
                )
            },
            {"role": "user", "content": state.objective}
        ]

        while not state.completed:
            if state.total_iterations >= state.max_iterations:
                state.final_answer = f"Agent reached step limit ({state.max_iterations})."
                state.completed = True
                break

            step_num = state.total_iterations + 1
            response = client.chat.completions.create(
                model=self.model if "gpt" in self.model else "gpt-4o-mini",
                messages=messages,
                tools=tools,
                temperature=0.1
            )

            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tcall in msg.tool_calls:
                    tname = tcall.function.name
                    targs = json.loads(tcall.function.arguments or "{}")

                    if state.is_cycle_detected(tname, targs):
                        observation = "Loop detected: You already called this tool with these arguments."
                    else:
                        observation = self.registry.execute(tname, targs)

                    step = AgentStep(
                        step_number=step_num,
                        thought=None,
                        tool_name=tname,
                        tool_arguments=targs,
                        observation=observation
                    )
                    state.record_step(step)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tcall.id,
                        "content": observation
                    })
            else:
                state.final_answer = msg.content
                state.completed = True

        return state


# ---------------------------------------------------------------------------
# 3. Interactive CLI Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" Personal AI Research Agent - Multi-Tool Autonomous ReAct Loop")
    print("=" * 65 + "\n")

    agent = ResearchAgent(max_iterations=6)

    # Example question that requires multi-step tool use:
    # 1. Check current date
    # 2. Perform a calculation or lookup
    # 3. Save research note
    demo_query = (
        "What is the current date and time? Also calculate (25 * 14) + 120, "
        "and save a research note titled 'Math Benchmark' with the result."
    )

    print(f"Goal: {demo_query}\n")
    state = agent.run(demo_query)

    print("\n" + "=" * 65)
    print(f"Agent Execution Summary: {state.total_iterations} steps taken in "
          f"{round(time.time() - state.start_time, 2)}s")
    print("=" * 65)
    for s in state.steps:
        print(f"\n[Step {s.step_number}]: Tool -> {s.tool_name}({s.tool_arguments})")
        print(f"Observation: {s.observation[:160]}...")

    print("\n" + "=" * 65)
    print("[FINAL ANSWER]:")
    print("=" * 65)
    print(state.final_answer)
