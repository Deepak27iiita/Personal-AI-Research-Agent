"""
Research Tools with Security Boundaries and Source Tracking.

Tools implemented:
1. web_search: Live search with source ID registration.
2. read_web_page: Web scraper with SSRF protection and HTML boilerplate stripping.
3. calculator: AST-based safe math evaluator.
4. get_current_time: System temporal anchor.
"""

import os
import ast
import operator
import datetime
import ipaddress
import urllib.parse
from typing import Dict, Any, Callable, List, Optional
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from src.models import ResearchState


# ---------------------------------------------------------------------------
# 1. SSRF Security Guard
# ---------------------------------------------------------------------------
def is_safe_url(url: str) -> (bool, Optional[str]):
    """
    Prevents Server-Side Request Forgery (SSRF) by validating that the target
    is an external HTTP/HTTPS host and not an internal network or cloud metadata IP.
    """
    try:
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return False, f"Blocked protocol '{parsed.scheme}://'. Only http and https allowed."

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: missing hostname."

        # Check for localhost / loopback aliases
        if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return False, "Access to localhost/loopback is forbidden."

        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"Access to private/internal IP {hostname} is blocked."
        except ValueError:
            # Hostname is a domain name, not a raw IP
            pass

        return True, None
    except Exception as e:
        return False, f"URL validation failed: {str(e)}"


# ---------------------------------------------------------------------------
# 2. Web Page Reading Tool
# ---------------------------------------------------------------------------
def read_web_page(
    url: str,
    max_chars: int = 4000,
    state: Optional[ResearchState] = None
) -> str:
    """
    Fetches the content of a web page, strips scripts/ads/HTML tags,
    and returns clean, readable text within token limits.
    """
    safe, err = is_safe_url(url)
    if not safe:
        return f"Security Policy Violation: {err}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 (PersonalAIResearchBot/1.0)"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    }

    try:
        with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            if response.status_code != 200:
                return f"HTTP Error {response.status_code}: Unable to access {url}"

            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type:
                return f"Notice: PDF content at {url} requires PDF parser. Summary unavailable."

            # Parse HTML and strip non-content elements
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "aside"]):
                element.extract()

            # Extract title and body text
            page_title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
            body_text = soup.get_text(separator="\n", strip=True)

            # Clean duplicate blank lines
            lines = [line.strip() for line in body_text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            # Cap character limit to prevent context window explosion
            truncated = clean_text[:max_chars]
            if len(clean_text) > max_chars:
                truncated += f"\n... [Truncated {len(clean_text) - max_chars} characters for brevity]"

            # Register full text in Source Tracker if state provided
            source_id = "N/A"
            if state:
                source = state.add_source(
                    url=url,
                    title=page_title,
                    snippet=truncated[:250],
                    full_text=truncated
                )
                source_id = source.id

            return (
                f"=== Page Content [{source_id}] ===\n"
                f"Title: {page_title}\n"
                f"URL: {url}\n"
                f"Text:\n{truncated}"
            )

    except httpx.TimeoutException:
        return f"Network Timeout: Server at {url} did not respond within 8 seconds."
    except Exception as e:
        return f"Error reading web page '{url}': {str(e)}"


# ---------------------------------------------------------------------------
# 3. Web Search Tool with Source Tracking
# ---------------------------------------------------------------------------
def web_search(
    query: str,
    max_results: int = 3,
    state: Optional[ResearchState] = None
) -> str:
    """
    Performs DuckDuckGo search and registers candidate sources with unique Source IDs.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No search results found for: '{query}'."

        formatted_lines = [f"Web Search Results for '{query}':\n"]
        for item in results:
            title = item.get("title", "No Title")
            snippet = item.get("body", "No Snippet")
            url = item.get("href", "")

            source_id = "N/A"
            if state:
                source = state.add_source(url=url, title=title, snippet=snippet)
                source_id = source.id

            formatted_lines.append(
                f"Source [{source_id}]: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {snippet}\n"
            )

        return "\n".join(formatted_lines)
    except Exception as e:
        return f"Web search failed: {str(e)}"


# ---------------------------------------------------------------------------
# 4. Safe Calculator
# ---------------------------------------------------------------------------
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported math operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return ALLOWED_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return ALLOWED_OPERATORS[op_type](_eval_node(node.operand))
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Safely calculates arithmetic expressions via AST."""
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(parsed.body)
        return f"Calculation Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


# ---------------------------------------------------------------------------
# 5. Current Time Tool
# ---------------------------------------------------------------------------
def get_current_time(timezone: str = "local") -> str:
    """Returns the current local date, time, and day."""
    now = datetime.datetime.now()
    return f"Current Date & Time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S (%A)')}"


# ---------------------------------------------------------------------------
# 6. Long-Term Memory Recall Tool
# ---------------------------------------------------------------------------
def recall_prior_research(query: str, memory_store: Optional[Any] = None) -> str:
    """
    Recalls verified research findings, comparison dimensions, and citations
    from previous research sessions stored in long-term memory.
    """
    try:
        from src.memory import MemoryStore
        store = memory_store or MemoryStore()
        matches = store.retrieve(query, limit=2)
        if not matches:
            return f"No relevant prior research found in long-term memory for: '{query}'."

        output = [f"Found {len(matches)} relevant prior research memories for '{query}':\n"]
        for mem, score in matches:
            output.append(mem.format_for_context())
            output.append(f"Relevance Confidence: {int(score * 100)}%\n")

        return "\n".join(output)
    except Exception as e:
        return f"Error recalling prior research: {str(e)}"


# ---------------------------------------------------------------------------
# 6. Research Tool Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """
    Manages research tools and injects the active ResearchState for provenance tracking.
    """

    def __init__(self, state: Optional[ResearchState] = None):
        self.state = state
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def set_state(self, state: ResearchState):
        self.state = state

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self._tools[name] = func
        self._schemas.append({
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters.get("properties", {}),
                "required": parameters.get("required", [])
            }
        })

    def get_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' does not exist. Available tools: {list(self._tools.keys())}"

        tool_func = self._tools[tool_name]
        try:
            # Inject state if the tool accepts it (e.g., for source tracking)
            if tool_name in ("web_search", "read_web_page") and self.state is not None:
                arguments["state"] = self.state

            return str(tool_func(**arguments))
        except TypeError as type_err:
            return f"Argument Error in '{tool_name}': {str(type_err)}"
        except Exception as exc:
            return f"Execution Error in '{tool_name}': {str(exc)}"


def create_research_registry(state: Optional[ResearchState] = None) -> ToolRegistry:
    """
    Builds the complete research tool suite.
    """
    registry = ToolRegistry(state=state)

    registry.register(
        name="web_search",
        description="Search the web for technical documentation, papers, and comparisons.",
        parameters={
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for (e.g., 'Agentic RAG vs Traditional RAG enterprise benchmark')."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of search results to retrieve (default: 3)."
                }
            },
            "required": ["query"]
        },
        func=web_search
    )

    registry.register(
        name="read_web_page",
        description="Fetch and read the full text of an article or documentation page from a URL.",
        parameters={
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact HTTP or HTTPS URL to read."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to extract (default: 4000)."
                }
            },
            "required": ["url"]
        },
        func=read_web_page
    )

    registry.register(
        name="calculator",
        description="Evaluate mathematical expressions, latency numbers, or cost estimates.",
        parameters={
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g., '120 * 0.00015')."
                }
            },
            "required": ["expression"]
        },
        func=calculator
    )

    registry.register(
        name="get_current_time",
        description="Check current system date and time.",
        parameters={
            "properties": {
                "timezone": {"type": "string", "description": "Optional timezone string"}
            },
            "required": []
        },
        func=get_current_time
    )

    registry.register(
        name="recall_prior_research",
        description="Search persistent long-term memory for verified research findings and conclusions from previous sessions.",
        parameters={
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or concept to recall from previous research sessions (e.g. 'Agentic RAG findings')."
                }
            },
            "required": ["query"]
        },
        func=recall_prior_research
    )

    return registry


# ---------------------------------------------------------------------------
# Backward Compatibility & Note Taking
# ---------------------------------------------------------------------------
RESEARCH_NOTES_FILE = os.path.join(os.path.dirname(__file__), "..", "research_notes.md")


def save_research_note(topic: str, content: str) -> str:
    """Appends notes to research_notes.md."""
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(RESEARCH_NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n\n### Note: {topic}\n- **Recorded at:** {ts}\n- **Content:** {content.strip()}\n---")
        return f"Success: Research note on '{topic}' was successfully saved to research_notes.md."
    except Exception as e:
        return f"Error saving research note: {str(e)}"


def create_default_registry() -> ToolRegistry:
    """Creates registry with all standard tools plus save_research_note."""
    reg = create_research_registry()
    reg.register(
        name="save_research_note",
        description="Save notes to research_notes.md",
        parameters={
            "properties": {
                "topic": {"type": "string", "description": "Note topic"},
                "content": {"type": "string", "description": "Note content"}
            },
            "required": ["topic", "content"]
        },
        func=save_research_note
    )
    return reg
