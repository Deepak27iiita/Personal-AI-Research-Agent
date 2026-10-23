import os
import sys
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Logging Setup
# ---------------------------------------------------------------------------
# Standard Python logging allows us to trace execution without polluting stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LLMClient")


# ---------------------------------------------------------------------------
# 2. Configuration & Secrets Management
# ---------------------------------------------------------------------------
# Load variables from the local .env file into os.environ
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")


# ---------------------------------------------------------------------------
# 3. Core LLM Client Function
# ---------------------------------------------------------------------------
def call_llm(
    prompt: str,
    system_instruction: Optional[str] = "You are a concise, helpful research assistant.",
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Sends a message to the configured LLM provider and returns the structured response.

    Args:
        prompt: The user query to send to the model.
        system_instruction: The guiding persona or constraint for the model.
        model: Model identifier.

    Returns:
        dict containing:
            - text: The generated string reply.
            - model: The model that generated the reply.
            - usage: Token usage statistics (prompt, completion, total).
    
    Raises:
        ValueError: If no valid API key is found.
        RuntimeError: If the remote API call fails.
    """
    logger.info(f"Preparing to call model='{model}' with prompt: '{prompt[:60]}...'")

    # Guard: Ensure at least one API key exists
    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        error_msg = (
            "No API key detected! Please create a .env file and set either "
            "GEMINI_API_KEY or OPENAI_API_KEY."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # -----------------------------------------------------------------------
    # Route A: Google Gemini (Preferred default)
    # -----------------------------------------------------------------------
    if GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=GEMINI_API_KEY)

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2  # Low temperature for factual consistency
            )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            # Extract token usage if returned by the API
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            logger.info("Successfully received response from Gemini API.")
            return {
                "text": response.text,
                "model": model,
                "usage": usage
            }

        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            raise RuntimeError(f"Gemini API execution error: {e}") from e

    # -----------------------------------------------------------------------
    # Route B: OpenAI Fallback
    # -----------------------------------------------------------------------
    elif OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model if "gpt" in model else "gpt-4o-mini",
                messages=messages,
                temperature=0.2
            )

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            logger.info("Successfully received response from OpenAI API.")
            return {
                "text": response.choices[0].message.content,
                "model": response.model,
                "usage": usage
            }

        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise RuntimeError(f"OpenAI API execution error: {e}") from e


# ---------------------------------------------------------------------------
# 4. Entrypoint (Interactive CLI Execution)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Personal AI Research Agent - Milestone 1: LLM Connectivity")
    print("=" * 60 + "\n")

    test_prompt = "In 2 sentences, what is an AI Agent and how does it differ from a standard LLM?"

    try:
        result = call_llm(test_prompt)
        print("\n[AI RESPONSE]:")
        print("-" * 60)
        print(result["text"].strip())
        print("-" * 60)
        if result["usage"]:
            print(f"[TOKEN USAGE]: Prompt={result['usage'].get('prompt_tokens')} | "
                  f"Completion={result['usage'].get('completion_tokens')} | "
                  f"Total={result['usage'].get('total_tokens')}\n")
    except ValueError as val_err:
        print(f"\n[CONFIGURATION ERROR]: {val_err}\n")
        sys.exit(1)
    except RuntimeError as run_err:
        print(f"\n[EXECUTION ERROR]: {run_err}\n")
        sys.exit(1)
