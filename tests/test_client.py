import pytest
from unittest.mock import patch
from src.client import call_llm


def test_call_llm_missing_api_keys():
    """
    Ensures that call_llm raises ValueError with a helpful message
    if neither GEMINI_API_KEY nor OPENAI_API_KEY is configured.
    """
    with patch("src.client.GEMINI_API_KEY", None), patch("src.client.OPENAI_API_KEY", None):
        with pytest.raises(ValueError) as excinfo:
            call_llm("Hello world")
        assert "No API key detected" in str(excinfo.value)


def test_call_llm_gemini_error_handling():
    """
    Ensures that remote runtime errors from the provider are caught
    and cleanly re-raised as RuntimeError with appropriate context.
    """
    with patch("src.client.GEMINI_API_KEY", "fake_key_for_testing"):
        with patch("google.genai.Client", side_effect=Exception("Simulated 401 Unauthorized")):
            with pytest.raises(RuntimeError) as excinfo:
                call_llm("Test prompt")
            assert "Gemini API execution error" in str(excinfo.value)
