"""
llm.py
Provides a unified interface to call LLMs (Groq, Gemini) and track usage stats
(calls, prompt/response characters, tokens).
"""

import os
import time
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Module-level usage counters
_usage_stats: dict[str, int] = {
    "calls": 0,
    "prompt_chars": 0,
    "response_chars": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


# ============================================================================
# Custom Exceptions
# ============================================================================

class LLMUnavailableError(Exception):
    """Raised when the AI service is unreachable, timed out, or returning 503/429 after retries."""
    pass


class LLMConfigError(Exception):
    """Raised when LLM configuration is invalid: bad API key, missing key, 401/403/404, model not found."""
    pass


def _is_config_error(exc: Exception) -> bool:
    """
    Determines if an exception is an unrecoverable configuration/authentication issue
    (which should never be retried) rather than a transient availability failure.
    """
    if isinstance(exc, LLMConfigError):
        return True

    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    config_indicators = (
        "auth",
        "api_key",
        "apikey",
        "invalid_api_key",
        "unauthorized",
        "401",
        "permission",
        "forbidden",
        "403",
        "not_found",
        "not found",
        "model_not_found",
        "does not exist",
        "404",
    )

    if any(ind in exc_type for ind in ("auth", "permission", "notfound")):
        return True

    return any(ind in exc_str for ind in config_indicators)


def reset_usage_stats() -> None:
    """Resets the LLM usage statistics counters to zero."""
    global _usage_stats
    _usage_stats = {
        "calls": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def get_usage_stats() -> dict[str, int]:
    """Returns a copy of the current LLM usage statistics."""
    return dict(_usage_stats)


def _get_provider() -> str:
    """
    Determines which LLM provider to use.
    Priority:
      1. Explicit LLM_PROVIDER in .env ("groq" or "gemini")
      2. If GROQ_API_KEY is present and not a placeholder -> "groq"
      3. Otherwise -> "gemini"
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key != "your_groq_key_here":
        return "groq"

    return "gemini"


def _call_groq(prompt: str) -> tuple[str, int, int]:
    """
    Calls the Groq API using the official groq SDK.
    Returns (response_text, input_tokens, output_tokens).
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "your_groq_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing or not set in .env. "
            "Please add your valid Groq API key."
        )

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key.strip())

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    content = chat_completion.choices[0].message.content or ""
    input_tokens = 0
    output_tokens = 0

    if hasattr(chat_completion, "usage") and chat_completion.usage:
        input_tokens = getattr(chat_completion.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(chat_completion.usage, "completion_tokens", 0) or 0

    return content.strip(), input_tokens, output_tokens


def _call_gemini(prompt: str) -> tuple[str, int, int]:
    """
    Calls the Google Gemini API using langchain-google-genai.
    Returns (response_text, input_tokens, output_tokens).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "your_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or not set in .env. "
            "Please add your valid Gemini API key."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key.strip(),
        temperature=0.2,
    )
    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, str):
        result_text = content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif hasattr(block, "type") and getattr(block, "type") == "text":
                text_parts.append(getattr(block, "text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        result_text = "".join(text_parts)
    else:
        result_text = str(content)

    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata.get("input_tokens", 0) or 0
        output_tokens = response.usage_metadata.get("output_tokens", 0) or 0
    elif hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
        meta = response.response_metadata
        token_usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
        input_tokens = token_usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0
        output_tokens = token_usage.get("output_tokens") or token_usage.get("completion_tokens") or 0

    return result_text.strip(), int(input_tokens), int(output_tokens)


def call_llm(prompt: str) -> str:
    """
    Calls the active LLM provider (Groq or Gemini) with automatic retry and usage tracking.

    Retries on transient network/service errors (connection errors, timeouts, 503, 429) up to 4 attempts.
    Raises LLMConfigError immediately without retrying on configuration or authentication errors.
    Raises LLMUnavailableError after retries are exhausted.

    Args:
        prompt (str): The prompt string.

    Returns:
        str: Clean response text.
    """
    provider = _get_provider()
    max_retries = 4

    last_exception = None
    for attempt in range(max_retries):
        try:
            if provider == "groq":
                final_text, in_toks, out_toks = _call_groq(prompt)
            elif provider == "gemini":
                final_text, in_toks, out_toks = _call_gemini(prompt)
            else:
                raise LLMConfigError(f"Unsupported LLM_PROVIDER: '{provider}'. Use 'groq' or 'gemini'.")

            # Fallback estimation if provider did not return token counts
            if in_toks <= 0:
                in_toks = max(1, len(prompt) // 4)
            if out_toks <= 0:
                out_toks = max(1, len(final_text) // 4)

            # Record usage statistics
            _usage_stats["calls"] += 1
            _usage_stats["prompt_chars"] += len(prompt)
            _usage_stats["response_chars"] += len(final_text)
            _usage_stats["input_tokens"] += in_toks
            _usage_stats["output_tokens"] += out_toks
            _usage_stats["total_tokens"] = _usage_stats["input_tokens"] + _usage_stats["output_tokens"]

            return final_text

        except Exception as e:
            if _is_config_error(e):
                raise LLMConfigError(f"LLM configuration error: {e}") from e

            last_exception = e
            print(f"[{provider}] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Error during {provider} LLM call after {max_retries} attempts: {e}")
                raise LLMUnavailableError(
                    f"AI service unreachable after {max_retries} attempts: {last_exception}"
                ) from last_exception

    raise LLMUnavailableError(f"AI service unreachable: {last_exception}")


def check_llm_available() -> tuple[bool, str]:
    """
    Performs a lightweight probe call to verify AI service connectivity and credentials.
    Makes a single attempt without updating usage statistics.

    Returns:
        tuple[bool, str]: (True, "") on success, or (False, friendly_message) on failure.
    """
    provider = _get_provider()
    probe_prompt = "Reply with OK"

    try:
        if provider == "groq":
            text, _, _ = _call_groq(probe_prompt)
        elif provider == "gemini":
            text, _, _ = _call_gemini(probe_prompt)
        else:
            return False, f"Unsupported LLM_PROVIDER: '{provider}'. Check your .env file."

        if not text:
            return False, "AI service returned an empty response."
        return True, ""

    except Exception as e:
        if _is_config_error(e):
            return False, f"AI service configuration problem (API key or model name): {e}"
        return False, f"Can't reach the AI service. Check your internet connection ({e})."


if __name__ == "__main__":
    # Standalone sanity test
    print(f"Testing call_llm() with provider: {_get_provider()}...")
    reset_usage_stats()
    try:
        result = call_llm("Respond with 'LLM connection successful!' if you can read this.")
        print("Response received:")
        print(result)
        print("Usage stats:")
        print(get_usage_stats())
    except Exception as err:
        print(f"Test encountered an issue: {err}")
