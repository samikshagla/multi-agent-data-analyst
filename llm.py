"""
llm.py
Provides a unified interface to call LLMs (Groq, Gemini) and track usage stats
(calls, prompt/response characters, tokens).
Supports thread-safe, session-scoped LLM configuration and usage tracking for multi-user hosting.
"""

import copy
import inspect
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Custom Exceptions
# ============================================================================

class LLMUnavailableError(Exception):
    """Raised when the AI service is unreachable, timed out, or returning 503/429 after retries."""
    pass


class LLMConfigError(Exception):
    """Raised when LLM configuration is invalid: bad API key, missing key, 401/403/404, model not found."""
    pass


class LLMBudgetError(LLMConfigError):
    """Raised when an LLM session has reached or exceeded its configured max_tokens budget."""
    pass


class LLMDisabledError(Exception):
    """Raised when LLM calls are disabled in demo mode and no custom context config is provided."""
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


# ============================================================================
# Key Hygiene & Redaction
# ============================================================================

def mask_key(key: str) -> str:
    """Masks an API key for safe display (e.g. 'gsk_***abcd')."""
    if not key:
        return ""
    k = key.strip()
    if len(k) <= 8:
        return "***" + (k[-3:] if len(k) >= 3 else "")
    prefix = k[:4]
    suffix = k[-4:]
    return f"{prefix}***{suffix}"


def redact(text: str, key: str | None = None) -> str:
    """
    Redacts sensitive API keys and any contiguous substrings >= 8 characters
    from text to prevent accidental exposure in errors, logs, and tracebacks.
    """
    if not text:
        return ""

    keys_to_redact: list[str] = []
    if key:
        keys_to_redact.append(key)
    try:
        ctx_cfg = _current_config.get()
        if ctx_cfg and ctx_cfg.api_key:
            keys_to_redact.append(ctx_cfg.api_key)
    except Exception:
        pass

    for env_var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        val = os.getenv(env_var, "").strip()
        if val and val not in ("your_groq_key_here", "your_key_here"):
            keys_to_redact.append(val)

    for k in keys_to_redact:
        k = k.strip()
        if not k:
            continue
        if len(k) < 8:
            text = text.replace(k, "[REDACTED_KEY]")
            continue
        if k in text:
            text = text.replace(k, "[REDACTED_KEY]")
        n = len(k)
        seen = set()
        for length in range(n - 1, 7, -1):
            for i in range(n - length + 1):
                sub = k[i:i + length]
                if sub in seen:
                    continue
                seen.add(sub)
                if sub in text:
                    text = text.replace(sub, "[REDACTED]")

    return text


# ============================================================================
# Multi-User LLM Config & Usage Tracking
# ============================================================================

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    max_tokens: int | None = None

    def __repr__(self) -> str:
        masked = mask_key(self.api_key)
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={masked!r}, max_tokens={self.max_tokens!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class UsageTracker:
    """Thread-safe usage tracker maintaining per-session and per-model metrics."""

    def __init__(self, max_tokens: int | None = None):
        self._lock = threading.Lock()
        self.max_tokens = max_tokens
        self._stats: dict[str, Any] = {
            "calls": 0,
            "prompt_chars": 0,
            "response_chars": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "by_model": {},
        }

    def check_budget(self) -> None:
        """Raises LLMBudgetError if tokens used so far meet or exceed max_tokens."""
        if self.max_tokens is not None:
            with self._lock:
                if self._stats["total_tokens"] >= self.max_tokens:
                    raise LLMBudgetError(
                        f"This session reached its token limit ({self._stats['total_tokens']}/{self.max_tokens} tokens)."
                    )

    def record_call(
        self,
        prompt: str,
        response: str,
        in_toks: int,
        out_toks: int,
        provider: str,
        model_name: str,
    ) -> None:
        """Records a successful LLM call into usage statistics."""
        with self._lock:
            self._stats["calls"] += 1
            self._stats["prompt_chars"] += len(prompt)
            self._stats["response_chars"] += len(response)
            self._stats["input_tokens"] += in_toks
            self._stats["output_tokens"] += out_toks
            self._stats["total_tokens"] += (in_toks + out_toks)

            by_model = self._stats.setdefault("by_model", {})
            if model_name not in by_model:
                by_model[model_name] = {
                    "provider": provider,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tokens": 0,
                    "total_tokens": 0,
                }
            by_model[model_name]["calls"] += 1
            by_model[model_name]["input_tokens"] += in_toks
            by_model[model_name]["output_tokens"] += out_toks
            by_model[model_name]["tokens"] += (in_toks + out_toks)
            by_model[model_name]["total_tokens"] += (in_toks + out_toks)

    def get_stats(self) -> dict[str, Any]:
        """Returns a deep copy of the tracked statistics."""
        with self._lock:
            return copy.deepcopy(self._stats)

    def reset(self) -> None:
        """Resets all usage counters to zero."""
        with self._lock:
            self._stats = {
                "calls": 0,
                "prompt_chars": 0,
                "response_chars": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "by_model": {},
            }


_global_tracker = UsageTracker()
_usage_stats: dict[str, Any] = _global_tracker._stats

_current_config: ContextVar[LLMConfig | None] = ContextVar("_current_config", default=None)
_current_tracker: ContextVar[UsageTracker | None] = ContextVar("_current_tracker", default=None)


@contextmanager
def llm_session(config: LLMConfig):
    """
    Context manager setting an isolated session LLM configuration and usage tracker.
    Nesting and exception safe; restores previous context variables on exit without touching os.environ.
    Suppresses provider SDK debug logging to prevent key leaks.
    """
    tracker = UsageTracker(max_tokens=config.max_tokens)
    token_cfg = _current_config.set(config)
    token_trk = _current_tracker.set(tracker)

    # Provider SDK debug logging suppression
    loggers_to_silence = ["groq", "google", "langchain_google_genai", "urllib3", "httpx", "httpcore"]
    prev_levels = {}
    for name in loggers_to_silence:
        log = logging.getLogger(name)
        prev_levels[log] = log.level
        if log.level < logging.WARNING:
            log.setLevel(logging.WARNING)

    try:
        yield tracker
    finally:
        for log, lvl in prev_levels.items():
            log.setLevel(lvl)
        _current_tracker.reset(token_trk)
        _current_config.reset(token_cfg)


def reset_usage_stats() -> None:
    """Resets the active LLM usage statistics counters to zero."""
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.reset()
    else:
        _global_tracker.reset()


def get_usage_stats() -> dict[str, Any]:
    """Returns a deep copy of the active LLM usage statistics."""
    tracker = _current_tracker.get()
    if tracker is not None:
        return tracker.get_stats()
    return _global_tracker.get_stats()


# ============================================================================
# Provider Helper Functions
# ============================================================================

def _get_model_name(provider: str) -> str:
    """
    Returns the configured model name for the specified provider.
    Groq defaults to 'openai/gpt-oss-120b', Gemini defaults to 'gemini-2.0-flash'.
    """
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    elif provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return os.getenv("LLM_MODEL", "unknown")


def _get_provider() -> str:
    """
    Determines which LLM provider to use from the environment.
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


def _safe_call_provider(
    fn: Any,
    prompt: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> tuple[str, int, int]:
    """
    Safely dispatches to a provider call function or test mock,
    adapting to either single-arg or multi-arg signatures without
    re-calling the function on internal application errors.
    """
    use_kwargs = False
    kwargs = {}
    use_single = False
    use_multi = False

    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if has_var_kw:
            use_kwargs = True
            kwargs = {"api_key": api_key, "model_name": model_name}
        else:
            param_names = [p.name for p in params]
            if "api_key" in param_names:
                kwargs["api_key"] = api_key
            if "model_name" in param_names:
                kwargs["model_name"] = model_name
            if kwargs:
                use_kwargs = True
            elif len(params) == 1:
                use_single = True
            else:
                use_multi = True
    except (TypeError, ValueError):
        pass

    if use_kwargs:
        return fn(prompt, **kwargs)
    elif use_single:
        return fn(prompt)
    elif use_multi:
        return fn(prompt, api_key, model_name)

    try:
        return fn(prompt, api_key=api_key, model_name=model_name)
    except TypeError:
        return fn(prompt)


def _call_groq(
    prompt: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> tuple[str, int, int]:
    """
    Calls the Groq API using the official groq SDK.
    Returns (response_text, input_tokens, output_tokens).
    """
    from groq import Groq

    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.strip() == "your_groq_key_here":
            raise ValueError(
                "GROQ_API_KEY is missing or not set in .env. "
                "Please add your valid Groq API key."
            )

    if model_name is None:
        model_name = _get_model_name("groq")

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


def _call_gemini(
    prompt: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> tuple[str, int, int]:
    """
    Calls the Google Gemini API using langchain-google-genai.
    Returns (response_text, input_tokens, output_tokens).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.strip() == "your_key_here":
            raise ValueError(
                "GEMINI_API_KEY is missing or not set in .env. "
                "Please add your valid Gemini API key."
            )

    if model_name is None:
        model_name = _get_model_name("gemini")

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
                text_parts.append(getattr(block, "type", ""))
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


# ============================================================================
# Main Invocation & Health Check API
# ============================================================================

def call_llm(prompt: str) -> str:
    """
    Calls the active LLM provider (Groq or Gemini) with automatic retry and usage tracking.

    Resolves configuration in order:
      1. Context config from llm_session (if active)
      2. Environment variables (.env)

    Enforces DEMO_ONLY guard, session token budgets, and key redaction across all exceptions.
    """
    # 1. DEMO_ONLY and HOSTED guards
    ctx_config = _current_config.get()
    if os.environ.get("DEMO_ONLY") == "1" and ctx_config is None:
        raise LLMDisabledError("LLM calls are disabled in demo mode")
    if os.environ.get("HOSTED") == "1" and ctx_config is None:
        raise LLMDisabledError("Server-side API keys are disabled on this host")

    # 2. Check budget BEFORE call
    tracker = _current_tracker.get() or _global_tracker
    tracker.check_budget()

    # 3. Resolve configuration
    if ctx_config is not None:
        provider = ctx_config.provider.strip().lower()
        model_name = ctx_config.model
        api_key = ctx_config.api_key
    else:
        provider = _get_provider()
        model_name = _get_model_name(provider)
        api_key = None

    max_retries = 4
    last_exception = None

    for attempt in range(max_retries):
        try:
            if provider == "groq":
                final_text, in_toks, out_toks = _safe_call_provider(
                    _call_groq, prompt, api_key=api_key, model_name=model_name
                )
            elif provider == "gemini":
                final_text, in_toks, out_toks = _safe_call_provider(
                    _call_gemini, prompt, api_key=api_key, model_name=model_name
                )
            else:
                raise LLMConfigError(f"Unsupported LLM_PROVIDER: '{provider}'. Use 'groq' or 'gemini'.")

            # Fallback estimation if provider did not return token counts
            if in_toks <= 0:
                in_toks = max(1, len(prompt) // 4)
            if out_toks <= 0:
                out_toks = max(1, len(final_text) // 4)

            # Record usage statistics
            tracker.record_call(prompt, final_text, in_toks, out_toks, provider, model_name)

            return final_text

        except Exception as e:
            if isinstance(e, (LLMBudgetError, LLMDisabledError)):
                raise
            redacted_err_str = redact(str(e), key=api_key)
            if _is_config_error(e):
                del e
                del api_key
                raise LLMConfigError(f"LLM configuration error: {redacted_err_str}") from None

            last_exception = e
            print(f"[{provider}] Attempt {attempt + 1}/{max_retries} failed: {redacted_err_str}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Error during {provider} LLM call after {max_retries} attempts: {redacted_err_str}")
                del e
                del api_key
                raise LLMUnavailableError(
                    f"AI service unreachable after {max_retries} attempts: {redacted_err_str}"
                ) from None

    redacted_last = redact(str(last_exception), key=api_key) if last_exception else "unknown error"
    del last_exception
    del api_key
    raise LLMUnavailableError(f"AI service unreachable: {redacted_last}") from None


def check_llm_available(config: LLMConfig | None = None) -> tuple[bool, str]:
    """
    Performs a lightweight probe call to verify AI service connectivity and credentials.
    Makes a single attempt without updating usage statistics.

    Returns:
        tuple[bool, str]: (True, "") on success, or (False, friendly_message) on failure.
    """
    ctx_config = config if config is not None else _current_config.get()
    if os.environ.get("DEMO_ONLY") == "1" and ctx_config is None:
        return False, "LLM calls are disabled in demo mode"
    if os.environ.get("HOSTED") == "1" and ctx_config is None:
        return False, "Server-side API keys are disabled on this host"

    if ctx_config is not None:
        provider = ctx_config.provider.strip().lower()
        model_name = ctx_config.model
        api_key = ctx_config.api_key
    else:
        provider = _get_provider()
        model_name = _get_model_name(provider)
        api_key = None

    probe_prompt = "Reply with OK"

    try:
        if provider == "groq":
            text, _, _ = _safe_call_provider(_call_groq, probe_prompt, api_key=api_key, model_name=model_name)
        elif provider == "gemini":
            text, _, _ = _safe_call_provider(_call_gemini, probe_prompt, api_key=api_key, model_name=model_name)
        else:
            return False, f"Unsupported LLM_PROVIDER: '{provider}'. Check your .env file."

        if not text:
            return False, "AI service returned an empty response."
        return True, ""

    except Exception as e:
        redacted_err = redact(str(e), key=api_key)
        if _is_config_error(e):
            return False, f"AI service configuration problem (API key or model name): {redacted_err}"
        return False, f"Can't reach the AI service. Check your internet connection ({redacted_err})."


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
