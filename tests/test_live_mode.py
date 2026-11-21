"""
tests/test_live_mode.py
Comprehensive test suite verifying visitor live-mode UI, user-provided API key handling,
strict key hygiene, prerequisite gating, token budget enforcement, and session isolation.
"""

import json
import os
import re
from pathlib import Path
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

from llm import LLMBudgetError, LLMConfig, redact
from components.results_data import review_json_bytes, cleaned_csv_bytes
from components.demo_snapshot import save_snapshot, load_snapshot


APP_PATH = str(Path(__file__).parent.parent / "app.py")
CANARY_KEY = "gsk_TESTKEY0123456789abcdef"


def get_canary_substrings(key: str = CANARY_KEY, min_len: int = 8) -> list[str]:
    """Generates all contiguous substrings of length >= min_len."""
    return [key[i:j] for i in range(len(key)) for j in range(i + min_len, len(key) + 1)]


def fake_groq_pipeline(prompt: str, api_key: str | None = None, model_name: str | None = None) -> tuple[str, int, int]:
    """Deterministic fake LLM provider returning valid responses for every agent node."""
    p = prompt.lower()
    if "reply with ok" in p:
        return "OK", 5, 5
    elif "cleaning plan" in p or ("clean" in p and "json" in p):
        text = json.dumps([{"action": "strip_whitespace", "column": "all", "reason": "Trim whitespace"}])
    elif "questions" in p or "business questions" in p:
        text = json.dumps(["What is total records?"])
    elif "verdict" in p or "critique" in p or "review" in p:
        text = json.dumps({"verdict": "pass", "reason": "Verified accurate", "confidence": 0.95})
    elif "chart" in p or "visual" in p:
        text = json.dumps({"chart_type": "metric", "title": "Total Records", "x": None, "y": None, "takeaway": "Total count is recorded."})
    elif "report" in p or "executive" in p:
        text = "# Executive Summary\nData analysis completed with high confidence."
    elif "code" in p or "question" in p:
        text = "result = len(df)"
    else:
        text = "OK"
    return text, 40, 40


def extract_all_rendered_text(at: AppTest) -> str:
    """Collects text from all rendered element types in an AppTest session."""
    texts = []
    for attr in ("markdown", "text", "caption", "error", "info", "warning", "code", "title", "header", "subheader", "success"):
        for elem in getattr(at, attr, []):
            val = getattr(elem, "value", None) or getattr(elem, "body", None) or str(elem)
            texts.append(str(val))
    return " ".join(texts)


# ============================================================================
# 1. Key Hygiene Tests (Step 6)
# ============================================================================

def test_key_hygiene_full_fake_run(monkeypatch, caplog):
    """
    Injects a known canary API key into an AppTest session in hosted mode, runs
    a full fake pipeline run, and asserts that neither the key nor any substring
    >= 8 characters leaks into rendered UI, logs, final state, report, review
    bytes, cleaned CSV bytes, query params, or environment variables.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    pre_groq = os.environ.get("GROQ_API_KEY")
    pre_gemini = os.environ.get("GEMINI_API_KEY")

    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run()
    assert not at.exception

    # 1. Enter and save canary API key
    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input(CANARY_KEY)
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break
    assert not at.exception

    # 2. Select sample data
    for b in at.button:
        if "Try with sample data" in b.label:
            b.click().run()
            break
    assert not at.exception

    # 3. Tick privacy consent checkbox
    for cb in at.checkbox:
        if "understand" in cb.label:
            cb.check().run()
            break
    assert not at.exception

    # 4. Run Analysis with fake LLM
    with patch("llm._call_groq", side_effect=fake_groq_pipeline):
        for b in at.button:
            if "Run Analysis" in b.label:
                b.click().run()
                break

    assert not at.exception
    final_state = at.session_state.final_state
    assert final_state is not None, "Pipeline must produce a final_state"

    canary_substrings = get_canary_substrings(CANARY_KEY, min_len=8)

    # (a) Rendered UI elements
    rendered_text = extract_all_rendered_text(at)
    for sub in canary_substrings:
        assert sub not in rendered_text, f"Canary substring '{sub}' leaked into rendered UI text!"

    # (b) final_state serialized to JSON
    fs_json = json.dumps(final_state, default=str)
    for sub in canary_substrings:
        assert sub not in fs_json, f"Canary substring '{sub}' leaked into final_state JSON!"

    # (c) Events in activity feed
    events_str = json.dumps(final_state.get("events", []), default=str)
    for sub in canary_substrings:
        assert sub not in events_str, f"Canary substring '{sub}' leaked into events!"

    # (d) Report markdown text
    report_text = final_state.get("final_report", "")
    for sub in canary_substrings:
        assert sub not in report_text, f"Canary substring '{sub}' leaked into final_report!"

    # (e) Review JSON bytes
    rev_str = review_json_bytes(final_state).decode("utf-8", errors="ignore")
    for sub in canary_substrings:
        assert sub not in rev_str, f"Canary substring '{sub}' leaked into review JSON bytes!"

    # (f) Cleaned CSV bytes
    clean_bytes = cleaned_csv_bytes(final_state) or b""
    clean_str = clean_bytes.decode("utf-8", errors="ignore")
    for sub in canary_substrings:
        assert sub not in clean_str, f"Canary substring '{sub}' leaked into cleaned CSV bytes!"

    # (g) UI errors and exceptions
    errors_str = " ".join(e.value for e in at.error)
    for sub in canary_substrings:
        assert sub not in errors_str, f"Canary substring '{sub}' leaked into UI error message!"

    # (h) caplog (Python logging)
    for sub in canary_substrings:
        assert sub not in caplog.text, f"Canary substring '{sub}' leaked into Python logging!"

    # (i) URL query params
    qp_str = str(at.query_params)
    for sub in canary_substrings:
        assert sub not in qp_str, f"Canary substring '{sub}' leaked into query_params!"

    # (j) Key exists ONLY inside st.session_state._user_llm_config.api_key
    user_cfg = at.session_state.get("_user_llm_config")
    assert user_cfg is not None
    assert user_cfg.api_key == CANARY_KEY

    # (k) os.environ unchanged
    assert os.environ.get("GROQ_API_KEY") == pre_groq
    assert os.environ.get("GEMINI_API_KEY") == pre_gemini


# ============================================================================
# 2. Additional AppTest Tests (Step 7)
# ============================================================================

def test_hosted_mode_prerequisite_reason_lines(monkeypatch):
    """
    In hosted mode (HOSTED=1), verifies that the Run button is disabled and
    displays the exact reason line for each missing prerequisite:
    - 'Save your API key first' when no key
    - 'Choose a dataset' when key saved and privacy ticked, but no dataset
    - 'Tick the privacy box' when key saved and dataset loaded, but box unchecked
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # Case 1: No key -> 'Save your API key first'
    run_btn = [b for b in at.button if "Run Analysis" in b.label][0]
    assert run_btn.disabled is True
    captions = [c.value for c in at.caption]
    assert "Save your API key first" in captions

    # Save key
    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input("gsk_TESTKEY0123456789abcdef")
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break

    # Tick privacy consent box
    for cb in at.checkbox:
        if "understand" in cb.label:
            cb.check().run()
            break

    # Case 2: Key saved & privacy checked, but NO dataset -> 'Choose a dataset'
    run_btn = [b for b in at.button if "Run Analysis" in b.label][0]
    assert run_btn.disabled is True
    captions = [c.value for c in at.caption]
    assert "Choose a dataset" in captions

    # Load sample dataset
    for b in at.button:
        if "Try with sample data" in b.label:
            b.click().run()
            break

    # Untick privacy
    for cb in at.checkbox:
        if "understand" in cb.label:
            cb.uncheck().run()
            break

    # Case 3: Key saved & dataset loaded, but privacy UNCHECKED -> 'Tick the privacy box'
    run_btn = [b for b in at.button if "Run Analysis" in b.label][0]
    assert run_btn.disabled is True
    captions = [c.value for c in at.caption]
    assert "Tick the privacy box" in captions


def test_key_save_masked_status_and_clear_key(monkeypatch):
    """
    Saving a key through the form sets _user_llm_config and shows the masked status line.
    'Clear key' button clears _user_llm_config and removes the masked status line.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input("gsk_TESTKEY0123456789abcdef")
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break

    # Assert key saved and masked status line appears
    cfg = at.session_state._user_llm_config
    assert cfg is not None
    assert cfg.api_key == "gsk_TESTKEY0123456789abcdef"
    success_msgs = [s.value for s in at.success]
    assert any("Key saved: gsk_***cdef" in msg for msg in success_msgs)

    # Click 'Clear key'
    for b in at.button:
        if "Clear key" in b.label:
            b.click().run()
            break

    assert at.session_state._user_llm_config is None
    captions = [c.value for c in at.caption]
    assert any("No API key configured" in c for c in captions)


def test_session_isolation_independent_keys(monkeypatch):
    """Verifies that two independent AppTest sessions do NOT share keys or session state."""
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at1 = AppTest.from_file(APP_PATH, default_timeout=20)
    at1.run()
    for ti in at1.text_input:
        if ti.label == "API Key":
            ti.input("gsk_SESSION1_0123456789abcdef")
            break
    for b in at1.button:
        if "Save key" in b.label:
            b.click().run()
            break
    assert at1.session_state._user_llm_config.api_key == "gsk_SESSION1_0123456789abcdef"

    # Second fresh session
    at2 = AppTest.from_file(APP_PATH, default_timeout=20)
    at2.run()
    assert at2.session_state._user_llm_config is None


def test_test_key_button_success_and_redaction(monkeypatch):
    """
    'Test key' button:
    - Success shows green message.
    - Failure with exception echoing the secret key redacts the key.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    secret = "gsk_SECRETLEAK0123456789abcdef"
    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input(secret)
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break

    # 1. Success test
    with patch("llm._call_groq", return_value=("OK", 1, 1)):
        for b in at.button:
            if "Test key" in b.label:
                b.click().run()
                break
    success_msgs = [s.value for s in at.success]
    assert any("API key verified successfully!" in msg for msg in success_msgs)

    # 2. Failure test with leaked key in exception
    leaking_exc = Exception(f"Unauthorized error for key {secret}")
    with patch("llm._call_groq", side_effect=leaking_exc):
        for b in at.button:
            if "Test key" in b.label:
                b.click().run()
                break

    error_msgs = [e.value for e in at.error]
    assert any("[REDACTED_KEY]" in msg for msg in error_msgs)
    assert not any(secret in msg for msg in error_msgs)


def test_full_fake_run_token_meter(monkeypatch):
    """
    Full fake run in hosted mode with sample data: results tabs appear,
    and token meter shows 'Tokens used: ... (your key)'.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run()

    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input("gsk_TESTKEY0123456789abcdef")
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break

    for b in at.button:
        if "Try with sample data" in b.label:
            b.click().run()
            break

    for cb in at.checkbox:
        if "understand" in cb.label:
            cb.check().run()
            break

    with patch("llm._call_groq", side_effect=fake_groq_pipeline):
        for b in at.button:
            if "Run Analysis" in b.label:
                b.click().run()
                break

    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["📋 Report", "📊 Charts", "🧹 Cleaning Log", "🛡️ Reviewer Findings"]

    info_msgs = [i.value for i in at.info]
    assert any("Tokens used:" in msg and "(your key)" in msg for msg in info_msgs)


def test_llm_budget_error_during_run(monkeypatch):
    """
    LLMBudgetError during a run displays:
    'Reached the token limit for one run (N tokens). Nothing was saved as a finished run.'
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input("gsk_TESTKEY0123456789abcdef")
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break

    for b in at.button:
        if "Try with sample data" in b.label:
            b.click().run()
            break

    for cb in at.checkbox:
        if "understand" in cb.label:
            cb.check().run()
            break

    def fake_groq_budget(prompt, api_key=None, model_name=None):
        if "reply with ok" in prompt.lower():
            return "OK", 1, 1
        raise LLMBudgetError("Exceeded token budget during analysis")

    with patch("llm._call_groq", side_effect=fake_groq_budget):
        for b in at.button:
            if "Run Analysis" in b.label:
                b.click().run()
                break
        if at.session_state.get("running"):
            at.run()

    error_msgs = [e.value for e in at.error]
    assert any("Reached the token limit for one run" in msg and "Nothing was saved as a finished run" in msg for msg in error_msgs)


def test_busy_gate_and_cooldown_messages(monkeypatch):
    """Busy gate and cooldown messages display as expected in the UI."""
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    # 1. Busy gate
    at1 = AppTest.from_file(APP_PATH, default_timeout=20)
    at1.run()
    for ti in at1.text_input:
        if ti.label == "API Key":
            ti.input("gsk_TESTKEY0123456789abcdef")
            break
    for b in at1.button:
        if "Save key" in b.label:
            b.click().run()
            break
    for b in at1.button:
        if "Try with sample data" in b.label:
            b.click().run()
            break
    for cb in at1.checkbox:
        if "understand" in cb.label:
            cb.check().run()
            break

    with patch("llm._call_groq", return_value=("OK", 1, 1)):
        with patch("tools.limits.RunGate.try_acquire", return_value=None):
            for b in at1.button:
                if "Run Analysis" in b.label:
                    b.click().run()
                    break
    warnings1 = [w.value for w in at1.warning]
    assert any("The server is busy with other runs" in w for w in warnings1)

    # 2. Cooldown
    at2 = AppTest.from_file(APP_PATH, default_timeout=20)
    at2.run()
    for ti in at2.text_input:
        if ti.label == "API Key":
            ti.input("gsk_TESTKEY0123456789abcdef")
            break
    for b in at2.button:
        if "Save key" in b.label:
            b.click().run()
            break
    for b in at2.button:
        if "Try with sample data" in b.label:
            b.click().run()
            break
    for cb in at2.checkbox:
        if "understand" in cb.label:
            cb.check().run()
            break

    with patch("llm._call_groq", return_value=("OK", 1, 1)):
        with patch("tools.limits.SessionCooldown.seconds_remaining", return_value=45):
            for b in at2.button:
                if "Run Analysis" in b.label:
                    b.click().run()
                    break
    warnings2 = [w.value for w in at2.warning]
    assert any("Please wait 45 second(s) before starting another analysis" in w for w in warnings2)


def test_model_name_sanitization(monkeypatch):
    """Model name with spaces or special characters is rejected with an error."""
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    for ti in at.text_input:
        if ti.label == "Model":
            ti.input("invalid model with spaces!").run()
            break

    error_msgs = [e.value for e in at.error]
    assert any("Invalid model name. Allowed: letters, digits, and . _ / : - (max 80 chars)." in err for err in error_msgs)


def test_key_format_validation(monkeypatch):
    """
    Key format checks:
    - Whitespace is rejected.
    - Length < 20 or > 200 is rejected.
    - Unexpected prefix shows a warning but allows save.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    # 1. Whitespace
    at1 = AppTest.from_file(APP_PATH, default_timeout=20)
    at1.run()
    for ti in at1.text_input:
        if ti.label == "API Key":
            ti.input("gsk_12345 67890123456789")
            break
    for b in at1.button:
        if "Save key" in b.label:
            b.click().run()
            break
    assert any("API key must not contain any whitespace." in e.value for e in at1.error)

    # 2. Length < 20
    at2 = AppTest.from_file(APP_PATH, default_timeout=20)
    at2.run()
    for ti in at2.text_input:
        if ti.label == "API Key":
            ti.input("gsk_tooshort")
            break
    for b in at2.button:
        if "Save key" in b.label:
            b.click().run()
            break
    assert any("API key must be between 20 and 200 characters." in e.value for e in at2.error)

    # 3. Unexpected prefix warning for Groq
    at3 = AppTest.from_file(APP_PATH, default_timeout=20)
    at3.run()
    for ti in at3.text_input:
        if ti.label == "API Key":
            ti.input("custom_prefix_key_123456789012345")
            break
    for b in at3.button:
        if "Save key" in b.label:
            b.click().run()
            break
    assert any("Warning: Groq API keys usually start with 'gsk_'." in w.value for w in at3.warning)
    assert at3.session_state._user_llm_config is not None

    # 4. Unexpected prefix warning for Gemini
    at4 = AppTest.from_file(APP_PATH, default_timeout=20)
    at4.run()
    for sb in at4.selectbox:
        if sb.label == "Provider":
            sb.select("Google Gemini").run()
            break
    for ti in at4.text_input:
        if ti.label == "API Key":
            ti.input("custom_prefix_gemini_123456789012")
            break
    for b in at4.button:
        if "Save key" in b.label:
            b.click().run()
            break
    assert any("Warning: Google Gemini API keys usually start with 'AIza'." in w.value for w in at4.warning)
    assert at4.session_state._user_llm_config is not None


def test_demo_only_hides_key_section(monkeypatch):
    """DEMO_ONLY=1 hides the whole key section and never calls the LLM layer."""
    monkeypatch.setenv("DEMO_ONLY", "1")

    def forbidden_call(*args, **kwargs):
        raise AssertionError("LLM layer must NEVER be called in DEMO_ONLY mode!")

    monkeypatch.setattr("llm.call_llm", forbidden_call)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # The key section must not exist
    headers = [h.value for h in at.header]
    assert not any("Use your own API key" in h for h in headers)
    labels = [ti.label for ti in at.text_input]
    assert "API Key" not in labels


def test_recorded_demo_works_in_hosted_mode_zero_llm_no_key(monkeypatch):
    """The recorded demo still works in hosted mode with zero LLM calls and no key."""
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    def forbidden_call(*args, **kwargs):
        raise AssertionError("LLM layer must NEVER be called during recorded demo playback!")

    monkeypatch.setattr("llm.call_llm", forbidden_call)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    for cb in at.checkbox:
        if "Skip animation" in cb.label:
            cb.check()
            break

    for b in at.button:
        if "Watch a recorded run" in b.label:
            b.click().run()
            break

    if at.session_state.replaying_demo:
        at.run()

    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["📋 Report", "📊 Charts", "🧹 Cleaning Log", "🛡️ Reviewer Findings"]


# ============================================================================
# 3. Privacy Notice & Model Default Verification Tests
# ============================================================================

def test_privacy_notice_content_and_phrases(monkeypatch):
    """
    Verifies the rewritten privacy notice:
    - Contains required phrases:
      * '5 most frequent values'
      * 'first 5 rows'
      * '10 rows'
      * '150 characters'
      * 'error message'
      * 'do not upload'
    - Is compact and readable (under 250 words).
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # Find the privacy notice text in markdown elements
    notice_text = None
    for m in at.markdown:
        val = str(m.value)
        if "What is sent" in val and "Cleaning & Questions" in val:
            notice_text = val
            break

    assert notice_text is not None, "Privacy notice markdown table not found!"

    # Verify all 6 mandatory phrases are present
    required_phrases = [
        "5 most frequent values",
        "first 5 rows",
        "10 rows",
        "150 characters",
        "error message",
        "do not upload",
    ]
    for phrase in required_phrases:
        assert phrase.lower() in notice_text.lower(), f"Mandatory phrase '{phrase}' missing from privacy notice!"

    # Verify word count is compact (under 250 words)
    words = notice_text.split()
    assert len(words) <= 250, f"Privacy notice is too long ({len(words)} words, expected <= 250)"


def test_gemini_default_model(monkeypatch):
    """
    Verifies that the Gemini default model is 'gemini-3.8-flash' in llm.py
    and is dynamically reflected in the Streamlit UI, while allowing env overrides.
    """
    from llm import _get_model_name

    # 1. Default without env var
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert _get_model_name("gemini") == "gemini-3.8-flash"

    # 2. Env override works
    monkeypatch.setenv("GEMINI_MODEL", "custom-gemini-test")
    assert _get_model_name("gemini") == "custom-gemini-test"

    # 3. AppTest UI reflects gemini-3.8-flash when Google Gemini is selected
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    for sb in at.selectbox:
        if sb.label == "Provider":
            sb.select("Google Gemini").run()
            break

    model_inputs = [ti for ti in at.text_input if ti.label == "Model"]
    assert len(model_inputs) > 0
    assert model_inputs[0].value == "gemini-3.8-flash"


def test_model_not_found_message(monkeypatch):
    """
    When the provider reports that a model is not found, the 'Test key'
    failure message displays:
    'This model name is not available for your key. Try another model name from the provider's model list.'
    and leaks no secret key.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    secret = "gsk_NOTFOUND0123456789abcdef"
    for ti in at.text_input:
        if ti.label == "API Key":
            ti.input(secret)
            break
    for b in at.button:
        if "Save key" in b.label:
            b.click().run()
            break

    # Mock provider returning a "model not found" error containing the key
    fake_error = Exception(f"404 NotFound: model not found for requested key {secret}")
    with patch("llm._call_groq", side_effect=fake_error):
        for b in at.button:
            if "Test key" in b.label:
                b.click().run()
                break

    error_msgs = [e.value for e in at.error]
    expected_msg = "This model name is not available for your key. Try another model name from the provider's model list."
    assert any(expected_msg in msg for msg in error_msgs)
    assert not any(secret in msg for msg in error_msgs)

