"""
app.py
Multi-Agent Data Analyst — Streamlit Web Interface.
Connects the Streamlit UI to the multi-agent pipeline with live agent cards,
activity timeline feed, progress tracking, and execution summary metrics.
Supports a zero-LLM recorded demo mode with step-by-step playback.
"""

import io
import os
from pathlib import Path
import re
import time
from typing import Any
import uuid
import streamlit as st
import pandas as pd

from components.data_utils import load_csv_safe, dataset_snapshot, save_upload
from components.ui_state import (
    AGENTS,
    initial_statuses,
    apply_node_update,
    new_events,
    event_to_html,
    summarize_run,
    get_final_status_message,
    get_final_agent_statuses,
    format_error_details,
)
from components.agent_cards import render_agent_cards
from components.activity_feed import render_feed
from components.results_tabs import render_results
from tools.limits import (
    get_settings,
    validate_upload_size,
    validate_dataframe,
)
from tools.workspace import RunWorkspace, cleanup_stale_workspaces
from components.run_control import run_guarded
from llm import (
    LLMConfig,
    check_llm_available,
    llm_session,
    mask_key,
    redact,
)

# Clean up any stale workspaces older than 2 hours at startup
cleanup_stale_workspaces()

# ============================================================================
# Mode Helpers
# ============================================================================

def is_demo_only() -> bool:
    """Returns True if the application is running in forced demo-only mode."""
    if os.environ.get("DEMO_ONLY") == "1":
        return True
    try:
        if hasattr(st, "secrets") and st.secrets.get("DEMO_ONLY") == "1":
            return True
    except Exception:
        pass
    return False


def has_env_keys() -> bool:
    """Checks whether local environment contains configured API keys."""
    groq_k = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_k = os.environ.get("GEMINI_API_KEY", "").strip()
    return bool(
        (groq_k and groq_k not in ("your_groq_key_here", "your_key_here"))
        or (gemini_k and gemini_k not in ("your_gemini_key_here", "your_key_here"))
    )


# ============================================================================
# 1. Page Configuration & Custom CSS Injection
# ============================================================================

st.set_page_config(
    page_title="Multi-Agent Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_file = Path("assets/styles.css")
if css_file.exists():
    css_content = css_file.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ============================================================================
# 2. Session State Initialization
# ============================================================================

if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "dataset_snapshot" not in st.session_state:
    st.session_state.dataset_snapshot = None

if "df_preview" not in st.session_state:
    st.session_state.df_preview = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "running" not in st.session_state:
    st.session_state.running = False

if "final_state" not in st.session_state:
    st.session_state.final_state = None

if "statuses" not in st.session_state:
    st.session_state.statuses = initial_statuses()

if "events_html" not in st.session_state:
    st.session_state.events_html = []

if "seen_events_count" not in st.session_state:
    st.session_state.seen_events_count = 0

if "_last_processed_upload" not in st.session_state:
    st.session_state._last_processed_upload = None

if "is_demo_mode" not in st.session_state:
    st.session_state.is_demo_mode = False

if "demo_meta" not in st.session_state:
    st.session_state.demo_meta = None

if "replaying_demo" not in st.session_state:
    st.session_state.replaying_demo = False

if "replaying_steps" not in st.session_state:
    st.session_state.replaying_steps = []

if "replaying_final" not in st.session_state:
    st.session_state.replaying_final = None

if "skip_animation" not in st.session_state:
    st.session_state.skip_animation = False

if "demo_missing_notice" not in st.session_state:
    st.session_state.demo_missing_notice = False

if "_active_workspace" not in st.session_state:
    st.session_state._active_workspace = None

if "cleaned_csv_bytes" not in st.session_state:
    st.session_state.cleaned_csv_bytes = None

if "cleaned_csv_oversized" not in st.session_state:
    st.session_state.cleaned_csv_oversized = False

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "_user_llm_config" not in st.session_state:
    st.session_state._user_llm_config = None

if "_key_test_result" not in st.session_state:
    st.session_state._key_test_result = None

if "privacy_acknowledged" not in st.session_state:
    st.session_state.privacy_acknowledged = False

if "last_tokens_used" not in st.session_state:
    st.session_state.last_tokens_used = None

if "last_max_tokens" not in st.session_state:
    st.session_state.last_max_tokens = None


def cleanup_active_workspace() -> None:
    """Safely cleans up any active temporary workspace associated with the session."""
    ws = st.session_state.get("_active_workspace")
    if ws is not None:
        try:
            ws.cleanup()
        except Exception:
            pass
        st.session_state._active_workspace = None


def clear_analysis_state() -> None:
    """Resets execution state when a new dataset is loaded or demo run starts."""
    st.session_state.final_state = None
    st.session_state.statuses = initial_statuses()
    st.session_state.events_html = []
    st.session_state.seen_events_count = 0
    st.session_state.running = False
    st.session_state.error_message = None
    st.session_state.cleaned_csv_bytes = None
    st.session_state.cleaned_csv_oversized = False
    st.session_state.last_tokens_used = None
    st.session_state.last_max_tokens = None


# ============================================================================
# 3. Sidebar: Recorded Demo, File Upload & How It Works
# ============================================================================

with st.sidebar:
    st.header("🎬 Recorded Demo")
    skip_anim = st.checkbox("⚡ Skip animation", value=st.session_state.skip_animation)
    st.session_state.skip_animation = skip_anim

    snapshot_file = os.environ.get(
        "DEMO_SNAPSHOT_PATH",
        os.path.join("demo", "sample_run.json"),
    )

    is_busy = st.session_state.running or st.session_state.replaying_demo

    if st.button("📺 Watch a recorded run", disabled=is_busy, use_container_width=True):
        cleanup_active_workspace()
        if not os.path.exists(snapshot_file):
            st.session_state.demo_missing_notice = True
            st.session_state.is_demo_mode = False
            clear_analysis_state()
            st.rerun()
        else:
            try:
                from components.demo_snapshot import load_snapshot
                final_state, steps, meta = load_snapshot(snapshot_file)
                st.session_state.demo_missing_notice = False
                st.session_state.is_demo_mode = True
                st.session_state.demo_meta = meta
                st.session_state.replaying_demo = True
                st.session_state.replaying_steps = steps
                st.session_state.replaying_final = final_state
                st.session_state.dataset_path = None
                st.session_state.dataset_name = meta.get("dataset_name", "sample.csv")
                clear_analysis_state()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load recorded run: {e}")

    demo_only_active = is_demo_only()

    if not demo_only_active:
        st.markdown("---")
        st.header("🔑 Use your own API key")

        provider_display = st.selectbox(
            "Provider",
            ["Groq", "Google Gemini"],
            index=0,
            disabled=is_busy,
            key="sb_provider",
        )
        provider_name = "groq" if provider_display == "Groq" else "gemini"
        default_model = "openai/gpt-oss-120b" if provider_name == "groq" else "gemini-2.0-flash"

        model_input = st.text_input(
            "Model",
            value=default_model,
            disabled=is_busy,
            key=f"sb_model_{provider_name}",
            help="Model identifier on the selected provider.",
        )

        model_clean = model_input.strip()
        model_valid = bool(re.match(r"^[a-zA-Z0-9._/:-]{1,80}$", model_clean))
        if not model_valid:
            st.error("Invalid model name. Allowed: letters, digits, and . _ / : - (max 80 chars).")

        with st.form("api_key_form", clear_on_submit=True):
            key_val = st.text_input(
                "API Key",
                type="password",
                help="Your key stays in this browser session and is sent only to the chosen provider.",
            )
            save_btn = st.form_submit_button("Save key", disabled=is_busy)

            if save_btn:
                if not model_valid:
                    st.error("Please enter a valid model name before saving the key.")
                elif not key_val:
                    st.error("API key cannot be empty.")
                elif any(c.isspace() for c in key_val):
                    st.error("API key must not contain any whitespace.")
                elif len(key_val.strip()) < 20 or len(key_val.strip()) > 200:
                    st.error("API key must be between 20 and 200 characters.")
                else:
                    clean_key = key_val.strip()
                    key_warn = None
                    if provider_name == "groq" and not clean_key.startswith("gsk_"):
                        key_warn = "Warning: Groq API keys usually start with 'gsk_'."
                    elif provider_name == "gemini" and not clean_key.startswith("AIza"):
                        key_warn = "Warning: Google Gemini API keys usually start with 'AIza'."

                    cfg = LLMConfig(
                        provider=provider_name,
                        model=model_clean,
                        api_key=clean_key,
                        max_tokens=get_settings().max_run_tokens,
                    )
                    st.session_state["_user_llm_config"] = cfg
                    st.session_state["_key_test_result"] = None
                    st.session_state["_key_warning"] = key_warn
                    st.rerun()

        saved_cfg = st.session_state.get("_user_llm_config")
        if saved_cfg is not None:
            masked = mask_key(saved_cfg.api_key)
            st.success(f"Key saved: {masked} ({saved_cfg.provider}, {saved_cfg.model})")
            if st.session_state.get("_key_warning"):
                st.warning(st.session_state["_key_warning"])

            t_col, c_col = st.columns(2)
            with t_col:
                if st.button("Test key", disabled=is_busy, use_container_width=True):
                    with st.spinner("Testing key..."):
                        with llm_session(saved_cfg):
                            ok, msg = check_llm_available(config=saved_cfg)
                    if ok:
                        st.session_state["_key_test_result"] = ("success", "API key verified successfully!")
                    else:
                        msg_lower = (msg or "").lower()
                        if any(term in msg_lower for term in ("auth", "401", "key", "unauthorized", "invalid", "forbidden", "403")):
                            friendly_prefix = "The provider rejected this key."
                        else:
                            friendly_prefix = "Can't reach the provider."
                        redacted_err = redact(msg or "", key=saved_cfg.api_key)
                        st.session_state["_key_test_result"] = ("error", f"{friendly_prefix} {redacted_err}".strip())
                    st.rerun()
            with c_col:
                if st.button("Clear key", disabled=is_busy, use_container_width=True):
                    for k in ("_user_llm_config", "_key_test_result", "_key_warning", "last_tokens_used", "last_max_tokens"):
                        st.session_state[k] = None
                    st.rerun()

            test_res = st.session_state.get("_key_test_result")
            if test_res:
                t_kind, t_msg = test_res
                if t_kind == "success":
                    st.success(t_msg)
                else:
                    st.error(t_msg)
            else:
                st.caption("Testing makes one tiny call using your key.")
        else:
            st.caption("No API key configured (using local default if available)")

        st.info("Your key stays in this browser session's server memory, is sent only to the provider you chose, and is never stored or logged. Close the tab or press Clear key to remove it. Use a key you can revoke.")

    st.markdown("---")
    st.header("📂 Data Source")

    if demo_only_active:
        st.info("ℹ️ **Demo-only mode**: File uploads and live execution are disabled. Click **Watch a recorded run** above to replay a real multi-agent run.")
    else:
        # CSV File Uploader
        settings = get_settings()
        uploaded_file = st.file_uploader(
            "Upload a CSV dataset",
            type=["csv"],
            disabled=is_busy,
            help=f"Upload a tabular CSV file up to {settings.max_upload_mb}MB.",
        )

        if uploaded_file is not None:
            upload_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state._last_processed_upload != upload_identifier:
                raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

                # Check upload size limit
                size_err = validate_upload_size(len(raw_bytes), settings)
                if size_err:
                    st.session_state.error_message = size_err
                    cleanup_active_workspace()
                    st.session_state.dataset_path = None
                    st.session_state.dataset_name = None
                    st.session_state.dataset_snapshot = None
                    st.session_state.df_preview = None
                    st.session_state.is_demo_mode = False
                    clear_analysis_state()
                    st.session_state._last_processed_upload = upload_identifier
                    st.rerun()

                df, error = load_csv_safe(io.BytesIO(raw_bytes))
                if error:
                    st.session_state.error_message = error
                    cleanup_active_workspace()
                    st.session_state.dataset_path = None
                    st.session_state.dataset_name = None
                    st.session_state.dataset_snapshot = None
                    st.session_state.df_preview = None
                    st.session_state.is_demo_mode = False
                    clear_analysis_state()
                    st.session_state._last_processed_upload = upload_identifier
                    st.rerun()

                # Validate rows and columns
                df_err = validate_dataframe(df, settings)
                if df_err:
                    st.session_state.error_message = df_err
                    cleanup_active_workspace()
                    st.session_state.dataset_path = None
                    st.session_state.dataset_name = None
                    st.session_state.dataset_snapshot = None
                    st.session_state.df_preview = None
                    st.session_state.is_demo_mode = False
                    clear_analysis_state()
                    st.session_state._last_processed_upload = upload_identifier
                    st.rerun()

                # Valid dataset
                st.session_state.error_message = None
                cleanup_active_workspace()

                if settings.hosted:
                    ws = RunWorkspace()
                    st.session_state._active_workspace = ws
                    raw_path = ws.get_raw_path(uploaded_file.name)
                    Path(raw_path).write_bytes(raw_bytes)
                    st.session_state.dataset_path = raw_path
                else:
                    saved_path = save_upload(uploaded_file)
                    st.session_state.dataset_path = str(saved_path)

                st.session_state.dataset_name = uploaded_file.name
                st.session_state.dataset_snapshot = dataset_snapshot(df)
                st.session_state.df_preview = df.head(5)
                st.session_state.is_demo_mode = False
                st.session_state.demo_missing_notice = False
                clear_analysis_state()
                st.session_state._last_processed_upload = upload_identifier
                st.rerun()

        # "Try with sample data" button
        if st.button("📊 Try with sample data", disabled=is_busy, use_container_width=True):
            sample_path = os.path.join("data", "sample.csv")
            if not os.path.exists(sample_path):
                st.error("Sample dataset 'data/sample.csv' is missing from the workspace.")
            else:
                df, error = load_csv_safe(sample_path)
                if error:
                    st.error(f"Failed to load sample dataset: {error}")
                else:
                    df_err = validate_dataframe(df, settings)
                    if df_err:
                        st.error(df_err)
                    else:
                        cleanup_active_workspace()
                        st.session_state.dataset_path = sample_path
                        st.session_state.dataset_name = "sample.csv (E-Commerce Deliveries)"
                        st.session_state.dataset_snapshot = dataset_snapshot(df)
                        st.session_state.df_preview = df.head(5)
                        st.session_state.is_demo_mode = False
                        st.session_state.demo_missing_notice = False
                        clear_analysis_state()
                        st.session_state._last_processed_upload = None
                        st.rerun()

        st.caption("Sample: 8,000 rows; a typical run uses about 22k tokens, up to about 58k when the reviewer asks for retries")

    st.markdown("---")

    # "How it works" expander
    with st.expander("ℹ️ How it works", expanded=True):
        st.markdown(
            """
- 🧹 **Cleaner**: Imputes missing values, drops duplicates, clips outliers, and standardizes formats.
- 🔍 **Analyst**: Plans analytical questions and executes verified pandas calculations.
- 📈 **Visualizer**: Generates optimal chart specs and derives quantitative takeaways.
- 🛡️ **Reviewer**: Independently recalculates results and double-checks the math.
            """
        )


# ============================================================================
# 4. Main Area: Header Banner & Demo Notices
# ============================================================================

st.markdown(
    """
<div class="header-banner">
    <h1>Multi-Agent Data Analyst</h1>
    <p>Upload a CSV. Four AI agents clean it, analyze it, chart it, and double-check the numbers.</p>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.demo_missing_notice:
    st.info("No recorded run is available yet")

if st.session_state.is_demo_mode and st.session_state.demo_meta:
    meta = st.session_state.demo_meta
    rec_raw = meta.get("recorded_at", "")
    date_val = str(rec_raw).split("T")[0] if "T" in str(rec_raw) else str(rec_raw or "recent date")
    models = meta.get("models_used", [])
    model_str = ", ".join(models) if models else "openai/gpt-oss-120b"
    commit_hash = meta.get("git_commit", "unknown")
    st.info(
        f"Recorded run: a replay of a real run on {date_val} with {model_str} (commit {commit_hash}). No AI calls are made in demo mode."
    )


# ============================================================================
# 5. Error Banner
# ============================================================================

if st.session_state.error_message:
    st.error(f"❌ {st.session_state.error_message}")


# ============================================================================
# 6. Main Area: Dataset Snapshot or Empty State
# ============================================================================

if st.session_state.is_demo_mode and st.session_state.demo_meta:
    meta = st.session_state.demo_meta
    ds_name = meta.get("dataset_name", "dataset.csv")

    st.markdown(
        f"### 📋 Dataset Snapshot &bull; <small style='color:#94A3B8;'>{ds_name}</small>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        raw_r = meta.get("raw_rows")
        st.metric(label="Raw Rows", value=f"{raw_r:,}" if raw_r is not None else "N/A")
    with m2:
        raw_c = meta.get("raw_cols")
        st.metric(label="Raw Columns", value=f"{raw_c}" if raw_c is not None else "N/A")
    with m3:
        clean_r = meta.get("clean_rows")
        st.metric(label="Cleaned Rows", value=f"{clean_r:,}" if clean_r is not None else "N/A")
    with m4:
        clean_c = meta.get("clean_cols")
        st.metric(label="Cleaned Columns", value=f"{clean_c}" if clean_c is not None else "N/A")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

elif st.session_state.dataset_path is not None and st.session_state.dataset_snapshot is not None:
    snap = st.session_state.dataset_snapshot

    st.markdown(
        f"### 📋 Dataset Snapshot &bull; <small style='color:#94A3B8;'>{st.session_state.dataset_name}</small>",
        unsafe_allow_html=True,
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Rows", value=f"{snap['rows']:,}")
    with m_col2:
        st.metric(label="Columns", value=f"{snap['columns']:,}")
    with m_col3:
        st.metric(label="Missing %", value=f"{snap['missing_pct']}%")
    with m_col4:
        st.metric(label="Duplicate Rows", value=f"{snap['duplicate_rows']:,}")

    st.markdown(
        f"""
<div class="col-breakdown">
    Column types: <span>{snap['numeric_cols']}</span> numeric &bull; <span>{snap['text_cols']}</span> text &bull; <span>{snap['date_like_cols']}</span> date-like
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    with st.expander("🔍 Preview First 5 Rows", expanded=False):
        if st.session_state.df_preview is not None:
            st.dataframe(st.session_state.df_preview, use_container_width=True)

else:
    # Empty state when no dataset or demo is loaded
    st.markdown(
        """
<div class="empty-state-card">
    <h3>Upload a CSV or try the sample data to get started</h3>
    <p>Select a file from your computer using the sidebar, or click <strong>Watch a recorded run</strong> to explore the system without an API key.</p>
    <div class="empty-state-features">
        <div class="feature-box">
            <strong>📋 Insights Report</strong>
            <span>Executive business summary answering key analytical questions about your dataset.</span>
        </div>
        <div class="feature-box">
            <strong>📊 Interactive Charts</strong>
            <span>Visual charts (bar, line, scatter) paired with data-backed takeaway observations.</span>
        </div>
        <div class="feature-box">
            <strong>🧹 Data Cleaning Log</strong>
            <span>Transparent audit trail of all imputation, outlier filtering, and formatting operations.</span>
        </div>
        <div class="feature-box">
            <strong>🛡️ Reviewer Audit Findings</strong>
            <span>Independent dual-computation verification catching numerical discrepancies.</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 7. Demo Replay Execution Loop
# ============================================================================

if st.session_state.replaying_demo:
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    cards_placeholder = st.empty()
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    feed_placeholder = st.empty()

    st.session_state.statuses = initial_statuses()
    st.session_state.statuses["cleaner"] = {"status": "running", "note": "Cleaning data..."}
    render_agent_cards(cards_placeholder, st.session_state.statuses)
    render_feed(feed_placeholder, st.session_state.events_html)

    steps = st.session_state.replaying_steps
    t_total = float(steps[-1].get("seconds", 1.0)) if steps else 1.0
    scale = min(1.0, 20.0 / max(t_total, 1.0))
    prev_t = 0.0

    for step in steps:
        curr_t = float(step.get("seconds", 0.0))
        dt = (curr_t - prev_t) * scale
        prev_t = curr_t

        if not st.session_state.skip_animation:
            time.sleep(max(0.3, dt))

        # 1. Update statuses
        st.session_state.statuses = apply_node_update(
            st.session_state.statuses,
            step.get("node_name", ""),
            step,
            retry_count=step.get("retry_count", 0),
        )

        # 2. Extract new timeline events
        if "events" in step and isinstance(step["events"], list):
            for ev in step["events"]:
                st.session_state.events_html.append(event_to_html(ev))

        # 3. Redraw live components
        render_agent_cards(cards_placeholder, st.session_state.statuses)
        render_feed(feed_placeholder, st.session_state.events_html)

    # Finalize replay
    st.session_state.final_state = st.session_state.replaying_final
    st.session_state.statuses = get_final_agent_statuses(
        st.session_state.final_state,
        st.session_state.statuses,
    )
    st.session_state.replaying_demo = False
    st.rerun()


# ============================================================================
# 8. Live Pipeline Execution Interface (Only when NOT demo-only)
# ============================================================================

if not demo_only_active and not st.session_state.is_demo_mode:
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Privacy Notice Expander
    with st.expander("🛡️ What happens to my data?", expanded=not st.session_state.get("privacy_acknowledged", False)):
        st.markdown(
            """
### What data is sent to the AI provider?
When an analysis is executed, the following parts of your data are formatted into prompts and sent to the selected AI provider using your API key:
- **Dataset Schema & Profile (Cleaner & Analyst)**: Total row count, column count, column names, pandas data types, null percentage, and unique value counts.
- **Summary Statistics (Cleaner & Analyst)**: Minimum, maximum, and mean for numeric columns; top 5 most frequent values and occurrence counts for categorical columns.
- **Sample Rows (Cleaner & Analyst)**: The first **5 sample rows** of the dataset (text fields over 50 characters are truncated).
- **Analysis Questions & Code (Analyst & Reviewer)**: Generated business questions, column names, and pandas code snippets executed in the sandbox.
- **Analysis Result Tables (Visualizer & Reviewer)**: Sandboxed results are capped at **50 rows** maximum. During chart generation and review, previews of the first **5 to 10 rows** are sent to the LLM to choose chart types, verify takeaway statements, and perform logic reviews.
- **Executive Report Insights (Report Synthesis)**: Verified takeaways and brief text summaries of key findings (up to 150 characters per result item).

### Data Retention & Privacy Guarantees
- **Temporary Server Processing**: The uploaded dataset is processed strictly in this session's isolated temporary server directory and server memory.
- **Automatic Deletion**: The temporary workspace directory and its files are automatically deleted when the run ends (on success, error, or timeout), or when you select a different dataset or close the app.
- **In-Memory Key Handling**: Your API key is held only in this browser session's server memory, is never written to disk, is never logged, and is never shared with the author.
- **Zero Author Data Collection**: Neither your CSV data nor your API key is ever collected, stored permanently, or shared with the author.
- **Safety Warning**: Do not upload personal, sensitive, private, or confidential data.
- **Portfolio Demonstration**: This project is a portfolio demo and its results must not be used for commercial, legal, medical, financial, or operational decision-making.
"""
        )
        st.checkbox(
            "I understand and want to continue",
            value=st.session_state.get("privacy_acknowledged", False),
            key="cb_privacy_agree",
        )
        st.session_state.privacy_acknowledged = st.session_state.get("cb_privacy_agree", False)

    settings = get_settings()
    user_cfg = st.session_state.get("_user_llm_config")
    has_dataset = st.session_state.dataset_path is not None
    has_key = (user_cfg is not None) or (not settings.hosted and has_env_keys())
    privacy_ticked = bool(st.session_state.get("privacy_acknowledged", False))

    reason_line = None
    if not has_key:
        reason_line = "Save your API key first"
    elif not has_dataset:
        reason_line = "Choose a dataset"
    elif not privacy_ticked:
        reason_line = "Tick the privacy box"

    is_run_enabled = (
        not st.session_state.running
        and not st.session_state.replaying_demo
        and reason_line is None
    )

    run_btn_col, _ = st.columns([1.5, 3.5])
    with run_btn_col:
        run_clicked = st.button(
            "🚀 Run Analysis",
            type="primary",
            disabled=not is_run_enabled,
            use_container_width=True,
            help="Initiates the multi-agent pipeline on the loaded dataset.",
        )

    if reason_line and not st.session_state.running and not st.session_state.replaying_demo:
        st.caption(reason_line)

    warning_placeholder = st.empty()
    cards_placeholder = st.empty()
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    feed_placeholder = st.empty()
    summary_placeholder = st.empty()

    if run_clicked:
        with st.spinner("Checking AI service..."):
            is_available, check_msg = check_llm_available(config=user_cfg)

        if not is_available:
            friendly_err, tech_err = format_error_details(check_msg)
            user_key = getattr(user_cfg, "api_key", None)
            with warning_placeholder.container():
                st.error(redact(friendly_err, key=user_key))
                with st.expander("Technical details"):
                    st.code(redact(tech_err, key=user_key))
        else:
            warning_placeholder.empty()
            st.session_state.running = True
            st.session_state.final_state = None
            st.session_state.statuses = initial_statuses()
            st.session_state.events_html = []
            st.session_state.seen_events_count = 0
            st.rerun()

    # Active Live Execution Loop
    if st.session_state.running:
        from graph import run_pipeline

        warning_placeholder.warning(
            "⏳ Analysis takes about 2 minutes. Please don't change anything until it finishes."
        )

        st.session_state.statuses["cleaner"] = {"status": "running", "note": "Cleaning data..."}
        render_agent_cards(cards_placeholder, st.session_state.statuses)
        render_feed(feed_placeholder, st.session_state.events_html)

        retry_box = [0]

        def on_stream_update(item: Any) -> None:
            if isinstance(item, tuple) and len(item) == 2:
                node_name, state_update = item
                if node_name == "__final__":
                    return

                if "retry_count" in state_update:
                    retry_box[0] = state_update["retry_count"]

                st.session_state.statuses = apply_node_update(
                    st.session_state.statuses,
                    node_name,
                    state_update,
                    retry_count=retry_box[0],
                )

                if "events" in state_update and isinstance(state_update["events"], list):
                    new_evs = new_events(state_update["events"], st.session_state.seen_events_count)
                    for ev in new_evs:
                        st.session_state.events_html.append(event_to_html(ev))
                    st.session_state.seen_events_count = len(state_update["events"])

                render_agent_cards(cards_placeholder, st.session_state.statuses)
                render_feed(feed_placeholder, st.session_state.events_html)

        active_ws = st.session_state.get("_active_workspace")
        if active_ws is None and settings.hosted:
            active_ws = RunWorkspace()
            st.session_state._active_workspace = active_ws
        ws_dir = active_ws.workspace_dir if active_ws else None

        def make_stream():
            return run_pipeline(
                st.session_state.dataset_path,
                stream=True,
                workspace_dir=ws_dir,
            )

        session_id = st.session_state.get("session_id", "default_session")
        res = run_guarded(
            settings=settings,
            session_id=session_id,
            make_stream=make_stream,
            on_update=on_stream_update,
            workspace=active_ws,
            session_config=user_cfg,
        )

        warning_placeholder.empty()
        status = res.get("status")
        final_state_obj = res.get("final_state")
        user_key = getattr(user_cfg, "api_key", None)

        if status == "ok" and final_state_obj:
            if "cleaned_csv_bytes" in final_state_obj and isinstance(final_state_obj["cleaned_csv_bytes"], bytes):
                st.session_state.cleaned_csv_bytes = final_state_obj["cleaned_csv_bytes"]
                st.session_state.cleaned_csv_oversized = False
            elif final_state_obj.get("cleaned_csv_oversized"):
                st.session_state.cleaned_csv_bytes = None
                st.session_state.cleaned_csv_oversized = True
            else:
                st.session_state.cleaned_csv_bytes = None
                st.session_state.cleaned_csv_oversized = False

            st.session_state.final_state = final_state_obj
            st.session_state.last_tokens_used = res.get("tokens_used", 0)
            st.session_state.last_max_tokens = res.get("max_tokens")
            st.session_state.statuses = get_final_agent_statuses(
                final_state_obj,
                st.session_state.statuses,
            )
            render_agent_cards(cards_placeholder, st.session_state.statuses, final_state=final_state_obj)
        elif status == "budget":
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "error"
                    v["note"] = "Budget reached"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
            render_feed(feed_placeholder, st.session_state.events_html)
            st.error(res.get("error", "Token budget reached."))
        elif status == "cooldown":
            st.warning(res.get("error", "Cooldown active."))
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "idle"
                    v["note"] = "Waiting"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
        elif status == "busy":
            st.warning(res.get("error", "Server is busy."))
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "idle"
                    v["note"] = "Waiting"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
        elif status == "timeout":
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "error"
                    v["note"] = "Timed out"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
            render_feed(feed_placeholder, st.session_state.events_html)
            st.error(res.get("error", "Run timed out."))
        else:
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "error"
                    v["note"] = "Failed"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
            render_feed(feed_placeholder, st.session_state.events_html)
            err_val = res.get("exception") or res.get("error", "Unknown error")
            friendly_err, tech_err = format_error_details(err_val)
            st.error(redact(friendly_err, key=user_key))
            with st.expander("Technical details"):
                st.code(redact(tech_err, key=user_key))

        st.session_state.running = False
        if active_ws is not None:
            st.session_state._active_workspace = None

        if status == "ok" and final_state_obj:
            st.rerun()

    else:
        render_agent_cards(
            cards_placeholder,
            st.session_state.statuses,
            final_state=st.session_state.final_state,
        )
        render_feed(feed_placeholder, st.session_state.events_html)


# ============================================================================
# 9. Results & Completion Display (Live or Demo Replay)
# ============================================================================

if (st.session_state.final_state is not None) and not st.session_state.running and not st.session_state.replaying_demo:
    # If in demo mode and not already rendered placeholders above
    if st.session_state.is_demo_mode or demo_only_active:
        cards_placeholder = st.empty()
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        feed_placeholder = st.empty()
        render_agent_cards(
            cards_placeholder,
            st.session_state.statuses,
            final_state=st.session_state.final_state,
        )
        render_feed(feed_placeholder, st.session_state.events_html)

    summary_placeholder = st.empty()
    summary = summarize_run(st.session_state.final_state)
    is_all_clean, status_title = get_final_status_message(
        st.session_state.final_state,
        st.session_state.statuses,
    )
    title_icon = "✅" if is_all_clean else "⚠️"

    with summary_placeholder.container():
        st.markdown(
            f"""
<div class="run-complete-card">
    <div class="run-complete-header">
        <div class="run-complete-title">{title_icon} {status_title}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Row 1: Execution metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Execution Time", f"{summary['duration_sec']:.1f}s")
        with c2:
            st.metric("LLM Invocations", f"{summary['llm_calls']}")
        with c3:
            st.metric("Tokens Consumed", f"{summary['tokens']:,}")
        with c4:
            st.metric("Review Retries", f"{summary['retries_used']}")

        tokens_u = st.session_state.get("last_tokens_used")
        if tokens_u is None and summary.get("tokens"):
            tokens_u = summary.get("tokens")
        max_t = st.session_state.get("last_max_tokens") or get_settings().max_run_tokens
        user_cfg = st.session_state.get("_user_llm_config")
        if user_cfg is not None and tokens_u is not None:
            if max_t is not None:
                st.info(f"Tokens used: {tokens_u:,} of {max_t:,} (your key)")
            else:
                st.info(f"Tokens used: {tokens_u:,} (your key)")

        # Row 2: Verification audit metrics
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Flagged (Pass 1)", f"{summary['items_flagged_first_pass']}")
        with c6:
            st.metric("Fixed by Retry", f"{summary['fixed_by_retry']}")
        with c7:
            st.metric("Unresolved Items", f"{summary['unresolved']}")
        with c8:
            if summary["report_path"]:
                st.metric("Saved Report", os.path.basename(summary["report_path"]))
            else:
                st.metric("Report Status", "Generated")

        if summary["report_path"]:
            st.markdown(
                f'<div class="report-path-box">Full Markdown report available at: <code>{summary["report_path"]}</code></div>',
                unsafe_allow_html=True,
            )

        if summary["final_status"] != "failed":
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            render_results(st.session_state.final_state)


# ============================================================================
# 10. Footer
# ============================================================================

st.markdown(
    """
<div class="app-footer">
    Powered by <strong>LangGraph</strong>, <strong>Gemini / Groq</strong>, <strong>Pandas</strong> & <strong>Streamlit</strong> &bull;
    Built by <a href="https://github.com/samikshagla" target="_blank">samikshagla</a>
</div>
""",
    unsafe_allow_html=True,
)
