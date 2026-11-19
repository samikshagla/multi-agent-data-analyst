"""
app.py
Multi-Agent Data Analyst — Streamlit Web Interface.
Connects the Streamlit UI to the multi-agent pipeline with live agent cards,
activity timeline feed, progress tracking, and execution summary metrics.
Supports a zero-LLM recorded demo mode with step-by-step playback.
"""

import os
from pathlib import Path
import time
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
from tools.workspace import cleanup_stale_workspaces

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


def clear_analysis_state() -> None:
    """Resets execution state when a new dataset is loaded or demo run starts."""
    st.session_state.final_state = None
    st.session_state.statuses = initial_statuses()
    st.session_state.events_html = []
    st.session_state.seen_events_count = 0
    st.session_state.running = False
    st.session_state.error_message = None


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

    st.markdown("---")
    st.header("📂 Data Source")

    demo_only_active = is_demo_only()

    if demo_only_active:
        st.info("ℹ️ **Demo-only mode**: File uploads and live execution are disabled. Click **Watch a recorded run** above to replay a real multi-agent run.")
    else:
        # CSV File Uploader
        uploaded_file = st.file_uploader(
            "Upload a CSV dataset",
            type=["csv"],
            disabled=is_busy,
            help="Upload a tabular CSV file up to 20MB.",
        )

        if uploaded_file is not None:
            upload_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state._last_processed_upload != upload_identifier:
                df, error = load_csv_safe(uploaded_file)
                if error:
                    st.session_state.error_message = error
                    st.session_state.dataset_path = None
                    st.session_state.dataset_name = None
                    st.session_state.dataset_snapshot = None
                    st.session_state.df_preview = None
                    st.session_state.is_demo_mode = False
                    clear_analysis_state()
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
                    st.session_state.dataset_path = sample_path
                    st.session_state.dataset_name = "sample.csv (E-Commerce Deliveries)"
                    st.session_state.dataset_snapshot = dataset_snapshot(df)
                    st.session_state.df_preview = df.head(5)
                    st.session_state.is_demo_mode = False
                    st.session_state.demo_missing_notice = False
                    clear_analysis_state()
                    st.session_state._last_processed_upload = None
                    st.rerun()

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

has_live_dataset = (
    not demo_only_active
    and not st.session_state.is_demo_mode
    and st.session_state.dataset_path is not None
)

if has_live_dataset:
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    run_btn_col, _ = st.columns([1.5, 3.5])
    with run_btn_col:
        run_clicked = st.button(
            "🚀 Run Analysis",
            type="primary",
            disabled=st.session_state.running,
            use_container_width=True,
            help="Initiates the multi-agent pipeline on the loaded dataset.",
        )

    warning_placeholder = st.empty()
    cards_placeholder = st.empty()
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    feed_placeholder = st.empty()
    summary_placeholder = st.empty()

    if run_clicked:
        from llm import check_llm_available, LLMUnavailableError, LLMConfigError

        with st.spinner("Checking AI service..."):
            is_available, check_msg = check_llm_available()

        if not is_available:
            friendly_err, tech_err = format_error_details(check_msg)
            with warning_placeholder.container():
                st.error(friendly_err)
                with st.expander("Technical details"):
                    st.code(tech_err)
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
        from llm import LLMUnavailableError, LLMConfigError

        warning_placeholder.warning(
            "⏳ Analysis takes about 2 minutes. Please don't change anything until it finishes."
        )

        st.session_state.statuses["cleaner"] = {"status": "running", "note": "Cleaning data..."}
        render_agent_cards(cards_placeholder, st.session_state.statuses)
        render_feed(feed_placeholder, st.session_state.events_html)

        final_state_obj = None
        try:
            current_retry = 0
            for node_name, state_update in run_pipeline(st.session_state.dataset_path, stream=True):
                if node_name == "__final__":
                    final_state_obj = state_update
                    continue

                if "retry_count" in state_update:
                    current_retry = state_update["retry_count"]

                st.session_state.statuses = apply_node_update(
                    st.session_state.statuses,
                    node_name,
                    state_update,
                    retry_count=current_retry,
                )

                if "events" in state_update and isinstance(state_update["events"], list):
                    new_evs = new_events(state_update["events"], st.session_state.seen_events_count)
                    for ev in new_evs:
                        st.session_state.events_html.append(event_to_html(ev))
                    st.session_state.seen_events_count = len(state_update["events"])

                render_agent_cards(cards_placeholder, st.session_state.statuses)
                render_feed(feed_placeholder, st.session_state.events_html)

            if final_state_obj:
                st.session_state.final_state = final_state_obj
                st.session_state.statuses = get_final_agent_statuses(
                    final_state_obj,
                    st.session_state.statuses,
                )
                render_agent_cards(cards_placeholder, st.session_state.statuses, final_state=final_state_obj)

        except LLMUnavailableError as ex:
            warning_placeholder.empty()
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "error"
                    v["note"] = "AI service unreachable"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
            render_feed(feed_placeholder, st.session_state.events_html)
            friendly_err, tech_err = format_error_details(ex)
            st.error(friendly_err)
            with st.expander("Technical details"):
                st.code(tech_err)
        except LLMConfigError as ex:
            warning_placeholder.empty()
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "error"
                    v["note"] = "Config error"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
            render_feed(feed_placeholder, st.session_state.events_html)
            friendly_err, tech_err = format_error_details(ex)
            st.error(friendly_err)
            with st.expander("Technical details"):
                st.code(tech_err)
        except Exception as ex:
            warning_placeholder.empty()
            for k, v in st.session_state.statuses.items():
                if isinstance(v, dict) and v.get("status") == "running":
                    v["status"] = "error"
                    v["note"] = "Failed"
            render_agent_cards(cards_placeholder, st.session_state.statuses)
            render_feed(feed_placeholder, st.session_state.events_html)
            friendly_err, tech_err = format_error_details(ex)
            st.error(friendly_err)
            with st.expander("Technical details"):
                st.code(tech_err)
        finally:
            st.session_state.running = False
            if final_state_obj:
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
