"""
app.py
Multi-Agent Data Analyst — Streamlit Web Interface (UI Frame).
Provides file upload handling, sample data loading, dataset snapshot preview,
and workflow initiation interface without invoking the agent pipeline.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd

from components.data_utils import load_csv_safe, dataset_snapshot, save_upload

# ============================================================================
# 1. Page Configuration & Custom CSS Injection
# ============================================================================

st.set_page_config(
    page_title="Multi-Agent Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load external CSS styles
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

if "run_requested" not in st.session_state:
    st.session_state.run_requested = False

if "_last_processed_upload" not in st.session_state:
    st.session_state._last_processed_upload = None


# ============================================================================
# 3. Sidebar: File Upload, Sample Data & How It Works
# ============================================================================

with st.sidebar:
    st.header("📂 Data Source")

    # CSV File Uploader
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
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
                st.session_state.run_requested = False
            else:
                saved_path = save_upload(uploaded_file)
                st.session_state.dataset_path = str(saved_path)
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.dataset_snapshot = dataset_snapshot(df)
                st.session_state.df_preview = df.head(5)
                st.session_state.error_message = None
                st.session_state.run_requested = False
            st.session_state._last_processed_upload = upload_identifier
            st.rerun()

    # "Try with sample data" button
    if st.button("📊 Try with sample data", use_container_width=True):
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
                st.session_state.error_message = None
                st.session_state.run_requested = False
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
# 4. Main Area: Header Banner
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


# ============================================================================
# 5. Error Banner (when validation fails)
# ============================================================================

if st.session_state.error_message:
    st.error(f"❌ {st.session_state.error_message}")


# ============================================================================
# 6. Main Area: Dataset Snapshot or Empty State
# ============================================================================

is_dataset_loaded = (
    st.session_state.dataset_path is not None
    and st.session_state.dataset_snapshot is not None
)

if is_dataset_loaded:
    snap = st.session_state.dataset_snapshot

    st.markdown(f"### 📋 Dataset Snapshot &bull; <small style='color:#94A3B8;'>{st.session_state.dataset_name}</small>", unsafe_allow_html=True)

    # 4 Metric Tiles: Rows, Columns, Missing %, Duplicate rows
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Rows", value=f"{snap['rows']:,}")
    with m_col2:
        st.metric(label="Columns", value=f"{snap['columns']:,}")
    with m_col3:
        st.metric(label="Missing %", value=f"{snap['missing_pct']}%")
    with m_col4:
        st.metric(label="Duplicate Rows", value=f"{snap['duplicate_rows']:,}")

    # Small line listing counts of numeric, text, and date-like columns
    st.markdown(
        f"""
<div class="col-breakdown">
    Column types: <span>{snap['numeric_cols']}</span> numeric &bull; <span>{snap['text_cols']}</span> text &bull; <span>{snap['date_like_cols']}</span> date-like
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Expandable preview of the first 5 rows
    with st.expander("🔍 Preview First 5 Rows", expanded=False):
        if st.session_state.df_preview is not None:
            st.dataframe(st.session_state.df_preview, use_container_width=True)

else:
    # Empty state when no valid dataset is selected yet
    st.markdown(
        """
<div class="empty-state-card">
    <h3>Upload a CSV or try the sample data to get started</h3>
    <p>Select a file from your computer using the sidebar, or click <strong>Try with sample data</strong> to explore the system.</p>
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
# 7. Prominent "Run Analysis" Button
# ============================================================================

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

btn_col, _ = st.columns([1.5, 3.5])
with btn_col:
    run_clicked = st.button(
        "🚀 Run Analysis",
        type="primary",
        disabled=not is_dataset_loaded,
        use_container_width=True,
        help="Initiates the multi-agent pipeline on the loaded dataset.",
    )

if run_clicked:
    st.session_state.run_requested = True

if st.session_state.run_requested:
    st.info("Pipeline connection comes in the next step", icon="ℹ️")


# ============================================================================
# 8. Footer
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
