"""
components/results_tabs.py
Streamlit UI component to display the pipeline analysis results across four tabs:
1. Executive Report (Markdown report, execution metrics & download button)
2. Interactive Charts (Plotly visualizer, metric cards, takeaway observations & badges)
3. Cleaning Audit Log (Before/after dataset snapshots, transform log & CSV download)
4. Reviewer Findings (Dual recomputations, discrepancy audits & JSON export)
"""

import html
import os
from typing import Any
import pandas as pd
import plotly.io as pio
import streamlit as st

from components.ui_state import summarize_run
from components.results_data import (
    build_chart_items,
    build_cleaning_rows,
    build_review_cards,
    review_summary,
    report_filename,
    cleaned_csv_bytes,
    review_json_bytes,
)


def _tag_badge_html(tag: str) -> str:
    """Generates styled HTML pill for verification tags."""
    tag_clean = tag.strip()
    tag_slug = tag_clean.lower().replace(" ", "-")
    escaped_tag = html.escape(tag_clean)
    return f'<span class="tag-pill tag-{tag_slug}">{escaped_tag}</span>'


def render_results(final_state: dict[str, Any] | None) -> None:
    """
    Renders the four result tabs for completed multi-agent analysis.

    Args:
        final_state: Dictionary containing finalized agent outputs and run stats.
    """
    if not isinstance(final_state, dict):
        st.info("No analysis results are available to display.")
        return

    tab_report, tab_charts, tab_clean, tab_review = st.tabs([
        "📋 Report",
        "📊 Charts",
        "🧹 Cleaning Log",
        "🛡️ Reviewer Findings",
    ])

    # ========================================================================
    # Tab 1: Executive Insights Report
    # ========================================================================
    with tab_report:
        try:
            summary = summarize_run(final_state)

            # Row 1: Execution & Verification Metric Tiles
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            with m1:
                st.metric("Time", f"{summary['duration_sec']:.1f}s")
            with m2:
                st.metric("LLM Calls", summary["llm_calls"])
            with m3:
                st.metric("Tokens", f"{summary['tokens']:,}")
            with m4:
                st.metric("Retries", summary["retries_used"])
            with m5:
                st.metric("Flagged", summary["items_flagged_first_pass"])
            with m6:
                st.metric("Fixed", summary["fixed_by_retry"])
            with m7:
                st.metric("Unresolved", summary["unresolved"])

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Report Download Button
            rep_md = str(final_state.get("final_report", "")).strip()
            rep_fname = report_filename(final_state)

            down_col, _ = st.columns([2, 5])
            with down_col:
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=rep_md.encode("utf-8"),
                    file_name=rep_fname,
                    mime="text/markdown",
                    key="btn_download_report_markdown",
                    use_container_width=True,
                )

            st.markdown("---")

            # Markdown Report (Never unsafe_allow_html as instructed)
            if rep_md:
                st.markdown(rep_md)
            else:
                st.info("No markdown report generated for this dataset.")

        except Exception as ex:
            st.error(f"Failed to render Report tab: {ex}")

    # ========================================================================
    # Tab 2: Interactive Charts & Takeaways
    # ========================================================================
    with tab_charts:
        try:
            chart_items = build_chart_items(final_state)

            if not chart_items:
                st.info("No charts or analysis items available.")
            else:
                st.markdown(
                    "<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.2rem;'>"
                    "Visualizations synthesized by the <strong>Visualizer Agent</strong> paired with verified quantitative takeaways."
                    "</p>",
                    unsafe_allow_html=True,
                )

                # 2-Column Grid Layout
                for idx in range(0, len(chart_items), 2):
                    col_left, col_right = st.columns(2)
                    pair = [chart_items[idx]]
                    if idx + 1 < len(chart_items):
                        pair.append(chart_items[idx + 1])

                    for col, item in zip((col_left, col_right), pair):
                        with col:
                            iid = item["id"]
                            tag = item["tag"]
                            title_esc = html.escape(item["title"])
                            badge_html = _tag_badge_html(tag)

                            st.markdown(
                                f"""
<div class="result-card">
    <div class="result-card-header">
        <span class="result-card-title">{title_esc}</span>
        {badge_html}
    </div>
</div>
""",
                                unsafe_allow_html=True,
                            )

                            # 1. Metric Type: big metric card display
                            if item["type"] == "metric":
                                metric_val = item["takeaway"].replace("Result: ", "").strip()
                                st.markdown(
                                    f"""
<div class="metric-big-card">
    <div class="metric-big-card-val">{html.escape(metric_val)}</div>
    <div class="metric-big-card-lbl">{html.escape(item['question'])}</div>
</div>
""",
                                    unsafe_allow_html=True,
                                )

                            # 2. Tabular Plotly Chart
                            elif item["status"] == "ok" and item["figure_json"]:
                                try:
                                    fig = pio.from_json(item["figure_json"])
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"plotly_chart_{iid}",
                                    )
                                except Exception as err:
                                    st.error(f"Could not render Plotly figure: {err}")

                            # 3. Skipped / Failed chart item
                            else:
                                reason_note = item["note"] or "Chart generation was skipped."
                                st.info(f"ℹ️ **No chart:** {reason_note}")

                            # Takeaway observation
                            if item["takeaway"]:
                                st.markdown(
                                    f"""
<div class="chart-takeaway-box">
    <strong>Takeaway:</strong> {html.escape(item['takeaway'])}
</div>
""",
                                    unsafe_allow_html=True,
                                )

                            # Takeaway warning if detected
                            if item.get("takeaway_warning"):
                                st.markdown(
                                    f'<div class="takeaway-warning-text">⚠️ {html.escape(item["takeaway_warning"])}</div>',
                                    unsafe_allow_html=True,
                                )

                            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        except Exception as ex:
            st.error(f"Failed to render Charts tab: {ex}")

    # ========================================================================
    # Tab 3: Data Cleaning Log & CSV Export
    # ========================================================================
    with tab_clean:
        try:
            cleaning_rows, clean_summary = build_cleaning_rows(final_state)

            # Before / After Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                raw_rows_val = f"{clean_summary['raw_rows']:,}" if clean_summary["raw_rows"] is not None else "N/A"
                st.metric("Raw Rows", raw_rows_val)
            with c2:
                raw_cols_val = f"{clean_summary['raw_cols']}" if clean_summary["raw_cols"] is not None else "N/A"
                st.metric("Raw Columns", raw_cols_val)
            with c3:
                clean_rows_val = f"{clean_summary['clean_rows']:,}" if clean_summary["clean_rows"] is not None else "N/A"
                st.metric("Cleaned Rows", clean_rows_val)
            with c4:
                clean_cols_val = f"{clean_summary['clean_cols']}" if clean_summary["clean_cols"] is not None else "N/A"
                st.metric("Cleaned Columns", clean_cols_val)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Cleaned CSV Download Button
            csv_bytes = cleaned_csv_bytes(final_state)
            clean_btn_col, _ = st.columns([2, 5])
            with clean_btn_col:
                if csv_bytes is not None:
                    raw_stem = os.path.splitext(os.path.basename(str(final_state.get("raw_df_path", "data"))))[0]
                    clean_fname = f"{raw_stem}_cleaned.csv"
                    st.download_button(
                        label="📥 Download Cleaned CSV",
                        data=csv_bytes,
                        file_name=clean_fname,
                        mime="text/csv",
                        key="btn_download_cleaned_csv",
                        use_container_width=True,
                    )
                else:
                    st.info("Cleaned CSV file is not available on disk for download.")

            st.markdown("---")

            # Cleaning Steps Table
            if cleaning_rows:
                df_log = pd.DataFrame(cleaning_rows)
                df_log.rename(
                    columns={
                        "step": "Step",
                        "kind": "Type",
                        "message": "Transformation / Audit Log",
                    },
                    inplace=True,
                )
                st.dataframe(df_log, use_container_width=True, hide_index=True)
            else:
                st.info("No data transformations were performed by the Cleaner agent.")

        except Exception as ex:
            st.error(f"Failed to render Cleaning Log tab: {ex}")

    # ========================================================================
    # Tab 4: Reviewer Findings & Verification Audit
    # ========================================================================
    with tab_review:
        try:
            rev_sum = review_summary(final_state)

            # Summary Metrics Row
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            with r1:
                st.metric("Evaluated", rev_sum["items_checked"])
            with r2:
                st.metric("Verified", rev_sum["verified"])
            with r3:
                st.metric("Unverified", rev_sum["unverified"])
            with r4:
                st.metric("Unresolved", rev_sum["unresolved"])
            with r5:
                st.metric("Flagged (Pass 1)", rev_sum["caught_first_pass"])
            with r6:
                st.metric("Fixed by Retry", rev_sum["fixed_by_retry"])

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color: #94A3B8; font-size: 0.92rem; font-style: italic; margin-bottom: 1.2rem;'>"
                "The reviewer recomputes each answer independently and compares the numbers."
                "</p>",
                unsafe_allow_html=True,
            )

            # Download Audit JSON Button
            audit_json_bytes = review_json_bytes(final_state)
            json_btn_col, _ = st.columns([2, 5])
            with json_btn_col:
                st.download_button(
                    label="📥 Download Review Audit JSON",
                    data=audit_json_bytes,
                    file_name="review_audit.json",
                    mime="application/json",
                    key="btn_download_review_audit_json",
                    use_container_width=True,
                )

            st.markdown("---")

            # Item Cards
            review_cards = build_review_cards(final_state)

            if not review_cards:
                st.info("No review audit findings recorded.")
            else:
                for card in review_cards:
                    cid = card["id"]
                    q_esc = html.escape(card["question"])
                    tag = card["tag"]
                    badge_html = _tag_badge_html(tag)

                    with st.container():
                        st.markdown(
                            f"""
<div class="result-card">
    <div class="result-card-header">
        <span class="result-card-title">Q{cid + 1}: {q_esc}</span>
        {badge_html}
    </div>
</div>
""",
                            unsafe_allow_html=True,
                        )

                        # Dual recomputed comparison columns (scalars or small dataframes)
                        orig_disp = card.get("original_display") or {
                            "kind": "text",
                            "formatted": str(card.get("original_value", "")),
                        }
                        recomp_disp = card.get("recomputed_display") or {
                            "kind": "text",
                            "formatted": str(card.get("recomputed_value", "")),
                        }

                        col_orig, col_recomp = st.columns(2)
                        with col_orig:
                            st.caption("🔍 Analyst Result")
                            if orig_disp.get("kind") == "scalar":
                                st.markdown(
                                    f"<div style='font-size: 1.35rem; font-weight: 700; color: #14B8A6; padding: 0.35rem 0;'>"
                                    f"{html.escape(orig_disp['formatted'])}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            elif orig_disp.get("kind") == "dataframe":
                                st.dataframe(orig_disp["value"], use_container_width=True)
                                if orig_disp.get("caption"):
                                    st.caption(orig_disp["caption"])
                            elif orig_disp.get("kind") == "missing":
                                st.caption("(None)")
                            else:
                                st.code(orig_disp.get("formatted") or "(None)", language="text")

                        with col_recomp:
                            st.caption("🛡️ Reviewer Recomputed")
                            if recomp_disp.get("kind") == "scalar":
                                st.markdown(
                                    f"<div style='font-size: 1.35rem; font-weight: 700; color: #38BDF8; padding: 0.35rem 0;'>"
                                    f"{html.escape(recomp_disp['formatted'])}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            elif recomp_disp.get("kind") == "dataframe":
                                st.dataframe(recomp_disp["value"], use_container_width=True)
                                if recomp_disp.get("caption"):
                                    st.caption(recomp_disp["caption"])
                            elif recomp_disp.get("kind") == "missing":
                                st.caption("(None)")
                            else:
                                st.code(recomp_disp.get("formatted") or "(None)", language="text")

                        # Highlight differing values when verdict is mismatch
                        diffs = card.get("differing_notes", [])
                        if diffs:
                            diff_items = "".join(f"<li>{html.escape(d)}</li>" for d in diffs[:3])
                            st.markdown(
                                f"""
<div style="background: rgba(239, 68, 68, 0.12); border-left: 3px solid #EF4444; padding: 0.5rem 0.85rem; border-radius: 6px; font-size: 0.85rem; margin: 0.5rem 0;">
    <strong style="color: #F87171;">Differences detected:</strong>
    <ul style="margin: 0.25rem 0 0 0; padding-left: 1.2rem; color: #FECACA;">
        {diff_items}
    </ul>
</div>
""",
                                unsafe_allow_html=True,
                            )

                        # Audit reason & finding
                        if card["reason"]:
                            st.markdown(f"**Audit Finding:** {card['reason']}")

                        # Reviewer notes (advisory only, from Check 3)
                        logic_notes = card.get("logic_warnings", [])
                        if logic_notes:
                            st.markdown("**Reviewer notes (advisory):**")
                            for note in logic_notes:
                                st.markdown(f"- ℹ️ *{html.escape(note)}*")

                        if card.get("takeaway_warning"):
                            st.caption(f"⚠️ Takeaway check: {card['takeaway_warning']}")

                        # Retry history if item failed on an earlier pass
                        if card["retry_history"]:
                            st.markdown("**Retry History:**")
                            for h in card["retry_history"]:
                                st.markdown(f"- **Retry {h['retry_round'] + 1}:** {h['reason']}")

                        # Expandable Analyst Python Code
                        if card["original_code"]:
                            with st.expander("🔍 View Analyst Code", expanded=False):
                                st.code(card["original_code"], language="python")

                        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        except Exception as ex:
            st.error(f"Failed to render Reviewer Findings tab: {ex}")
