"""
Fed-AgentGEO: Public Demo with De-identified Examples
with Generative Engine Optimization (GEO) & Explainable AI (XAI)

Professional, Fully Interactive Streamlit Interface (app.py)
No pre-filled or hardcoded inputs/outputs.
Interactive search, demo case entry, and live federated retrieval.
"""

import os
import json
import time
import uuid
from pathlib import Path
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Fed-AgentGEO | Clinical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Internal module imports with fallback
try:
    from src.local_node import HospitalNode, GuidelinesNode
    from src.agent_geo_optimizer import optimize_clinical_record
    from src.federated_ranker import FederatedAggregator, USE_MOCK_LLM
    from src.speech_to_text import render_speech_dictation_widget
    from src.report_parser import parse_clinical_document
except ImportError:
    from local_node import HospitalNode, GuidelinesNode
    from agent_geo_optimizer import optimize_clinical_record
    from federated_ranker import FederatedAggregator, USE_MOCK_LLM
    from speech_to_text import render_speech_dictation_widget
    from report_parser import parse_clinical_document

# Professional Clean CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
  }

  /* Card Containers */
  .geo-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  }
  @media (prefers-color-scheme: dark) {
    .geo-card {
      background: #14171A;
      border-color: #272B30;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
  }

  /* KPI Stat Boxes */
  .kpi-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
  }
  @media (prefers-color-scheme: dark) {
    .kpi-box {
      background: #181B1E;
      border-color: #2C3238;
    }
  }
  .kpi-val {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
  }
  @media (prefers-color-scheme: dark) {
    .kpi-val { color: #F8FAFC; }
  }
  .kpi-lbl {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-top: 2px;
  }

  /* Semantic Badges */
  .geo-badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 6px 0;
  }
  .geo-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
  }
  .geo-badge-dept { background: #E0F2FE; color: #0284C7; }
  .geo-badge-dx { background: #F1F5F9; color: #334155; }
  .geo-badge-renal { background: #FEF3C7; color: #B45309; }
  .geo-badge-endocrine { background: #EDE9FE; color: #6D28D9; }
  .geo-badge-cm { background: #F3F4F6; color: #4B5563; }
  .geo-badge-allergy { background: #FFE4E6; color: #BE123C; }
  .geo-badge-success { background: #DCFCE7; color: #166534; font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }
  .geo-badge-warning { background: #FEF9C3; color: #854D0E; font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }
  
  .geo-alert {
    display: inline-block;
    background: #FEE2E2;
    color: #991B1B;
    border-left: 3px solid #EF4444;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 3px;
    margin: 2px 0;
  }

  /* Citation Pill */
  .citation-pill {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    font-weight: 600;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
  }
  .citation-pill-empty {
    background: #FFFBEB;
    border: 1px dashed #D97706;
    color: #B45309;
    font-weight: 600;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
  }

  .no-match-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
  }
  @media (prefers-color-scheme: dark) {
    .no-match-card {
      background: #231C10;
      border-color: #78350F;
      border-left-color: #D97706;
    }
  }

  /* 3-Column Stream Headers */
  .stream-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    padding-bottom: 5px;
    margin-bottom: 8px;
    border-bottom: 2px solid;
  }
  .stream-local { border-color: #2563EB; color: #2563EB; }
  .stream-peer { border-color: #7C3AED; color: #7C3AED; }
  .stream-guideline { border-color: #059669; color: #059669; }

  /* Sidebar header */
  .sb-brand {
    padding-bottom: 8px;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 12px;
  }
  @media (prefers-color-scheme: dark) {
    .sb-brand { border-color: #2D3748; }
  }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SESSION STATE MANAGEMENT (CLEAN, USER-DRIVEN)
# ------------------------------------------------------------------------------
if "aggregator" not in st.session_state:
    st.session_state.aggregator = FederatedAggregator()

if "active_hospital" not in st.session_state:
    st.session_state.active_hospital = "Hospital_A"

if "signed_ehr_logs" not in st.session_state:
    st.session_state.signed_ehr_logs = []

if "extracted_vitals" not in st.session_state:
    st.session_state.extracted_vitals = {}

# Start with an empty query so the user enters their own input
if "query_input_text" not in st.session_state:
    st.session_state.query_input_text = ""

# Start with None so NO predefined output is displayed initially
if "consultation_results" not in st.session_state:
    st.session_state.consultation_results = None

# Custom draft order in session state
if "custom_rx_order" not in st.session_state:
    st.session_state.custom_rx_order = ""

if "custom_rx_rationale" not in st.session_state:
    st.session_state.custom_rx_rationale = ""

# Ingestion form user-entered state (starts empty)
if "ingest_fields" not in st.session_state:
    st.session_state.ingest_fields = {
        "patient_id": "",
        "department": "Cardiology",
        "physician": "",
        "age": 0,
        "gender": "Male",
        "primary_dx": "",
        "comorbidities": [],
        "allergies": "",
        "bp_d1": "",
        "hr_d1": "",
        "cr_d1": "",
        "egfr_d1": "",
        "symp_d1": "",
        "rx_d1": "",
        "bp_d4": "",
        "hr_d4": "",
        "cr_d4": "",
        "egfr_d4": "",
        "symp_d4": "",
        "rx_d4": "",
        "outcome_status": "Improved",
        "effective_meds": "",
        "contraindicated_meds": "",
        "notes_raw": ""
    }

def clear_ingestion_form():
    st.session_state.ingest_fields = {
        "patient_id": "",
        "department": "Cardiology",
        "physician": "",
        "age": 0,
        "gender": "Male",
        "primary_dx": "",
        "comorbidities": [],
        "allergies": "",
        "bp_d1": "",
        "hr_d1": "",
        "cr_d1": "",
        "egfr_d1": "",
        "symp_d1": "",
        "rx_d1": "",
        "bp_d4": "",
        "hr_d4": "",
        "cr_d4": "",
        "egfr_d4": "",
        "symp_d4": "",
        "rx_d4": "",
        "outcome_status": "Improved",
        "effective_meds": "",
        "contraindicated_meds": "",
        "notes_raw": ""
    }
    st.session_state.extracted_vitals = {}

# ------------------------------------------------------------------------------
# SIDEBAR: INSTITUTIONAL CONTROLS & LIVE STATS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div style="font-size: 17px; font-weight: 700; color: #1E293B;">🩺 Fed-AgentGEO</div>
      <div style="font-size: 11px; color: #64748B;">Clinical Decision Support & XAI</div>
    </div>
    """, unsafe_allow_html=True)

    # Active Hospital Selector
    st.markdown("<div style='font-size: 12px; font-weight: 600; color: #475569;'>Institutional Node:</div>", unsafe_allow_html=True)
    hosp_map = {
        "Hospital_A": "Hospital A (My Hospital)",
        "Hospital_B": "Hospital B (Peer Center)",
        "Hospital_C": "Hospital C (Regional Center)"
    }
    selected_node = st.selectbox(
        "Active Node:",
        options=list(hosp_map.keys()),
        format_func=lambda x: hosp_map[x],
        index=list(hosp_map.keys()).index(st.session_state.active_hospital),
        label_visibility="collapsed"
    )
    if selected_node != st.session_state.active_hospital:
        st.session_state.active_hospital = selected_node
        st.session_state.consultation_results = None
        st.rerun()

    current_node = st.session_state.aggregator.get_node(st.session_state.active_hospital)
    vault_metrics = current_node.get_vault_metrics()

    # Privacy Status Card
    st.markdown(f"""
    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 6px 10px; margin: 10px 0 14px 0;">
      <div style="font-size: 11px; font-weight: 600; color: #854D0E;">🟡 Public Demo · De-identified Examples</div>
      <div style="font-size: 10px; color: #15803D; margin-top: 2px;">ChromaDB Collection: <code>node_{st.session_state.active_hospital.lower()}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Local Node Metrics
    st.markdown("<div style='font-size: 12px; font-weight: 600; color: #475569;'>Node Overview:</div>", unsafe_allow_html=True)
    sb_c1, sb_c2 = st.columns(2)
    with sb_c1:
        st.metric("Indexed Cases", vault_metrics["total_cases"])
    with sb_c2:
        st.metric("Signal Boost", f"+{vault_metrics['snr_gain_pct']}%")

    st.markdown("---")
    # Audit Trail Downloads
    st.markdown("<div style='font-size: 12px; font-weight: 600; color: #475569;'>Audit Logs:</div>", unsafe_allow_html=True)
    if st.session_state.signed_ehr_logs:
        st.download_button(
            f"📥 Download Signed EHR Orders ({len(st.session_state.signed_ehr_logs)})",
            data=json.dumps(st.session_state.signed_ehr_logs, indent=2),
            file_name=f"ehr_audit_{st.session_state.active_hospital.lower()}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.caption("No orders signed in this session.")

    st.caption("AgentGEO research demo · de-identified example data")

# ------------------------------------------------------------------------------
# MAIN TOP HEADER
# ------------------------------------------------------------------------------
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
      <h3 style="margin: 0; font-weight: 700; letter-spacing: -0.3px;">Fed-AgentGEO</h3>
      <span class="geo-badge geo-badge-dept">Active: {st.session_state.active_hospital.replace('_', ' ')}</span>
      <span class="geo-badge geo-badge-warning">Public Demo · Not for Clinical Use</span>
    </div>
    <div style="font-size: 12px; color: #64748B; margin-bottom: 12px;">
      Generative Engine Optimization (GEO) & Explainable AI for Multi-Center Decision Support
    </div>
    """, unsafe_allow_html=True)
with top_col2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 6px;">
      <span class="citation-pill">Node: {st.session_state.active_hospital}</span>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3 PRIMARY TABS
# ------------------------------------------------------------------------------
st.warning("Public demo with de-identified example cases. Their synthetic status has not been independently verified. Do not enter real patient information. Outputs are for research demonstration and are not medical advice or a substitute for professional care.")
tab_board, tab_ingest, tab_vault = st.tabs([
    "🤝 Tab 1: Federated Consultation & Clinical Board",
    "📥 Tab 2: Patient Entry (Demo Only)",
    "🏛️ Tab 3: Local GEO Knowledge Vault (Audit)"
])

# ==============================================================================
# TAB 1: FEDERATED CONSULTATION & CLINICAL BOARD
# ==============================================================================
with tab_board:
    # Query Input Card (Clean, Interactive, User-Driven)
    with st.container():
        st.markdown('<div class="geo-card">', unsafe_allow_html=True)
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #1E293B; margin-bottom: 4px;'>Patient Clinical Query:</div>", unsafe_allow_html=True)

        # Voice dictation widget
        render_speech_dictation_widget("query_dictate", height=100, placeholder="Dictate clinical symptoms or presentation...")

        # Clinical Query text area with clear placeholder
        user_query_text = st.text_area(
            "Enter clinical presentation, demographics, lab biomarkers, or comorbidity profile:",
            value=st.session_state.query_input_text,
            placeholder="e.g., 64-year-old male with unstable angina, eGFR 38 mL/min, type-2 diabetes, and hypertension...",
            height=75,
            key="user_query_input_box"
        )
        st.session_state.query_input_text = user_query_text

        c_flt1, c_flt2, c_btn = st.columns([1, 1, 2])
        with c_flt1:
            dept_filter = st.selectbox("Department Scope:", ["All", "Cardiology", "Nephrology", "Endocrinology"], index=0)
        with c_flt2:
            top_k_val = st.slider("Evidence Depth (k):", 1, 5, 3)
        with c_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_query_btn = st.button("🔍 Run Federated Evidence Retrieval", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Execute Federated Query only when triggered by the user
    if run_query_btn:
        if not user_query_text.strip():
            st.warning("⚠️ Please enter a patient clinical query or speak via the voice dictation tool above before running retrieval.")
        else:
            with st.spinner("Executing federated retrieval across local node, external peer hospitals, and guidelines..."):
                st.session_state.consultation_results = st.session_state.aggregator.run_federated_consultation(
                    active_hospital_id=st.session_state.active_hospital,
                    query_text=user_query_text,
                    department_filter=dept_filter,
                    top_k=top_k_val
                )

    results = st.session_state.consultation_results

    # If no query has been run yet, show clean interactive ready state
    if results is None:
        st.markdown("""
        <div class="geo-card" style="text-align: center; padding: 45px 20px; color: #64748B;">
          <div style="font-size: 32px; margin-bottom: 6px;">🩺</div>
          <div style="font-size: 16px; font-weight: 700; color: #1E293B;">Ready for Clinical Consultation</div>
          <div style="font-size: 13px; margin-top: 4px; max-width: 580px; margin-left: auto; margin-right: auto;">
            Enter patient symptoms, biomarkers (e.g. eGFR, creatinine), or comorbidities above and click <b>Run Federated Evidence Retrieval</b> to search across isolated hospital nodes.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        stream_1 = results["stream_1_local"]
        stream_2 = results["stream_2_peer"]
        stream_3 = results["stream_3_guideline"]

        # ----------------------------------------------------------------------
        # THE 3-COLUMN COMPARATIVE BOARD
        # ----------------------------------------------------------------------
        st.markdown("<h4 style='margin: 14px 0 6px 0; font-weight: 700;'>The 3-Column Comparative Board</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        # COLUMN 1: LOCAL HOSPITAL
        with col1:
            st.markdown('<div class="geo-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown(f'<div class="stream-header stream-local">🏥 STREAM 1: {stream_1["hospital_id"].replace("_", " ")} (LOCAL)</div>', unsafe_allow_html=True)
            
            if stream_1.get("has_match") and stream_1.get("top_case"):
                st.markdown(f"**Cohort Match**: <span class='geo-badge-success'>{stream_1['cohort_match_pct']}% Match</span>", unsafe_allow_html=True)
                st.markdown(f"**Citation**: <span class='citation-pill'>{stream_1['citation']}</span>", unsafe_allow_html=True)
                st.markdown("---")
                top_loc = stream_1["top_case"]
                meta = top_loc["metadata"]
                st.markdown(f"**Top Match**: `#{top_loc['case_id']}` ({meta.get('department')})")
                st.markdown(f"**Profile**: {meta.get('age')}yo {meta.get('gender')} · {meta.get('primary_diagnosis')}")
                st.markdown(f"**Comorbidities**: {meta.get('comorbidities')}")
                st.markdown(f"**Outcome**: <span class='geo-badge-success'>{meta.get('outcome_status')}</span>", unsafe_allow_html=True)
                with st.expander("📄 View Local Evidence Chunk"):
                    st.markdown(top_loc["evidence_chunk"], unsafe_allow_html=True)
            else:
                st.markdown(f"**Cohort Match**: <span class='geo-badge-warning'>0.0% Match</span>", unsafe_allow_html=True)
                st.markdown(f"**Citation**: <span class='citation-pill-empty'>{stream_1['citation']}</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"""
                <div class="no-match-card">
                  <div style="font-size: 14px; font-weight: 700; color: #B45309; margin-bottom: 4px;">
                    ⚠️ Response 1: No Perfect Match Found
                  </div>
                  <div style="font-size: 12px; color: #78350F; margin-bottom: 8px; line-height: 1.4;">
                    <b>{stream_1["hospital_id"].replace("_", " ")} (Own Hospital)</b> has no historical patient records with clinical relevance matching this presentation.
                  </div>
                  <div style="font-size: 11px; color: #92400E; background: #FEF3C7; padding: 6px 10px; border-radius: 6px; border: 1px solid #FDE68A;">
                    {stream_1.get("no_match_message", "No matching cases meeting relevance criteria.")}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # COLUMN 2: TOP PEER CENTER
        with col2:
            st.markdown('<div class="geo-card" style="height: 100%;">', unsafe_allow_html=True)
            peer_name = stream_2["hospital_id"].replace("_", " ")
            st.markdown(f'<div class="stream-header stream-peer">🏆 STREAM 2: {peer_name} (TOP PEER)</div>', unsafe_allow_html=True)
            
            if stream_2.get("has_match") and stream_2.get("top_case"):
                st.markdown(f"**Hybrid Score**: <span class='geo-badge geo-badge-endocrine'>Score: {stream_2['hybrid_score']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Citation**: <span class='citation-pill'>{stream_2['citation']}</span>", unsafe_allow_html=True)
                st.markdown("---")
                top_p = stream_2["top_case"]
                p_meta = top_p["metadata"]
                st.markdown(f"**Benchmark**: `#{top_p['case_id']}` ({p_meta.get('department')})")
                st.markdown(f"**Profile**: {p_meta.get('age')}yo {p_meta.get('gender')} · {p_meta.get('primary_diagnosis')}")
                st.markdown(f"**Comorbidities**: {p_meta.get('comorbidities')}")
                st.markdown(f"**Outcome**: <span class='geo-badge-success'>{p_meta.get('outcome_status')}</span>", unsafe_allow_html=True)
                st.caption(f"Score Formula: 0.45×Sim ({stream_2['cohort_similarity_pct']}%) + 0.40×Outcome ({stream_2['positive_outcome_rate_pct']}%) + 0.15×Vol ({stream_2['normalized_volume']})")
                with st.expander("📄 View Peer Evidence Chunk"):
                    st.markdown(top_p["evidence_chunk"], unsafe_allow_html=True)
            else:
                st.markdown(f"**Hybrid Score**: <span class='geo-badge-warning'>Score: 0.0</span>", unsafe_allow_html=True)
                st.markdown(f"**Citation**: <span class='citation-pill-empty'>{stream_2['citation']}</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"""
                <div class="no-match-card">
                  <div style="font-size: 14px; font-weight: 700; color: #B45309; margin-bottom: 4px;">
                    ⚠️ Response 2: No Perfect Match Found
                  </div>
                  <div style="font-size: 12px; color: #78350F; margin-bottom: 8px; line-height: 1.4;">
                    <b>External Peer Centers</b> (Hospital B & Hospital C) have no benchmark patient cohort matching this clinical presentation.
                  </div>
                  <div style="font-size: 11px; color: #92400E; background: #FEF3C7; padding: 6px 10px; border-radius: 6px; border: 1px solid #FDE68A;">
                    {stream_2.get("no_match_message", "Zero external hospital cases met clinical relevance criteria.")}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # COLUMN 3: GLOBAL REFERENCE GUIDELINES
        with col3:
            st.markdown('<div class="geo-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown(f'<div class="stream-header stream-guideline">🌐 STREAM 3: GLOBAL PRACTICE GUIDELINE</div>', unsafe_allow_html=True)
            
            if stream_3.get("has_match") and stream_3.get("top_guideline"):
                top_g = stream_3["top_guideline"]
                body_name = top_g['issuing_body'] if top_g else 'Global Authority'
                st.markdown(f"**Authority**: <span class='geo-badge geo-badge-dept'>{body_name}</span>", unsafe_allow_html=True)
                st.markdown(f"**Citation**: <span class='citation-pill'>{stream_3['citation']}</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"**Standard**: {top_g['title']}")
                with st.expander("📖 View Guideline Rules & Dosing", expanded=False):
                    st.markdown(top_g["evidence_chunk"])
            else:
                st.markdown(f"**Authority**: <span class='geo-badge-warning'>Global Standards</span>", unsafe_allow_html=True)
                st.markdown(f"**Citation**: <span class='citation-pill-empty'>{stream_3['citation']}</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"""
                <div class="no-match-card">
                  <div style="font-size: 14px; font-weight: 700; color: #B45309; margin-bottom: 4px;">
                    ⚠️ Response 3: No Perfect Match Found
                  </div>
                  <div style="font-size: 12px; color: #78350F; margin-bottom: 8px; line-height: 1.4;">
                    <b>Public Practice Guidelines</b> (KDIGO, ACC/AHA, ADA) in the repository do not contain a protocol directly matching this presentation.
                  </div>
                  <div style="font-size: 11px; color: #92400E; background: #FEF3C7; padding: 6px 10px; border-radius: 6px; border: 1px solid #FDE68A;">
                    {stream_3.get("no_match_message", "No guideline scored above clinical relevance threshold.")}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # BOTTOM PANEL: EXPLAINABLE AI (XAI) & DOCTOR DECISION PAD
        # ----------------------------------------------------------------------
        st.markdown("<h4 style='margin: 16px 0 6px 0; font-weight: 700;'>Explainable AI (XAI) Transparency & Decision Pad</h4>", unsafe_allow_html=True)
        
        # 1. Feature Match Breakdown Table
        with st.container():
            st.markdown('<div class="geo-card">', unsafe_allow_html=True)
            st.markdown("**1. Cohort Feature Match Matrix:**")
            df_xai = pd.DataFrame(results["xai_feature_breakdown"])
            st.dataframe(df_xai, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 2. Statistical Transparency & Triangulation Synthesis
        c_stat, c_synth = st.columns([1, 2])
        with c_stat:
            st.markdown('<div class="geo-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("**2. Statistical Response Rate:**")
            stats = results["statistical_transparency"]
            if stats.get("has_cohort", True):
                st.markdown(f"""
                <div style="margin: 8px 0;">
                  <div style="font-size: 26px; font-weight: 700; color: #10B981;">{stats['improved_pct']}%</div>
                  <div style="font-size: 11px; color: #64748B;">Improved Response Across Cohort</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(stats['improved_pct']) / 100.0)
                st.markdown(f"- **Evaluated Cohort**: `{stats['total_evaluated']} patients`")
                st.markdown(f"- **Adverse Reactions**: `{stats['adverse_pct']}%`")
                st.markdown(f"- **Stable / Neutral**: `{stats['stable_pct']}%`")
                st.markdown(f"- **Winning Peer Hybrid Score**: `{stats['peer_hybrid_score']}`")
            else:
                st.markdown("""
                <div style="margin: 8px 0;">
                  <div style="font-size: 20px; font-weight: 700; color: #94A3B8;">N/A</div>
                  <div style="font-size: 11px; color: #64748B;">No cohort cases matched</div>
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 6px; line-height: 1.4;">
                  Statistical outcome rates unavailable because no historical patient records met clinical criteria.
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_synth:
            st.markdown('<div class="geo-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("**3. Consensus Triangulation Note:**")
            st.markdown(results["cross_stream_synthesis"])
            st.markdown('</div>', unsafe_allow_html=True)

        # 3. Doctor Prescription Pad (Interactive, User-Driven)
        with st.container():
            st.markdown('<div class="geo-card" style="border-top: 3px solid #2563EB;">', unsafe_allow_html=True)
            
            c_head1, c_head2 = st.columns([3, 1])
            with c_head1:
                st.markdown("**4. Attending Physician Prescription Pad & EHR Sign-off:**")
            with c_head2:
                # Interactive button to auto-draft from evidence if user desires
                if st.button("⚡ Auto-Draft Orders", help="Draft orders based on retrieved evidence"):
                    has_any_match = stream_1.get("has_match") or stream_2.get("has_match") or stream_3.get("has_match")
                    if has_any_match:
                        st.session_state.custom_rx_order = (
                            "1. Continue indicated cardioprotective/anti-ischemic therapy\n"
                            "2. Dose-adjust or withhold high-dose RAAS blockade if renal markers borderline\n"
                            "3. Evaluate guideline-directed SGLT2i/protective therapy once stabilized"
                        )
                        cites = []
                        if stream_1.get("has_match"): cites.append(f"Local [{stream_1['citation']}]")
                        if stream_2.get("has_match"): cites.append(f"Peer [{stream_2['citation']}]")
                        if stream_3.get("has_match"): cites.append(f"Guidelines [{stream_3['citation']}]")
                        st.session_state.custom_rx_rationale = f"Grounded in consensus from {', '.join(cites)}."
                    else:
                        st.session_state.custom_rx_order = (
                            "1. Diagnostic workup and bedside clinical assessment\n"
                            "2. Order baseline laboratory panel and imaging\n"
                            "3. Refer to subspecialty clinical guidance"
                        )
                        st.session_state.custom_rx_rationale = (
                            "No direct matches found in Local Hospital, Peer Centers, or Practice Guidelines. "
                            "Treatment plan formulated based on primary bedside examination."
                        )
                    st.rerun()

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                rx_order_text = st.text_area(
                    "Enter Final Pharmacological Orders & Dosages:",
                    value=st.session_state.custom_rx_order,
                    placeholder="Enter final medication regimen, dosage, frequency, and administration route...",
                    height=110,
                    key="rx_order_box"
                )
                st.session_state.custom_rx_order = rx_order_text
            with p_col2:
                rationale_text = st.text_area(
                    "Enter Clinical Rationale & Decision Justification:",
                    value=st.session_state.custom_rx_rationale,
                    placeholder="Document clinical rationale, contraindication considerations, and evidence grounding...",
                    height=110,
                    key="rx_rationale_box"
                )
                st.session_state.custom_rx_rationale = rationale_text

            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                if st.button("✍️ Sign & Log Clinical Decision to Hospital EHR", type="primary", use_container_width=True):
                    if not rx_order_text.strip():
                        st.warning("Please enter pharmacological orders before signing.")
                    else:
                        record = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "hospital_id": st.session_state.active_hospital,
                            "query": user_query_text,
                            "local_citation": stream_1["citation"] if stream_1.get("has_match") else "No Perfect Match",
                            "peer_citation": stream_2["citation"] if stream_2.get("has_match") else "No Perfect Match",
                            "guideline_citation": stream_3["citation"] if stream_3.get("has_match") else "No Perfect Match",
                            "prescription": rx_order_text,
                            "rationale": rationale_text,
                            "physician": "Synthetic Clinician"
                        }
                        st.session_state.signed_ehr_logs.append(record)
                        st.success("✓ Clinical order cryptographically signed and logged into local hospital EHR audit trail!")
            with b_col2:
                if st.session_state.signed_ehr_logs:
                    st.download_button(
                        f"📥 Export EHR Audit ({len(st.session_state.signed_ehr_logs)})",
                        data=json.dumps(st.session_state.signed_ehr_logs, indent=2),
                        file_name=f"ehr_audit_{st.session_state.active_hospital.lower()}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: CLINICAL INGESTION PORTAL (DOCTOR CASE ENTRY)
# ==============================================================================
with tab_ingest:
    st.markdown("<h4 style='margin: 0 0 6px 0; font-weight: 700;'>Add a Patient Case to the Demo Index</h4>", unsafe_allow_html=True)
    st.info("Demo only. Add fictional or de-identified cases. Submitted cases are saved to this public demo's hospital case store and search index, where other visitors may see or retrieve them. Never enter real patient information.")

    with st.form("demo_patient_entry"):
        st.markdown("**Patient profile**")
        profile_cols = st.columns(4)
        with profile_cols[0]:
            demo_age = st.number_input("Age", min_value=0, max_value=115, value=0, key="demo_age")
        with profile_cols[1]:
            demo_gender = st.selectbox("Gender", ["Select", "Female", "Male", "Other"], key="demo_gender")
        with profile_cols[2]:
            demo_department = st.selectbox("Department", ["Cardiology", "Nephrology", "Endocrinology", "Internal Medicine"], key="demo_department")
        with profile_cols[3]:
            demo_outcome = st.selectbox("Outcome", ["Improved", "Stable", "Unresponsive", "Adverse Reaction"], key="demo_outcome")

        demo_diagnosis = st.text_input("Primary diagnosis", placeholder="Example: Stage 3 chronic kidney disease", key="demo_diagnosis")
        demo_comorbidities = st.text_input("Comorbidities", placeholder="Separate entries with commas", key="demo_comorbidities")
        demo_allergies = st.text_input("Allergies", placeholder="Separate entries with commas", key="demo_allergies")

        st.markdown("**Initial observations (optional)**")
        vitals_cols = st.columns(4)
        with vitals_cols[0]:
            demo_bp = st.text_input("Blood pressure", placeholder="128/78 mmHg", key="demo_bp")
        with vitals_cols[1]:
            demo_hr = st.text_input("Heart rate", placeholder="72 bpm", key="demo_hr")
        with vitals_cols[2]:
            demo_creatinine = st.text_input("Creatinine", placeholder="1.1 mg/dL", key="demo_creatinine")
        with vitals_cols[3]:
            demo_egfr = st.text_input("eGFR", placeholder="68 mL/min", key="demo_egfr")

        demo_symptoms = st.text_input("Symptoms", placeholder="Describe the presentation", key="demo_symptoms")
        demo_intervention = st.text_input("Interventions or medications", placeholder="Separate entries with commas", key="demo_intervention")
        demo_effective = st.text_input("Effective medications", placeholder="Separate entries with commas", key="demo_effective")
        demo_contraindicated = st.text_input("Withheld or contraindicated medications", placeholder="Separate entries with commas", key="demo_contraindicated")
        demo_notes = st.text_area("Clinical narrative", placeholder="Enter a fictional clinical summary", height=110, key="demo_notes")
        preview_clicked = st.form_submit_button("Add case to demo index", type="primary", use_container_width=True)

    if preview_clicked:
        st.session_state.pop("demo_optimization_preview", None)
        if demo_age < 1 or demo_gender == "Select" or not demo_diagnosis.strip() or not demo_notes.strip():
            st.warning("Enter an age, gender, primary diagnosis, and clinical narrative to create a preview.")
        else:
            case_preview = {
                "patient_id": f"DEMO-USER-{uuid.uuid4().hex[:10].upper()}",
                "hospital_id": st.session_state.active_hospital,
                "department": demo_department,
                "demographics": {"age": int(demo_age), "gender": demo_gender},
                "primary_diagnosis": demo_diagnosis.strip(),
                "comorbidities": [item.strip() for item in demo_comorbidities.split(",") if item.strip()],
                "allergies": [item.strip() for item in demo_allergies.split(",") if item.strip()],
                "longitudinal_timeline": [{
                    "day": 1,
                    "symptoms": demo_symptoms,
                    "vitals": {"bp": demo_bp, "hr": demo_hr, "creatinine": demo_creatinine, "eGFR": demo_egfr},
                    "intervention": demo_intervention,
                }],
                "treatment_outcome": {
                    "status": demo_outcome,
                    "effective_medications": [item.strip() for item in demo_effective.split(",") if item.strip()],
                    "adverse_or_contraindicated_drugs": [item.strip() for item in demo_contraindicated.split(",") if item.strip()],
                    "clinical_notes_raw": demo_notes.strip(),
                },
            }
            preview_result = current_node.ingest_patient_case(case_preview)
            preview_result.pop("original_raw_text", None)
            st.session_state.demo_optimization_preview = preview_result

    preview_result = st.session_state.get("demo_optimization_preview")
    if preview_result:
        st.success("Case added to the selected hospital's demo case store and search index.")
        result_cols = st.columns(3)
        result_cols[0].metric("Raw SNR", f"{preview_result['raw_snr']:.3f}")
        result_cols[1].metric("Optimized SNR", f"{preview_result['optimized_snr']:.3f}", f"+{preview_result['improvement_pct']}%")
        result_cols[2].metric("Repairs applied", len(preview_result["repairs_applied"]))
        st.markdown(f"**Diagnosed issues:** {', '.join(preview_result['diagnosed_flaws']) or 'None'}")
        st.markdown(f"**Repairs:** {', '.join(preview_result['repairs_applied'])}")
        with st.expander("View optimized evidence preview", expanded=True):
            st.code(preview_result["optimized_evidence_chunk"], language="markdown")

# ==============================================================================
# TAB 3: LOCAL GEO KNOWLEDGE VAULT
# ==============================================================================
with tab_vault:
    st.markdown("<h4 style='margin: 0 0 6px 0; font-weight: 700;'>Institutional Knowledge Vault & Audit</h4>", unsafe_allow_html=True)
    st.caption(f"Inspection vault for **{st.session_state.active_hospital}**. Search, filter, and audit local patient cases.")

    # Top KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-val">{vault_metrics['total_cases']}</div>
          <div class="kpi-lbl">Total Indexed Cases</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-val">{vault_metrics['avg_raw_snr']:.3f}</div>
          <div class="kpi-lbl">Raw SNR (Prose)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-val">{vault_metrics['avg_optimized_snr']:.3f}</div>
          <div class="kpi-lbl">Optimized SNR (GEO)</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-val" style="color: #10B981;">+{vault_metrics['snr_gain_pct']}%</div>
          <div class="kpi-lbl">Signal Density Gain</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Interactive Filter & Case Selector
    raw_cases = current_node.get_all_raw_cases()
    if raw_cases:
        with st.container():
            st.markdown('<div class="geo-card">', unsafe_allow_html=True)
            
            c_f_dept, c_f_search = st.columns([1, 2])
            with c_f_dept:
                dept_filter_vault = st.selectbox("Filter Department:", ["All"] + list(set(c.get("department", "General") for c in raw_cases)))
            with c_f_search:
                search_query_vault = st.text_input("Search Cases by Keyword (ID, Diagnosis, or Drug):", placeholder="e.g., Angina, Nephropathy, Metoprolol...")

            # Apply interactive filtering
            filtered_cases = raw_cases
            if dept_filter_vault != "All":
                filtered_cases = [c for c in filtered_cases if c.get("department") == dept_filter_vault]
            if search_query_vault.strip():
                sq = search_query_vault.strip().lower()
                filtered_cases = [c for c in filtered_cases if sq in c.get("patient_id", "").lower() or sq in c.get("primary_diagnosis", "").lower() or sq in str(c.get("treatment_outcome", {})).lower()]

            if filtered_cases:
                case_map = {c["patient_id"]: f"{c['patient_id']} · {c.get('department','')} · {c.get('primary_diagnosis','')}" for c in filtered_cases}
                selected_cid = st.selectbox("Select Case to Inspect:", options=list(case_map.keys()), format_func=lambda x: case_map[x])
                target_case = next((c for c in filtered_cases if c["patient_id"] == selected_cid), filtered_cases[0])
            else:
                st.info("No cases match the selected filter.")
                target_case = None

            st.markdown('</div>', unsafe_allow_html=True)

        if target_case:
            opt_audit = optimize_clinical_record(target_case)

            # Side-by-Side Comparator
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                st.markdown("""
                <div class="geo-card" style="border-top: 3px solid #EF4444; height: 100%;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:700; color:#DC2626; font-size:12px;">RAW DOCTOR NOTES (UNOPTIMIZED)</span>
                    <span style="background:#FEE2E2; color:#DC2626; font-size:10px; padding:2px 6px; border-radius:10px; font-weight:600;">Low Citable Signal</span>
                  </div>
                """, unsafe_allow_html=True)
                st.markdown(f"_{target_case.get('treatment_outcome', {}).get('clinical_notes_raw', 'No raw text')}_")
                st.markdown("---")
                st.markdown(f"**SNR**: `{opt_audit['raw_snr']:.3f}`")
                st.markdown(f"**Citation Vulnerabilities**:\n" + "\n".join([f"- ⚠️ `{f}`" for f in opt_audit["diagnosed_flaws"]]))
                st.markdown("</div>", unsafe_allow_html=True)

            with cmp_col2:
                st.markdown("""
                <div class="geo-card" style="border-top: 3px solid #10B981; height: 100%;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:700; color:#059669; font-size:12px;">AGENTGEO CITABLE EVIDENCE CHUNK</span>
                    <span style="background:#DCFCE7; color:#166534; font-size:10px; padding:2px 6px; border-radius:10px; font-weight:600;">High Citable Signal</span>
                  </div>
                """, unsafe_allow_html=True)
                st.markdown(opt_audit["optimized_evidence_chunk"], unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"**Optimized SNR**: `{opt_audit['optimized_snr']:.3f}` (`+{opt_audit['improvement_pct']}% boost`)")
                st.markdown(f"**Repairs Applied**:\n" + "\n".join([f"- ✓ `{r}`" for r in opt_audit["repairs_applied"]]))
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No cases currently indexed in this node.")
