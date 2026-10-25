"""Streamlit interface for the ResumeCoach AI application.

This module collects a resume and target job description, sends the extracted
content to the coaching layer, displays the analysis, and offers downloadable
tailored documents.
"""

import streamlit as st
import requests
from uuid import uuid4
from bs4 import BeautifulSoup
import parser
import coach
import chat_coach
import doc_generator

# Page Configuration
st.set_page_config(
    page_title="Resume Coach AI",
    page_icon="🎯",
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172a3a;
        --muted: #607080;
        --line: #dce5ea;
        --paper: #ffffff;
        --wash: #f4f8f8;
        --teal: #087f8c;
        --coral: #d76552;
    }

    .stApp {
        background: var(--wash);
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1500px;
        padding: 1rem 2.5rem 4rem;
    }

    [data-testid="stHeader"] {
        height: 0;
        background: transparent;
        overflow: visible;
    }

    [data-testid="stHeader"] [data-testid="stToolbar"] {
        position: absolute;
        top: 0.35rem;
        right: 0.75rem;
        z-index: 1000;
    }

    [data-testid="stSidebar"] {
        background: var(--ink);
        border-right: 1px solid #294254;
    }

    [data-testid="stSidebar"] * {
        color: #eef6f6;
    }

    [data-testid="stSidebar"] [data-testid="stChatMessage"] {
        background: #223d4d;
        border: 1px solid #345365;
    }

    [data-testid="stSidebar"] [data-testid="stChatInput"] {
        background: #ffffff;
    }

    [data-testid="stSidebar"] [data-testid="stChatInput"] textarea,
    [data-testid="stSidebar"] [data-testid="stChatInput"] input {
        background: #ffffff !important;
        color: #172a3a !important;
        caret-color: #172a3a !important;
    }

    [data-testid="stSidebar"] [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stSidebar"] [data-testid="stChatInput"] input::placeholder {
        color: #607080 !important;
        opacity: 1;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--ink);
        letter-spacing: 0;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }

    [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(23, 42, 58, 0.05);
    }

    [data-testid="stExpander"] {
        background: var(--paper);
        border: 1px solid var(--line);
    }

    .section-kicker {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid4())

def fetch_jd_from_url(url: str) -> str:
    """Fetches and extracts clean text from a job description webpage."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove navigation, script, and footer clutter
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        text = soup.get_text(separator="\n")
        # Clean extra blank lines
        clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(clean_lines)
    except Exception as e:
        st.error(f"Failed to fetch job description from URL: {e}")
        return ""

# Header
st.markdown('<div class="section-kicker">AI career coaching workspace</div>', unsafe_allow_html=True)
st.title("Resume Coach")
st.markdown(
    '<div class="hero-copy">Turn a resume and a target role into clear, evidence-based application advice.</div>',
    unsafe_allow_html=True,
)
st.divider()

# Layout: 2 Columns
col_input, col_output = st.columns([1, 1], gap="large")


with col_input:
    st.subheader("📋 Step 1: Candidate Inputs")

    # 1. Resume File Uploader
    uploaded_file = st.file_uploader(
        "Upload Your Resume (.pdf or .docx)",
        type=["pdf", "docx"],
        help="Upload your current resume in PDF or Word format."
    )

    # 2. Target Job Description Input (Tabs for Text vs URL)
    st.markdown("### Target Job Description")
    jd_tab1, jd_tab2 = st.tabs(["📝 Paste Text", "🔗 Paste Link (URL)"])

    jd_text = ""
    with jd_tab1:
        jd_input_text = st.text_area(
            "Paste the complete job description here:",
            height=250,
            placeholder="Paste role responsibilities, requirements, qualifications..."
        )
        if jd_input_text:
            jd_text = jd_input_text

    with jd_tab2:
        jd_url = st.text_input(
            "Paste Job Posting URL:",
            placeholder="https://company.com/careers/job-role"
        )
        if jd_url:
            with st.spinner("Fetching job description from URL..."):
                fetched_text = fetch_jd_from_url(jd_url)
                if fetched_text:
                    st.success("Successfully fetched job description!")
                    jd_text = fetched_text

    analyze_button = st.button("🚀 Analyze & Optimize Resume", type="primary", use_container_width=True)


with st.sidebar:
    st.markdown("## 💬 Coach Chat")
    st.caption("Your conversation stays connected to the active resume analysis.")

    sidebar_results = st.session_state.get("analysis")
    if not sidebar_results:
        st.info("Run an analysis to unlock personalized follow-up coaching.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about fit, gaps, rewrites, or interviews")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if not sidebar_results:
            st.warning("Analyze a resume first so I can give personalized advice.")
        else:
            try:
                with st.spinner("Thinking..."):
                    answer = chat_coach.ask_coach(
                        st.session_state.chat_session_id,
                        st.session_state.get("resume_text", ""),
                        st.session_state.get("job_description", ""),
                        sidebar_results,
                        question,
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as ex:
                error_text = str(ex).lower()
                if any(marker in error_text for marker in (
                    "rate_limit_exceeded", "request too large", "tokens per minute",
                    "max completion tokens",
                )):
                    st.error("The chat request was too large. Try a shorter question.")
                else:
                    st.error(f"Unable to answer the chat question: {ex}")


with col_output:
    st.subheader("📊 Step 2: Coaching Insights")

    if analyze_button:
        # Input Validation
        if not uploaded_file:
            st.warning("⚠️ Please upload your resume first.")
        elif not jd_text.strip():
            st.warning("⚠️ Please provide a target job description (via text or URL).")
        else:
            with st.spinner("Analyzing resume against job requirements..."):
                try:
                    # 1. Parse Resume
                    resume_text = parser.parse_resume(uploaded_file)

                    # 2. Analyze with AI Coach
                    results = coach.analyze_resume(resume_text, jd_text)
                    st.session_state.analysis = results
                    st.session_state.resume_text = resume_text
                    st.session_state.job_description = jd_text
                    st.session_state.chat_messages = []
                    st.session_state.chat_session_id = str(uuid4())
                except ValueError as ve:
                    error_text = str(ve)
                    if "AI" in error_text or "JSON" in error_text:
                        st.error(f"⚠️ {error_text}")
                    else:
                        st.error(f"⚠️ Document error: {error_text}")
                except Exception as ex:
                    error_text = str(ex)
                    lower_error = error_text.lower()
                    if any(marker in lower_error for marker in (
                        "rate_limit_exceeded",
                        "request too large",
                        "tokens per minute",
                        "requested tokens",
                    )):
                        st.error(
                            "❌ The AI request was too large for the current service limit. "
                            "The app now shortens long inputs; please try again with a shorter job description if needed."
                        )
                    elif any(marker in lower_error for marker in (
                        "max completion tokens",
                        "failed to generate json",
                        "json_validate_failed",
                        "failed to validate json",
                        "failed_generation",
                    )):
                        st.error(
                            "❌ The AI could not finish a valid response. "
                            "Please try again, or shorten the resume or job description."
                        )
                    else:
                        st.error(f"❌ An unexpected error occurred: {ex}")

    results = st.session_state.get("analysis")
    if results:
        resume_text = st.session_state.get("resume_text", "")

        # 3. Display Match Score Metric
        score = results.get("match_score", 0)
        st.metric(label="🎯 ATS Match Score", value=f"{score}%")
        st.progress(score / 100)

        # 4. Executive Summary
        st.markdown("#### 💡 Executive Summary")
        st.info(results.get("summary", "Analysis complete."))

        # 5. Skills and coaching advice
        col_match, col_miss = st.columns(2)
        with col_match:
            st.markdown("##### ✅ Matching Skills")
            for skill in results.get("matching_skills", []):
                st.markdown(f"- **{skill}**")

        with col_miss:
            st.markdown("##### ⚠️ Missing Critical Skills")
            for skill in results.get("missing_skills", []):
                st.markdown(f"- 🔴 {skill}")

        col_strengths, col_actions = st.columns(2)
        with col_strengths:
            st.markdown("##### 🌟 Strengths to Emphasize")
            for strength in results.get("strengths_to_emphasize", []):
                st.markdown(f"- {strength}")

        with col_actions:
            st.markdown("##### 📝 Recommended Actions")
            for action in results.get("coaching_actions", []):
                st.markdown(f"- {action}")

        # 6. Interview preparation
        interview = results.get("interview_preparation", {})
        with st.expander("🎤 Interview Preparation", expanded=True):
            interview_columns = st.columns(3)
            for column, title, key in zip(
                interview_columns,
                ("Likely Questions", "Talking Points", "Questions to Ask"),
                ("likely_questions", "talking_points", "questions_to_ask"),
            ):
                with column:
                    st.markdown(f"**{title}**")
                    for item in interview.get(key, []):
                        st.markdown(f"- {item}")

        # 7. STAR Rewrites Section
        st.divider()
        st.markdown("#### 🌟 Bullet Point Optimizer (STAR Framework)")
        st.caption(results.get("rewrites_summary", "No rewrite summary was returned."))

        star_rewrites = results.get("star_rewrites", [])
        if star_rewrites:
            for idx, item in enumerate(star_rewrites, 1):
                with st.expander(f"📌 Optimization #{idx}", expanded=True):
                    st.markdown(f"**❌ Original:**\n> *{item.get('original')}*")
                    st.markdown(f"**✅ STAR-Optimized:**\n> **{item.get('rewritten')}**")
                    st.caption(f"💡 **Why:** {item.get('reason')}")
                    if item.get("grounded") is False:
                        st.warning(
                            "This rewrite may contain unsupported claims: "
                            + ", ".join(item.get("unsupported_claims", []))
                        )
                    else:
                        st.success("Grounding check passed: no unsupported claims were reported.")
        else:
            st.write("No weak bullet points detected. Good job!")

        # 8. Dual Download Options
        st.divider()
        st.markdown("#### 📥 Download Tailored Deliverables")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            if uploaded_file:
                docx_buffer = doc_generator.generate_tailored_resume(
                    uploaded_file, resume_text, star_rewrites
                )
                st.download_button(
                    label="📄 Tailored Resume (.docx)",
                    data=docx_buffer,
                    file_name="Tailored_Resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
        with col_dl2:
            txt_report = doc_generator.generate_text_coaching_report(results)
            st.download_button(
                label="📋 Coaching Report (.txt)",
                data=txt_report,
                file_name="Resume_Coaching_Report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.caption("Your personalized coach is always available in the sidebar.")
    else:
        st.info("👈 Upload your resume and paste a job description on the left, then click **Analyze & Optimize Resume**.")