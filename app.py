"""Streamlit interface for the ResumeCoach AI application.

This module collects a resume and target job description, sends the extracted
content to the coaching layer, displays the analysis, and offers downloadable
tailored documents.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import parser
import coach
import doc_generator

# Page Configuration
st.set_page_config(
    page_title="Resume Coach AI",
    page_icon="🎯",
    layout="wide"
)

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
st.title("🎯 Resume Coach — AI Job Application Assistant")
st.markdown("Tailor your resume to any target role with instant ATS gap analysis and metric-driven STAR rewrites.")
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

                    # 3. Display Match Score Metric
                    score = results.get("match_score", 0)
                    st.metric(label="🎯 ATS Match Score", value=f"{score}%")
                    st.progress(score / 100)

                    # 4. Executive Summary
                    st.markdown("#### 💡 Executive Summary")
                    st.info(results.get("summary", "Analysis complete."))

                    # 5. Skills Gap Breakdown
                    col_match, col_miss = st.columns(2)
                    with col_match:
                        st.markdown("##### ✅ Matching Strengths")
                        for skill in results.get("matching_skills", []):
                            st.markdown(f"- **{skill}**")

                    with col_miss:
                        st.markdown("##### ⚠️ Missing Critical Skills")
                        for skill in results.get("missing_skills", []):
                            st.markdown(f"- 🔴 {skill}")

                    # 6. STAR Rewrites Section
                    st.divider()
                    st.markdown("#### 🌟 Bullet Point Optimizer (STAR Framework)")
                    
                    rewrites_summary = results.get(
                        "rewrites_summary", 
                        f"Identified weak bullet points. Top items prioritized below:"
                    )
                    st.caption(f"ℹ️ {rewrites_summary}")

                    star_rewrites = results.get("star_rewrites", [])
                    if star_rewrites:
                        for idx, item in enumerate(star_rewrites, 1):
                            with st.expander(f"📌 Optimization #{idx}", expanded=True):
                                st.markdown(f"**❌ Original:**\n> *{item.get('original')}*")
                                st.markdown(f"**✅ STAR-Optimized:**\n> **{item.get('rewritten')}**")
                                st.caption(f"💡 **Why:** {item.get('reason')}")
                    else:
                        st.write("No weak bullet points detected. Good job!")

                    # 7. Dual Download Options
                    st.divider()
                    st.markdown("#### 📥 Download Tailored Deliverables")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Tailored Word Resume
                        docx_buffer = doc_generator.generate_tailored_resume(
                            uploaded_file, resume_text, star_rewrites
                        )
                        st.download_button(
                            label="📄 Tailored Resume (.docx)",
                            data=docx_buffer,
                            file_name="Tailored_Resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                        
                    with col_dl2:
                        # Plain Text Coaching Summary Report
                        txt_report = doc_generator.generate_text_coaching_report(results)
                        st.download_button(
                            label="📋 Coaching Report (.txt)",
                            data=txt_report,
                            file_name="Resume_Coaching_Report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                except ValueError as ve:
                    st.error(f"⚠️ Document Error: {ve}")
                except Exception as ex:
                    st.error(f"❌ An unexpected error occurred: {ex}")
    else:
        st.info("👈 Upload your resume and paste a job description on the left, then click **Analyze & Optimize Resume**.")