import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

# 1. Load the secret API key from the .env file
load_dotenv()

def get_groq_client() -> Groq:
    """Initializes and returns the Groq client securely."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
    return Groq(api_key=api_key)

def analyze_resume(resume_text: str, job_description: str) -> dict:
    """
    Analyzes resume text against a target job description.
    Returns:
        dict: match_score, summary, matching_skills, missing_skills,
              total_weak_bullets_found, rewrites_summary, star_rewrites.
    """
    client = get_groq_client()

     # 2. Build the System Prompt (The Persona and Guidelines)
    system_prompt = """
        You are an Executive Career Coach and an Applicant Tracking System (ATS) Expert.
        Your task is to evaluate the candidate's resume against the target Job Description objectively and constructively.

        Follow these strict rules:
        1. MATCH SCORE: Provide an integer match score between 0 and 100 based on core skills, requirements, and domain alignment.
        2. SKILL GAPS:
        - matching_skills: List specific hard/soft skills present in both the resume and JD.
        - missing_skills: List critical skills, tools, or certifications required by the JD but absent/weak in the resume.
        3. BULLET POINT OPTIMIZATION (STAR METHOD):
        - Scan the entire resume for passive or weak bullet points that lack quantifiable impact.
        - Count the total number of weak bullet points identified across the resume (total_weak_bullets_found).
        - Select and rewrite the top 3 most critical weak bullets using the STAR framework (Situation, Task, Action, Result).
        - Each rewrite must start with a strong action verb and include realistic quantifiable metrics/KPIs.
        - Do not fabricate false companies or degrees; elevate the existing achievement.
        - In 'rewrites_summary', explain how many total weak bullets were found and that the top 3 were prioritized.

        OUTPUT FORMAT:
        You MUST output a valid JSON object matching this schema:
        {
        "match_score": 85,
        "summary": "Executive summary of the candidate fit...",
        "matching_skills": ["Skill A", "Skill B"],
        "missing_skills": ["Skill C", "Skill D"],
        "total_weak_bullets_found": 5,
        "rewrites_summary": "Identified 5 weak bullets across the resume. Prioritized rewriting the top 3 most critical ones below.",
        "star_rewrites": [
            {
            "original": "Exact text from resume",
            "rewritten": "STAR-optimized bullet with action verb and metrics",
            "reason": "Why this change improves ATS score and hiring manager impact"
            }
        ]
        }
        """

# 3. Build the User Input
    user_prompt = f"""
        --- TARGET JOB DESCRIPTION ---
        {job_description}

        --- CANDIDATE RESUME ---
        {resume_text}
        """
        # 4. Call Groq API with JSON mode enabled
    # We use llama-3.3-70b-versatile: fast, highly intelligent, and free on Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,  # Low temperature for precise, consistent evaluation
    )

    raw_content = response.choices[0].message.content

    # 5. Defensive JSON Parsing (strips any markdown backticks if present)
    clean_json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())
    
    try:
        parsed_data = json.loads(clean_json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse AI response as JSON: {e}\nRaw output: {raw_content}")