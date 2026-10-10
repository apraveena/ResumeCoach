import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

# 1. Load the secret API key from the .env file
load_dotenv()

def get_groq_client() -> Groq:
    """Initializes and returns the Groq client securely."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
    return Groq(api_key=api_key)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def load_system_prompt(version: str = "v2") -> str:
    if version not in ("v1", "v2"):
        raise ValueError(f"Unknown prompt version: {version}")
    path = PROMPTS_DIR / f"prompt_{version}.txt"
    return path.read_text(encoding="utf-8")

def analyze_resume(resume_text: str, job_description: str, prompt_version: str = "v2") -> dict:
    """
    Analyzes resume text against a target job description.
    Returns:
        dict: match_score, summary, matching_skills, missing_skills,
              total_weak_bullets_found, rewrites_summary, star_rewrites.
    """
    client = get_groq_client()

     # 2. Build the System Prompt (The Persona and Guidelines)
    system_prompt = load_system_prompt(prompt_version)

# 3. Build the User Input
    user_prompt = f"""
        --- TARGET JOB DESCRIPTION ---
        {job_description}

        --- CANDIDATE RESUME ---
        {resume_text}
        """
        # 4. Call Groq API with JSON mode enabled
    # Model for resume analysis; keep constant across prompt comparisons
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
