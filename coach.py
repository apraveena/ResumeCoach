"""AI coaching service for comparing resumes with job descriptions.

The module loads a versioned system prompt, calls the Groq chat API, and
converts the model's JSON response into a Python dictionary for the UI and
evaluation runner.
"""

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
MAX_ANALYSIS_CONTEXT_CHARS = 14000

CHAT_SYSTEM_PROMPT = """You are a practical resume and interview coach.
Answer the user's question using only the supplied resume, job description, and
analysis. Do not invent experience, metrics, employers, certifications, or
skills. If the information is not available, say that clearly. Keep answers
concise and actionable for a job seeker.
"""

def load_system_prompt(version: str = "v2") -> str:
    if version not in ("v1", "v2"):
        raise ValueError(f"Unknown prompt version: {version}")
    path = PROMPTS_DIR / f"prompt_{version}.txt"
    return path.read_text(encoding="utf-8")


def _is_generation_error(error: Exception) -> bool:
    """Return whether the provider rejected an incomplete model generation."""
    message = str(error).lower()
    return any(marker in message for marker in (
        "json_validate_failed",
        "failed_generation",
        "max completion tokens",
    ))


def _is_request_size_error(error: Exception) -> bool:
    """Return whether Groq rejected the request for exceeding its token limit."""
    message = str(error).lower()
    return any(marker in message for marker in (
        "rate_limit_exceeded",
        "request too large",
        "tokens per minute",
        "requested tokens",
    ))


def _limit_text(text: str, max_chars: int) -> str:
    """Keep both the beginning and end of long user input for analysis."""
    if len(text) <= max_chars:
        return text
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + "\n\n[Middle of this input was shortened to fit the model limit.]\n\n"
        + text[-tail_chars:]
    )


def _parse_json_response(raw_content: str) -> dict:
    """Parse JSON even when a model wraps it in a markdown code fence."""
    clean_json = (raw_content or "").strip()
    clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean_json)
    if not clean_json:
        raise ValueError("The AI returned an empty response instead of a coaching analysis.")

    try:
        return json.loads(clean_json)
    except json.JSONDecodeError:
        # A fallback response may contain a short sentence before the JSON.
        start = clean_json.find("{")
        end = clean_json.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(clean_json[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError("The AI returned incomplete or invalid JSON. Please try again.")

def analyze_resume(resume_text: str, job_description: str, prompt_version: str = "v2") -> dict:
    """
    Analyzes resume text against a target job description.
    Returns:
          dict: structured fit analysis, strengths, coaching actions, interview
              preparation, and grounded STAR rewrites.
    """
    client = get_groq_client()

     # 2. Build the System Prompt (The Persona and Guidelines)
    system_prompt = load_system_prompt(prompt_version)

    # Keep the request below the provider's 8,000-token service limit.
    resume_budget = int(MAX_ANALYSIS_CONTEXT_CHARS * 0.6)
    job_budget = MAX_ANALYSIS_CONTEXT_CHARS - resume_budget

    # 3. Build the User Input
    user_prompt = f"""
        --- TARGET JOB DESCRIPTION ---
        {_limit_text(job_description, job_budget)}

        --- CANDIDATE RESUME ---
        {_limit_text(resume_text, resume_budget)}
        """
        # 4. Call Groq API with JSON mode enabled
    # Model for resume analysis; keep constant across prompt comparisons
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request_options = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": 0.2,
        "max_completion_tokens": 2400,
        "reasoning_effort": "low",
    }

    try:
        response = client.chat.completions.create(
            **request_options,
            response_format={"type": "json_object"},
        )
        return _parse_json_response(response.choices[0].message.content)
    except Exception as first_error:
        if _is_request_size_error(first_error):
            raise ValueError(
                "The resume and job description are too large for the AI service. "
                "The app already shortens inputs, so please try a shorter job description."
            ) from first_error

        # JSON mode can return an empty response for reasoning models. Retry
        # with a smaller model, without provider-side validation, and parse the
        # result locally.
        try:
            fallback_options = {
                **request_options,
                "model": "openai/gpt-oss-20b",
                "max_completion_tokens": 1600,
            }
            fallback_options.pop("reasoning_effort", None)
            response = client.chat.completions.create(**fallback_options)
            return _parse_json_response(response.choices[0].message.content)
        except Exception as retry_error:
            raise ValueError(
                "The AI could not finish a valid coaching response. "
                "Please try again with a shorter resume or job description."
            ) from retry_error


def chat_with_coach(
    resume_text: str,
    job_description: str,
    analysis: dict,
    messages: list[dict],
) -> str:
    """Answer a follow-up question using the saved resume analysis."""
    client = get_groq_client()
    context = f"""
--- CANDIDATE RESUME ---
{resume_text}

--- TARGET JOB DESCRIPTION ---
{job_description}

--- COMPLETED COACHING ANALYSIS ---
{json.dumps(analysis, ensure_ascii=False)}
"""
    chat_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT + context}]
    chat_messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in messages[-10:]
        if message.get("role") in ("user", "assistant")
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=chat_messages,
        temperature=0.3,
        max_completion_tokens=1200,
    )
    return response.choices[0].message.content.strip()
