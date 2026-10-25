"""LangChain conversation chain for the ResumeCoach follow-up chat.

Streamlit keeps the visible page state. This module keeps the AI message
history for each Streamlit session and sends that history through a simple
LangChain prompt chain backed by the Groq chat model.
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()

_histories: dict[str, InMemoryChatMessageHistory] = {}
_coach_chain: RunnableWithMessageHistory | None = None
MAX_CHAT_HISTORY_MESSAGES = 4


def _limit_text(text: str, max_chars: int) -> str:
    """Keep chat context within the model's request limit."""
    if len(text) <= max_chars:
        return text
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + "\n\n[Middle of this context was shortened.]\n\n"
        + text[-tail_chars:]
    )


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the conversation history belonging to one browser session."""
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]


def _get_coach_chain() -> RunnableWithMessageHistory:
    """Build the LangChain prompt and model once per application process."""
    global _coach_chain
    if _coach_chain is not None:
        return _coach_chain

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a practical resume and interview coach.
Use only the supplied resume, job description, and completed analysis.
Do not invent experience, metrics, employers, certifications, or skills.
If the information is not available, say so clearly.
Keep answers concise and actionable.

--- CANDIDATE RESUME ---
{resume_text}

--- TARGET JOB DESCRIPTION ---
{job_description}

--- COMPLETED COACHING ANALYSIS ---
{analysis}
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=800,
        api_key=api_key,
    )
    chain = prompt | model
    _coach_chain = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return _coach_chain


def ask_coach(
    session_id: str,
    resume_text: str,
    job_description: str,
    analysis: dict,
    question: str,
) -> str:
    """Ask a follow-up question and save the exchange in LangChain memory."""
    history = _get_session_history(session_id)
    if len(history.messages) > MAX_CHAT_HISTORY_MESSAGES:
        history.messages = history.messages[-MAX_CHAT_HISTORY_MESSAGES:]
    for message in history.messages:
        if isinstance(message.content, str):
            message.content = _limit_text(message.content, 1000)

    response = _get_coach_chain().invoke(
        {
            "resume_text": _limit_text(resume_text, 4500),
            "job_description": _limit_text(job_description, 3000),
            "analysis": _limit_text(json.dumps(analysis, ensure_ascii=False), 3500),
            "question": question,
        },
        config={"configurable": {"session_id": session_id}},
    )
    return response.content.strip()