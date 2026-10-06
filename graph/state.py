"""
graph/state.py
Shared state schema passed between all agent nodes in the LangGraph workflow.
"""

import operator
from typing import TypedDict, List, Optional, Annotated


class AgentState(TypedDict):
    # ---- Inputs (set by Streamlit UI) ----
    resume_text: str
    jd_text: str

    # ---- Resume Parser Agent output ----
    resume_skills: List[str]
    resume_experience: List[str]
    resume_summary: str

    # ---- JD Analyzer Agent output ----
    jd_required_skills: List[str]
    jd_nice_to_have_skills: List[str]
    jd_role_title: str
    jd_key_responsibilities: List[str]

    # ---- Skill Gap Agent output ----
    matched_skills: List[str]
    missing_skills: List[str]
    match_percentage: float
    gap_summary: str

    # ---- Cover Letter Agent output ----
    cover_letter: str

    # ---- Resume Bullet Agent output ----
    suggested_bullets: List[str]

    # ---- Q&A Agent (optional) ----
    qa_question: Optional[str]
    qa_answer: Optional[str]

    # ---- Meta ----
    # Annotated with operator.add so parallel nodes (resume_parser + jd_analyzer)
    # can BOTH write to "errors" in the same step — LangGraph concatenates the
    # lists instead of raising InvalidUpdateError on a conflicting write.
    errors: Annotated[List[str], operator.add]