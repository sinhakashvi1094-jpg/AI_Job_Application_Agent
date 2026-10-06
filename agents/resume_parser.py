"""
agents/resume_parser.py
Extracts structured skills, experience, and summary from raw resume text.
"""

import json
import re
from config import get_llm


PARSE_PROMPT = """You are a resume parsing assistant. Extract structured information from the resume text below.

Return ONLY valid JSON, no preamble, no markdown code fences, no explanation. Use exactly this schema:

{{
  "skills": ["skill1", "skill2", ...],
  "experience": ["short bullet describing role/achievement", ...],
  "summary": "2-3 sentence professional summary of this candidate"
}}

Rules:
- "skills" should include technical skills, tools, frameworks, languages mentioned anywhere in the resume.
- "experience" should be 3-6 concise bullets capturing key roles/projects/achievements.
- Do not invent information not present in the resume text.

Resume text:
---
{resume_text}
---
"""


def _extract_json(raw_text: str) -> dict:
    """Strips markdown fences if present and parses JSON safely."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to salvage by grabbing the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def resume_parser_node(state: dict) -> dict:
    """
    LangGraph node. Reads state['resume_text'], writes:
    resume_skills, resume_experience, resume_summary
    """
    errors = state.get("errors", [])

    try:
        llm = get_llm(temperature=0.1)  # low temp for extraction accuracy
        prompt = PARSE_PROMPT.format(resume_text=state["resume_text"])
        response = llm.invoke(prompt)
        parsed = _extract_json(response.content)

        return {
            "resume_skills": parsed.get("skills", []),
            "resume_experience": parsed.get("experience", []),
            "resume_summary": parsed.get("summary", ""),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"resume_parser_node failed: {str(e)}")
        return {
            "resume_skills": [],
            "resume_experience": [],
            "resume_summary": "",
            "errors": errors,
        }