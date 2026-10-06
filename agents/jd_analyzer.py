"""
agents/jd_analyzer.py
Extracts structured requirements from a pasted job description.
"""

import json
import re
from config import get_llm


JD_PROMPT = """You are a job description parsing assistant. Extract structured information from the job description below.

Return ONLY valid JSON, no preamble, no markdown code fences, no explanation. Use exactly this schema:

{{
  "role_title": "exact job title as stated",
  "required_skills": ["skill1", "skill2", ...],
  "nice_to_have_skills": ["skill1", "skill2", ...],
  "key_responsibilities": ["responsibility 1", "responsibility 2", ...]
}}

Rules:
- "required_skills" = must-have technical skills/tools explicitly stated as required/must-have.
- "nice_to_have_skills" = skills mentioned as preferred/bonus/plus.
- If the JD doesn't clearly separate required vs nice-to-have, use your best judgment based on language ("must have" vs "preferred").
- "key_responsibilities" should be 3-6 concise bullets.
- Do not invent information not present in the job description text.

Job description text:
---
{jd_text}
---
"""


def _extract_json(raw_text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def jd_analyzer_node(state: dict) -> dict:
    """
    LangGraph node. Reads state['jd_text'], writes:
    jd_role_title, jd_required_skills, jd_nice_to_have_skills, jd_key_responsibilities
    """
    errors = state.get("errors", [])

    try:
        llm = get_llm(temperature=0.1)
        prompt = JD_PROMPT.format(jd_text=state["jd_text"])
        response = llm.invoke(prompt)
        parsed = _extract_json(response.content)

        return {
            "jd_role_title": parsed.get("role_title", ""),
            "jd_required_skills": parsed.get("required_skills", []),
            "jd_nice_to_have_skills": parsed.get("nice_to_have_skills", []),
            "jd_key_responsibilities": parsed.get("key_responsibilities", []),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"jd_analyzer_node failed: {str(e)}")
        return {
            "jd_role_title": "",
            "jd_required_skills": [],
            "jd_nice_to_have_skills": [],
            "jd_key_responsibilities": [],
            "errors": errors,
        }