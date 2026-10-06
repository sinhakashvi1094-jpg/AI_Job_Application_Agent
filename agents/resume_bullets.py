"""
agents/resume_bullets.py
Generates ATS-optimized resume bullets tailored to a specific JD,
using the candidate's actual experience as the factual base.
"""

from config import get_llm
import json
import re


BULLETS_PROMPT = """You are an ATS resume optimization expert. Rewrite the candidate's experience bullets to be tailored for this specific job, while staying factually accurate to what they actually did.

Candidate's original experience bullets: {experience}
Candidate's matched skills relevant to this role: {matched_skills}
Role title: {role_title}
Key responsibilities to align language with: {responsibilities}

Rules:
- Return ONLY a valid JSON array of strings, no markdown fences, no preamble.
- Each bullet must start with a strong action verb (Built, Architected, Optimized, Deployed, Implemented, Reduced, etc.) — never "Responsible for" or "Worked on".
- Include a quantifiable metric ONLY if implied by the original text; otherwise use a placeholder in the format [ADD METRIC] instead of inventing a fake number.
- Weave in exact keyword phrasing from "matched_skills" and "responsibilities" naturally — this is for ATS keyword matching.
- Do NOT fabricate tools, technologies, or achievements not present in the original experience.
- Produce 3-5 bullets, each under 25 words.

Example format: ["Built a RAG-based document retrieval pipeline using LangChain and FAISS, reducing manual search time by [ADD METRIC]", "..."]

Return the JSON array now:
"""


def _extract_json_array(raw_text: str) -> list:
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def resume_bullets_node(state: dict) -> dict:
    """
    LangGraph node. Reads resume_experience, matched_skills, jd_role_title,
    jd_key_responsibilities. Writes: suggested_bullets
    """
    errors = state.get("errors", [])

    try:
        llm = get_llm(temperature=0.4)
        prompt = BULLETS_PROMPT.format(
            experience="; ".join(state.get("resume_experience", [])),
            matched_skills=", ".join(state.get("matched_skills", [])),
            role_title=state.get("jd_role_title", "the role"),
            responsibilities="; ".join(state.get("jd_key_responsibilities", [])),
        )
        response = llm.invoke(prompt)
        bullets = _extract_json_array(response.content)

        return {
            "suggested_bullets": bullets,
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"resume_bullets_node failed: {str(e)}")
        return {
            "suggested_bullets": [],
            "errors": errors,
        }