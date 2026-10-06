"""
agents/skill_gap.py
Compares resume skills against JD skills using semantic (FAISS) matching,
then asks the LLM to write a short human-readable gap summary.
"""

from config import get_llm
from vectorstore.faiss_utils import semantic_skill_match


GAP_SUMMARY_PROMPT = """You are a career coach. A candidate is applying for the role: {role_title}

Matched skills (candidate already has): {matched}
Missing skills (candidate should learn or highlight if transferable experience exists): {missing}

Write a short, encouraging 3-4 sentence summary for the candidate:
1. Acknowledge their strong overlap with the role.
2. Name the 1-3 most important missing skills to address.
3. Suggest how they could bridge the gap (a project, certification, or reframing existing experience).

Return plain text only, no markdown headers.
"""


def skill_gap_node(state: dict) -> dict:
    """
    LangGraph node. Reads resume_skills + jd_required_skills, writes:
    matched_skills, missing_skills, match_percentage, gap_summary
    """
    errors = state.get("errors", [])

    try:
        resume_skills = state.get("resume_skills", [])
        jd_required = state.get("jd_required_skills", [])
        jd_nice_to_have = state.get("jd_nice_to_have_skills", [])

        all_jd_skills = jd_required + jd_nice_to_have
        matched, missing = semantic_skill_match(resume_skills, all_jd_skills)

        match_percentage = round((len(matched) / len(all_jd_skills)) * 100, 1) if all_jd_skills else 0.0

        llm = get_llm(temperature=0.4)
        prompt = GAP_SUMMARY_PROMPT.format(
            role_title=state.get("jd_role_title", "this role"),
            matched=", ".join(matched) if matched else "none identified",
            missing=", ".join(missing) if missing else "none",
        )
        response = llm.invoke(prompt)

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "match_percentage": match_percentage,
            "gap_summary": response.content.strip(),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"skill_gap_node failed: {str(e)}")
        return {
            "matched_skills": [],
            "missing_skills": [],
            "match_percentage": 0.0,
            "gap_summary": "",
            "errors": errors,
        }