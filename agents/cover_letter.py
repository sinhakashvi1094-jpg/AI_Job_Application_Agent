"""
agents/cover_letter.py
Generates a tailored cover letter using resume summary, matched skills,
and JD responsibilities. Explicitly avoids generic filler language.
"""

from config import get_llm


COVER_LETTER_PROMPT = """You are an expert career writer. Write a concise, specific, non-generic cover letter for this candidate applying to this role.

Candidate summary: {resume_summary}
Candidate's matched relevant skills: {matched_skills}
Role title: {role_title}
Key responsibilities of the role: {responsibilities}
Known skill gaps (do NOT mention these as weaknesses; instead, if relevant, frame nearby matched skills as transferable): {missing_skills}

Rules:
- 3-4 paragraphs, under 300 words total.
- Open with a specific hook related to the role/company mission, not "I am writing to apply for...".
- Reference 2-3 concrete matched skills tied to specific responsibilities, not a generic list.
- Do NOT use generic phrases like "team player", "hard worker", "passionate about technology", "fast learner".
- Do NOT mention missing skills directly or apologize for gaps.
- Close with a confident, specific call to action (not "I look forward to hearing from you").
- Return plain text only, no markdown, no placeholders like [Company Name] — instead phrase generically enough to not need a company name (e.g. "your team" instead of "[Company]").

Write the cover letter now:
"""


def cover_letter_node(state: dict) -> dict:
    """
    LangGraph node. Reads resume_summary, matched_skills, jd_role_title,
    jd_key_responsibilities, missing_skills. Writes: cover_letter
    """
    errors = state.get("errors", [])

    try:
        llm = get_llm(temperature=0.6)  # slightly higher temp for natural writing tone
        prompt = COVER_LETTER_PROMPT.format(
            resume_summary=state.get("resume_summary", ""),
            matched_skills=", ".join(state.get("matched_skills", [])),
            role_title=state.get("jd_role_title", "the role"),
            responsibilities="; ".join(state.get("jd_key_responsibilities", [])),
            missing_skills=", ".join(state.get("missing_skills", [])),
        )
        response = llm.invoke(prompt)

        return {
            "cover_letter": response.content.strip(),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"cover_letter_node failed: {str(e)}")
        return {
            "cover_letter": "",
            "errors": errors,
        }