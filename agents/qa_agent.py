"""
agents/qa_agent.py
RAG-based agent that answers application questions using resume + JD +
cover letter context. Only runs if the user provides a question.
"""

from config import get_llm


QA_PROMPT = """You are helping a candidate answer a job application question. Use ONLY the context below — do not invent facts about the candidate.

Candidate resume summary: {resume_summary}
Candidate experience: {experience}
Candidate matched skills for this role: {matched_skills}
Role title: {role_title}
Role responsibilities: {responsibilities}

Application question: "{question}"

Rules:
- Answer in first person, as the candidate.
- Ground the answer in the actual resume/experience context given — do not fabricate projects, companies, or numbers not present above.
- Keep it concise: 100-150 words unless the question clearly needs more.
- Be specific and concrete, not generic ("I am passionate about...").
- Return plain text only.

Answer:
"""


def qa_agent_node(state: dict) -> dict:
    """
    LangGraph node. Only runs meaningfully if state['qa_question'] is set.
    Reads resume_summary, resume_experience, matched_skills, jd_role_title,
    jd_key_responsibilities, qa_question. Writes: qa_answer
    """
    errors = state.get("errors", [])
    question = state.get("qa_question")

    if not question:
        # Nothing to answer — skip cleanly, no error
        return {"qa_answer": None, "errors": errors}

    try:
        llm = get_llm(temperature=0.5)
        prompt = QA_PROMPT.format(
            resume_summary=state.get("resume_summary", ""),
            experience="; ".join(state.get("resume_experience", [])),
            matched_skills=", ".join(state.get("matched_skills", [])),
            role_title=state.get("jd_role_title", "the role"),
            responsibilities="; ".join(state.get("jd_key_responsibilities", [])),
            question=question,
        )
        response = llm.invoke(prompt)

        return {
            "qa_answer": response.content.strip(),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"qa_agent_node failed: {str(e)}")
        return {
            "qa_answer": "",
            "errors": errors,
        }