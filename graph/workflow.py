"""
graph/workflow.py
Wires all 5 agents into a LangGraph StateGraph:

  START
    |--> resume_parser  --\
    |--> jd_analyzer     --> skill_gap --> cover_letter --> resume_bullets --> [conditional] --> qa_agent or END
                                                                                      |
                                                                                      +--> END (if no question)

Resume Parser and JD Analyzer run in PARALLEL (both fan out from START,
fan into skill_gap, which waits for both to finish).
Q&A Agent only runs if state['qa_question'] is set (conditional edge).
"""

from langgraph.graph import StateGraph, START, END
from typing import Literal

from graph.state import AgentState
from agents.resume_parser import resume_parser_node
from agents.jd_analyzer import jd_analyzer_node
from agents.skill_gap import skill_gap_node
from agents.cover_letter import cover_letter_node
from agents.resume_bullets import resume_bullets_node
from agents.qa_agent import qa_agent_node


def route_after_bullets(state: AgentState) -> Literal["qa_agent", "__end__"]:
    """Router: only call the Q&A agent if the user actually provided a question."""
    if state.get("qa_question"):
        return "qa_agent"
    return "__end__"


def build_graph():
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("resume_parser", resume_parser_node)
    builder.add_node("jd_analyzer", jd_analyzer_node)
    builder.add_node("skill_gap", skill_gap_node)
    builder.add_node("cover_letter", cover_letter_node)
    builder.add_node("resume_bullets", resume_bullets_node)
    builder.add_node("qa_agent", qa_agent_node)

    # Fan-out: both parsers run in parallel straight from START
    builder.add_edge(START, "resume_parser")
    builder.add_edge(START, "jd_analyzer")

    # Fan-in: skill_gap waits for BOTH resume_parser AND jd_analyzer to finish
    builder.add_edge(["resume_parser", "jd_analyzer"], "skill_gap")

    # Sequential from here
    builder.add_edge("skill_gap", "cover_letter")
    builder.add_edge("cover_letter", "resume_bullets")

    # Conditional: only go to qa_agent if a question was provided
    builder.add_conditional_edges(
        "resume_bullets",
        route_after_bullets,
        {
            "qa_agent": "qa_agent",
            "__end__": END,
        },
    )
    builder.add_edge("qa_agent", END)

    return builder.compile()


# Singleton compiled graph, imported by app.py
job_agent_graph = build_graph()


def run_pipeline(resume_text: str, jd_text: str, qa_question: str | None = None) -> dict:
    """
    Convenience wrapper for Streamlit app.py — runs the full pipeline
    and returns the final state dict.
    """
    initial_state: AgentState = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "qa_question": qa_question,
        "errors": [],
    }
    final_state = job_agent_graph.invoke(initial_state)
    return final_state