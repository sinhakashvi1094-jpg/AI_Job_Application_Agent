# 🎯 AI Job Application Agent

A multi-agent LangGraph system that analyzes job descriptions against your resume, identifies skill gaps using semantic (embedding-based) matching, and generates tailored application materials — a cover letter and ATS-optimized resume bullets — grounded strictly in your actual experience.

Built as a demonstration of production-oriented GenAI engineering: multi-agent orchestration, parallel execution, semantic search, retry/fallback resilience, and rate-limit-aware API design.

## What it does

1. **Parses your resume** (PDF/DOCX/text) into structured skills, experience, and a summary
2. **Parses a job description** into required skills, nice-to-have skills, and key responsibilities
3. **Compares them semantically** using FAISS + Gemini embeddings — catches conceptual matches like "GenAI" ≈ "Generative AI" that simple keyword matching would miss
4. **Generates a tailored cover letter** grounded in your real matched skills, with explicit prompt constraints against generic filler language
5. **Generates ATS-optimized resume bullets** — never fabricates metrics; flags where you need to add a real number instead of hallucinating one
6. **Optionally answers application questions** (e.g. "Why do you want to work here?") using RAG over your resume + the JD

## Architecture
┌─────────────────┐
                │   Streamlit UI   │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  LangGraph State  │
                │      Machine      │
                └────────┬─────────┘
                         │
    ┌──────────┬────────┼────────┬──────────┐
    ▼          ▼        ▼        ▼          ▼
FAISS Vector Store
(semantic skill matching)
**Why LangGraph:** the Resume Parser and JD Analyzer agents run in **parallel** (both fan out from `START`), then fan into the Skill Gap agent, which waits for both to complete. The Q&A agent only runs via a **conditional edge** — it's skipped entirely if the user didn't ask a question. A plain LangChain sequential chain can't express this branching + conditional routing as cleanly.

## Key engineering decisions

- **Semantic skill matching over keyword matching.** Skills are embedded with Gemini's embedding model and compared via FAISS cosine similarity (threshold tuned to 0.83 after testing — high enough to avoid false positives like matching "LangGraph" to "LangChain" just because they share a name, low enough to still catch genuine synonyms).
- **No hallucinated metrics.** The resume bullet agent is explicitly prompted to insert `[ADD METRIC]` placeholders rather than inventing numbers — the model doesn't know your actual impact, so it shouldn't guess.
- **Global rate limiting.** All Gemini calls pass through a process-wide throttle (`config.py`) so that parallel LangGraph nodes don't simultaneously blow through the free-tier requests-per-minute limit — a real failure mode discovered while building this, not a hypothetical one.
- **Model fallback chain.** If the primary model (`gemini-2.5-flash-lite`) is rate-limited or unavailable, requests automatically retry with exponential backoff, then fall back to `gemini-2.5-flash`.
- **Errors don't crash the pipeline.** Each agent catches its own exceptions and logs them into a shared `errors` list in the graph state (merged safely across parallel writes via a `operator.add` reducer) — one failed agent doesn't take down the others.

## Tech stack

- **Orchestration:** LangGraph (StateGraph, parallel fan-out/fan-in, conditional edges)
- **LLM:** Google Gemini (`gemini-2.5-flash-lite` / `gemini-2.5-flash`, with fallback)
- **Semantic search:** FAISS + Gemini embeddings
- **UI:** Streamlit
- **File parsing:** pypdf, python-docx

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:
Run the app:

```bash
streamlit run app.py
```

## Limitations / honest notes

- Free-tier Gemini API keys have strict rate limits (10-15 requests/minute), so a full pipeline run takes 30-60+ seconds due to intentional throttling. A paid tier or higher-quota key removes this bottleneck.
- Semantic skill matching is a genuine tradeoff: a similarity threshold that's too loose produces false-positive matches (e.g. treating "LangGraph" as equivalent to "LangChain"); too strict misses real synonyms. Threshold was tuned empirically, not theoretically derived.
- Does not scrape or auto-submit to job boards — by design. Automating LinkedIn/Indeed interactions typically violates their Terms of Service. This tool assists with materials; the user still applies manually.







