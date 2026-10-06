"""
vectorstore/faiss_utils.py
Semantic skill matching using FAISS + Gemini embeddings.
"""

import os
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY

_embedder = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=GOOGLE_API_KEY,
)

SIMILARITY_THRESHOLD = 0.83  # cosine similarity above this = "same skill"


def _embed_texts(texts: list[str]) -> np.ndarray:
    vectors = _embedder.embed_documents(texts)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)  # normalize so inner product = cosine similarity
    return arr


def semantic_skill_match(resume_skills: list[str], jd_skills: list[str]):
    """
    For each JD skill, finds the closest resume skill via FAISS cosine similarity.
    Returns (matched_skills, missing_skills) — both lists of JD skill names.
    """
    if not jd_skills:
        return [], []
    if not resume_skills:
        return [], list(jd_skills)

    resume_vectors = _embed_texts(resume_skills)
    jd_vectors = _embed_texts(jd_skills)

    index = faiss.IndexFlatIP(resume_vectors.shape[1])  # inner product on normalized vecs = cosine sim
    index.add(resume_vectors)

    matched, missing = [], []
    top_k = 1
    scores, indices = index.search(jd_vectors, top_k)

    for i, jd_skill in enumerate(jd_skills):
        best_score = scores[i][0]
        if best_score >= SIMILARITY_THRESHOLD:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return matched, missing