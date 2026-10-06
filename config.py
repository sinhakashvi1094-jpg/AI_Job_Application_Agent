"""
config.py
Central configuration: Gemini API key, model selection, retry/fallback,
and a global rate limiter to respect free-tier RPM limits.
"""

import os
import time
import random
import threading
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini fallback chain (fastest/cheapest first, most reliable last).
# NOTE: gemini-2.0-flash was deprecated and retired March 3, 2026 — do not use it.
# gemini-1.5-flash is also retired. Current free-tier models as of mid-2026:
GEMINI_MODEL_CHAIN = [
    "gemini-2.5-flash-lite",  # 15 RPM / 1000 RPD free tier
    "gemini-2.5-flash",       # 10 RPM / 250 RPD free tier
]

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds, exponential backoff base

# ---- Global rate limiter ----
# Free-tier Gemini keys have a strict requests-per-minute cap. flash-lite allows
# 15 RPM (4s/call), flash allows only 10 RPM (6s/call). Since our fallback chain
# may drop into the stricter model, we throttle to the more conservative interval
# so we never exceed either model's limit. This pipeline fires several LLM calls
# back-to-back, some in parallel (resume_parser + jd_analyzer) — without this
# throttle, parallel nodes hit the API at the same instant and blow the RPM cap.
MIN_INTERVAL_SECONDS = 6.5
_rate_lock = threading.Lock()
_last_call_time = [0.0]


def _throttle():
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time[0]
        wait = MIN_INTERVAL_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()


def get_llm(temperature: float = 0.3):
    """
    Returns an LLM wrapper with automatic retry + model fallback across the
    Gemini model chain, throttled to respect rate limits.
    Exposes .invoke(prompt) like a normal LangChain LLM.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not found in .env")

    return _GeminiFallbackLLM(GEMINI_MODEL_CHAIN, GOOGLE_API_KEY, temperature)


class _GeminiFallbackLLM:
    def __init__(self, model_chain, api_key, temperature):
        self.temperature = temperature
        self.model_chain = model_chain
        self._clients = [
            ChatGoogleGenerativeAI(model=m, google_api_key=api_key, temperature=temperature)
            for m in model_chain
        ]

    def invoke(self, prompt):
        last_err = None
        for model_name, client in zip(self.model_chain, self._clients):
            for attempt in range(MAX_RETRIES):
                _throttle()  # wait for our turn before every single API call
                try:
                    return client.invoke(prompt)
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        delay = (BASE_DELAY ** attempt) + random.uniform(0, 1)
                        print(f"[{model_name}] retry {attempt+1}/{MAX_RETRIES} after {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        break  # non-retryable error, skip to next model in chain
            continue  # this model exhausted retries, try next model

        raise RuntimeError(f"All Gemini models in fallback chain failed. Last error: {last_err}")