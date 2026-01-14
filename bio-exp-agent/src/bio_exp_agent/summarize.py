from __future__ import annotations

from typing import List

from transformers import pipeline

from .config import CONFIG
from .utils import chunk_text


def _get_summarizer():
    return pipeline("text2text-generation", model=CONFIG.summary_model)


def summarize_text(text: str) -> str:
    if not text.strip():
        return ""
    summarizer = _get_summarizer()
    parts: List[str] = []
    for chunk in chunk_text(text, CONFIG.max_chunk_chars):
        prompt = (
            "Summarize the following scientific text in 5-8 concise bullet points. "
            "Keep key methods, results, and constraints.\n\n" + chunk
        )
        out = summarizer(prompt, max_length=256, min_length=40, do_sample=False)
        parts.append(out[0]["generated_text"].strip())
    return "\n".join(parts)


def summarize_sections(sections: List[dict]) -> str:
    merged = "\n\n".join(s["content"] for s in sections if s.get("content"))
    return summarize_text(merged)
