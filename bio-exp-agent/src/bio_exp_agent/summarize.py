from __future__ import annotations

from typing import List

from .config import CONFIG, get_model_spec
from .utils import chunk_text


def _strip_prompt(prompt: str, generated: str) -> str:
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated.strip()


def summarize_text(text: str, model_key: str | None = None) -> str:
    if not text.strip():
        return ""
    spec = get_model_spec(model_key)
    from transformers import pipeline

    summarizer = pipeline(spec["task"], model=spec["id"])
    parts: List[str] = []
    for chunk in chunk_text(text, CONFIG.max_chunk_chars):
        prompt = (
            "Summarize the following scientific text in 5-8 concise bullet points. "
            "Keep key methods, results, and constraints.\n\n" + chunk
        )
        if spec["task"] == "text-generation":
            out = summarizer(prompt, max_new_tokens=256, do_sample=False)
            parts.append(_strip_prompt(prompt, out[0]["generated_text"]))
        else:
            out = summarizer(prompt, max_length=256, min_length=40, do_sample=False)
            parts.append(out[0]["generated_text"].strip())
    return "\n".join(parts)


def summarize_sections(sections: List[dict], model_key: str | None = None) -> str:
    merged = "\n\n".join(s["content"] for s in sections if s.get("content"))
    return summarize_text(merged, model_key=model_key)
