from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from .config import CONFIG, get_model_spec


def _load_index(index_path: Path) -> List[dict]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return data["chunks"]


def _retrieve_chunks(chunks: List[dict], query: str, top_k: int) -> List[dict]:
    embedder = SentenceTransformer(CONFIG.embedding_model)
    query_vec = embedder.encode([query]).astype("float32")

    emb = np.array([c["embedding"] for c in chunks], dtype="float32")
    nn = NearestNeighbors(n_neighbors=min(top_k, len(chunks)), metric="cosine")
    nn.fit(emb)
    _, idx = nn.kneighbors(query_vec)

    return [chunks[i] for i in idx[0]]


def _build_context(chunks: List[dict]) -> str:
    protocol = [c for c in chunks if c.get("is_protocol")]
    non_protocol = [c for c in chunks if not c.get("is_protocol")]

    selected = protocol + non_protocol
    parts = []
    total = 0
    for c in selected:
        block = f"[Section: {c.get('title','').strip()}] {c['text']}"
        if total + len(block) > CONFIG.max_context_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _strip_prompt(prompt: str, generated: str) -> str:
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated.strip()


def generate_design(index_path: Path, rules: str, model_key: str | None = None) -> str:
    chunks = _load_index(index_path)
    retrieved = _retrieve_chunks(chunks, rules, CONFIG.top_k)
    context = _build_context(retrieved)

    prompt = (
        "You are a biology experiment design assistant. Based on the user rules and context from research papers, "
        "generate a detailed experiment design.\n\n"
        "IMPORTANT: If the rules ask for a protocol, provide EXPLICIT STEP-BY-STEP instructions. "
        "Number each step clearly (1., 2., 3., etc). Include reagents, quantities, temperatures, and times.\n\n"
        "Structure your response as:\n"
        "GOAL: [Clear statement of objective]\n"
        "HYPOTHESIS: [Predicted outcome]\n"
        "PROTOCOL STEPS:\n"
        "1. [First step with specific details]\n"
        "2. [Second step with specific details]\n"
        "3. [Continue with all necessary steps]\n"
        "CONTROLS: [Positive and negative controls]\n"
        "EXPECTED RESULTS: [What success looks like]\n"
        "ESTIMATED COST: [Cost reduction strategies if applicable]\n\n"
        f"User rules: {rules}\n\n"
        f"Reference context from papers:\n{context}"
    )

    spec = get_model_spec(model_key)
    from transformers import pipeline

    gen = pipeline(spec["task"], model=spec["id"])
    if spec["task"] == "text-generation":
        out = gen(prompt, max_new_tokens=512, do_sample=False)
        return _strip_prompt(prompt, out[0]["generated_text"])
    out = gen(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"].strip()
