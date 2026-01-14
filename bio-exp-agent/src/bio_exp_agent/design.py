from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from .config import CONFIG


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


def generate_design(index_path: Path, rules: str) -> str:
    chunks = _load_index(index_path)
    retrieved = _retrieve_chunks(chunks, rules, CONFIG.top_k)
    context = _build_context(retrieved)

    prompt = (
        "You are a biology experiment design assistant. Use the context from papers and the user rules. "
        "Keep protocol-style steps explicit and do not summarize them into one sentence. "
        "Provide a structured design with: Goal, Hypothesis, Variables, Protocol Steps, Controls, Measurements, "
        "Risks, and Expected Results.\n\n"
        f"User rules: {rules}\n\n"
        f"Context from papers:\n{context}"
    )

    gen = pipeline("text2text-generation", model=CONFIG.design_model)
    out = gen(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"].strip()
