from dataclasses import dataclass
from typing import Dict


MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "flan-t5-small": {
        "id": "google/flan-t5-small",
        "task": "text2text-generation",
        "label": "Flan-T5 Small (fast, lightweight)",
    },
    "flan-t5-base": {
        "id": "google/flan-t5-base",
        "task": "text2text-generation",
        "label": "Flan-T5 Base (balanced)",
    },
    "scit5-base": {
        "id": "allenai/scit5-base",
        "task": "text2text-generation",
        "label": "SciT5 Base (science tuned)",
    },
    "llama-3.2-1b-instruct": {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "task": "text-generation",
        "label": "Llama 3.2 1B Instruct (requires HF access)",
    },
}

DEFAULT_MODEL_KEY = "flan-t5-small"


def get_model_spec(model_key_or_id: str | None) -> Dict[str, str]:
    if not model_key_or_id:
        return MODEL_CATALOG[DEFAULT_MODEL_KEY]
    if model_key_or_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_key_or_id]
    for spec in MODEL_CATALOG.values():
        if spec["id"] == model_key_or_id:
            return spec
    return MODEL_CATALOG[DEFAULT_MODEL_KEY]


@dataclass(frozen=True)
class Config:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_chunk_chars: int = 1500
    min_text_chars_per_page: int = 200
    ocr_lang: str = "en"
    max_context_chars: int = 6000
    top_k: int = 6


CONFIG = Config()
