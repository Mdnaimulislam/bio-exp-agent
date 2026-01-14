from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    summary_model: str = "google/flan-t5-small"
    design_model: str = "google/flan-t5-small"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_chunk_chars: int = 1500
    min_text_chars_per_page: int = 200
    ocr_lang: str = "en"
    max_context_chars: int = 6000
    top_k: int = 6


CONFIG = Config()
