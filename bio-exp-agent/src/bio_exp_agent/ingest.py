from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import fitz  # pymupdf
from PIL import Image
from tqdm import tqdm

from .config import CONFIG
from .summarize import summarize_sections
from .utils import clean_text, split_sections, mark_protocol_sections, chunk_text


def _page_to_image(page: fitz.Page) -> Image.Image:
    pix = page.get_pixmap(dpi=200)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img


def _ocr_page(img: Image.Image) -> str:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError("PaddleOCR is not available. Install paddleocr.") from exc

    ocr = PaddleOCR(use_angle_cls=True, lang=CONFIG.ocr_lang)
    result = ocr.ocr(img, cls=True)
    lines = []
    for page in result:
        for det in page:
            lines.append(det[1][0])
    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    for page in doc:
        text = page.get_text() or ""
        if len(text.strip()) < CONFIG.min_text_chars_per_page:
            img = _page_to_image(page)
            try:
                text = _ocr_page(img)
            except RuntimeError:
                text = ""
        pages_text.append(text)
    return clean_text("\n\n".join(pages_text))


def build_paper_record(pdf_path: Path, model_key: str | None = None) -> Dict:
    text = extract_text_from_pdf(pdf_path)
    sections = mark_protocol_sections(split_sections(text))
    protocol_blocks = [s for s in sections if s["is_protocol"]]
    non_protocol_blocks = [s for s in sections if not s["is_protocol"]]
    summary = summarize_sections(non_protocol_blocks, model_key=model_key)

    return {
        "paper_id": pdf_path.stem,
        "source": str(pdf_path),
        "summary": summary,
        "sections": sections,
        "protocols": protocol_blocks,
        "non_protocol_sections": non_protocol_blocks,
    }


def write_paper_json(record: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['paper_id']}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


def build_index(paper_records: List[Dict], out_path: Path) -> Path:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(CONFIG.embedding_model)
    chunks = []

    for record in paper_records:
        for section in record["sections"]:
            for chunk in chunk_text(section["content"], CONFIG.max_chunk_chars):
                if not chunk:
                    continue
                chunks.append({
                    "paper_id": record["paper_id"],
                    "is_protocol": section["is_protocol"],
                    "title": section.get("title", ""),
                    "text": chunk,
                })

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    for c, e in zip(chunks, embeddings):
        c["embedding"] = e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"chunks": chunks}, indent=2), encoding="utf-8")
    return out_path


def ingest_pdfs(input_dir: Path, out_dir: Path, model_key: str | None = None) -> Path:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")

    records = []
    for pdf in tqdm(pdfs, desc="Ingesting PDFs"):
        record = build_paper_record(pdf, model_key=model_key)
        write_paper_json(record, out_dir)
        records.append(record)

    index_path = out_dir / "index.json"
    return build_index(records, index_path)
