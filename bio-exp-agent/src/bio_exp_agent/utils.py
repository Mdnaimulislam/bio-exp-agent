import re
from typing import Iterable, List, Dict


PROTOCOL_KEYWORDS = [
    "materials and methods",
    "methods",
    "experimental procedures",
    "methodology",
    "protocol",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 120:
        return False
    if stripped.endswith(":"):
        return True
    if re.match(r"^\d+\.?\s+\w+", stripped):
        return True
    if stripped.isupper() and any(c.isalpha() for c in stripped):
        return True
    return False


def split_sections(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    sections: List[Dict[str, str]] = []
    current_title = ""
    current_lines: List[str] = []

    def flush():
        if not current_lines:
            return
        content = clean_text("\n".join(current_lines))
        title = current_title or ""
        sections.append({"title": title, "content": content})

    for line in lines:
        if is_heading(line):
            flush()
            current_title = line.strip().rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return sections


def mark_protocol_sections(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    marked = []
    for sec in sections:
        title = sec.get("title", "").lower()
        is_protocol = any(k in title for k in PROTOCOL_KEYWORDS)
        marked.append({
            "title": sec.get("title", ""),
            "content": sec.get("content", ""),
            "is_protocol": is_protocol,
        })
    return marked


def chunk_text(text: str, max_chars: int) -> Iterable[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        yield text
        return
    paragraphs = re.split(r"\n\n+", text)
    current = []
    count = 0
    for para in paragraphs:
        if not para.strip():
            continue
        if count + len(para) + 2 > max_chars and current:
            yield clean_text("\n\n".join(current))
            current = [para]
            count = len(para)
        else:
            current.append(para)
            count += len(para) + 2
    if current:
        yield clean_text("\n\n".join(current))
