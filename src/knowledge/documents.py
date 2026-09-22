from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


def stable_id(kind: str, *parts: str) -> str:
    import json
    raw = json.dumps(parts, ensure_ascii=False).encode("utf-8")
    return f"{kind}:" + hashlib.sha256(raw).hexdigest()


@dataclass
class Passage:
    id: str
    document_id: str
    location: str
    text: str
    extraction_method: str = "text"


@dataclass
class Document:
    id: str
    source: str
    title: str
    sha256: str
    format: str
    passages: list[Passage]

    def to_dict(self):
        return asdict(self)


def read_document(path: Path, *, chunk_size: int = 5000, overlap: int = 300, ocr: bool = False) -> Document:
    """Keep page/character locations; never silently treat binary data as text."""
    path = path.resolve()
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size must exceed nonnegative overlap")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    document_id = stable_id("document", digest)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        pages = [("text", raw.decode("utf-8-sig"), "text")]
    elif suffix == ".epub":
        from knowledge.formats import epub_pages
        pages = epub_pages(raw)
    elif suffix == ".pdf":
        from io import BytesIO
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF: supply a decrypted copy")
        pages = []
        for i, page in enumerate(reader.pages):
            text, method = page.extract_text() or "", "text"
            if not text.strip():
                if not ocr:
                    raise ValueError("PDF has pages without extractable text; OCR/review is required")
                from knowledge.formats import ocr_page
                text, method = ocr_page(path, i + 1), "ocr"
            pages.append((f"page {i + 1}", text, method))
    else:
        raise ValueError(f"Unsupported format: {suffix}; supported: TXT, Markdown, EPUB, PDF")
    passages = []
    for page, text, method in pages:
        for start in range(0, len(text), chunk_size - overlap):
            end = min(start + chunk_size, len(text))
            fragment = text[start:end]
            if fragment.strip():
                location = f"{page}, characters {start}:{end}"
                identity = (document_id, location, method, fragment) if method == "ocr" else (document_id, location)
                passages.append(Passage(stable_id("passage", *identity), document_id, location, fragment, method))
            if end == len(text):
                break
    if not passages:
        raise ValueError("Document contains no readable text")
    return Document(document_id, str(path), path.stem, digest, suffix, passages)
