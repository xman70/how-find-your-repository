"""Reading TXT / DOCX / PDF documents."""
from __future__ import annotations

import io
from pathlib import Path


class DocumentReadError(Exception):
    pass


def read_bytes(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md", ".text", ""):
        for enc in ("utf-8-sig", "utf-16", "cp1253", "cp1252", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        raise DocumentReadError("Could not decode the text file.")
    if ext == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise DocumentReadError("python-docx is not installed.") from exc
        try:
            d = docx.Document(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            raise DocumentReadError(f"Could not read the DOCX file: {exc}") from exc
        return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentReadError("pypdf is not installed.") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # noqa: BLE001
            raise DocumentReadError(f"Could not read the PDF file: {exc}") from exc
        text = "\n\n".join(pages)
        if not text.strip():
            raise DocumentReadError("The PDF contains no extractable text (it may be a scanned image).")
        return text
    raise DocumentReadError(f"Unsupported file type '{ext}'. Use TXT, DOCX or PDF.")


def read_document(path: str | Path) -> str:
    p = Path(path)
    return read_bytes(p.read_bytes(), p.name)
