from __future__ import annotations

import io
from typing import Optional


def _safe_decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _is_text_like(filename: str, content_type: str) -> bool:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    if ctype.startswith("text/"):
        return True
    return name.endswith((".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".yml", ".toml", ".ini"))


def _extract_pdf_text(data: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    parts: list[str] = []
    for page in doc:
        t = page.get_text("text")
        if t:
            parts.append(t)
    return "\n".join(parts)


def _extract_docx_text(data: bytes) -> str:
    import docx  # python-docx

    d = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def _extract_image_ocr(data: bytes) -> str:
    from PIL import Image
    import pytesseract

    img = Image.open(io.BytesIO(data))
    # Convert to RGB for broader compatibility
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img)
    return text or ""


def extract_text_from_upload(
    filename: str,
    content_type: str,
    data: bytes,
    max_chars: int = 250_000,
) -> str:
    """
    Tier-1 lightweight ingestion:
      - text/* or text-like extensions: decode to utf-8 (replace errors)
      - PDFs: extract embedded text via PyMuPDF (fitz)
      - DOCX: extract paragraphs via python-docx
      - Images: OCR via pytesseract + Pillow
      - other: best-effort decode if small; otherwise return empty
    """
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    try:
        if _is_text_like(filename, content_type):
            return _safe_decode(data)[:max_chars]

        if ctype == "application/pdf" or name.endswith(".pdf"):
            return _extract_pdf_text(data)[:max_chars]

        if ctype in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) or name.endswith(".docx"):
            # Note: .doc (legacy) is not supported by python-docx; we treat it as unsupported.
            if name.endswith(".doc") and not name.endswith(".docx"):
                return ""
            return _extract_docx_text(data)[:max_chars]

        if ctype.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return _extract_image_ocr(data)[:max_chars]

        # Best-effort decode for small unknown binaries (helps with weird text mime types)
        if len(data) <= 200_000:
            decoded = _safe_decode(data)
            # Heuristic: if it contains many replacement chars, treat as binary and ignore
            if decoded.count("\ufffd") > 1000:
                return ""
            return decoded[:max_chars]

        return ""
    except Exception:
        # Fail gracefully: don't break the chat due to a bad file
        return ""