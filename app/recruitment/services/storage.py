"""
Local file storage for candidate documents (Phase 12).

MVP uses the local filesystem under UPLOAD_DIR. For production, swap this module
for S3/object storage (keep the same save/read/delete interface).
"""
import os
import uuid
import re

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
DOCS_SUBDIR = "documents"

ALLOWED_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}
ALLOWED_EXTS = {".pdf", ".doc", ".docx", ".txt"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


class StorageError(Exception):
    pass


def _safe_name(filename: str) -> str:
    base = os.path.basename(filename or "file")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base[:200] or "file"


def validate(filename: str, content_type: str, size: int):
    ext = os.path.splitext(filename or "")[1].lower()
    if size > MAX_BYTES:
        raise StorageError("File is larger than the 10 MB limit.")
    if ext not in ALLOWED_EXTS and content_type not in ALLOWED_TYPES:
        raise StorageError("Unsupported file type. Upload a PDF, Word document, or text file.")


def save_bytes(data: bytes, filename: str) -> str:
    target_dir = os.path.join(UPLOAD_DIR, DOCS_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)
    stored = f"{uuid.uuid4().hex}_{_safe_name(filename)}"
    path = os.path.join(target_dir, stored)
    with open(path, "wb") as f:
        f.write(data)
    return path


def delete_path(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
