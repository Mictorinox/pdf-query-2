# configs/__init__.py
# This file makes the 'configs' directory a Python package.

from .config import (
    API_CONFIG,
    PROJECT_ROOT,
    CHROMA_DB_PATH,
    TEMP_UPLOADS_DIR,
    EMBEDDING_MODEL,
    DOCUMENT_PROCESSING,
    SUPPORTED_FILE_TYPES,
    UI_CONFIG
)

__all__ = [
    "API_CONFIG",
    "PROJECT_ROOT",
    "CHROMA_DB_PATH",
    "TEMP_UPLOADS_DIR",
    "EMBEDDING_MODEL",
    "DOCUMENT_PROCESSING",
    "SUPPORTED_FILE_TYPES",
    "UI_CONFIG"
]