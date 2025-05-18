# knowledge_base/__init__.py
# This file makes the 'knowledge_base' directory a Python package.

from .kb_manager import (
    create_kb,
    load_kb,
    list_kbs,
    add_documents_to_kb,
    get_kb_path,
    DEFAULT_KB_ROOT_DIR
)

__all__ = [
    "create_kb",
    "load_kb",
    "list_kbs",
    "add_documents_to_kb",
    "get_kb_path",
    "DEFAULT_KB_ROOT_DIR"
]