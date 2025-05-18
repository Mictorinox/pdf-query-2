# utils/__init__.py
# This file makes the 'utils' directory a Python package.

from .file_utils import load_document
from .vector_utils import split_documents, get_embedding_function

__all__ = [
    "load_document",
    "split_documents",
    "get_embedding_function",
]