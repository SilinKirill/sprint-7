"""Explicit imports of shared implementation; importing does not load private data."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
for name in ('task-3', 'task-4', 'task-5'):
    sys.path.insert(0, str(ROOT / name))
from common import MODEL, DIMENSION, QUERY_PROMPT, chunk_document, load_model, encode
import rag
from pipeline import ProtectedRAG
from security import filter_passages
from prepare_examples import make_examples


def load_base():
    return rag.load_task3_index()
