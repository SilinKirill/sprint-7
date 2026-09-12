"""Import reusable implementation without loading any source documents."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / 'task-6'))
from snapshot import read_snapshot, write_snapshot, digest
from runtime6 import LiveRAG
from shared6 import rag, DIMENSION
