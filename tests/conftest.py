import sys
from pathlib import Path

# analysis.py lives at the repo root, next to this tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
