"""Pytest bootstrap: put the repo root and scripts/ on sys.path so tests can
`import src...` (and `import fetch_notams`) without per-file sys.path
boilerplate. Auto-loaded by pytest before test collection."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
