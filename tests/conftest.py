"""Make the repo root importable so `import agent...` / `import eval...` work
regardless of the directory pytest is launched from."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
