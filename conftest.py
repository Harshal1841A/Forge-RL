# conftest.py — ensures project root is on sys.path for pytest and direct runs
# Updated by FORGE v2 merge: also covers tests/forge_ma/ sub-suite
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
# Primary root (original FORGE v1 modules: env, agents, tools, server, training)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))