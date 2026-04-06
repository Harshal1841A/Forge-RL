# conftest.py  — ensures project root is on sys.path for pytest and direct runs
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
