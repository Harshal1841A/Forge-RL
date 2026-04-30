"""
conftest.py for FORGE-MA tests (tests/forge_ma/).
Adds the merged project root to sys.path so all forge_ma modules resolve correctly.
"""
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
