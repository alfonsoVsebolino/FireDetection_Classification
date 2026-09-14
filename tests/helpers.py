"""Common testing helpers and dynamic module importers for progressive verification."""

import importlib
import os
import sys
import pytest

# Guarantee project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def import_or_skip(module_name: str, attribute_name: str = None):
    """Conditionally import or skip tests when modules are not yet implemented."""
    try:
        mod = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Module '{module_name}' not yet available: {e}")
    if attribute_name is not None:
        if not hasattr(mod, attribute_name):
            pytest.skip(f"Module '{module_name}' missing expected attribute '{attribute_name}'")
        return getattr(mod, attribute_name)
    return mod
