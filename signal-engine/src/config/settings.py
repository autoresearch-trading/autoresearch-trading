from __future__ import annotations

"""Backward compatibility shim for legacy imports."""

# Import from the root config module to avoid circular imports
import sys
from pathlib import Path

# Add the project root to the path so we can import from the root config
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings

__all__ = ["Settings"]
