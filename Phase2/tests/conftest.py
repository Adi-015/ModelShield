"""
Pytest configuration for ModelShield Phase 2.
Ensures Phase2 directory has precedence on sys.path so Phase 2 config is loaded.
"""

import sys
from pathlib import Path

phase2_dir = str(Path(__file__).resolve().parent.parent)
root_dir = str(Path(__file__).resolve().parent.parent.parent)

# Prioritize Phase2 directory
if phase2_dir in sys.path:
    sys.path.remove(phase2_dir)
sys.path.insert(0, phase2_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)

# If root config was already imported, purge it
if "config" in sys.modules:
    del sys.modules["config"]
