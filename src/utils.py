"""
Small utility helpers for this lab.
"""
from __future__ import annotations

from pathlib import Path
import json


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))
