from __future__ import annotations
from pathlib import Path
import json


def to_json(self, path: str | Path | None = None, indent: int | None = 2) -> str:
    """Serialize the summary to JSON, optionally writing it to `path`."""
    text = json.dumps(self.to_dict(), indent=indent, default=str)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
