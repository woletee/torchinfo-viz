from __future__ import annotations
from importlib.resources import files
from pathlib import Path
import html


def read_asset(name: str) -> str:
    return files("torchinfo._export").joinpath(name).read_text(encoding="utf-8")


def to_html(self, path: str | Path | None = None) -> str:
    """Render a self-contained interactive HTML report, optionally saving it."""
    data = self.to_json(indent=None).replace("</", r"<\/")
    page = (
        read_asset("report.html")
        .replace("/*__CSS__*/", read_asset("report.css"))
        .replace("/*__JS__*/", read_asset("report.js"))
        .replace("__TITLE__", html.escape(f"{self.to_dict()['model']} · torchinfo"))
        .replace("__DATA__", data)
    )
    if path is not None:
        Path(path).write_text(page, encoding="utf-8")
    return page
