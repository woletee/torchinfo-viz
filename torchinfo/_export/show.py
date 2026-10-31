from __future__ import annotations
from pathlib import Path
import html
import tempfile
import webbrowser


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


def show(self, height: int = 640) -> Path | None:
    """Open the interactive report inline in Jupyter, otherwise in a browser."""
    page = self.to_html()
    if in_notebook():
        from IPython.display import HTML, display

        frame = f'<iframe srcdoc="{html.escape(page)}" style="width:100%;height:{height}px;border:0"></iframe>'
        display(HTML(frame))
        return None
    path = Path(tempfile.mkdtemp(prefix="torchinfo-")) / "report.html"
    path.write_text(page, encoding="utf-8")
    webbrowser.open(path.as_uri())
    return path
