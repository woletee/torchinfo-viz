"""Run torchinfo-viz on well-known models and build a gallery of HTML reports.

    python examples/model_zoo.py --out zoo_reports
    python examples/model_zoo.py --only resnet,unet     # substring filter

Every model is built with random weights (no downloads). For each one it checks
that the summary runs, the data-flow graph was recorded, and the per-layer
parameter counts add up to the model's real parameter count.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from torchinfo import summary

Builder = Callable[[], nn.Module]
ZOO: list[tuple[str, str, Builder, tuple[int, ...] | None, dict[str, Any]]] = []


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def add(group: str, name: str, build: Builder, size=None, **kwargs: Any) -> None:
    ZOO.append((group, name, build, size, kwargs))


def register_torchvision() -> None:
    from torchvision import models as m
    from torchvision.models import detection, segmentation

    img = (1, 3, 224, 224)
    for name in [
        "resnet18", "resnet50", "vgg11", "densenet121", "googlenet",
        "mobilenet_v3_small", "efficientnet_b0", "regnet_y_400mf",
        "shufflenet_v2_x1_0", "convnext_tiny", "vit_b_16", "swin_t",
    ]:
        kw = {"aux_logits": False, "init_weights": False} if name == "googlenet" else {}
        add("torchvision", name, lambda n=name, kw=kw: getattr(m, n)(weights=None, **kw), img)
    add("torchvision", "fcn_resnet50", lambda: segmentation.fcn_resnet50(
        weights=None, weights_backbone=None), (2, 3, 224, 224))
    add("torchvision", "deeplabv3_mobilenet_v3", lambda: segmentation.deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=None), (2, 3, 224, 224))
    add("torchvision", "ssdlite320 (eval)", lambda: detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None).eval(), (1, 3, 320, 320))


def register_monai() -> None:
    from monai.networks import nets

    vol = (1, 1, 64, 64, 64)
    add("monai", "UNet", lambda: nets.UNet(3, 1, 1, (16, 32, 64, 128), (2, 2, 2)), vol)
    add("monai", "BasicUNet", lambda: nets.BasicUNet(3, 1, 1), vol)
    add("monai", "AttentionUnet", lambda: nets.AttentionUnet(
        3, 1, 1, (16, 32, 64, 128), (2, 2, 2)), vol)
    add("monai", "SegResNet", lambda: nets.SegResNet(3, 8, 1, 1), vol)
    add("monai", "DynUNet", lambda: nets.DynUNet(
        3, 1, 1, [3, 3, 3, 3], [1, 2, 2, 2], [2, 2, 2], deep_supervision=False), vol)
    add("monai", "UNETR", lambda: nets.UNETR(1, 1, (64, 64, 64), feature_size=16), vol)
    add("monai", "SwinUNETR", lambda: nets.SwinUNETR(
        in_channels=1, out_channels=1, feature_size=24), vol)
    add("monai", "DenseNet121 (3D)", lambda: nets.DenseNet121(3, 1, 2), vol)


def register_transformers() -> None:
    from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

    ids = {"input_data": [torch.randint(0, 1000, (1, 32))], "dtypes": [torch.long]}
    small = dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=2,
                 intermediate_size=256, vocab_size=1000)
    add("transformers", "BERT (2 layers)", lambda: BertModel(BertConfig(**small)), **ids)
    add("transformers", "GPT-2 (2 layers)", lambda: GPT2Model(GPT2Config(
        n_layer=2, n_embd=128, n_head=2, vocab_size=1000)), **ids)


def register_torch() -> None:
    add("torch.nn", "Transformer", lambda: nn.Transformer(
        64, 4, 2, 2, 128, batch_first=True), input_data=[
        torch.randn(1, 10, 64), torch.randn(1, 12, 64)])
    add("torch.nn", "LSTM (2 layers)", lambda: nn.LSTM(32, 64, 2, batch_first=True),
        (1, 20, 32))


def unique_params(model: nn.Module) -> int:
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def run_one(entry, out: Path) -> dict[str, Any]:
    group, name, build, size, kwargs = entry
    row: dict[str, Any] = {"group": group, "name": name}
    start = time.perf_counter()
    try:
        model = build()
        result = summary(model, size, verbose=0, device="cpu", **kwargs)
        data = result.to_dict()
        graph = data["graph"]
        own = sum(layer["self_params"] for layer in data["layers"])
        expected = unique_params(model)
        file = out / f"{group}_{''.join(c if c.isalnum() else '_' for c in name)}.html"
        result.to_html(file)
        row.update(
            ok=True, file=file.name, params=expected, layers=len(data["layers"]),
            nodes=len(graph["nodes"]) if graph else 0,
            edges=len(graph["edges"]) if graph else 0,
            graph=graph is not None, params_match=own == expected,
            macs=data["totals"]["total_mult_adds"],
        )
    except Exception as error:  # report and keep going
        row.update(ok=False, error=f"{type(error).__name__}: {error}",
                   trace=traceback.format_exc())
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def status(row: dict[str, Any]) -> str:
    if not row["ok"]:
        return "FAIL"
    if not row["graph"]:
        return "NO GRAPH"
    return "OK" if row["params_match"] else "PARAM MISMATCH"


def write_index(rows: list[dict[str, Any]], out: Path) -> Path:
    colors = {"OK": "#1a7f37", "FAIL": "#cf222e"}
    body = []
    for r in rows:
        s = status(r)
        link = (f'<a href="{r["file"]}">{html.escape(r["name"])}</a>'
                if r["ok"] else html.escape(r["name"]))
        info = (f'{r["params"]:,}</td><td>{r["layers"]}</td><td>{r["nodes"]} / {r["edges"]}'
                f'</td><td>{r["macs"] / 1e9:.2f} G' if r["ok"]
                else f'</td><td colspan="3"><code>{html.escape(r["error"])}</code>')
        body.append(
            f'<tr><td>{r["group"]}</td><td>{link}</td>'
            f'<td style="color:{colors.get(s, "#9a6700")};font-weight:600">{s}</td>'
            f'<td>{info}</td><td>{r["seconds"]}s</td></tr>')
    passed = sum(status(r) == "OK" for r in rows)
    page = f"""<!doctype html><meta charset="utf-8"><title>torchinfo-viz model zoo</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;background:#fff;color:#1f2328}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 10px;border-bottom:1px solid #d0d7de;
text-align:left}}th{{background:#f6f8fa}}a{{color:#0969da}}code{{font-size:12px}}
@media (prefers-color-scheme:dark){{body{{background:#0d1117;color:#e6edf3}}th{{background:#161b22}}
td,th{{border-color:#30363d}}a{{color:#4493f8}}}}</style>
<h1>torchinfo-viz model zoo</h1><p>{passed} / {len(rows)} models passed.
Click a model to open its interactive report.</p>
<table><tr><th>Library</th><th>Model</th><th>Status</th><th>Params</th><th>Layers</th>
<th>Graph nodes / edges</th><th>Mult-adds</th><th>Time</th></tr>{''.join(body)}</table>"""
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="zoo_reports", help="folder for the reports")
    parser.add_argument("--only", default="", help="comma-separated name filters")
    args = parser.parse_args()
    register_torch()
    for module, register in [("torchvision", register_torchvision),
                             ("monai", register_monai),
                             ("transformers", register_transformers)]:
        if not has(module):
            print(f"skipping {module} (not installed)")
            continue
        try:
            register()
        except Exception as error:  # broken install or too-old version
            print(f"skipping {module} ({type(error).__name__}: {error})"[:150])
    filters = [f.strip().lower() for f in args.only.split(",") if f.strip()]
    entries = [e for e in ZOO if not filters or any(f in e[1].lower() for f in filters)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        row = run_one(entry, out)
        rows.append(row)
        extra = (f'{row["params"]:>12,} params  {row["nodes"]:>4} nodes'
                 if row["ok"] else row["error"][:90])
        print(f'{status(row):<15} {row["group"]:<13} {row["name"]:<26} {extra}')
    index = write_index(rows, out)
    passed = sum(status(r) == "OK" for r in rows)
    print(f"\n{passed}/{len(rows)} passed. Open {index.resolve()}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
