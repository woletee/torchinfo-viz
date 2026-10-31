from collections import Counter

import torch
import torchvision
from torch import nn

from torchinfo import summary


class Block(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = Block(1, 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = Block(8, 8)
        self.up = nn.ConvTranspose2d(8, 8, 2, 2)
        self.dec = Block(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc(x)
        up = self.up(self.mid(self.pool(skip)))
        return torch.sigmoid(self.dec(torch.cat([up, skip], dim=1)))


class SharedBlocks(nn.Module):
    """Registers the same submodule under two parents, like MONAI's DynUNet."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.path = nn.Sequential(self.blocks[0], self.blocks[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)


def graph_of(model: nn.Module, **kwargs: object) -> dict:
    return summary(model, verbose=0, **kwargs).to_dict()["graph"]


def test_skip_connection_reaches_cat() -> None:
    graph = graph_of(TinyUNet(), input_size=(1, 1, 16, 16))
    nodes = {node["key"]: node for node in graph["nodes"]}
    cat = next(key for key, node in nodes.items() if node.get("name") == "cat")
    sources = [edge["src"] for edge in graph["edges"] if edge["dst"] == cat]

    assert len(sources) == 2
    assert {edge["shape"][1] for edge in graph["edges"] if edge["dst"] == cat} == {8}
    assert Counter(node["kind"] for node in graph["nodes"])["input"] == 1
    assert [n for n in graph["nodes"] if n["kind"] == "output"][0]["shape"] == [
        1,
        8,
        16,
        16,
    ]
    assert (
        Counter(n["name"] for n in graph["nodes"] if n["kind"] == "op")["sigmoid"] == 1
    )


def test_residual_adds_have_two_inputs() -> None:
    graph = graph_of(torchvision.models.resnet18(), input_size=(1, 3, 64, 64))
    adds = [node["key"] for node in graph["nodes"] if node.get("name") == "add"]

    assert len(adds) == 8
    for key in adds:
        assert sum(edge["dst"] == key for edge in graph["edges"]) == 2


def test_graph_needs_a_forward_pass() -> None:
    assert graph_of(TinyUNet()) is None


def test_shared_modules_are_counted_once() -> None:
    results = summary(SharedBlocks(), input_size=(1, 4), verbose=0)
    layers = results.to_dict()["layers"]

    assert sum(layer["self_params"] for layer in layers) == results.total_params == 40


def test_attention_self_params_cover_uncalled_submodules() -> None:
    model = nn.TransformerEncoder(nn.TransformerEncoderLayer(16, 2, 32), 2)
    data = summary(model, (5, 1, 16), verbose=0).to_dict()
    total = sum(p.numel() for p in model.parameters())
    assert sum(layer["self_params"] for layer in data["layers"]) == total
