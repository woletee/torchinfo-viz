import json
from pathlib import Path

import torchvision

from tests.fixtures.models import LinearModel, SingleInputNet
from torchinfo import summary


def test_to_dict_links_parents_and_totals() -> None:
    results = summary(SingleInputNet(), input_size=(2, 1, 28, 28), verbose=0)
    data = results.to_dict()

    assert data["model"] == "SingleInputNet"
    assert data["has_input"]
    assert data["totals"]["total_params"] == results.total_params
    layers = data["layers"]
    assert layers[0]["parent"] is None
    assert all(layer["parent"] == 0 for layer in layers[1:])
    assert sum(layer["self_params"] for layer in layers) == results.total_params


def test_self_params_add_up_for_nested_model() -> None:
    results = summary(torchvision.models.resnet18(), verbose=0)
    layers = results.to_dict()["layers"]

    assert not results.to_dict()["has_input"]
    assert sum(layer["self_params"] for layer in layers) == results.total_params
    for layer in layers[1:]:
        assert layers[layer["parent"]]["depth"] == layer["depth"] - 1


def test_to_json_round_trips(tmp_path: Path) -> None:
    results = summary(LinearModel(), input_size=(1, 128), verbose=0)
    path = tmp_path / "summary.json"

    text = results.to_json(path)

    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(text)
    assert json.loads(text)["layers"][0]["input_size"] == [1, 128]


def test_to_html_is_self_contained(tmp_path: Path) -> None:
    results = summary(LinearModel(), input_size=(1, 128), verbose=0)
    path = tmp_path / "report.html"

    page = results.to_html(path)

    assert path.read_text(encoding="utf-8") == page
    for placeholder in ("__DATA__", "__TITLE__", "/*__CSS__*/", "/*__JS__*/"):
        assert placeholder not in page
    assert "<title>LinearModel · torchinfo</title>" in page
    assert "<script src=" not in page
    assert '"name": "Linear"' in page or '"name":"Linear"' in page


def test_to_html_escapes_script_end_tags() -> None:
    results = summary(LinearModel(), input_size=(1, 128), verbose=0)
    results.summary_list[1].class_name = "</script><b>x"

    page = results.to_html()

    assert "</script><b>x" not in page
