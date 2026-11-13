import json

from typer.testing import CliRunner

from gnn_fraud.cli import app
from gnn_fraud.synthetic import make_synthetic_graph, write_elliptic_csvs

runner = CliRunner()


def test_prepare_and_train_from_local_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("GNNFRAUD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GNNFRAUD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    # 49 steps so the default 1-29 / 30-34 / 35-49 split applies; 93 local features as in Elliptic
    graph = make_synthetic_graph(n_steps=49, nodes_per_step=20, n_local=93, n_aggregated=4, seed=1)
    write_elliptic_csvs(graph, tmp_path / "data" / "raw")

    result = runner.invoke(app, ["prepare"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "processed" / "elliptic.npz").exists()

    result = runner.invoke(app, ["train", "--model", "lr", "--features", "local"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.output)
    assert info["trained_on_steps"] == [1, 34]
    assert (tmp_path / "artifacts" / "lr-local" / "meta.json").exists()


def test_train_rejects_unknown_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GNNFRAUD_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["train", "--model", "xgboost"])
    assert result.exit_code != 0


def test_train_without_data_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("GNNFRAUD_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["train", "--model", "lr"])
    assert result.exit_code == 1
    assert "gnnfraud prepare" in result.output
