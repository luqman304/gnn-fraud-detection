import pytest
from fastapi.testclient import TestClient

from gnn_fraud.api import create_app
from gnn_fraud.training import GNNConfig, train_model


@pytest.fixture(scope="module")
def client(graph, split, tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    graph.save(root / "graph.npz")
    model = train_model("sage", graph, split, "all", seed=0, gnn_cfg=GNNConfig(hidden=16, max_epochs=30))
    model.save(root / "model")
    with TestClient(create_app(root / "graph.npz", root / "model")) as c:
        yield c


def test_health_and_model_card(client, graph):
    assert client.get("/healthz").json() == {"status": "ok"}
    card = client.get("/model").json()
    assert card["kind"] == "sage"
    assert card["nodes_scored"] == graph.num_nodes


def test_score_transaction_with_neighbourhood(client, graph):
    tx = int(graph.tx_id[graph.edge_index[0, 0]])
    r = client.get(f"/transactions/{tx}", params={"hops": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["tx_id"] == tx
    assert 0.0 <= body["illicit_probability"] <= 1.0
    assert body["degree"] >= 1
    assert body["neighbourhood_size"] >= body["degree"]
    hops = [n["hop"] for n in body["neighbourhood"]]
    assert hops == sorted(hops) and set(hops) <= {1, 2}
    steps = {n["time_step"] for n in body["neighbourhood"]}
    assert steps == {body["time_step"]}


def test_max_neighbours_truncates(client, graph):
    tx = int(graph.tx_id[graph.edge_index[0, 0]])
    body = client.get(f"/transactions/{tx}", params={"hops": 3, "max_neighbours": 2}).json()
    assert len(body["neighbourhood"]) <= 2


def test_unknown_transaction_is_404(client):
    assert client.get("/transactions/1").status_code == 404


def test_batch_scoring_reports_missing_ids(client, graph):
    ids = [int(graph.tx_id[0]), int(graph.tx_id[1]), 1]
    body = client.post("/score", json={"tx_ids": ids}).json()
    assert [s["tx_id"] for s in body["scores"]] == ids[:2]
    assert body["missing"] == [1]


def test_batch_validation(client):
    assert client.post("/score", json={"tx_ids": []}).status_code == 422
