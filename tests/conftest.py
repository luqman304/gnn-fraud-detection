import pytest

from gnn_fraud.data import TemporalSplit
from gnn_fraud.synthetic import make_synthetic_graph
from gnn_fraud.training import GNNConfig, RFConfig


@pytest.fixture(scope="session")
def graph():
    return make_synthetic_graph(n_steps=12, nodes_per_step=80, seed=0)


@pytest.fixture(scope="session")
def split():
    return TemporalSplit(train_end=6, val_end=8)


@pytest.fixture(scope="session")
def gnn_cfg():
    return GNNConfig(hidden=32, max_epochs=80, patience=20)


@pytest.fixture(scope="session")
def rf_cfg():
    return RFConfig(n_estimators=30, max_features=8)
