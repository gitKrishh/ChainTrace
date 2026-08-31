"""Tests for M1/M2/M3 model comparison experiment runner (Phase 5)."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from backend.domain.network import NetworkObservation
from backend.domain.types import ScriptType
from backend.generator.export import observations_to_dataframe
from backend.ml.experiment import ComparisonResult, run_comparison_experiment
from backend.ml.feature_registry import (
    ALL_TX_FEATURES,
    GRAPH_FEATURES,
    NETWORK_FEATURES,
)


@pytest.fixture
def mock_transaction_df():
    """Create a mock transaction dataframe spanning multiple time_steps."""
    np.random.seed(42)
    n_samples = 100

    # 50 rows in train (time_steps 1..34), 50 rows in test (time_steps 35..49)
    time_steps = np.concatenate([
        np.random.randint(1, 35, size=50),
        np.random.randint(35, 50, size=50),
    ])
    labels = np.random.choice([1, 2], size=n_samples, p=[0.3, 0.7])  # 1=Illicit, 2=Licit

    data = {
        "txId": list(range(1000, 1000 + n_samples)),
        "time_step": time_steps,
        "label": labels,
    }

    # Populate all 182 blockchain features
    for feat in ALL_TX_FEATURES:
        data[feat] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def mock_graph_features(mock_transaction_df):
    """Create mock graph features dataframe for mock transactions."""
    tx_ids = mock_transaction_df["txId"].tolist()
    data = {"txId": tx_ids}
    for feat in GRAPH_FEATURES:
        data[feat] = np.random.rand(len(tx_ids))
    return pd.DataFrame(data)


@pytest.fixture
def mock_network_observations(mock_transaction_df):
    """Create mock network observations dataframe for mock transactions."""
    tx_ids = mock_transaction_df["txId"].tolist()
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    obs = []
    for txid in tx_ids:
        obs.append(
            NetworkObservation(
                txid=txid,
                src_ip="192.168.1.1",
                dst_ip="10.0.0.1",
                src_port=8333,
                dst_port=8333,
                timestamp=now,
                asn=15169,
                country="US",
                script_type=ScriptType.P2PKH,
                scenario_id="normal_propagation",
            )
        )
    return observations_to_dataframe(obs)


def test_run_comparison_experiment_m1_only(mock_transaction_df, tmp_path):
    """Test comparison experiment running M1 baseline when no graph/network data provided."""
    result = run_comparison_experiment(
        tx_df=mock_transaction_df,
        models_dir=tmp_path,
        graph_feature_df=None,
        network_observations_df=None,
        train_cutoff=34,
        n_estimators=10,
        random_seed=42,
        experiment_id="test_m1_only",
    )

    assert isinstance(result, ComparisonResult)
    assert "M1" in result.model_results
    assert "M2" not in result.model_results
    assert "M3" not in result.model_results

    m1 = result.model_results["M1"]
    assert m1.n_features == len(ALL_TX_FEATURES)
    assert m1.model_path.exists()
    assert m1.test_metrics["test_f1"] is not None

    # Check report saved
    report_file = tmp_path / f"comparison_{result.experiment_id}.json"
    assert report_file.exists()


def test_run_comparison_experiment_full_m1_m2_m3(
    mock_transaction_df,
    mock_graph_features,
    mock_network_observations,
    tmp_path,
):
    """Test full M1/M2/M3 comparison run with all feature sets and artifact generation."""
    result = run_comparison_experiment(
        tx_df=mock_transaction_df,
        models_dir=tmp_path,
        graph_feature_df=mock_graph_features,
        network_observations_df=mock_network_observations,
        train_cutoff=34,
        n_estimators=10,
        random_seed=42,
        experiment_id="test_m1_m2_m3_full",
    )

    assert "M1" in result.model_results
    assert "M2" in result.model_results
    assert "M3" in result.model_results

    m1 = result.model_results["M1"]
    m2 = result.model_results["M2"]
    m3 = result.model_results["M3"]

    # Feature counts progression
    assert m1.n_features == len(ALL_TX_FEATURES)
    assert m2.n_features == len(ALL_TX_FEATURES) + len(GRAPH_FEATURES)
    assert m3.n_features == len(ALL_TX_FEATURES) + len(GRAPH_FEATURES) + len(NETWORK_FEATURES)

    # Check artifacts exist on disk
    for mr in (m1, m2, m3):
        assert mr.model_path.exists()
        metadata_path = mr.model_path.with_name(mr.model_path.stem + "_metadata.json")
        assert metadata_path.exists()
        if mr.anomaly_model_path:
            assert mr.anomaly_model_path.exists()

    # Verify summary table structure
    assert len(result.summary_table) == 3
    summary_models = [r["model"] for r in result.summary_table]
    assert summary_models == ["M1", "M2", "M3"]

    # Verify JSON report on disk
    report_path = tmp_path / f"comparison_{result.experiment_id}.json"
    assert report_path.exists()
    with open(report_path) as f:
        report_data = json.load(f)
    assert "models" in report_data
    assert "M3" in report_data["models"]
    assert report_data["models"]["M3"]["n_features"] == m3.n_features
