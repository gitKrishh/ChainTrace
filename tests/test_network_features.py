"""Tests for network feature aggregation and merging (Phase 5)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from backend.domain.network import NetworkObservation
from backend.domain.types import ScriptType
from backend.generator.export import observations_to_dataframe
from backend.ml.feature_registry import NETWORK_FEATURES
from backend.ml.network_features import (
    aggregate_network_features,
    merge_network_features,
)


@pytest.fixture
def sample_observations():
    """Create sample synthetic observations for multiple transactions."""
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    obs = [
        # Tx 100: 2 observations with delay, normal
        NetworkObservation(
            txid=100,
            src_ip="192.168.1.1",
            dst_ip="10.0.0.1",
            src_port=8333,
            dst_port=8333,
            timestamp=now,
            asn=15169,
            country="US",
            script_type=ScriptType.P2PKH,
            scenario_id="normal_propagation",
        ),
        NetworkObservation(
            txid=100,
            src_ip="192.168.1.2",
            dst_ip="10.0.0.2",
            src_port=8333,
            dst_port=8333,
            timestamp=now + timedelta(milliseconds=500),
            asn=15169,
            country="US",
            script_type=ScriptType.P2PKH,
            scenario_id="normal_propagation",
        ),
        # Tx 200: 1 observation, tor scenario
        NetworkObservation(
            txid=200,
            src_ip="10.10.10.10",
            dst_ip="10.0.0.3",
            src_port=9050,
            dst_port=8333,
            timestamp=now + timedelta(seconds=10),
            asn=9009,
            country="DE",
            script_type=ScriptType.P2WPKH,
            scenario_id="tor_relay_propagation",
        ),
    ]
    return observations_to_dataframe(obs)


def test_aggregate_network_features_structure(sample_observations):
    """Verify aggregated dataframe contains expected columns and rows."""
    agg = aggregate_network_features(sample_observations)

    assert len(agg) == 2  # Tx 100 and Tx 200
    assert "txId" in agg.columns
    for feat in NETWORK_FEATURES:
        assert feat in agg.columns


def test_aggregate_network_features_values(sample_observations):
    """Verify values calculated during network aggregation."""
    agg = aggregate_network_features(sample_observations).set_index("txId")

    # Tx 100
    tx100 = agg.loc[100]
    assert tx100["num_observations"] == 2
    assert tx100["num_unique_src_ips"] == 2
    assert tx100["num_unique_dst_ips"] == 2
    assert tx100["num_unique_countries"] == 1
    assert tx100["uses_tor"] == 0
    assert tx100["uses_vpn"] == 0
    assert tx100["mean_propagation_delay_ms"] == pytest.approx(500.0)

    # Tx 200
    tx200 = agg.loc[200]
    assert tx200["num_observations"] == 1
    assert tx200["uses_tor"] == 1
    assert tx200["uses_vpn"] == 0


def test_aggregate_network_features_empty():
    """Verify aggregation on empty dataframe returns empty DataFrame with right schema."""
    empty_df = pd.DataFrame()
    agg = aggregate_network_features(empty_df)
    assert agg.empty
    assert list(agg.columns) == ["txId"] + NETWORK_FEATURES


def test_merge_network_features():
    """Verify merging network features onto transaction dataframe."""
    tx_df = pd.DataFrame({
        "txId": [100, 200, 300],  # 300 has no network observations
        "time_step": [1, 1, 2],
        "total_BTC": [1.5, 0.5, 10.0],
    })

    net_feature_df = pd.DataFrame({
        "txId": [100, 200],
        "num_observations": [2, 1],
        "num_unique_src_ips": [2, 1],
        "num_unique_dst_ips": [2, 1],
        "num_unique_countries": [1, 1],
        "num_unique_asns": [1, 1],
        "mean_propagation_delay_ms": [500.0, 0.0],
        "max_propagation_delay_ms": [500.0, 0.0],
        "uses_tor": [0, 1],
        "uses_vpn": [0, 0],
    })

    merged = merge_network_features(tx_df, net_feature_df)

    assert len(merged) == 3
    for feat in NETWORK_FEATURES:
        assert feat in merged.columns

    # Tx 300 should have 0s filled
    tx300 = merged[merged["txId"] == 300].iloc[0]
    assert tx300["num_observations"] == 0
    assert tx300["uses_tor"] == 0
