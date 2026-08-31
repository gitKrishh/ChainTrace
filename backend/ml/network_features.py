"""Network-derived feature aggregation for M3 experiments.

Aggregates synthetic network observations into per-transaction feature
vectors suitable for merging with the transaction feature matrix.

These features are 100% derived from synthetic data and MUST carry
``contains_synthetic_input=True`` in any downstream ``MLScore``.

See CONTEXT.md — Experimental Progression (M3: blockchain + graph + network).
"""

import logging

import pandas as pd

from backend.ml.feature_registry import NETWORK_FEATURES

logger = logging.getLogger(__name__)


def aggregate_network_features(
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate network observations into per-transaction feature rows.

    For each transaction (``txid``), computes:

    - ``num_observations``: total observation count
    - ``num_unique_src_ips``: distinct source IPs observed
    - ``num_unique_dst_ips``: distinct destination IPs observed
    - ``num_unique_countries``: distinct country codes seen
    - ``num_unique_asns``: distinct ASN numbers seen
    - ``mean_propagation_delay_ms``: average inter-observation time (ms)
    - ``max_propagation_delay_ms``: maximum inter-observation time (ms)
    - ``uses_tor``: 1 if any observation used a Tor exit node, 0 otherwise
    - ``uses_vpn``: 1 if any observation used a VPN node, 0 otherwise

    Args:
        observations_df: DataFrame of synthetic network observations,
            as produced by ``generator.export.observations_to_dataframe()``.
            Must contain at least ``txid``, ``src_ip``, ``dst_ip``.

    Returns:
        DataFrame with ``txId`` column plus the 9 network feature columns.
        One row per unique transaction.
    """
    if observations_df.empty:
        logger.warning("Empty observations DataFrame; returning empty network features")
        return pd.DataFrame(columns=["txId"] + NETWORK_FEATURES)

    # Normalize txid column name
    tx_col = "txid" if "txid" in observations_df.columns else "txId"
    df = observations_df.copy()
    if tx_col != "txId":
        df = df.rename(columns={tx_col: "txId"})

    # Parse timestamps for propagation delay computation (handle ISO8601 with/without microsecond)
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", format="ISO8601", utc=True)
    else:
        df["_ts"] = pd.NaT

    # Detect Tor/VPN from scenario_id or ASN patterns
    df["_is_tor"] = _detect_tor(df)
    df["_is_vpn"] = _detect_vpn(df)

    # Group by transaction and aggregate basic counts
    grouped = df.groupby("txId")

    agg = pd.DataFrame({
        "num_observations": grouped.size(),
        "num_unique_src_ips": grouped["src_ip"].nunique(),
        "num_unique_dst_ips": grouped["dst_ip"].nunique(),
    })

    # Country/ASN uniqueness (handle missing columns gracefully)
    if "country" in df.columns:
        agg["num_unique_countries"] = grouped["country"].nunique()
    else:
        agg["num_unique_countries"] = 0

    if "asn" in df.columns:
        agg["num_unique_asns"] = grouped["asn"].nunique()
    else:
        agg["num_unique_asns"] = 0

    # Vectorized propagation delay computation
    mean_delays, max_delays = _compute_delays_vectorized(df)
    agg["mean_propagation_delay_ms"] = mean_delays
    agg["max_propagation_delay_ms"] = max_delays
    agg["mean_propagation_delay_ms"] = agg["mean_propagation_delay_ms"].fillna(0.0)
    agg["max_propagation_delay_ms"] = agg["max_propagation_delay_ms"].fillna(0.0)

    # Tor / VPN flags (any observation within the tx group)
    agg["uses_tor"] = grouped["_is_tor"].max().astype(int)
    agg["uses_vpn"] = grouped["_is_vpn"].max().astype(int)

    # Reset index to get txId as a column
    agg = agg.reset_index()

    # Ensure column order matches NETWORK_FEATURES
    for col in NETWORK_FEATURES:
        if col not in agg.columns:
            agg[col] = 0

    result = agg[["txId"] + NETWORK_FEATURES].copy()

    logger.info(
        "Aggregated network features for %d transactions from %d observations",
        len(result),
        len(observations_df),
    )

    return result


def merge_network_features(
    tx_df: pd.DataFrame,
    network_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge network features onto the transaction feature DataFrame.

    Performs a left join so all transactions are preserved.
    Missing network features (transactions with no observations) are
    filled with 0 — giving them neutral/baseline network signal.

    Args:
        tx_df: Transaction DataFrame with a ``txId`` column.
        network_feature_df: DataFrame from ``aggregate_network_features()``.

    Returns:
        The transaction DataFrame augmented with network feature columns.

    Raises:
        ValueError: If ``txId`` column is missing from either DataFrame.
    """
    if "txId" not in tx_df.columns:
        msg = "tx_df is missing 'txId' column"
        raise ValueError(msg)
    if "txId" not in network_feature_df.columns:
        msg = "network_feature_df is missing 'txId' column"
        raise ValueError(msg)

    merged = tx_df.merge(network_feature_df, on="txId", how="left")

    # Fill NaN for transactions with no network observations
    for col in NETWORK_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    n_missing = len(tx_df) - len(
        network_feature_df.merge(tx_df[["txId"]], on="txId", how="inner")
    )
    if n_missing > 0:
        logger.info(
            "%d transactions had no network features (filled with 0)",
            n_missing,
        )

    return merged


# --- Private helpers ---


def _compute_delays_vectorized(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute mean and max propagation delay (ms) per txId vectorially."""
    if "_ts" not in df.columns or df["_ts"].isna().all():
        unique_txs = df["txId"].unique()
        return (
            pd.Series(0.0, index=unique_txs),
            pd.Series(0.0, index=unique_txs),
        )

    # Sort by txId and timestamp
    valid_ts = df.dropna(subset=["_ts"]).sort_values(["txId", "_ts"]).copy()
    if valid_ts.empty:
        unique_txs = df["txId"].unique()
        return (
            pd.Series(0.0, index=unique_txs),
            pd.Series(0.0, index=unique_txs),
        )

    valid_ts["_diff_ms"] = valid_ts.groupby("txId")["_ts"].diff().dt.total_seconds() * 1000.0
    diffs = valid_ts.dropna(subset=["_diff_ms"])

    if diffs.empty:
        unique_txs = df["txId"].unique()
        return (
            pd.Series(0.0, index=unique_txs),
            pd.Series(0.0, index=unique_txs),
        )

    mean_delays = diffs.groupby("txId")["_diff_ms"].mean()
    max_delays = diffs.groupby("txId")["_diff_ms"].max()
    return mean_delays, max_delays


def _detect_tor(df: pd.DataFrame) -> pd.Series:
    """Heuristic: flag observations likely from Tor nodes.

    Checks scenario_id for 'tor' patterns. In the synthetic generator,
    Tor scenarios are explicitly labeled.
    """
    if "scenario_id" in df.columns:
        return df["scenario_id"].str.contains("tor", case=False, na=False)
    return pd.Series(False, index=df.index)


def _detect_vpn(df: pd.DataFrame) -> pd.Series:
    """Heuristic: flag observations likely from VPN nodes.

    Checks scenario_id for 'vpn' patterns.
    """
    if "scenario_id" in df.columns:
        return df["scenario_id"].str.contains("vpn", case=False, na=False)
    return pd.Series(False, index=df.index)
