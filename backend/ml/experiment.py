"""M1/M2/M3 model comparison experiment runner.

Trains and evaluates three classifier configurations to demonstrate
incremental value through the experimental progression defined in
CONTEXT.md:

    M1: blockchain features only
    M2: blockchain + graph-derived features
    M3: blockchain + graph + network-derived features

Each model is trained using temporal splits, evaluated per-time-step,
and saved as a versioned artifact under ``models/``.

The network layer must not be claimed to improve detection unless the
experiments demonstrate measurable improvement.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.ml.dataset import prepare_ml_splits
from backend.ml.evaluation import TemporalEvaluation, evaluate_temporal, summarize_evaluation
from backend.ml.feature_registry import (
    ANONYMIZED_TX_FEATURES,
    FEATURE_SETS,
)
from backend.ml.graph_features import merge_graph_features
from backend.ml.network_features import aggregate_network_features, merge_network_features

logger = logging.getLogger(__name__)


# ======================================================================
# Comparison result container
# ======================================================================


@dataclass
class ModelResult:
    """Result of training and evaluating a single model configuration."""

    feature_set_name: str
    model_version: str
    n_features: int
    train_samples: int
    test_samples: int
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    temporal_evaluation: TemporalEvaluation | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    model_path: Path | None = None
    anomaly_model_path: Path | None = None


@dataclass
class ComparisonResult:
    """Results of the full M1/M2/M3 comparison experiment."""

    experiment_id: str
    timestamp: str
    model_results: dict[str, ModelResult] = field(default_factory=dict)
    summary_table: list[dict] = field(default_factory=list)


# ======================================================================
# Experiment runner
# ======================================================================


def run_comparison_experiment(
    tx_df: pd.DataFrame,
    models_dir: Path,
    graph_feature_df: pd.DataFrame | None = None,
    network_observations_df: pd.DataFrame | None = None,
    train_cutoff: int = 34,
    random_seed: int = 42,
    n_estimators: int = 200,
    experiment_id: str | None = None,
    train_anomaly: bool = True,
) -> ComparisonResult:
    """Run the full M1 → M2 → M3 comparison experiment.

    Trains three classifiers (one per feature set), evaluates each
    with temporal metrics, saves versioned model artifacts, and
    produces a comparison summary.

    Args:
        tx_df: Transaction DataFrame from ``load_transaction_dataset()``.
            Must contain ``txId``, ``time_step``, ``label``, and all
            blockchain feature columns.
        models_dir: Directory to save model artifacts.
        graph_feature_df: Optional DataFrame of graph-derived features
            (from ``extract_graph_feature_df()``). Required for M2/M3.
            If ``None``, M2/M3 are skipped.
        network_observations_df: Optional DataFrame of synthetic network
            observations (from generator pipeline). Required for M3.
            If ``None``, M3 is skipped.
        train_cutoff: Last ``time_step`` in training set (default 34).
        random_seed: Random seed for reproducibility.
        n_estimators: Number of trees in the Random Forest.
        experiment_id: Unique experiment identifier. Auto-generated if None.
        train_anomaly: Whether to train an anomaly detector alongside
            each classifier (default True).

    Returns:
        ``ComparisonResult`` with per-model metrics and comparison summary.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    exp_id = experiment_id or f"comparison_{timestamp}"

    logger.info("=" * 70)
    logger.info("Starting M1/M2/M3 comparison experiment: %s", exp_id)
    logger.info("=" * 70)

    result = ComparisonResult(experiment_id=exp_id, timestamp=timestamp)

    # --- Determine which models to run ---
    models_to_run: list[str] = ["M1"]

    if graph_feature_df is not None and not graph_feature_df.empty:
        models_to_run.append("M2")
    else:
        logger.info("No graph features provided — skipping M2")

    if (
        graph_feature_df is not None
        and not graph_feature_df.empty
        and network_observations_df is not None
        and not network_observations_df.empty
    ):
        models_to_run.append("M3")
    else:
        logger.info("No network observations provided — skipping M3")

    # --- Augment transaction DataFrame with additional features ---
    augmented_df = tx_df.copy()

    if "M2" in models_to_run or "M3" in models_to_run:
        logger.info("Merging graph features into transaction DataFrame...")
        augmented_df = merge_graph_features(augmented_df, graph_feature_df)

    if "M3" in models_to_run:
        logger.info("Aggregating and merging network features...")
        net_features = aggregate_network_features(network_observations_df)
        augmented_df = merge_network_features(augmented_df, net_features)

    # --- Train and evaluate each model ---
    for model_name in models_to_run:
        logger.info("-" * 50)
        logger.info("Training model: %s", model_name)
        logger.info("-" * 50)

        model_result = _train_and_evaluate_model(
            model_name=model_name,
            df=augmented_df,
            models_dir=models_dir,
            train_cutoff=train_cutoff,
            random_seed=random_seed,
            n_estimators=n_estimators,
            experiment_id=exp_id,
            train_anomaly=train_anomaly,
        )
        result.model_results[model_name] = model_result

    # --- Build comparison summary ---
    result.summary_table = _build_summary_table(result.model_results)

    # --- Save comparison report ---
    _save_comparison_report(result, models_dir)

    logger.info("=" * 70)
    logger.info("Comparison experiment complete: %s", exp_id)
    _log_summary_table(result.summary_table)
    logger.info("=" * 70)

    return result


# ======================================================================
# Internal helpers
# ======================================================================


def _train_and_evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    models_dir: Path,
    train_cutoff: int,
    random_seed: int,
    n_estimators: int,
    experiment_id: str,
    train_anomaly: bool,
) -> ModelResult:
    """Train and evaluate a single model configuration."""
    feature_cols = _resolve_feature_columns(model_name, df)

    # Prepare splits with leakage audit
    X_train, y_train, X_test, y_test, config = prepare_ml_splits(
        df=df,
        feature_cols=feature_cols,
        train_cutoff=train_cutoff,
        experiment_id=experiment_id,
        dataset_version="elliptic_pp_v1",
        feature_configuration=model_name,
        random_seed=random_seed,
    )

    # Train classifier
    classifier = TransactionClassifier(
        feature_set_name=model_name,
        n_estimators=n_estimators,
        random_state=random_seed,
    )
    train_metrics = classifier.train(X_train, y_train, feature_names=feature_cols)

    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test)

    # Temporal evaluation (per-time-step breakdown)
    # Extract time_steps for test set
    supervised_df = df[df["label"].isin({1, 2})].copy()
    test_df = supervised_df[supervised_df["time_step"] > train_cutoff]
    test_time_steps = test_df["time_step"].values[:len(y_test)]

    temporal_eval = evaluate_temporal(classifier, X_test, y_test, test_time_steps)

    # Feature importances (top 20)
    importances = classifier.feature_importances()
    top_importances = dict(list(importances.items())[:20])

    # Save classifier
    model_path = classifier.save(
        models_dir=models_dir,
        experiment_config=config,
        test_metrics=test_metrics,
    )

    # Optionally train anomaly detector on anonymized features
    anomaly_path = None
    if train_anomaly:
        anomaly_path = _train_anomaly_detector(
            df=df,
            train_cutoff=train_cutoff,
            random_seed=random_seed,
            models_dir=models_dir,
            model_name=model_name,
        )

    return ModelResult(
        feature_set_name=model_name,
        model_version=classifier.model_version,
        n_features=len(feature_cols),
        train_samples=len(y_train),
        test_samples=len(y_test),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        temporal_evaluation=temporal_eval,
        feature_importances=top_importances,
        model_path=model_path,
        anomaly_model_path=anomaly_path,
    )


def _train_anomaly_detector(
    df: pd.DataFrame,
    train_cutoff: int,
    random_seed: int,
    models_dir: Path,
    model_name: str,
) -> Path | None:
    """Train an Isolation Forest on anonymized features for a given model stage.

    Uses the anonymized features (``Local_feature_*`` +
    ``Aggregate_feature_*``) since these feed the anomaly detector's
    internal score per ARCHITECTURE.md.
    """
    # Filter to available anonymized feature columns
    available_anon = [c for c in ANONYMIZED_TX_FEATURES if c in df.columns]
    if not available_anon:
        logger.warning("No anonymized features available for anomaly detector")
        return None

    supervised_df = df[df["label"].isin({1, 2})].copy()
    train_df = supervised_df[supervised_df["time_step"] <= train_cutoff]

    X_anomaly = train_df[available_anon].fillna(0.0).values.astype(np.float64)

    if len(X_anomaly) == 0:
        logger.warning("Empty training set for anomaly detector")
        return None

    detector = AnomalyDetector(
        model_version=f"iforest_{model_name}_v1",
        random_state=random_seed,
    )
    detector.fit(X_anomaly)

    return detector.save(models_dir)


def _resolve_feature_columns(model_name: str, df: pd.DataFrame) -> list[str]:
    """Resolve the feature columns for a model, filtering to what exists in df.

    Falls back gracefully: if a feature set references columns that don't
    exist in the DataFrame (e.g. graph features weren't merged), those
    columns are dropped with a warning.
    """
    requested = FEATURE_SETS[model_name]
    available = [c for c in requested if c in df.columns]

    missing = set(requested) - set(available)
    if missing:
        logger.warning(
            "%s: %d/%d requested features missing from DataFrame (dropped): %s",
            model_name,
            len(missing),
            len(requested),
            sorted(missing)[:10],
        )

    if not available:
        msg = f"No features available for model {model_name}"
        raise ValueError(msg)

    return available


def _build_summary_table(model_results: dict[str, ModelResult]) -> list[dict]:
    """Build a comparison summary table from model results."""
    rows = []
    for name, result in model_results.items():
        row = {
            "model": name,
            "n_features": result.n_features,
            "train_samples": result.train_samples,
            "test_samples": result.test_samples,
            "test_accuracy": result.test_metrics.get("test_accuracy"),
            "test_precision": result.test_metrics.get("test_precision"),
            "test_recall": result.test_metrics.get("test_recall"),
            "test_f1": result.test_metrics.get("test_f1"),
            "test_roc_auc": result.test_metrics.get("test_roc_auc"),
        }
        rows.append(row)
    return rows


def _log_summary_table(summary: list[dict]) -> None:
    """Log the comparison summary table."""
    logger.info("\n--- MODEL COMPARISON SUMMARY ---")
    header = (
        f"{'Model':<6} {'Features':>8} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for row in summary:
        logger.info(
            "%-6s %8d %10.4f %10.4f %8.4f %8.4f %9s",
            row["model"],
            row["n_features"],
            row.get("test_accuracy", 0) or 0,
            row.get("test_precision", 0) or 0,
            row.get("test_recall", 0) or 0,
            row.get("test_f1", 0) or 0,
            f"{row['test_roc_auc']:.4f}" if row.get("test_roc_auc") is not None else "N/A",
        )


def _save_comparison_report(result: ComparisonResult, models_dir: Path) -> Path:
    """Save the comparison results as a JSON report."""
    models_dir.mkdir(parents=True, exist_ok=True)
    report_path = models_dir / f"comparison_{result.experiment_id}.json"

    report = {
        "experiment_id": result.experiment_id,
        "timestamp": result.timestamp,
        "summary": result.summary_table,
        "models": {},
    }

    for name, mr in result.model_results.items():
        model_entry = {
            "model_version": mr.model_version,
            "feature_set_name": mr.feature_set_name,
            "n_features": mr.n_features,
            "train_samples": mr.train_samples,
            "test_samples": mr.test_samples,
            "train_metrics": _safe_serialize(mr.train_metrics),
            "test_metrics": _safe_serialize(mr.test_metrics),
            "top_feature_importances": {
                k: round(v, 6) for k, v in list(mr.feature_importances.items())[:20]
            },
            "model_path": str(mr.model_path) if mr.model_path else None,
            "anomaly_model_path": str(mr.anomaly_model_path) if mr.anomaly_model_path else None,
        }

        if mr.temporal_evaluation:
            model_entry["temporal_evaluation"] = summarize_evaluation(
                mr.temporal_evaluation
            )

        report["models"][name] = model_entry

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Comparison report saved to %s", report_path)
    return report_path


def _safe_serialize(obj: dict) -> dict:
    """Ensure all values in a metrics dict are JSON-serializable."""
    result = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, np.integer | np.floating):
            result[k] = v.item()
        elif isinstance(v, Path):
            result[k] = str(v)
        else:
            result[k] = v
    return result
