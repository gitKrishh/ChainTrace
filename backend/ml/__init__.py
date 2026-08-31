"""Machine learning module.

Responsible for supervised classification, anomaly detection,
feature aggregation, M1/M2/M3 comparison experiments, and inference.
Training and inference must remain separate.
Training must not execute during API requests.
"""

from .anomaly import AnomalyDetector
from .classifier import TransactionClassifier
from .dataset import load_transaction_dataset, prepare_ml_splits
from .evaluation import TemporalEvaluation, evaluate_temporal, summarize_evaluation
from .experiment import ComparisonResult, ModelResult, run_comparison_experiment
from .feature_registry import (
    ALL_TX_FEATURES,
    ALL_WALLET_FEATURES,
    ANONYMIZED_TX_FEATURES,
    BINARY_TARGET_MAP,
    FEATURE_SETS,
    FORBIDDEN_FEATURE_COLUMNS,
    GRAPH_FEATURES,
    INTERPRETABLE_TX_FEATURES,
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    NETWORK_FEATURES,
    TRAIN_LABELS,
    WALLET_SCALAR_FEATURES,
    WALLET_STATS_FEATURES,
)
from .graph_features import extract_graph_feature_df, merge_graph_features
from .inference import InferenceEngine
from .leakage import LeakageAuditResult, audit_leakage
from .network_features import aggregate_network_features, merge_network_features
from .splitting import temporal_train_test_split

__all__ = [
    # Anomaly
    "AnomalyDetector",
    # Classifier
    "TransactionClassifier",
    # Dataset
    "load_transaction_dataset",
    "prepare_ml_splits",
    # Evaluation
    "TemporalEvaluation",
    "evaluate_temporal",
    "summarize_evaluation",
    # Experiment comparison
    "ComparisonResult",
    "ModelResult",
    "run_comparison_experiment",
    # Inference
    "InferenceEngine",
    # Feature registry
    "ALL_TX_FEATURES",
    "ALL_WALLET_FEATURES",
    "ANONYMIZED_TX_FEATURES",
    "BINARY_TARGET_MAP",
    "FEATURE_SETS",
    "FORBIDDEN_FEATURE_COLUMNS",
    "GRAPH_FEATURES",
    "INTERPRETABLE_TX_FEATURES",
    "LABEL_ILLICIT",
    "LABEL_LICIT",
    "LABEL_UNKNOWN",
    "NETWORK_FEATURES",
    "TRAIN_LABELS",
    "WALLET_SCALAR_FEATURES",
    "WALLET_STATS_FEATURES",
    # Graph features
    "extract_graph_feature_df",
    "merge_graph_features",
    # Network features
    "aggregate_network_features",
    "merge_network_features",
    # Splitting
    "temporal_train_test_split",
    # Leakage
    "LeakageAuditResult",
    "audit_leakage",
]
