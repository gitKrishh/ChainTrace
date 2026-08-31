"""Model inference pipeline.

Loads saved/versioned model artifacts from the ``models/`` directory
and performs prediction on new or existing transaction/wallet data.

CRITICAL RULE:
Training must NEVER execute during inference or API requests (see AGENTS.md §7).

Outputs canonical ``MLScore`` domain models with complete provenance,
confidence probabilities, and optional anomaly scores.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.domain.ml import MLScore
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.ml.feature_registry import (
    INTERPRETABLE_TX_FEATURES,
    NETWORK_FEATURES,
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Inference engine for transaction classification and anomaly scoring.

    Loads a trained ``TransactionClassifier`` and an optional
    ``AnomalyDetector`` to score transactions without triggering training.

    Attributes:
        classifier: The loaded ``TransactionClassifier``.
        anomaly_detector: Optional loaded ``AnomalyDetector``.
        model_version: Model identifier of the classifier.
        feature_set_name: Feature set configuration name (M1/M2/M3).
    """

    def __init__(
        self,
        classifier: TransactionClassifier,
        anomaly_detector: AnomalyDetector | None = None,
    ) -> None:
        """Initialize the inference engine with pre-loaded models.

        Args:
            classifier: Fitted TransactionClassifier instance.
            anomaly_detector: Optional fitted AnomalyDetector instance.
        """
        if not classifier.is_fitted:
            msg = "Classifier passed to InferenceEngine is not fitted."
            raise ValueError(msg)
        if anomaly_detector is not None and not anomaly_detector.is_fitted:
            msg = "AnomalyDetector passed to InferenceEngine is not fitted."
            raise ValueError(msg)

        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.model_version = classifier.model_version
        self.feature_set_name = classifier.feature_set_name
        self.feature_names = classifier.feature_names

    @classmethod
    def load(
        cls,
        model_path: Path,
        anomaly_model_path: Path | None = None,
    ) -> "InferenceEngine":
        """Load an inference engine from serialized model artifacts on disk.

        Args:
            model_path: Path to the classifier ``.joblib`` artifact.
            anomaly_model_path: Optional path to the anomaly detector ``.joblib`` artifact.

        Returns:
            Instantiated ``InferenceEngine``.
        """
        logger.info("Loading classifier from %s", model_path)
        classifier = TransactionClassifier.load(model_path)

        anomaly_detector = None
        if anomaly_model_path is not None and anomaly_model_path.exists():
            logger.info("Loading anomaly detector from %s", anomaly_model_path)
            anomaly_detector = AnomalyDetector.load(anomaly_model_path)

        return cls(classifier=classifier, anomaly_detector=anomaly_detector)

    def predict_batch(
        self,
        df: pd.DataFrame,
        entity_type: str = "transaction",
    ) -> list[MLScore]:
        """Score a batch of records and return canonical MLScore domain objects.

        Args:
            df: DataFrame containing the transaction records with columns
                matching ``self.feature_names``, plus ``txId`` / ``txid`` or ``address``.
            entity_type: Entity type ('transaction' or 'wallet').

        Returns:
            List of ``MLScore`` domain models.
        """
        if df.empty:
            return []

        # Identify entity ID column
        id_col = _resolve_id_column(df, entity_type)
        entity_ids = [str(x) for x in df[id_col]]

        # Prepare feature matrix X aligned with model feature_names
        X = self._prepare_features(df)

        # Predict classes and probabilities
        preds = self.classifier.predict(X)
        probas = self.classifier.predict_proba(X)

        # Anomaly scores if detector is available
        anomaly_scores: np.ndarray | None = None
        if self.anomaly_detector is not None:
            anomaly_scores = self.anomaly_detector.score(X)

        # Determine if input contains synthetic features
        contains_synthetic = any(col in NETWORK_FEATURES for col in self.feature_names)

        results: list[MLScore] = []
        for i, eid in enumerate(entity_ids):
            pred_class = int(preds[i])
            # Probability corresponding to positive class (1 = Illicit)
            prob_val = float(probas[i, 1]) if probas.shape[1] > 1 else float(probas[i, 0])
            anom_val = float(anomaly_scores[i]) if anomaly_scores is not None else None

            score = MLScore(
                entity_id=eid,
                entity_type=entity_type,
                prediction=pred_class,
                probability=round(prob_val, 4),
                anomaly_score=round(anom_val, 4) if anom_val is not None else None,
                model_version=self.model_version,
                feature_set=self.feature_set_name,
                contains_synthetic_input=contains_synthetic,
            )
            results.append(score)

        return results

    def predict_single(
        self,
        record: dict | pd.Series,
        entity_id: str | int | None = None,
        entity_type: str = "transaction",
    ) -> MLScore:
        """Score a single entity record.

        Args:
            record: Mapping or Series of feature names to values.
            entity_id: Optional entity ID if not in record.
            entity_type: Entity type ('transaction' or 'wallet').

        Returns:
            ``MLScore`` domain model.
        """
        df = pd.DataFrame([record])
        if entity_id is not None:
            id_col = "txId" if entity_type == "transaction" else "address"
            df[id_col] = entity_id

        scores = self.predict_batch(df, entity_type=entity_type)
        if not scores:
            msg = "Failed to produce ML score for input record."
            raise ValueError(msg)
        return scores[0]

    def explain_prediction(
        self,
        record: dict | pd.Series,
        top_k: int = 5,
    ) -> dict[str, float | str | None]:
        """Generate human-facing explanation for a prediction.

        MUST use only interpretable features per ARCHITECTURE.md §ml/
        and DATASET.md §7. Anonymized features are NEVER surfaced.

        Args:
            record: Mapping or Series of feature names to values.
            top_k: Number of top contributing interpretable features to return.

        Returns:
            Dictionary with prediction summary, top interpretable signals,
            and high-level anomaly severity description.
        """
        score = self.predict_single(record)
        importances = self.classifier.feature_importances()

        # Filter importances to only interpretable features
        interpretable_signals: dict[str, float] = {}
        for feat, imp in importances.items():
            if feat in INTERPRETABLE_TX_FEATURES:
                feat_val = float(record.get(feat, 0.0) or 0.0)
                interpretable_signals[feat] = {
                    "importance": round(imp, 4),
                    "value": round(feat_val, 4),
                }
                if len(interpretable_signals) >= top_k:
                    break

        anomaly_explanation = None
        if score.anomaly_score is not None and self.anomaly_detector is not None:
            anomaly_explanation = AnomalyDetector.explain_score(score.anomaly_score)

        return {
            "entity_id": score.entity_id,
            "prediction": score.prediction,
            "probability": score.probability,
            "classification": "Illicit" if score.prediction == 1 else "Licit",
            "anomaly_score": score.anomaly_score,
            "anomaly_severity": anomaly_explanation,
            "top_interpretable_signals": interpretable_signals,
            "model_version": score.model_version,
            "contains_synthetic_input": score.contains_synthetic_input,
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and align feature matrix X matching ``self.feature_names``."""
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            logger.warning(
                "Missing %d features in input data; imputing 0.0: %s",
                len(missing),
                missing[:5],
            )

        X_df = pd.DataFrame(index=df.index)
        for col in self.feature_names:
            if col in df.columns:
                X_df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                X_df[col] = 0.0

        return X_df.values.astype(np.float64)


def _resolve_id_column(df: pd.DataFrame, entity_type: str) -> str:
    """Find the ID column in the DataFrame."""
    candidates = (
        ["txId", "txid", "id"]
        if entity_type == "transaction"
        else ["address", "wallet_address", "addr", "id"]
    )
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback to index if no named ID column is found
    df["_entity_id"] = df.index.astype(str)
    return "_entity_id"
