"""Tests for the model inference pipeline (Phase 5)."""

import numpy as np
import pandas as pd
import pytest

from backend.domain.ml import MLScore
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.ml.feature_registry import INTERPRETABLE_TX_FEATURES
from backend.ml.inference import InferenceEngine


@pytest.fixture
def trained_models(tmp_path):
    """Fixture providing a trained classifier, anomaly detector, and saved paths."""
    feature_names = ["total_BTC", "fees", "size", "in_txs_degree", "out_txs_degree"]
    X = np.array([
        [1.0, 0.001, 250, 1, 2],
        [50.0, 0.05, 1200, 10, 20],
        [0.1, 0.0001, 180, 1, 1],
        [100.0, 0.1, 2000, 15, 30],
    ])
    y = np.array([0, 1, 0, 1])

    classifier = TransactionClassifier(
        model_version="test_clf_v1",
        feature_set_name="M1",
        n_estimators=10,
        random_state=42,
    )
    classifier.train(X, y, feature_names=feature_names)
    clf_path = classifier.save(tmp_path)

    detector = AnomalyDetector(
        model_version="test_anom_v1",
        contamination=0.25,
        n_estimators=10,
        random_state=42,
    )
    detector.fit(X)
    anom_path = detector.save(tmp_path)

    return classifier, detector, clf_path, anom_path, feature_names


def test_inference_engine_instantiation(trained_models):
    """Verify InferenceEngine initializes with trained models."""
    clf, det, _, _, _ = trained_models
    engine = InferenceEngine(classifier=clf, anomaly_detector=det)

    assert engine.model_version == "test_clf_v1"
    assert engine.feature_set_name == "M1"
    assert len(engine.feature_names) == 5


def test_inference_engine_unfitted_raises():
    """Verify initializing with unfitted models raises ValueError."""
    unfitted_clf = TransactionClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        InferenceEngine(classifier=unfitted_clf)


def test_inference_engine_load_from_disk(trained_models):
    """Verify InferenceEngine loads properly from saved .joblib artifacts."""
    _, _, clf_path, anom_path, _ = trained_models
    engine = InferenceEngine.load(model_path=clf_path, anomaly_model_path=anom_path)

    assert engine.model_version == "test_clf_v1"
    assert engine.anomaly_detector is not None


def test_inference_engine_predict_batch(trained_models):
    """Verify batch prediction returns valid MLScore domain objects."""
    clf, det, _, _, _ = trained_models
    engine = InferenceEngine(classifier=clf, anomaly_detector=det)

    df = pd.DataFrame({
        "txId": [101, 102],
        "total_BTC": [0.2, 85.0],
        "fees": [0.0001, 0.08],
        "size": [200, 1500],
        "in_txs_degree": [1, 12],
        "out_txs_degree": [1, 25],
    })

    scores = engine.predict_batch(df)

    assert len(scores) == 2
    assert all(isinstance(s, MLScore) for s in scores)
    assert scores[0].entity_id == "101"
    assert scores[1].entity_id == "102"
    assert scores[0].prediction in (0, 1)
    assert 0.0 <= scores[0].probability <= 1.0
    assert 0.0 <= scores[0].anomaly_score <= 1.0
    assert scores[0].model_version == "test_clf_v1"
    assert scores[0].feature_set == "M1"


def test_inference_engine_predict_single(trained_models):
    """Verify single record prediction."""
    clf, det, _, _, _ = trained_models
    engine = InferenceEngine(classifier=clf, anomaly_detector=det)

    record = {
        "txId": 999,
        "total_BTC": 100.0,
        "fees": 0.1,
        "size": 2500,
        "in_txs_degree": 20,
        "out_txs_degree": 40,
    }

    score = engine.predict_single(record)
    assert isinstance(score, MLScore)
    assert score.entity_id == "999"
    assert score.prediction == 1  # High values correspond to illicit in our training


def test_inference_engine_explain_prediction(trained_models):
    """Verify human-facing explanation only contains interpretable signals."""
    clf, det, _, _, _ = trained_models
    engine = InferenceEngine(classifier=clf, anomaly_detector=det)

    record = {
        "txId": 555,
        "total_BTC": 50.0,
        "fees": 0.05,
        "size": 1200,
        "in_txs_degree": 10,
        "out_txs_degree": 20,
    }

    explanation = engine.explain_prediction(record)

    assert explanation["entity_id"] == "555"
    assert "classification" in explanation
    assert "top_interpretable_signals" in explanation
    assert "anomaly_severity" in explanation

    # Ensure all top signals are in the approved interpretable feature list
    for feat_name in explanation["top_interpretable_signals"]:
        assert feat_name in INTERPRETABLE_TX_FEATURES


def test_inference_engine_synthetic_flag():
    """Verify contains_synthetic_input flag is True for models with network features."""
    feature_names = ["total_BTC", "num_observations", "uses_tor"]
    X = np.array([[1.0, 2, 0], [10.0, 5, 1]])
    y = np.array([0, 1])

    clf = TransactionClassifier(feature_set_name="M3", n_estimators=5, random_state=42)
    clf.train(X, y, feature_names=feature_names)

    engine = InferenceEngine(classifier=clf)
    sample_rec = {
        "txId": 1,
        "total_BTC": 1.0,
        "num_observations": 2,
        "uses_tor": 0,
    }
    score = engine.predict_single(sample_rec)

    assert score.contains_synthetic_input is True
