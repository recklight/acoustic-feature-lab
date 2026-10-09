"""Pipeline: prepare, train and predict on synthetic recordings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from speechdsp import write_wav

from acoustic_feature_lab import (
    DatasetError,
    load_model,
    predict_inputs,
    prepare_features,
    save_model,
    train_model,
    utterance_features,
)
from acoustic_feature_lab.pipeline import recording_frames


def test_prepare_gives_one_vector_per_recording(synthetic_dataset, fast_config):
    features = prepare_features(synthetic_dataset, fast_config)
    index = pd.read_csv(synthetic_dataset)
    assert features.X.shape == (len(index), 78)
    assert np.allclose(features.y, index["severity"])
    assert features.groups.tolist() == index["group"].tolist()
    assert features.config == fast_config.to_dict()
    assert features.feature_names[0] == "mean_c1" and len(features.feature_names) == 78


def test_features_carry_the_severity_signal(synthetic_features):
    correlations = [
        abs(np.corrcoef(column, synthetic_features.y)[0, 1]) for column in synthetic_features.X.T
    ]
    assert max(correlations) > 0.7


def test_train_then_predict_reproduces_the_training_features(
    tmp_path, synthetic_dataset, synthetic_features, fast_config
):
    model = train_model(synthetic_features, fast_config)
    path = save_model(tmp_path / "model.joblib", model, fast_config)
    bundle = load_model(path)
    files = [synthetic_features.paths[0], synthetic_features.paths[5]]
    table = predict_inputs(files, bundle)
    assert table.columns.tolist() == ["path", "predicted"]
    expected = model.predict(synthetic_features.X[[0, 5]])
    assert np.allclose(table["predicted"], expected)


def test_classification_predictions_have_scores(tmp_path, synthetic_dataset, classification_config):
    features = prepare_features(synthetic_dataset, classification_config)
    assert set(features.y) == {"healthy", "dysphonic"}
    model = train_model(features, classification_config)
    path = save_model(tmp_path / "m.joblib", model, classification_config, ["dysphonic", "healthy"])
    table = predict_inputs(features.paths[:3].tolist(), load_model(path))
    assert table.columns.tolist() == ["path", "predicted", "score_dysphonic", "score_healthy"]


def test_too_short_recordings_are_all_reported(tmp_path, fast_config):
    for name in ("a", "b"):
        write_wav(tmp_path / f"{name}.wav", np.zeros(100), 16_000)
    with pytest.raises(DatasetError, match=r"2 recording\(s\).*a\.wav.*b\.wav"):
        recording_frames([tmp_path / "a.wav", tmp_path / "b.wav"], fast_config)
    with pytest.raises(ValueError, match="shorter than one"):
        utterance_features(np.zeros(100), 16_000, fast_config)


def test_predict_rejects_missing_inputs(tmp_path, synthetic_features, fast_config):
    model = train_model(synthetic_features, fast_config)
    bundle = load_model(save_model(tmp_path / "m.joblib", model, fast_config))
    with pytest.raises(FileNotFoundError):
        predict_inputs([tmp_path / "nothing.wav"], bundle)
    with pytest.raises(ValueError, match="no input"):
        predict_inputs([], bundle)
