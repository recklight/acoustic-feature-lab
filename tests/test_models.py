"""Model registry, reducers, nested tuning and model files."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from acoustic_feature_lab import (
    MODEL_NAMES,
    Config,
    ConfigError,
    ModelFileError,
    build_model,
    fit_model,
    load_model,
    save_model,
)
from acoustic_feature_lab.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    model_filename,
    search_space,
)


@pytest.fixture
def regression_data(rng):
    X = rng.normal(size=(60, 5))
    return X, X @ np.array([2.0, -1.0, 0.0, 0.5, 0.0]) * 10 + 50 + rng.normal(0, 2, 60)


@pytest.fixture
def class_data(rng):
    X = rng.normal(size=(60, 4))
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=60) > 0, "dysphonic", "healthy")
    return X, y


@pytest.mark.parametrize("name", [n for n in REGRESSION_MODELS if n != "torch_mlp"])
def test_every_regressor_fits_and_predicts(name, regression_data):
    X, y = regression_data
    config = Config().override({"model.name": name, "model.n_estimators": 20})
    model = fit_model(X, y, config)
    assert [step for step, _ in model.steps] == ["scale", "reduce", "model"]
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.8


@pytest.mark.parametrize("name", [n for n in CLASSIFICATION_MODELS if n != "torch_mlp"])
def test_every_classifier_fits_and_scores(name, class_data):
    X, y = class_data
    config = Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": name,
            "ablation.models": [],
            "model.n_estimators": 20,
        }
    )
    model = fit_model(X, y, config)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (60, 2) and np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) > 0.8


def test_names_cover_both_tasks():
    assert set(MODEL_NAMES) == set(REGRESSION_MODELS) | set(CLASSIFICATION_MODELS)
    with pytest.raises(ConfigError, match=r"data\.task"):
        build_model(Config().override({"model.name": "logistic"}))


def test_reducers_are_fitted_inside_the_pipeline(regression_data, class_data):
    X, y = regression_data
    pca = fit_model(X, y, Config().override({"model.reducer": "pca", "model.n_components": 2}))
    assert pca.named_steps["reduce"].n_components_ == 2
    Xc, yc = class_data
    lda_config = Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": "logistic",
            "model.reducer": "lda",
            "ablation.models": [],
        }
    )
    lda = fit_model(Xc, yc, lda_config)
    assert lda.named_steps["reduce"].transform(Xc).shape == (60, 1)
    with pytest.raises(ConfigError, match="discriminant directions"):
        fit_model(Xc, yc, lda_config.override({"model.n_components": 3}))


def test_tuning_searches_the_grid_on_inner_folds(regression_data):
    X, y = regression_data
    config = Config().override({"model.tune": True})
    assert search_space(config) == {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    groups = np.repeat([f"s{i}" for i in range(12)], 5)
    model = fit_model(X, y, config, groups=groups)
    assert model.named_steps["model"].alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
    untuned = fit_model(X, y, Config().override({"model.name": "linear", "model.tune": True}))
    assert untuned.named_steps["model"].__class__.__name__ == "LinearRegression"


def test_model_file_round_trip(tmp_path, regression_data):
    X, y = regression_data
    config = Config()
    model = fit_model(X, y, config)
    path = save_model(tmp_path / model_filename(config), model, config)
    bundle = load_model(path)
    assert bundle["format"] == "acoustic_feature_lab/model-v1"
    assert Config.from_dict(bundle["config"]) == config
    assert np.allclose(bundle["estimator"].predict(X), model.predict(X))


def test_foreign_and_missing_model_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_model(tmp_path / "absent.joblib")
    joblib.dump({"format": "something-else"}, tmp_path / "other.joblib")
    with pytest.raises(ModelFileError, match="not written by"):
        load_model(tmp_path / "other.joblib")
    (tmp_path / "junk.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ModelFileError, match="not a readable"):
        load_model(tmp_path / "junk.joblib")
    assert model_filename(Config().override({"model.name": "torch_mlp"})) == "model.pt"
