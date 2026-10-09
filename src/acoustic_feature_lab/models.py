"""Estimator registry, nested hyperparameter tuning and model files.

Every model is a scikit-learn :class:`~sklearn.pipeline.Pipeline`

``scale`` (StandardScaler) -> ``reduce`` (none, PCA or LDA) -> ``model``

so standardization and dimensionality reduction are learned on the training
fold only, never on the data they are evaluated on. Regression estimators are
ordinary least squares, ridge (Hoerl & Kennard, 1970), support vector
regression (Drucker et al., 1997), random forests (Breiman, 2001) and a
multilayer perceptron; classification estimators are logistic regression,
a support vector classifier, random forests and a perceptron with a logistic
hidden layer (all from scikit-learn, Pedregosa et al., 2011), plus a PyTorch
perceptron for either task (``dl`` extra). Support vector regression and the
regression perceptron predict a standardized target, so ``svr_epsilon`` is in
standard deviations of the rating. Linear discriminant analysis (Fisher, 1936)
keeps at most ``n_classes - 1`` directions; asking for more is reported as a
configuration error before anything is fitted.

With ``model.tune`` each training fold runs its own inner cross-validation
over a small grid (:data:`SEARCH_SPACES`) and refits the best setting, so the
outer folds measure the whole selection procedure (Varma & Simon, 2006).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .errors import ConfigError, ModelFileError, require_module
from .neural import TorchMLPClassifier, TorchMLPRegressor
from .splitting import split_indices

if TYPE_CHECKING:
    from .config import Config, ModelConfig

_LOGGER = logging.getLogger(__name__)

#: Hidden activations accepted by ``model.activation``.
ACTIVATIONS: tuple[str, ...] = ("logistic", "relu", "tanh")
#: Dimensionality reductions accepted by ``model.reducer``.
REDUCERS: tuple[str, ...] = ("none", "pca", "lda")
#: Value of the ``format`` key of every model file written here.
MODEL_FORMAT: str = "acoustic_feature_lab/model-v1"
#: Share of variance kept by PCA when ``model.n_components`` is null.
_PCA_VARIANCE: float = 0.95


def _linear(config: ModelConfig, seed: int) -> BaseEstimator:
    return LinearRegression()


def _ridge(config: ModelConfig, seed: int) -> BaseEstimator:
    return Ridge(alpha=config.ridge_alpha)


def _svr(config: ModelConfig, seed: int) -> BaseEstimator:
    return TransformedTargetRegressor(
        regressor=SVR(C=config.svr_c, epsilon=config.svr_epsilon), transformer=StandardScaler()
    )


def _forest_regressor(config: ModelConfig, seed: int) -> BaseEstimator:
    return RandomForestRegressor(
        n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=seed, n_jobs=1
    )


def _mlp_regressor(config: ModelConfig, seed: int) -> BaseEstimator:
    return TransformedTargetRegressor(
        regressor=MLPRegressor(
            hidden_layer_sizes=tuple(config.hidden_units),
            activation=config.activation,
            alpha=config.mlp_alpha,
            learning_rate_init=config.learning_rate,
            max_iter=config.max_iter,
            early_stopping=config.early_stopping,
            n_iter_no_change=config.patience,
            random_state=seed,
        ),
        transformer=StandardScaler(),
    )


def _torch_regressor(config: ModelConfig, seed: int) -> BaseEstimator:
    return TorchMLPRegressor(
        hidden_units=tuple(config.hidden_units),
        activation=config.activation,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.mlp_alpha,
        random_state=seed,
    )


def _logistic(config: ModelConfig, seed: int) -> BaseEstimator:
    return LogisticRegression(C=config.logistic_c, max_iter=config.max_iter)


def _svc(config: ModelConfig, seed: int) -> BaseEstimator:
    return SVC(C=config.svc_c, probability=True, random_state=seed)


def _forest_classifier(config: ModelConfig, seed: int) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=seed, n_jobs=1
    )


class CodedMLPClassifier(MLPClassifier):
    """Multilayer perceptron classifier that trains on integer class codes.

    Behaves like :class:`sklearn.neural_network.MLPClassifier` (same
    parameters, ``classes_`` holds the original names) but encodes the labels
    before fitting, because the early-stopping score of some scikit-learn
    releases fails on string labels.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> CodedMLPClassifier:
        """Fit on the codes of the sorted class names."""
        names, codes = np.unique(np.asarray(y).astype(str), return_inverse=True)
        super().fit(X, codes)
        self.class_names_ = names
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Class names of the most probable classes."""
        return self.class_names_[np.asarray(super().predict(X), dtype=np.intp)]

    @property
    def classes_(self) -> np.ndarray:
        """Class names, in the column order of :meth:`predict_proba`."""
        return self.class_names_

    @classes_.setter
    def classes_(self, value: np.ndarray) -> None:
        # The parent assigns its integer codes here during fitting; they stay internal.
        self._codes = value


def _mlp_classifier(config: ModelConfig, seed: int) -> BaseEstimator:
    return CodedMLPClassifier(
        hidden_layer_sizes=tuple(config.hidden_units),
        activation=config.activation,
        alpha=config.mlp_alpha,
        learning_rate_init=config.learning_rate,
        max_iter=config.max_iter,
        early_stopping=config.early_stopping,
        n_iter_no_change=config.patience,
        random_state=seed,
    )


def _torch_classifier(config: ModelConfig, seed: int) -> BaseEstimator:
    return TorchMLPClassifier(
        hidden_units=tuple(config.hidden_units),
        activation=config.activation,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.mlp_alpha,
        random_state=seed,
    )


_Factory = Callable[["ModelConfig", int], BaseEstimator]

_REGRESSORS: dict[str, _Factory] = {
    "linear": _linear,
    "ridge": _ridge,
    "svr": _svr,
    "random_forest": _forest_regressor,
    "mlp": _mlp_regressor,
    "torch_mlp": _torch_regressor,
}
_CLASSIFIERS: dict[str, _Factory] = {
    "logistic": _logistic,
    "svc": _svc,
    "random_forest": _forest_classifier,
    "mlp": _mlp_classifier,
    "torch_mlp": _torch_classifier,
}
#: Estimators accepted for ``data.task = "regression"``.
REGRESSION_MODELS: tuple[str, ...] = tuple(_REGRESSORS)
#: Estimators accepted for ``data.task = "classification"``.
CLASSIFICATION_MODELS: tuple[str, ...] = tuple(_CLASSIFIERS)
#: Model names accepted by :func:`build_model` and ``model.name``.
MODEL_NAMES: tuple[str, ...] = tuple(dict.fromkeys((*_REGRESSORS, *_CLASSIFIERS)))

#: Hyperparameter grids searched by the inner loop when ``model.tune`` is true.
SEARCH_SPACES: dict[tuple[str, str], dict[str, list[Any]]] = {
    ("regression", "ridge"): {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ("regression", "svr"): {
        "model__regressor__C": [0.1, 1.0, 10.0],
        "model__regressor__epsilon": [0.05, 0.1, 0.5],
    },
    ("regression", "random_forest"): {
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 3],
    },
    ("regression", "mlp"): {"model__regressor__alpha": [1e-4, 1e-2, 1.0]},
    ("classification", "logistic"): {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ("classification", "svc"): {"model__C": [0.1, 1.0, 10.0]},
    ("classification", "random_forest"): {
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 3],
    },
    ("classification", "mlp"): {"model__alpha": [1e-4, 1e-2, 1.0]},
}
#: Inner-loop score of the hyperparameter search (higher is better).
_SEARCH_SCORING: dict[str, str] = {
    "regression": "neg_root_mean_squared_error",
    "classification": "balanced_accuracy",
}


def models_for_task(task: str) -> tuple[str, ...]:
    """Model names that fit a task.

    Examples
    --------
    >>> models_for_task("classification")
    ('logistic', 'svc', 'random_forest', 'mlp', 'torch_mlp')
    """
    if task == "regression":
        return REGRESSION_MODELS
    if task == "classification":
        return CLASSIFICATION_MODELS
    raise ConfigError(
        f"data.task={task!r} is not supported; choose 'regression' or 'classification'"
    )


def _reducer(config: ModelConfig) -> BaseEstimator | str:
    if config.reducer == "pca":
        components = _PCA_VARIANCE if config.n_components is None else config.n_components
        return PCA(n_components=components, svd_solver="full")
    if config.reducer == "lda":
        return LinearDiscriminantAnalysis(n_components=config.n_components)
    return "passthrough"


def build_model(config: Config) -> BaseEstimator:
    """Untrained pipeline for ``config.model`` and ``config.data.task``.

    Parameters
    ----------
    config : Config
        Run configuration; ``config.seed`` seeds every randomized estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps ``scale``, ``reduce`` and ``model``. The PyTorch perceptron
        standardizes internally, so its pipeline has the single step ``model``.

    Raises
    ------
    ConfigError
        If the model does not fit the task.

    Examples
    --------
    >>> from acoustic_feature_lab.config import Config
    >>> pipeline = build_model(Config().override({"model.name": "svr", "model.reducer": "pca"}))
    >>> [name for name, _ in pipeline.steps], type(pipeline[-1].regressor).__name__
    (['scale', 'reduce', 'model'], 'SVR')
    """
    task = config.data.task
    registry = _REGRESSORS if task == "regression" else _CLASSIFIERS
    name = config.model.name
    if name not in registry:
        raise ConfigError(
            f"model.name={name!r} does not fit data.task={task!r}; choose one of {list(registry)}"
        )
    estimator = registry[name](config.model, config.seed)
    if name == "torch_mlp":
        return Pipeline([("model", estimator)])
    return Pipeline(
        [("scale", StandardScaler()), ("reduce", _reducer(config.model)), ("model", estimator)]
    )


def search_space(config: Config) -> dict[str, list[Any]]:
    """Hyperparameter grid of the configured model (empty when nothing is tuned).

    Examples
    --------
    >>> from acoustic_feature_lab.config import Config
    >>> search_space(Config())
    {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    """
    grid = SEARCH_SPACES.get((config.data.task, config.model.name), {})
    return {key: list(values) for key, values in grid.items()}


def _check_reducer(config: Config, y: np.ndarray, n_features: int) -> None:
    if config.model.reducer != "lda":
        return
    n_classes = np.unique(y.astype(str)).size
    limit = min(n_classes - 1, n_features)
    requested = config.model.n_components
    if requested is not None and requested > limit:
        raise ConfigError(
            f"model.n_components={requested} exceeds what LDA can extract from {n_classes} "
            f"classes and {n_features} features (at most {limit} discriminant directions)"
        )


def fit_model(
    X: ArrayLike, y: ArrayLike, config: Config, *, groups: ArrayLike | None = None
) -> BaseEstimator:
    """Fit the configured pipeline, tuning it by inner cross-validation if asked.

    Parameters
    ----------
    X : array_like, shape (n_items, n_features)
        Training features.
    y : array_like, shape (n_items,)
        Training targets.
    config : Config
        Run configuration (``model``, ``data.task``, ``evaluation.n_inner_splits``, ``seed``).
    groups : array_like, shape (n_items,), optional
        Speakers; the inner folds never split one.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline (the refitted best one when tuning).

    Examples
    --------
    >>> from acoustic_feature_lab.config import Config
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 3))
    >>> y = X @ np.array([1.0, 0.0, -2.0]) + rng.normal(0, 0.1, 40)
    >>> tuned = fit_model(X, y, Config().override({"model.tune": True}))
    >>> float(tuned[-1].alpha) in (0.01, 0.1, 1.0)
    True
    """
    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y)
    if config.data.task == "regression":
        targets = targets.astype(np.float64)
    else:
        targets = targets.astype(str)
    _check_reducer(config, targets, features.shape[1])
    pipeline = build_model(config)
    grid = search_space(config) if config.model.tune else {}
    if not grid:
        if config.model.tune:
            _LOGGER.info(
                "model %r has no hyperparameter grid; fitting it as configured", config.model.name
            )
        return pipeline.fit(features, targets)
    folds = split_indices(
        targets,
        groups,
        task=config.data.task,
        n_splits=config.evaluation.n_inner_splits,
        random_state=config.seed,
    )
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=_SEARCH_SCORING[config.data.task],
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(features, targets)
    _LOGGER.debug("inner search chose %s", search.best_params_)
    return search.best_estimator_


def model_filename(config: Config) -> str:
    """``model.pt`` for the PyTorch perceptron, ``model.joblib`` for everything else."""
    return "model.pt" if config.model.name == "torch_mlp" else "model.joblib"


def save_model(
    path: str | Path,
    estimator: Any,
    config: Config,
    classes: Sequence[str] | None = None,
) -> Path:
    """Write a trained pipeline with the configuration that produced it.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file (``model.joblib``, or ``model.pt`` for ``torch_mlp``).
    estimator : sklearn.pipeline.Pipeline
        Trained pipeline.
    config : Config
        Complete configuration; ``predict`` extracts features with it.
    classes : sequence of str, optional
        Class names (classification only).

    Returns
    -------
    pathlib.Path
        The written file.
    """
    from . import __version__  # deferred: the package __init__ imports this module

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "package_version": __version__,
        "config": config.to_dict(),
    }
    if classes is not None:
        bundle["classes"] = [str(name) for name in classes]
    if config.model.name == "torch_mlp":
        torch = require_module("torch", extra="dl", feature="saving the PyTorch perceptron")
        bundle["state_dict"] = estimator[-1].network_state()
        torch.save(bundle, destination)
    else:
        bundle["estimator"] = estimator
        joblib.dump(bundle, destination)
    _LOGGER.info("model written to %s", destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model file written by :func:`save_model`.

    Only open model files from sources you trust: joblib and PyTorch files
    written with pickle can execute code when loaded.

    Parameters
    ----------
    path : str or pathlib.Path
        ``model.joblib`` or ``model.pt``.

    Returns
    -------
    dict
        Keys ``format``, ``package_version``, ``config``, ``classes`` (when
        present) and ``estimator`` (a ready-to-use pipeline, rebuilt from the
        ``state_dict`` for PyTorch models).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ModelFileError
        If the file was not written by this package or is incomplete.
    """
    from .config import Config

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"model file not found: {source}. Run 'acoustic-feature-lab train' first"
        )
    if source.suffix == ".pt":
        torch = require_module("torch", extra="dl", feature="loading a PyTorch model")
        bundle = torch.load(source, map_location="cpu", weights_only=True)
    else:
        try:
            bundle = joblib.load(source)
        except Exception as exc:  # joblib raises many unrelated types for foreign files
            raise ModelFileError(f"{source} is not a readable model file: {exc}") from exc
    if not isinstance(bundle, Mapping) or bundle.get("format") != MODEL_FORMAT:
        raise ModelFileError(
            f"{source} was not written by acoustic-feature-lab (expected format {MODEL_FORMAT!r})"
        )
    missing = [key for key in ("config", "package_version") if key not in bundle]
    if missing:
        raise ModelFileError(f"{source} lacks {missing}")
    loaded = dict(bundle)
    config = Config.from_dict(loaded["config"])
    if "state_dict" in loaded:
        pipeline = build_model(config)
        pipeline[-1].load_network_state(loaded.pop("state_dict"), loaded.get("classes"))
        loaded["estimator"] = pipeline
    elif "estimator" not in loaded:
        raise ModelFileError(f"{source} holds no trained estimator")
    if config.data.task == "classification" and "classes" not in loaded:
        raise ModelFileError(f"{source} is a classification model without its class names")
    return loaded
