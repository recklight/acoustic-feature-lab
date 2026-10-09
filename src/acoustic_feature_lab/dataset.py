"""Dataset index files, utterance-level feature files and analysis tables.

A dataset is described by one CSV index, ``dataset.csv``:

=========  ========  ======================================================
column     required  meaning
=========  ========  ======================================================
path       yes       audio file, relative to the folder holding the CSV
<target>   yes       the rating (regression) or class name (classification)
group      no        speaker or session; folds never split a group
=========  ========  ======================================================

Targets and groups are read only from these columns, never parsed from file names.
``prepare`` turns the index into ``features.npz`` (one pooled feature vector
per recording); ``regress``, ``stepwise``, ``order-sweep`` and ``correlate``
read either such a file or any CSV table of numeric columns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import FEATURE_SECTIONS, Config
from .corpus import iter_audio_files
from .errors import ConfigError, DatasetError
from .pooling import pooled_feature_names

_LOGGER = logging.getLogger(__name__)

#: File name of a dataset index.
DATASET_INDEX_NAME: str = "dataset.csv"
#: ``data`` keys that shape ``features.npz`` besides the feature sections.
_TARGET_KEYS: tuple[str, ...] = ("task", "target")
#: Items an error message lists before "... and N more".
_LISTED = 5


def _listing(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_LISTED])
    extra = len(items) - _LISTED
    return shown + (f" ... and {extra} more" if extra > 0 else "")


def load_dataset_index(
    path: str | Path, *, target: str | None = None, task: str = "regression"
) -> pd.DataFrame:
    """Read and validate a dataset index.

    Parameters
    ----------
    path : str or pathlib.Path
        The CSV file.
    target : str, optional
        Column that must be present and complete; numeric for regression.
    task : {"regression", "classification"}, optional
        How the target column is checked.

    Returns
    -------
    pandas.DataFrame
        The index with ``path`` resolved to absolute POSIX paths and a
        ``group`` column (empty strings when the file has none).

    Raises
    ------
    FileNotFoundError
        If the index itself does not exist.
    DatasetError
        For missing columns, empty or non-numeric targets, missing audio files,
        or an audio file listed more than once (one recording in two rows could
        end up on both sides of a fold).
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"dataset index not found: {source}. Point data.dataset (or --dataset) at your own "
            "dataset.csv, or run 'acoustic-feature-lab synthesize' first"
        )
    try:
        table = pd.read_csv(source, dtype={"path": str, "group": str}, keep_default_na=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{source} is not a readable CSV file: {exc}") from None
    required = ["path"] + ([target] if target else [])
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise DatasetError(f"{source} lacks column(s) {missing}; found {list(table.columns)}")
    if table.empty:
        raise DatasetError(f"{source} lists no recordings")
    if table["path"].isna().any():
        raise DatasetError(f"{source} has rows without a path")
    if target:
        empty = table.index[table[target].isna()].tolist()
        if empty:
            raise DatasetError(
                f"{source}: column {target!r} is empty in row(s) "
                f"{_listing([str(i + 2) for i in empty])}"
            )
        if task == "regression":
            numeric = pd.to_numeric(table[target], errors="coerce")
            bad = table.index[numeric.isna()].tolist()
            if bad:
                raise DatasetError(
                    f"{source}: column {target!r} must be numeric for regression; "
                    f"bad value(s) in row(s) {_listing([str(i + 2) for i in bad])}"
                )
            table[target] = numeric.astype(np.float64)
        else:
            table[target] = table[target].astype(str)
    table["group"] = table["group"].fillna("").astype(str) if "group" in table else ""
    folder = source.resolve().parent
    resolved = [(folder / Path(str(item))).resolve() for item in table["path"]]
    absent = [
        str(item) for item, file in zip(table["path"], resolved, strict=True) if not file.is_file()
    ]
    if absent:
        raise DatasetError(
            f"{len(absent)} file(s) listed in {source} do not exist: {_listing(absent)}"
        )
    rows_of: dict[Path, list[int]] = {}
    for row, file in enumerate(resolved, start=2):
        rows_of.setdefault(file, []).append(row)
    repeated = [
        f"{table['path'].iloc[rows[0] - 2]} (rows {', '.join(map(str, rows))})"
        for rows in rows_of.values()
        if len(rows) > 1
    ]
    if repeated:
        raise DatasetError(
            f"{source} lists the same audio file more than once: {_listing(repeated)}"
        )
    table["path"] = [file.as_posix() for file in resolved]
    return table


@dataclass(frozen=True, eq=False)
class FeatureSet:
    """Utterance-level model inputs, one row per recording.

    Attributes
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
        Pooled features.
    y : numpy.ndarray, shape (n_items,)
        Ratings (float) or class names (str).
    groups : numpy.ndarray, shape (n_items,)
        Speaker of each recording (empty strings when ungrouped).
    paths : numpy.ndarray, shape (n_items,)
        Audio file of each row.
    config : dict
        The complete configuration ``prepare`` used (``Config.to_dict()``).
    """

    X: NDArray[np.float64]
    y: NDArray[Any]
    groups: NDArray[np.str_]
    paths: NDArray[np.str_]
    config: dict[str, Any]

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Column names of ``X``, derived from the stored configuration."""
        stored = Config.from_dict(self.config)
        return pooled_feature_names(stored.frontend, stored.pooling)


def save_features(path: str | Path, features: FeatureSet) -> Path:
    """Write ``features.npz``: keys ``X``, ``y``, ``groups``, ``paths`` and ``config_json``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        X=np.asarray(features.X, dtype=np.float64),
        y=np.asarray(features.y),
        groups=np.asarray(features.groups).astype(str),
        paths=np.asarray(features.paths).astype(str),
        config_json=np.asarray(json.dumps(features.config, sort_keys=True)),
    )
    return destination


def _differences(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    keys = []
    for section in FEATURE_SECTIONS:
        old, new = stored.get(section, {}), current.get(section, {})
        keys.extend(
            f"{section}.{key}"
            for key in sorted(set(old) | set(new))
            if old.get(key) != new.get(key)
        )
    old_data, new_data = stored.get("data", {}), current.get("data", {})
    keys.extend(f"data.{key}" for key in _TARGET_KEYS if old_data.get(key) != new_data.get(key))
    return keys


def load_features(path: str | Path, config: Config | None = None) -> FeatureSet:
    """Read ``features.npz``; with ``config``, insist on the same feature settings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ConfigError
        If ``config`` differs from the settings ``prepare`` used in
        :data:`~acoustic_feature_lab.config.FEATURE_SECTIONS` or in
        ``data.task`` / ``data.target``.
    DatasetError
        If the file is not a feature file of this package.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"feature file not found: {source}. Run 'acoustic-feature-lab prepare' first"
        )
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{source} is not a readable .npz file: {exc}") from None
    needed = {"X", "y", "groups", "paths", "config_json"}
    if not needed <= set(arrays):
        raise DatasetError(
            f"{source} lacks {sorted(needed - set(arrays))}; was it written by 'prepare'?"
        )
    stored = json.loads(str(arrays["config_json"]))
    if config is not None:
        changed = _differences(stored, config.to_dict())
        if changed:
            raise ConfigError(
                f"feature settings differ from the ones 'prepare' used: {', '.join(changed)}; "
                "run 'prepare' again or use the same --config"
            )
    return FeatureSet(
        X=arrays["X"].astype(np.float64),
        y=arrays["y"],
        groups=arrays["groups"].astype(str),
        paths=arrays["paths"].astype(str),
        config=stored,
    )


def build_dataset_index(
    audio_root: str | Path,
    targets: str | Path,
    *,
    id_column: str = "id",
) -> Path:
    """Write ``<audio_root>/dataset.csv`` by matching audio files to a ratings table.

    Every row of ``targets`` is matched to the audio file whose path relative
    to ``audio_root`` without the suffix, or whose bare file stem, equals the
    row's ``id``. All other columns of ``targets`` (ratings, ``label``,
    ``group``, ...) are copied.

    Parameters
    ----------
    audio_root : str or pathlib.Path
        Folder searched recursively for ``.wav`` files.
    targets : str or pathlib.Path
        CSV with an ``id`` column and the ratings.
    id_column : str, optional
        Name of the identifier column.

    Returns
    -------
    pathlib.Path
        The written index.

    Raises
    ------
    DatasetError
        For duplicate identifiers, ambiguous stems, unmatched rows, or two
        identifiers that match the same file (one by relative path, the other
        by stem).
    """
    root = Path(audio_root)
    if not root.is_dir():
        raise FileNotFoundError(f"audio folder not found: {root}")
    table_path = Path(targets)
    if not table_path.is_file():
        raise FileNotFoundError(f"ratings table not found: {table_path}")
    table = pd.read_csv(table_path, dtype={id_column: str})
    if id_column not in table.columns:
        raise DatasetError(
            f"{table_path} lacks the {id_column!r} column; found {list(table.columns)}"
        )
    duplicated = table[id_column][table[id_column].duplicated()].tolist()
    if duplicated:
        raise DatasetError(f"{table_path} repeats identifier(s) {_listing(duplicated)}")
    files = iter_audio_files(root)
    by_relative = {file.relative_to(root).with_suffix("").as_posix(): file for file in files}
    by_stem: dict[str, list[Path]] = {}
    for file in files:
        by_stem.setdefault(file.stem, []).append(file)
    matched, unmatched = [], []
    claimed: dict[Path, list[str]] = {}
    for identifier in table[id_column]:
        key = str(identifier)
        if key in by_relative:
            matched.append(by_relative[key])
        elif len(by_stem.get(key, [])) == 1:
            matched.append(by_stem[key][0])
        elif key in by_stem:
            raise DatasetError(f"identifier {key!r} matches several files: {by_stem[key]}")
        else:
            unmatched.append(key)
            continue
        claimed.setdefault(matched[-1], []).append(key)
    if unmatched:
        raise DatasetError(
            f"{len(unmatched)} identifier(s) in {table_path} have no audio file under {root}: "
            f"{_listing(unmatched)}"
        )
    shared = [
        f"{file.relative_to(root).as_posix()} ({', '.join(keys)})"
        for file, keys in claimed.items()
        if len(keys) > 1
    ]
    if shared:
        raise DatasetError(
            f"identifiers in {table_path} match the same audio file: {_listing(shared)}"
        )
    unused = len(files) - len(set(matched))
    if unused:
        _LOGGER.warning("%d audio file(s) under %s have no row in %s", unused, root, table_path)
    index = table.drop(columns=[id_column])
    index.insert(0, "path", [file.relative_to(root).as_posix() for file in matched])
    destination = root / DATASET_INDEX_NAME
    index.to_csv(destination, index=False, lineterminator="\n")
    return destination


def load_analysis_table(
    path: str | Path,
    *,
    target: str | None = None,
    predictors: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Predictors and response for the regression-analysis commands.

    Parameters
    ----------
    path : str or pathlib.Path
        ``features.npz`` from ``prepare`` (the response is its target), or a
        CSV table (``target`` names the response column).
    target : str, optional
        Response column of a CSV table; ignored for ``.npz`` files.
    predictors : sequence of str, optional
        Columns to use; default every numeric column except the response.

    Returns
    -------
    X : pandas.DataFrame
        Predictor columns.
    y : pandas.Series
        Numeric response.

    Raises
    ------
    DatasetError
        For unknown columns, a non-numeric response or too few complete rows.
    """
    source = Path(path)
    if source.suffix == ".npz":
        features = load_features(source)
        frame = pd.DataFrame(features.X, columns=list(features.feature_names))
        response_name = str(features.config.get("data", {}).get("target", "target"))
        try:
            response = pd.Series(features.y.astype(np.float64), name=response_name)
        except ValueError:
            raise DatasetError(
                f"{source} holds class labels; regression analysis needs a numeric rating"
            ) from None
    else:
        if not source.is_file():
            raise FileNotFoundError(f"table not found: {source}")
        frame = pd.read_csv(source)
        if not target:
            raise ConfigError("a CSV table needs --target to name the response column")
        if target not in frame.columns:
            raise DatasetError(f"{source} has no column {target!r}; found {list(frame.columns)}")
        response = pd.to_numeric(frame.pop(target), errors="coerce").rename(target)
    if predictors:
        unknown = [name for name in predictors if name not in frame.columns]
        if unknown:
            raise DatasetError(
                f"unknown predictor(s) {unknown}; available: {list(frame.columns)[:20]}"
            )
        frame = frame[list(predictors)]
    else:
        frame = frame.select_dtypes(include=[np.number])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    complete = frame.notna().all(axis=1) & response.notna()
    dropped = int((~complete).sum())
    if dropped:
        _LOGGER.warning("%d row(s) with missing or non-numeric values were dropped", dropped)
    frame, response = (
        frame[complete].reset_index(drop=True),
        response[complete].reset_index(drop=True),
    )
    if frame.shape[1] == 0:
        raise DatasetError(f"{source} has no numeric predictor columns")
    if len(response) < 3:
        raise DatasetError(f"{source} has only {len(response)} complete row(s)")
    return frame, response
