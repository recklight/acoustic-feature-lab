"""Dataset index, feature files, index building and analysis tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from speechdsp import write_wav

from acoustic_feature_lab import (
    Config,
    ConfigError,
    DatasetError,
    FeatureSet,
    build_dataset_index,
    load_analysis_table,
    load_dataset_index,
    load_features,
    make_vowel,
    save_features,
)


@pytest.fixture
def small_index(tmp_path):
    for name in ("a", "b", "c"):
        write_wav(tmp_path / "audio" / f"{name}.wav", make_vowel(20.0, rng=0)[:4000], 16_000)
    index = tmp_path / "audio" / "dataset.csv"
    index.write_text(
        "path,severity,label,group\na.wav,10,healthy,s1\nb.wav,55.5,dysphonic,s2\nc.wav,80,dysphonic,\n",
        encoding="utf-8",
    )
    return index


def test_paths_resolve_relative_to_the_index_folder(small_index):
    table = load_dataset_index(small_index, target="severity")
    assert table["path"].str.endswith("/audio/a.wav").iloc[0]
    assert table["severity"].dtype == np.float64
    assert table["group"].tolist() == ["s1", "s2", ""]


def test_index_without_group_column_gets_empty_groups(tmp_path, small_index):
    frame = pd.read_csv(small_index).drop(columns="group")
    frame.to_csv(small_index, index=False)
    assert load_dataset_index(small_index)["group"].tolist() == ["", "", ""]


def test_missing_columns_are_listed(small_index):
    with pytest.raises(DatasetError, match=r"lacks column\(s\) \['cape_v'\]; found"):
        load_dataset_index(small_index, target="cape_v")


def test_missing_files_are_listed(tmp_path):
    index = tmp_path / "dataset.csv"
    rows = "\n".join(f"missing_{i}.wav,{i}" for i in range(8))
    index.write_text(f"path,severity\n{rows}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"8 file\(s\).*and 3 more"):
        load_dataset_index(index, target="severity")


def test_non_numeric_ratings_are_rejected_for_regression(small_index):
    with pytest.raises(DatasetError, match="must be numeric"):
        load_dataset_index(small_index, target="label", task="regression")
    labels = load_dataset_index(small_index, target="label", task="classification")["label"]
    assert labels.tolist() == ["healthy", "dysphonic", "dysphonic"]


def test_a_file_listed_twice_is_rejected(small_index):
    small_index.write_text(
        "path,severity,group\na.wav,10,s1\nb.wav,20,s2\n./a.wav,80,s3\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match=r"more than once: a\.wav \(rows 2, 4\)"):
        load_dataset_index(small_index, target="severity")


def test_absent_index_explains_the_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthesize"):
        load_dataset_index(tmp_path / "nowhere.csv")


def test_feature_file_round_trip_and_setting_check(tmp_path):
    config = Config()
    features = FeatureSet(
        X=np.arange(6.0).reshape(3, 2),
        y=np.array([1.0, 2.0, 3.0]),
        groups=np.array(["a", "b", ""]),
        paths=np.array(["x.wav", "y.wav", "z.wav"]),
        config=config.to_dict(),
    )
    path = save_features(tmp_path / "features.npz", features)
    loaded = load_features(path, config)
    assert np.array_equal(loaded.X, features.X) and loaded.groups.tolist() == ["a", "b", ""]
    assert loaded.config == config.to_dict()
    assert load_features(path, config.override({"model.name": "svr"})).X.shape == (3, 2)
    with pytest.raises(ConfigError, match=r"frontend\.n_mels"):
        load_features(path, config.override({"frontend.n_mels": 30}))
    with pytest.raises(ConfigError, match=r"data\.target"):
        load_features(path, config.override({"data.target": "grade"}))
    assert len(loaded.feature_names) == 78


def test_build_dataset_index_matches_stems_and_relative_paths(tmp_path):
    audio = tmp_path / "audio"
    for relative in ("visit1/p1.wav", "visit1/p2.wav", "visit2/p1.wav", "other/q.wav"):
        write_wav(audio / relative, np.zeros(800), 16_000)
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("id,severity\nvisit1/p1,10\nvisit2/p1,20\np2,30\n", encoding="utf-8")
    index = pd.read_csv(build_dataset_index(audio, ratings))
    assert index["path"].tolist() == ["visit1/p1.wav", "visit2/p1.wav", "visit1/p2.wav"]
    ratings.write_text("id,severity\np1,10\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="several files"):
        build_dataset_index(audio, ratings)
    ratings.write_text("id,severity\nzz,10\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no audio file"):
        build_dataset_index(audio, ratings)


def test_two_identifiers_for_one_file_are_rejected(tmp_path):
    # "p001" matches by stem and "sub/p001" by relative path: one recording with
    # two ratings in two groups would sit on both sides of a fold.
    audio = tmp_path / "audio"
    write_wav(audio / "sub" / "p001.wav", np.zeros(800), 16_000)
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("id,severity,group\np001,10,a\nsub/p001,80,b\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"same audio file: sub/p001\.wav \(p001, sub/p001\)"):
        build_dataset_index(audio, ratings)
    assert not (audio / "dataset.csv").exists()


def test_analysis_table_from_csv_and_npz(tmp_path, synthetic_features):
    table = tmp_path / "table.csv"
    table.write_text("a,b,label,y\n1,2,x,3\n2,1,x,5\n3,5,y,7\n4,3,y,\n", encoding="utf-8")
    X, y = load_analysis_table(table, target="y")
    assert X.columns.tolist() == ["a", "b"] and y.tolist() == [3.0, 5.0, 7.0]
    with pytest.raises(ConfigError, match="--target"):
        load_analysis_table(table)
    with pytest.raises(DatasetError, match="unknown predictor"):
        load_analysis_table(table, target="y", predictors=["c"])
    path = save_features(tmp_path / "f.npz", synthetic_features)
    X, y = load_analysis_table(path, predictors=["mean_c1", "std_c0"])
    assert X.columns.tolist() == ["mean_c1", "std_c0"] and y.name == "severity"
