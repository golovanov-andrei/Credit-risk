import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURES, NUMERIC, normalize_features, synthetic_target
from credit_risk.model import (
    explain_one, load_artifact, predict_frame, score_band, select_threshold,
)
from conftest import ROOT


def sample(raw_data):
    return raw_data.head(1).copy()


def test_rule_boundaries(raw_data):
    frame = pd.concat([sample(raw_data)] * 4, ignore_index=True)
    frame["Credit amount"] = [8000, 8001, 9000, 9000]
    frame["Duration"] = [36, 25, 24, 36]
    frame["Checking account"] = ["little", "little", "little", None]
    assert synthetic_target(normalize_features(frame)).tolist() == [0, 1, 0, 0]


def test_input_and_missing_account_preserved(raw_data):
    row = sample(raw_data)
    row["Saving accounts"] = None
    row["Checking account"] = None
    original = row.copy(deep=True)
    normalized = normalize_features(row)
    pd.testing.assert_frame_equal(row, original)
    assert list(normalized.columns) == FEATURES
    assert normalized["Checking account"].isna().all()


@pytest.mark.parametrize("column,value", [
    ("Age", 17), ("Age", 20.5), ("Age", True), ("Age", None),
    ("Credit amount", 0), ("Credit amount", -1), ("Credit amount", float("inf")),
    ("Duration", 0), ("Duration", 12.5), ("Job", 4), ("Job", "unknown"),
    ("Housing", None), ("Purpose", "unknown station"),
])
def test_invalid_inputs_rejected(raw_data, column, value):
    row = sample(raw_data).astype(object)
    row[column] = value
    with pytest.raises(ValueError):
        normalize_features(row)


def test_missing_column_and_empty_batch(raw_data):
    with pytest.raises(ValueError, match="Missing columns"):
        normalize_features(raw_data.drop(columns="Purpose"))
    with pytest.raises(ValueError, match="At least one"):
        normalize_features(raw_data.head(0))


def test_preprocessing_fitted_only_on_train(raw_data, experiment):
    bundle, report, heldout = experiment
    splits = bundle["split_indices"]
    groups = [set(indices) for indices in splits.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert set.union(*groups) == set(range(len(raw_data)))
    assert len(heldout) == report["splits"]["test"]["rows"]
    scaler = bundle["pipeline"].named_steps["preprocess"].named_transformers_["numeric"]
    train_means = raw_data.loc[splits["train"], NUMERIC].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_means)
    assert not np.allclose(scaler.mean_, raw_data[NUMERIC].mean().to_numpy())


def test_training_and_inference_use_identical_features(raw_data, experiment):
    bundle, _, _ = experiment
    rows = raw_data.iloc[[0, 17, 22, 42, 70]].copy()
    batch = predict_frame(bundle, rows)
    for i, row_id in enumerate(rows.index):
        one = predict_frame(bundle, rows.loc[[row_id]])
        assert one.iloc[0]["score"] == pytest.approx(batch.iloc[i]["score"])
    expected = bundle["pipeline"].predict_proba(normalize_features(rows))[:, 1]
    np.testing.assert_allclose(batch["score"], expected)
    rows["Sex"] = "ignored for modeling"
    np.testing.assert_allclose(predict_frame(bundle, rows)["score"], expected)


def test_threshold_selected_from_validation(raw_data, experiment):
    bundle, report, _ = experiment
    X = normalize_features(raw_data)
    ids = bundle["split_indices"]["validation"]
    scores = bundle["pipeline"].predict_proba(X.loc[ids])[:, 1]
    expected = select_threshold(synthetic_target(X).loc[ids], scores)
    assert report["threshold"] == expected
    assert select_threshold([0, 1], [0.1, 0.9]) == 0.5


def test_exact_explanation_reconstructs_model_log_odds(raw_data, experiment):
    bundle, _, _ = experiment
    row = raw_data.iloc[[17]]
    baseline, table = explain_one(bundle, row)
    reconstructed = baseline + table["log_odds_contribution"].sum()
    expected = bundle["pipeline"].decision_function(normalize_features(row))[0]
    assert reconstructed == pytest.approx(expected, abs=1e-10)
    assert 1 / (1 + np.exp(-reconstructed)) == pytest.approx(
        predict_frame(bundle, row).iloc[0]["score"]
    )


def test_serialized_pipeline_keeps_predictions(raw_data, experiment, artifact_dir):
    loaded = load_artifact(artifact_dir / "model.joblib")
    np.testing.assert_allclose(
        predict_frame(loaded, raw_data.head(10))["score"],
        predict_frame(experiment[0], raw_data.head(10))["score"],
    )
    report = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["test"]["label_rule_reference"]["f1"] == 1
    assert sum(map(sum, report["test"]["logistic_regression"]["confusion_matrix"])) == 200


@pytest.mark.parametrize("score,band", [(0, "A"), (0.2, "B"), (0.5, "C"), (1, "C")])
def test_display_band_boundaries(score, band):
    assert score_band(score) == band


def test_cli_prediction_and_invalid_json(artifact_dir, tmp_path):
    command = [sys.executable, "-m", "credit_risk.cli", "predict",
               "--model", str(artifact_dir / "model.joblib")]
    result = subprocess.run(command + ["--input", str(ROOT / "examples/applicant.json")],
                            capture_output=True, text=True, check=True)
    prediction = json.loads(result.stdout)[0]
    assert 0 <= prediction["score"] <= 1
    assert prediction["rule_label"] == 1
    bad = tmp_path / "invalid.json"
    bad.write_text("[]", encoding="utf-8")
    result = subprocess.run(command + ["--input", str(bad)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Error:" in result.stderr


def test_training_repeat_is_deterministic(raw_data, experiment):
    from credit_risk.model import fit_experiment
    bundle, report, heldout = fit_experiment(raw_data)
    assert report == experiment[1]
    pd.testing.assert_frame_equal(heldout, experiment[2])

