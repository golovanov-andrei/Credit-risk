"""Train/validation/test experiment, artifact and exact linear explanations."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import (
    CATEGORICAL, FEATURES, NUMERIC, TARGET_DESCRIPTION,
    normalize_features, synthetic_target,
)

SEED = 42
SCHEMA_VERSION = 1


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=SEED,
        )),
    ])


def metrics(y_true, scores, threshold: float) -> dict:
    labels = (np.asarray(scores) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]).tolist(),
    }


def select_threshold(y_validation, validation_scores) -> float:
    """Choose F1 on validation only; prefer the closest to 0.5 in a tie."""
    candidates = np.round(np.arange(0.05, 1.0, 0.05), 2)
    return float(max(candidates, key=lambda t: (
        f1_score(y_validation, np.asarray(validation_scores) >= t, zero_division=0),
        -abs(float(t) - 0.5), float(t),
    )))


def fit_experiment(raw: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Return a training artifact, honest report and held-out predictions."""
    X = normalize_features(raw).reset_index(drop=True)
    y = synthetic_target(X)
    if y.value_counts().min() < 10 or y.nunique() != 2:
        raise ValueError("This 60/20/20 experiment requires at least 10 rows per class")
    train_val, test = train_test_split(
        X.index.to_numpy(), test_size=0.2, random_state=SEED, stratify=y,
    )
    train, validation = train_test_split(
        train_val, test_size=0.25, random_state=SEED, stratify=y.loc[train_val],
    )
    pipeline = build_pipeline().fit(X.loc[train], y.loc[train])
    validation_scores = pipeline.predict_proba(X.loc[validation])[:, 1]
    threshold = select_threshold(y.loc[validation], validation_scores)
    test_scores = pipeline.predict_proba(X.loc[test])[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(X.loc[train], y.loc[train])
    dummy_scores = dummy.predict_proba(X.loc[test])[:, 1]

    transformed_train = pipeline.named_steps["preprocess"].transform(X.loc[train])
    split_indices = {
        "train": train.tolist(), "validation": validation.tolist(), "test": test.tolist(),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_DESCRIPTION,
        "seed": SEED,
        "features": FEATURES,
        "excluded_columns": [str(c) for c in raw.columns if c not in FEATURES],
        "splits": {
            name: {"rows": len(indices), "positives": int(y.loc[indices].sum()),
                   "negatives": int((y.loc[indices] == 0).sum())}
            for name, indices in split_indices.items()
        },
        "threshold": threshold,
        "threshold_selection": "max validation F1; grid 0.05..0.95; test not used",
        "validation": metrics(y.loc[validation], validation_scores, threshold),
        "test": {
            "logistic_regression": metrics(y.loc[test], test_scores, threshold),
            "majority_baseline": metrics(y.loc[test], dummy_scores, 0.5),
            "label_rule_reference": metrics(y.loc[test], y.loc[test].to_numpy(), 0.5),
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "The target is a deterministic rule over input features, not observed default.",
            "The exact rule is a perfect reference by construction; ML is unnecessary for it.",
            "Only a few positives per held-out split; metrics have high variance.",
            "Scores from class-weighted logistic regression are not calibrated default probabilities.",
            "Historical data provenance, units and redistribution license are not verified.",
        ],
    }
    bundle = {
        "schema_version": SCHEMA_VERSION, "pipeline": pipeline,
        "threshold": threshold,
        "background_mean": np.asarray(transformed_train).mean(axis=0),
        "split_indices": split_indices, "report": report,
    }
    heldout = pd.DataFrame({
        "row_id": test, "synthetic_risk": y.loc[test].to_numpy(),
        "score": test_scores, "predicted_class": (test_scores >= threshold).astype(int),
    }).sort_values("row_id").reset_index(drop=True)
    return bundle, report, heldout


def train(data_path: Path, output_dir: Path) -> dict:
    data_path, output_dir = Path(data_path), Path(output_dir)
    raw = pd.read_csv(data_path)
    bundle, report, heldout = fit_experiment(raw)
    # Ignore transport-only CRLF differences, not data/content differences.
    report["data"] = {
        "filename": data_path.name,
        "sha256_lf_bytes": hashlib.sha256(
            data_path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "rows": len(raw),
        "missing": {str(k): int(v) for k, v in raw.isna().sum().items() if v},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    heldout.to_csv(output_dir / "test_predictions.csv", index=False, lineterminator="\n")
    return report


def load_artifact(path: Path) -> dict:
    """Only load trusted, locally trained joblib artifacts."""
    bundle = joblib.load(path)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported artifact schema; retrain the model")
    saved_version = bundle["report"]["versions"]["scikit_learn"]
    if saved_version != sklearn.__version__:
        raise ValueError(
            f"Artifact uses scikit-learn {saved_version}; retrain with {sklearn.__version__}"
        )
    return bundle


def score_band(score: float) -> str:
    """Display-only categories; unrelated to the selected decision threshold."""
    return "A" if score < 0.2 else ("B" if score < 0.5 else "C")


def predict_frame(bundle: dict, raw: pd.DataFrame) -> pd.DataFrame:
    X = normalize_features(raw)
    scores = bundle["pipeline"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "score": scores,
        "predicted_class": (scores >= bundle["threshold"]).astype(int),
        "display_band": [score_band(float(value)) for value in scores],
        "rule_label": synthetic_target(X).to_numpy(),
    }, index=raw.index)


def explain_one(bundle: dict, raw: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Exact additive log-odds contributions relative to training mean."""
    if len(raw) != 1:
        raise ValueError("Explanation requires exactly one row")
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    values = preprocessor.transform(normalize_features(raw))[0]
    background = bundle["background_mean"]
    weights = classifier.coef_[0]
    baseline = float(classifier.intercept_[0] + np.dot(background, weights))
    contributions = (values - background) * weights
    table = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
        "log_odds_contribution": contributions,
    }).sort_values("log_odds_contribution", key=abs, ascending=False)
    return baseline, table.reset_index(drop=True)

