"""Input contract shared by training, CLI, notebook and UI."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["Age", "Credit amount", "Duration"]
CATEGORICAL = ["Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
FEATURES = NUMERIC + CATEGORICAL
CATEGORIES = {
    "Job": ["0", "1", "2", "3"],
    "Housing": ["own", "rent", "free"],
    "Saving accounts": ["little", "moderate", "quite rich", "rich"],
    "Checking account": ["little", "moderate", "rich"],
    "Purpose": ["business", "car", "domestic appliances", "education",
                "furniture/equipment", "radio/TV", "repairs", "vacation/others"],
}
NULLABLE = {"Saving accounts", "Checking account"}
TARGET_DESCRIPTION = (
    "Synthetic rule: Credit amount > 8000 AND Duration > 24 "
    "AND Checking account == little. Not observed default."
)


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw inputs; retain the same eight model features everywhere."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("At least one row is required")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    result = frame[FEATURES].copy()
    for name in NUMERIC + ["Job"]:
        if result[name].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{name}: boolean values are not numeric inputs")
        try:
            values = pd.to_numeric(result[name], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}: expected numeric values") from exc
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name}: expected finite, non-missing values")
        if name in {"Age", "Duration", "Job"} and (values != np.floor(values)).any():
            raise ValueError(f"{name}: expected integers")
        if name == "Age" and (values < 18).any():
            raise ValueError("Age must be at least 18")
        if name in {"Credit amount", "Duration"} and (values <= 0).any():
            raise ValueError(f"{name} must be positive")
        if name == "Job":
            if not values.between(0, 3).all():
                raise ValueError("Job must be between 0 and 3")
            result[name] = values.astype(int).astype(str)
        else:
            result[name] = values

    for name in CATEGORICAL:
        result[name] = result[name].map(
            lambda value: np.nan if pd.isna(value) or str(value).strip() == ""
            else str(value).strip()
        )
        if name not in NULLABLE and result[name].isna().any():
            raise ValueError(f"{name}: missing values are not allowed")
        unknown = set(result[name].dropna()) - set(CATEGORIES[name])
        if unknown:
            raise ValueError(f"{name}: unsupported values {sorted(unknown)}")
    return result


def synthetic_target(frame: pd.DataFrame) -> pd.Series:
    """Label the existing educational rule; never substitute it for default."""
    return (
        (frame["Credit amount"] > 8000)
        & (frame["Duration"] > 24)
        & (frame["Checking account"] == "little")
    ).astype(int).rename("synthetic_risk")

