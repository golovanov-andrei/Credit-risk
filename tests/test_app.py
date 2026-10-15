from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from credit_risk.model import load_artifact, predict_frame
from conftest import ROOT


def test_missing_artifact_shows_training_command(monkeypatch, tmp_path):
    monkeypatch.setenv("CREDIT_RISK_MODEL", str(tmp_path / "missing.joblib"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "credit-risk train" in app.warning[0].value


def test_app_predictions_match_shared_pipeline(monkeypatch, artifact_dir, applicant):
    path = artifact_dir / "model.joblib"
    monkeypatch.setenv("CREDIT_RISK_MODEL", str(path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    # Submit a positive-rule profile using the actual form.
    app.number_input[1].set_value(9000.0)
    app.number_input[2].set_value(36)
    app.selectbox[3].set_value("little")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3
    raw = pd.DataFrame([applicant])
    expected = predict_frame(load_artifact(path), raw).iloc[0]
    assert app.metric[0].value == f"{expected['score']:.1%}"
    assert app.metric[2].value == "1"

