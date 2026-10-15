from pathlib import Path

import pandas as pd
import pytest

from credit_risk.model import fit_experiment, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw_data():
    return pd.read_csv(ROOT / "credit.csv")


@pytest.fixture(scope="session")
def experiment(raw_data):
    return fit_experiment(raw_data)


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    output = tmp_path_factory.mktemp("model")
    train(ROOT / "credit.csv", output)
    return output

