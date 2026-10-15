"""Small, scriptable entry points; no notebook execution is needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from credit_risk.model import load_artifact, predict_frame, train


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic-rule classification demo")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and evaluate with fixed splits")
    training.add_argument("--data", type=Path, default=Path("credit.csv"))
    training.add_argument("--output", type=Path, default=Path("artifacts"))
    prediction = commands.add_parser("predict", help="Score one JSON object or a list")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            report = train(args.data, args.output)
            print(json.dumps({
                "output": str(args.output), "threshold": report["threshold"],
                "test": report["test"], "target": report["target"],
            }, ensure_ascii=False, indent=2))
        else:
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
                raise ValueError("JSON must contain an object or a list of objects")
            result = predict_frame(load_artifact(args.model), pd.DataFrame(payload))
            print(result.to_json(orient="records", force_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

