"""Predict a penguin photograph using a trained local model."""

import argparse
import json
import logging
from pathlib import Path

from penguin_classifier.inference import ArtifactError, load_classifier, predict
from penguin_classifier.preprocessing import ImageInputError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.top_k < 1 or not 0 <= args.confidence_threshold <= 1:
        parser.error("top-k must be positive and confidence-threshold must be between 0 and 1")
    try:
        model, metadata = load_classifier(args.model_dir)
        result = predict(model, metadata, args.image)
    except (ArtifactError, ImageInputError, OSError, ValueError) as exc:
        logging.error("Prediction failed: %s", exc)
        return 2
    result["low_confidence"] = result["confidence"] < args.confidence_threshold
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Prediction: {result['prediction']} ({result['confidence']:.1%} model score)")
        if result["low_confidence"]:
            print("Low score: threshold is a heuristic, not calibrated uncertainty.")
        for name, score in sorted(result["scores"].items(), key=lambda item: item[1], reverse=True)[
            : args.top_k
        ]:
            print(f"  {name}: {score:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
