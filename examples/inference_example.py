#!/usr/bin/env python3
"""Run one prediction from a project-generated checkpoint."""

import argparse

from src.inference import DefectDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Checkpoint created by src.train")
    parser.add_argument("image_path", help="Image to classify")
    args = parser.parse_args()

    detector = DefectDetector(args.model_path)
    result = detector.predict(args.image_path)

    print(f"class={result['class']}")
    print(f"confidence={result['confidence']:.6f}")
    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}={probability:.6f}")


if __name__ == "__main__":
    main()
