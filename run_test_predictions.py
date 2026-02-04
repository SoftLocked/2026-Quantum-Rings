#!/usr/bin/env python3
"""
Run predictions on test circuits from test_circuits directory.
Uses the ThresholdClassifier and RuntimeRegressor from predict.py
"""

from predict import ThresholdClassifier, RuntimeRegressor
from pathlib import Path

# Test circuits and their configurations from QR-iQuHACK Settings.txt
test_cases = [
    {"file": "graphstate_indep_qiskit_30.qasm", "backend": "CPU", "precision": "single", "given_threshold": 16},
    {"file": "pricingcall_indep_qiskit_25.qasm", "backend": "CPU", "precision": "single", "given_threshold": 4},
    {"file": "qnn_indep_qiskit_30.qasm", "backend": "CPU", "precision": "single", "given_threshold": 32},
    {"file": "shor_9_4_indep_qiskit_18.qasm", "backend": "CPU", "precision": "single", "given_threshold": 16},
]

circuits_dir = Path("test_circuits")


def main():
    # Train models
    print("=" * 70)
    print("Training models...")
    print("=" * 70)

    classifier = ThresholdClassifier()
    classifier.train()
    print("Threshold classifier trained.")

    regressor = RuntimeRegressor()
    regressor.train()
    print("Runtime regressor trained.")

    print()
    print("=" * 70)
    print("PREDICTIONS FOR TEST CIRCUITS")
    print("=" * 70)
    print()
    print(f"{'Circuit':<40} {'Backend':<6} {'Prec':<8} {'Pred Thresh':>12} {'Given Thresh':>13} {'Pred Runtime':>13}")
    print("-" * 100)

    for tc in test_cases:
        qasm_path = circuits_dir / tc["file"]

        # Predict threshold (without using given threshold)
        pred_threshold = classifier.predict(
            qasm_path,
            tc["backend"],
            tc["precision"],
            conservative=True,
            confidence_threshold=0.6
        )

        # Predict runtime using the GIVEN threshold (as per settings file)
        pred_runtime = regressor.predict(
            qasm_path,
            tc["backend"],
            tc["precision"],
            tc["given_threshold"]
        )

        print(f"{tc['file']:<40} {tc['backend']:<6} {tc['precision']:<8} {pred_threshold:>12} {tc['given_threshold']:>13} {pred_runtime:>12.2f}s")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Predicted Threshold: Calculated by classifier (without using given values)")
    print("Predicted Runtime: Uses the GIVEN threshold from settings file")


if __name__ == "__main__":
    main()
