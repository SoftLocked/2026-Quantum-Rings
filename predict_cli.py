#!/usr/bin/env python3
"""
predict_cli.py — Quantum Circuit Simulation Predictor
======================================================
Predicts, from a raw OpenQASM 2.0 file:
  1. Minimum approximation threshold for fidelity >= 0.99
  2. Expected wall-clock runtime for a 10,000-shot forward simulation

Modes
-----
  Interactive  (no args)      guided prompt session
  Single file                 predict_cli.py circuit.qasm
  Directory                   predict_cli.py circuits/ --precision single
  Challenge batch             predict_cli.py --tasks holdout.json \\
                                            --circuits circuits_new/ \\
                                            --out predictions.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from comprehensive_features import QASMFeatureExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_THRESHOLDS = [1, 2, 4, 8, 16, 32, 64]
VALID_PRECISIONS = ["single", "double"]
DEFAULT_PRECISION = "single"

_HERE = Path(__file__).parent
CLASSIFIER_PATH = _HERE / "models" / "threshold_classifier.pkl"
REGRESSOR_PATH = _HERE / "models" / "runtime_model.pkl"


# ---------------------------------------------------------------------------
# Feature Engineering  (must match the notebook code exactly)
# ---------------------------------------------------------------------------

def _engineer_classifier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matches engineer_features() in Classifier.ipynb cell 1."""
    X = df.copy()

    # Interaction features
    X["degree_x_qubits"] = X["avg_qubit_degree"] * X["n_qubits"]
    X["degree_x_depth"] = X["avg_qubit_degree"] * X["crude_depth"]
    X["degree_x_2q"] = X["avg_qubit_degree"] * X["n_2q_gates"]
    X["entanglement_complexity"] = X["n_unique_edges"] * X["avg_qubit_degree"]
    X["entanglement_per_qubit"] = X["n_unique_edges"] / (X["n_qubits"] + 1)

    # Ratio features
    X["cx_ratio"] = X["n_cx"] / (X["n_total_gates"] + 1)
    X["rotation_ratio"] = X["n_rotation_gates"] / (X["n_total_gates"] + 1)
    X["multi_qubit_ratio"] = (X["n_2q_gates"] + X["n_3q_gates"]) / (X["n_total_gates"] + 1)
    X["gates_per_depth"] = X["n_total_gates"] / (X["crude_depth"] + 1)
    X["depth_per_qubit"] = X["crude_depth"] / (X["n_qubits"] + 1)
    X["edge_density"] = X["n_unique_edges"] / (X["n_qubits"] * (X["n_qubits"] - 1) / 2 + 1)
    X["edge_repetition_ratio"] = X["n_edge_repetitions"] / (X["n_unique_edges"] + 1)

    # Polynomial / log features
    X["degree_squared"] = X["avg_qubit_degree"] ** 2
    X["qubits_squared"] = X["n_qubits"] ** 2
    X["depth_squared"] = X["crude_depth"] ** 2
    X["log_qubits"] = np.log1p(X["n_qubits"])
    X["log_depth"] = np.log1p(X["crude_depth"])
    X["log_gates"] = np.log1p(X["n_total_gates"])

    # Complexity scores
    X["complexity_score"] = X["n_qubits"] * X["crude_depth"] * X["avg_qubit_degree"] / 1000
    X["entanglement_burden"] = X["n_2q_gates"] * X["avg_qubit_degree"] / (X["n_qubits"] + 1)
    X["sim_difficulty"] = X["n_qubits"] ** 1.5 * X["entanglement_pressure"]

    # Pattern features
    X["n_patterns"] = (
        X["has_qft_pattern"] + X["has_iqft_pattern"] +
        X["has_grover_pattern"] + X["has_variational_pattern"] + X["has_ghz_pattern"]
    )
    X["variational_complexity"] = X["has_variational_pattern"] * X["n_rotation_gates"]

    return X


def _engineer_regressor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matches engineer_features() in Regressor.ipynb cell 1."""
    X = df.copy()

    # Interaction features
    X["degree_x_qubits"] = X["avg_qubit_degree"] * X["n_qubits"]
    X["degree_x_depth"] = X["avg_qubit_degree"] * X["crude_depth"]
    X["entanglement_complexity"] = X["n_unique_edges"] * X["avg_qubit_degree"]
    X["entanglement_per_qubit"] = X["n_unique_edges"] / (X["n_qubits"] + 1)

    # Ratio features
    X["cx_ratio"] = X["n_cx"] / (X["n_total_gates"] + 1)
    X["multi_qubit_ratio"] = (X["n_2q_gates"] + X["n_3q_gates"]) / (X["n_total_gates"] + 1)
    X["gates_per_depth"] = X["n_total_gates"] / (X["crude_depth"] + 1)
    X["depth_per_qubit"] = X["crude_depth"] / (X["n_qubits"] + 1)
    X["edge_density"] = X["n_unique_edges"] / (X["n_qubits"] * (X["n_qubits"] - 1) / 2 + 1)
    X["edge_repetition_ratio"] = X["n_edge_repetitions"] / (X["n_unique_edges"] + 1)

    # Log features
    X["log_qubits"] = np.log1p(X["n_qubits"])
    X["log_depth"] = np.log1p(X["crude_depth"])
    X["log_gates"] = np.log1p(X["n_total_gates"])
    X["log_threshold"] = np.log2(X["min_threshold"] + 1)

    # Complexity scores
    X["complexity_score"] = X["n_qubits"] * X["crude_depth"] * X["avg_qubit_degree"] / 1000
    X["sim_difficulty"] = X["n_qubits"] ** 1.5 * X["entanglement_pressure"]

    # Threshold interaction features
    X["threshold_x_qubits"] = X["min_threshold"] * X["n_qubits"]
    X["threshold_x_gates"] = X["min_threshold"] * X["n_total_gates"]

    return X


def _prepare_features(
    raw_features: dict,
    engineer_fn,
    model_features: list,
    scaler,
) -> np.ndarray:
    """
    raw_features dict  ->  engineered DataFrame  ->  aligned  ->  scaled  ->  ndarray
    Columns not produced by engineering are zero-filled to match training schema.
    """
    df = pd.DataFrame([raw_features])
    X_eng = engineer_fn(df)

    # One-hot encode any remaining categorical columns
    cat_cols = X_eng.select_dtypes(exclude=[np.number]).columns.tolist()
    X_eng = pd.get_dummies(X_eng, columns=cat_cols)

    # Align with training feature set (add missing columns as 0)
    for col in model_features:
        if col not in X_eng.columns:
            X_eng[col] = 0

    X = X_eng[model_features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(scaler.transform(X), -10, 10)
    return X


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_models():
    """Load both trained model artifacts from disk."""
    missing = [p for p in (CLASSIFIER_PATH, REGRESSOR_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model file(s) not found:\n" +
            "\n".join(f"  {p}" for p in missing) +
            "\nRun Classifier.ipynb and Regressor.ipynb to train and save the models."
        )
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    return classifier, regressor


# ---------------------------------------------------------------------------
# Core Prediction
# ---------------------------------------------------------------------------

def predict_single(
    qasm_path: Path,
    precision: str,
    classifier: dict,
    regressor: dict,
    conservative: bool = False,
    confidence_threshold: float = 0.6,
    threshold_override: Optional[int] = None,
) -> dict:
    """
    Run threshold + runtime prediction for one circuit.

    Parameters
    ----------
    qasm_path           : Path to a .qasm file
    precision           : "single" or "double"
    classifier          : loaded artifact from threshold_classifier.pkl
    regressor           : loaded artifact from runtime_model.pkl
    conservative        : if True, bump predicted threshold up one rung when
                          classifier confidence falls below confidence_threshold
    confidence_threshold: confidence level below which conservative bumping triggers
    threshold_override  : skip the classifier and use this threshold directly

    Returns
    -------
    dict with all prediction details and basic circuit stats
    """
    # -- Extract base features -------------------------------------------------
    extractor = QASMFeatureExtractor(str(qasm_path))
    base_features = extractor.extract_all()
    base_features["backend"] = "CPU"
    base_features["precision"] = precision

    # -- Threshold prediction --------------------------------------------------
    if threshold_override is not None:
        predicted_threshold = threshold_override
        confidence = None
        probabilities = {}
    else:
        clf_model = classifier["model"]
        clf_scaler = classifier["scaler"]
        clf_features = classifier["features"]
        le = classifier["label_encoder"]
        threshold_classes = classifier["threshold_classes"]

        X_clf = _prepare_features(
            base_features, _engineer_classifier_features, clf_features, clf_scaler
        )

        pred_encoded = clf_model.predict(X_clf)[0]
        confidence = 0.0
        probabilities = {}

        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(X_clf)[0]
            confidence = float(proba.max())
            # Map class index → actual threshold value → probability
            for i, cls in enumerate(clf_model.classes_):
                probabilities[int(threshold_classes[cls])] = float(proba[i])
            # Conservative mode: step up one rung if confidence is low
            if conservative and confidence < confidence_threshold:
                pred_encoded = min(pred_encoded + 1, len(threshold_classes) - 1)

        predicted_threshold = int(le.inverse_transform([pred_encoded])[0])

    # -- Runtime prediction ----------------------------------------------------
    reg_features = {**base_features, "min_threshold": predicted_threshold}

    X_reg = _prepare_features(
        reg_features,
        _engineer_regressor_features,
        regressor["features"],
        regressor["scaler"],
    )

    predicted_runtime = float(np.expm1(regressor["model"].predict(X_reg)[0]))
    predicted_runtime = max(predicted_runtime, 0.0)

    return {
        "file": qasm_path.name,
        "precision": precision,
        "predicted_threshold": predicted_threshold,
        "predicted_runtime_s": predicted_runtime,
        "confidence": confidence,
        "probabilities": probabilities,
        # Circuit stats for display
        "n_qubits": base_features.get("n_qubits", 0),
        "n_2q_gates": base_features.get("n_2q_gates", 0),
        "n_total_gates": base_features.get("n_total_gates", 0),
    }


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def _fmt_runtime(seconds: float) -> str:
    """Human-readable runtime string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m  ({seconds:.0f}s)"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h  ({seconds:.0f}s)"
    elif seconds < 86400 * 14:
        return f"{seconds / 86400:.1f}d  ({seconds:.0f}s)"
    else:
        return f"{seconds / 86400:.0f}d  ({seconds:.0f}s)"


def _fmt_conf_bar(conf: Optional[float], width: int = 18) -> str:
    if conf is None:
        return "N/A (manual override)"
    filled = round(conf * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {conf * 100:.0f}%"


def _print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print a single circuit prediction."""
    sep = "-" * 56
    print()
    print(f"+{sep}+")
    name = result["file"]
    print(f"|  Circuit : {name:<45}|")
    print(f"+{sep}+")
    print(
        f"|  Qubits  : {result['n_qubits']:<6}"
        f"  2Q gates : {result['n_2q_gates']:<6}"
        f"  Total : {result['n_total_gates']:<9}|"
    )
    print(f"|  Precision: {result['precision']:<45}|")
    print(f"+{sep}+")
    print(f"|  Threshold: {result['predicted_threshold']:<45}|")
    print(f"|  Runtime  : {_fmt_runtime(result['predicted_runtime_s']):<45}|")
    conf_str = _fmt_conf_bar(result["confidence"])
    print(f"|  Confidence: {conf_str:<44}|")
    print(f"+{sep}+")

    if verbose and result["probabilities"]:
        print()
        print("  Threshold probability distribution:")
        for thresh, prob in sorted(result["probabilities"].items()):
            filled = round(prob * 22)
            bar = "#" * filled + "." * (22 - filled)
            marker = " <<" if thresh == result["predicted_threshold"] else "   "
            print(f"    {thresh:>3}: [{bar}] {prob * 100:5.1f}%{marker}")


def _print_batch_table(results: list) -> None:
    """Print a summary table for multiple circuit results."""
    W = 95
    print()
    print("=" * W)
    print(
        f"  {'File':<42} {'Prec':<8} {'Threshold':>10}"
        f" {'Runtime':>16} {'Confidence':>10}"
    )
    print("-" * W)
    for r in results:
        conf_str = f"{r['confidence'] * 100:.0f}%" if r["confidence"] is not None else "N/A"
        print(
            f"  {r['file']:<42} {r['precision']:<8} {r['predicted_threshold']:>10}"
            f" {_fmt_runtime(r['predicted_runtime_s']):>16} {conf_str:>10}"
        )
    print("=" * W)
    print(f"  Total: {len(results)} circuit(s)")


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------

def _prompt(msg: str, default: str = "") -> str:
    display = f"  {msg} [{default}]: " if default else f"  {msg}: "
    try:
        answer = input(display).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer if answer else default


def _prompt_choice(msg: str, choices: list, default: str = "") -> str:
    label = "/".join(choices)
    while True:
        answer = _prompt(f"{msg} ({label})", default=default)
        if answer in choices:
            return answer
        print(f"    Please enter one of: {label}")


def _prompt_yn(msg: str, default: bool = False) -> bool:
    return _prompt_choice(msg, ["y", "n"], default="y" if default else "n") == "y"


def _interactive_mode(classifier: dict, regressor: dict) -> None:
    print()
    print("=" * 60)
    print("  Quantum Circuit Simulation Predictor")
    print("  iQuHACK 2026 - Quantum Rings Challenge")
    print("=" * 60)
    print()
    print("  Predicts fidelity threshold + forward runtime from a .qasm file.")
    print("  Enter 'q' or Ctrl-C at any prompt to quit.")
    print()

    session_results = []
    first = True

    while True:
        if not first:
            print()
            if not _prompt_yn("Predict another circuit?", default=False):
                break
        first = False

        # -- Circuit path -------------------------------------------------
        print()
        circuit_input = _prompt("Path to .qasm file or directory")
        if circuit_input.lower() in ("q", "quit", "exit", ""):
            break

        input_path = Path(circuit_input)
        if not input_path.exists():
            print(f"  Error: path not found: {input_path}")
            continue

        if input_path.is_dir():
            qasm_files = sorted(input_path.glob("*.qasm"))
            if not qasm_files:
                print(f"  No .qasm files found in {input_path}")
                continue
            print(f"  Found {len(qasm_files)} .qasm file(s).")
        else:
            if input_path.suffix.lower() != ".qasm":
                print(f"  Warning: file extension is not .qasm")
            qasm_files = [input_path]

        # -- Options ------------------------------------------------------
        precision = _prompt_choice("Precision", VALID_PRECISIONS, default=DEFAULT_PRECISION)
        conservative = _prompt_yn(
            "Conservative mode? (bumps threshold up if confidence < 60%)", default=False
        )
        verbose = _prompt_yn("Show probability distribution?", default=False)

        threshold_override = None
        if _prompt_yn("Manually override the predicted threshold?", default=False):
            valid_str = ", ".join(str(t) for t in VALID_THRESHOLDS)
            while True:
                raw = _prompt(f"Threshold value ({valid_str})")
                try:
                    t = int(raw)
                    if t in VALID_THRESHOLDS:
                        threshold_override = t
                        break
                    print(f"    Must be one of: {valid_str}")
                except ValueError:
                    print("    Please enter an integer.")

        # -- Run predictions ----------------------------------------------
        print()
        batch = []
        for i, qasm_path in enumerate(qasm_files):
            if len(qasm_files) > 1:
                print(f"  [{i + 1}/{len(qasm_files)}] {qasm_path.name} ...", end="\r")
            try:
                result = predict_single(
                    qasm_path, precision, classifier, regressor,
                    conservative=conservative,
                    threshold_override=threshold_override,
                )
                batch.append(result)
            except Exception as exc:
                print(f"\n  Error: {qasm_path.name}: {exc}")

        if len(qasm_files) > 1:
            print()  # clear \r line
            _print_batch_table(batch)
        elif batch:
            _print_result(batch[0], verbose=verbose)

        session_results.extend(batch)

        # -- Save option --------------------------------------------------
        if batch and _prompt_yn("Save these results to JSON?", default=False):
            out_str = _prompt("Output file path", default="predictions.json")
            _save_json(batch, Path(out_str), challenge_fmt=False)
            print(f"  Saved to {out_str}")

    if session_results:
        print()
        print(f"  Session complete. {len(session_results)} prediction(s) made.")


# ---------------------------------------------------------------------------
# Challenge Batch Mode
# ---------------------------------------------------------------------------

def _run_challenge_batch(
    tasks_path: Path,
    circuits_dir: Path,
    out_path: Path,
    classifier: dict,
    regressor: dict,
    precision_override: Optional[str] = None,
    conservative: bool = False,
    threshold_override: Optional[int] = None,
) -> list:
    """
    Official iQuHACK challenge batch prediction.

    Input tasks JSON schema
    -----------------------
    {"schema": "...", "entries": [{"id": "H001", "qasm_file": "foo.qasm", ...}]}
    or a bare list of entry objects.

    Output JSON schema
    ------------------
    [{"id": "H001", "predicted_threshold_min": 4, "predicted_forward_wall_s": 120.5}]
    """
    with open(tasks_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    entries = raw if isinstance(raw, list) else raw.get("entries", [])
    if not entries:
        print("No entries found in tasks file.")
        sys.exit(1)

    print(f"Tasks  : {tasks_path}  ({len(entries)} entries)")
    print(f"Circuits: {circuits_dir}")
    print(f"Output : {out_path}")
    print()

    predictions = []
    errors = []

    for i, entry in enumerate(entries):
        task_id = entry.get("id", str(i))
        qasm_file = entry.get("qasm_file", "")
        precision = precision_override or entry.get("precision", DEFAULT_PRECISION)
        if precision not in VALID_PRECISIONS:
            precision = DEFAULT_PRECISION

        qasm_path = circuits_dir / qasm_file
        prefix = f"  [{i + 1:>{len(str(len(entries)))}}/{len(entries)}]"

        if not qasm_path.exists():
            print(f"{prefix} {task_id}: {qasm_file} — FILE MISSING")
            errors.append(task_id)
            continue

        try:
            result = predict_single(
                qasm_path, precision, classifier, regressor,
                conservative=conservative,
                threshold_override=threshold_override,
            )
            predictions.append({
                "id": task_id,
                "predicted_threshold_min": result["predicted_threshold"],
                "predicted_forward_wall_s": round(result["predicted_runtime_s"], 3),
            })
            conf_str = (
                f"  conf={result['confidence'] * 100:.0f}%"
                if result["confidence"] is not None else ""
            )
            print(
                f"{prefix} {task_id}: threshold={result['predicted_threshold']}"
                f"  runtime={_fmt_runtime(result['predicted_runtime_s'])}{conf_str}"
            )
        except Exception as exc:
            print(f"{prefix} {task_id}: ERROR — {exc}")
            errors.append(task_id)

    print()
    print(f"Done: {len(predictions)}/{len(entries)} predictions")
    if errors:
        sample = ", ".join(errors[:5])
        more = f" (+ {len(errors) - 5} more)" if len(errors) > 5 else ""
        print(f"Errors ({len(errors)}): {sample}{more}")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)
    print(f"Saved  : {out_path}")

    return predictions


# ---------------------------------------------------------------------------
# JSON Output
# ---------------------------------------------------------------------------

def _save_json(results: list, out_path: Path, challenge_fmt: bool = False) -> None:
    if challenge_fmt:
        data = [
            {
                "id": r.get("id", r["file"]),
                "predicted_threshold_min": r["predicted_threshold"],
                "predicted_forward_wall_s": round(r["predicted_runtime_s"], 3),
            }
            for r in results
        ]
    else:
        # Full results — drop large probability dict to keep file readable
        data = [
            {k: v for k, v in r.items() if k != "probabilities"}
            for r in results
        ]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_cli.py",
        description="Quantum Circuit Simulation Predictor — iQuHACK 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Interactive guided session
  python predict_cli.py

  # Single circuit (single precision, default)
  python predict_cli.py path/to/circuit.qasm

  # Single circuit, double precision, conservative mode, show probabilities
  python predict_cli.py circuit.qasm -p double --conservative --verbose

  # All circuits in a directory, save results
  python predict_cli.py circuits/ -p single --output results.json

  # Skip classifier: predict runtime at a known threshold
  python predict_cli.py circuit.qasm --threshold 8

  # Official challenge batch format
  python predict_cli.py --tasks data/holdout_public.json \\
                        --circuits circuits_new/ \\
                        --out predictions.json
""",
    )

    parser.add_argument(
        "circuit", nargs="?", metavar="CIRCUIT",
        help="Path to a .qasm file or directory of .qasm files "
             "(omit to enter interactive mode)",
    )
    parser.add_argument(
        "-p", "--precision", choices=VALID_PRECISIONS, default=None,
        metavar="PRECISION",
        help="Simulation precision: single or double  (default: single)",
    )
    parser.add_argument(
        "--threshold", type=int, choices=VALID_THRESHOLDS, metavar="T",
        help="Override threshold — skip the classifier and go straight to "
             "runtime prediction using this value  (one of: 1 2 4 8 16 32 64)",
    )
    parser.add_argument(
        "--conservative", action="store_true",
        help="Bump predicted threshold up one rung when classifier confidence < 60%%",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print full probability distribution over threshold classes",
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="Save predictions to a JSON file",
    )

    batch = parser.add_argument_group("challenge batch mode")
    batch.add_argument(
        "--tasks", metavar="TASKS_JSON",
        help="Path to holdout tasks JSON (official challenge format)",
    )
    batch.add_argument(
        "--circuits", metavar="DIR",
        help="Directory containing QASM files for batch mode",
    )
    batch.add_argument(
        "--out", metavar="PREDICTIONS_JSON",
        help="Output predictions file (challenge format, required with --tasks)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # -- Load models -----------------------------------------------------------
    print("Loading models...", end=" ", flush=True)
    try:
        classifier, regressor = load_models()
        print("ready.")
    except FileNotFoundError as exc:
        print()
        print(f"Error: {exc}")
        sys.exit(1)

    # -- Challenge batch mode --------------------------------------------------
    if args.tasks:
        if not args.circuits:
            parser.error("--circuits DIR is required when using --tasks")
        if not args.out:
            parser.error("--out PREDICTIONS_JSON is required when using --tasks")

        tasks_path = Path(args.tasks)
        circuits_dir = Path(args.circuits)
        out_path = Path(args.out)

        if not tasks_path.exists():
            print(f"Error: tasks file not found: {tasks_path}")
            sys.exit(1)
        if not circuits_dir.is_dir():
            print(f"Error: circuits directory not found: {circuits_dir}")
            sys.exit(1)

        _run_challenge_batch(
            tasks_path, circuits_dir, out_path,
            classifier, regressor,
            precision_override=args.precision,
            conservative=args.conservative,
            threshold_override=args.threshold,
        )
        return

    # -- Interactive mode (no circuit argument) --------------------------------
    if not args.circuit:
        _interactive_mode(classifier, regressor)
        return

    # -- Single file or directory ----------------------------------------------
    input_path = Path(args.circuit)
    if not input_path.exists():
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        qasm_files = sorted(input_path.glob("*.qasm"))
        if not qasm_files:
            print(f"No .qasm files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(qasm_files)} .qasm file(s) in {input_path}")
    else:
        qasm_files = [input_path]

    precision = args.precision or DEFAULT_PRECISION
    results = []

    for i, qasm_path in enumerate(qasm_files):
        if len(qasm_files) > 1:
            print(f"[{i + 1}/{len(qasm_files)}] {qasm_path.name} ...", end="\r")
        try:
            result = predict_single(
                qasm_path, precision, classifier, regressor,
                conservative=args.conservative,
                threshold_override=args.threshold,
            )
            results.append(result)
            if len(qasm_files) == 1:
                _print_result(result, verbose=args.verbose)
        except Exception as exc:
            print(f"\nError: {qasm_path.name}: {exc}")

    if len(qasm_files) > 1:
        print()  # clear \r
        _print_batch_table(results)

    if args.output and results:
        out_path = Path(args.output)
        _save_json(results, out_path, challenge_fmt=False)
        print(f"\nSaved {len(results)} prediction(s) to {out_path}")


if __name__ == "__main__":
    main()
