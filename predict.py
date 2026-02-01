#!/usr/bin/env python3
"""
Circuit Fingerprint Challenge - Prediction Script
iQuHACK 2026

This script predicts:
1. predicted_threshold_min: Minimum threshold rung meeting fidelity target
2. predicted_forward_wall_s: Runtime (seconds) for the 10,000-shot forward run

Usage:
    python predict.py --tasks <TASKS_JSON> --circuits <CIRCUITS_DIR> --id-map <ID_MAP> --out predictions.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

from comprehensive_features import QASMFeatureExtractor

warnings.filterwarnings("ignore")

# =============================================================================
# BEST HYPERPARAMETERS (from Optuna tuning in notebooks)
# =============================================================================

# Threshold Classifier - RandomForest (from Classifier_Hari.ipynb)
CLASSIFIER_PARAMS = {
    'n_estimators': 368,
    'max_depth': 5,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'criterion': 'entropy',
    'class_weight': 'balanced_subsample',
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1
}

# Runtime Regressor - XGBoost (from Regression_Hari.ipynb)
REGRESSOR_PARAMS = {
    'n_estimators': 470,
    'max_depth': 19,
    'learning_rate': 0.368110,
    'min_child_weight': 14,
    'subsample': 0.520515,
    'colsample_bytree': 0.567303,
    'colsample_bylevel': 0.529540,
    'reg_alpha': 0.000032,
    'reg_lambda': 0.0,
    'gamma': 0.036937,
    'random_state': 42,
    'verbosity': 0
}

# Top features for runtime regression (from feature importance analysis)
TOP_15_RUNTIME_FEATURES = [
    'max_gate_span', 'std_gate_span', 'avg_gate_span', 'gates_per_depth',
    'sim_difficulty', 'n_h', 'degree_x_qubits', 'precision_single',
    'precision_double', 'n_1q_gates', 'threshold_x_qubits', 'backend_GPU',
    'backend_CPU', 'midpoint_cut_crossings', 'log_qubits'
]


def engineer_classifier_features(df):
    """Create features for threshold classification."""
    X = df.copy()

    # Interaction features
    X['degree_x_qubits'] = X['avg_qubit_degree'] * X['n_qubits']
    X['degree_x_depth'] = X['avg_qubit_degree'] * X['crude_depth']
    X['degree_x_2q'] = X['avg_qubit_degree'] * X['n_2q_gates']
    X['entanglement_complexity'] = X['n_unique_edges'] * X['avg_qubit_degree']
    X['entanglement_per_qubit'] = X['n_unique_edges'] / (X['n_qubits'] + 1)

    # Ratio features
    X['cx_ratio'] = X['n_cx'] / (X['n_total_gates'] + 1)
    X['rotation_ratio'] = X['n_rotation_gates'] / (X['n_total_gates'] + 1)
    X['multi_qubit_ratio'] = (X['n_2q_gates'] + X['n_3q_gates']) / (X['n_total_gates'] + 1)
    X['gates_per_depth'] = X['n_total_gates'] / (X['crude_depth'] + 1)
    X['depth_per_qubit'] = X['crude_depth'] / (X['n_qubits'] + 1)
    X['edge_density'] = X['n_unique_edges'] / (X['n_qubits'] * (X['n_qubits'] - 1) / 2 + 1)
    X['edge_repetition_ratio'] = X['n_edge_repetitions'] / (X['n_unique_edges'] + 1)

    # Polynomial features
    X['degree_squared'] = X['avg_qubit_degree'] ** 2
    X['qubits_squared'] = X['n_qubits'] ** 2
    X['depth_squared'] = X['crude_depth'] ** 2
    X['log_qubits'] = np.log1p(X['n_qubits'])
    X['log_depth'] = np.log1p(X['crude_depth'])
    X['log_gates'] = np.log1p(X['n_total_gates'])

    # Complexity scores
    X['complexity_score'] = X['n_qubits'] * X['crude_depth'] * X['avg_qubit_degree'] / 1000
    X['entanglement_burden'] = X['n_2q_gates'] * X['avg_qubit_degree'] / (X['n_qubits'] + 1)
    X['sim_difficulty'] = X['n_qubits'] ** 1.5 * X['entanglement_pressure']

    # Pattern features
    X['n_patterns'] = (X['has_qft_pattern'] + X['has_iqft_pattern'] +
                       X['has_grover_pattern'] + X['has_variational_pattern'] + X['has_ghz_pattern'])
    X['variational_complexity'] = X['has_variational_pattern'] * X['n_rotation_gates']

    return X


def engineer_regressor_features(df):
    """Create features for runtime regression."""
    X = df.copy()

    X['degree_x_qubits'] = X['avg_qubit_degree'] * X['n_qubits']
    X['degree_x_depth'] = X['avg_qubit_degree'] * X['crude_depth']
    X['entanglement_complexity'] = X['n_unique_edges'] * X['avg_qubit_degree']
    X['entanglement_per_qubit'] = X['n_unique_edges'] / (X['n_qubits'] + 1)
    X['cx_ratio'] = X['n_cx'] / (X['n_total_gates'] + 1)
    X['multi_qubit_ratio'] = (X['n_2q_gates'] + X['n_3q_gates']) / (X['n_total_gates'] + 1)
    X['gates_per_depth'] = X['n_total_gates'] / (X['crude_depth'] + 1)
    X['depth_per_qubit'] = X['crude_depth'] / (X['n_qubits'] + 1)
    X['log_qubits'] = np.log1p(X['n_qubits'])
    X['log_depth'] = np.log1p(X['crude_depth'])
    X['log_gates'] = np.log1p(X['n_total_gates'])
    X['log_threshold'] = np.log2(X['min_threshold'] + 1)
    X['complexity_score'] = X['n_qubits'] * X['crude_depth'] * X['avg_qubit_degree'] / 1000
    X['sim_difficulty'] = X['n_qubits'] ** 1.5 * X['entanglement_pressure']
    X['threshold_x_qubits'] = X['min_threshold'] * X['n_qubits']
    X['threshold_x_gates'] = X['min_threshold'] * X['n_total_gates']

    return X


class ThresholdClassifier:
    """Predicts minimum threshold rung for a circuit."""

    def __init__(self, training_csv='training_data_75.csv'):
        self.training_csv = training_csv
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.drop_cols = [
            "min_threshold", "file", "family", "forward_runtime",
            "max_fidelity_achieved", "forward_shots", "forward_peak_rss_mb", "n_thresholds_tested"
        ]

    def train(self):
        """Train the threshold classifier on training data."""
        # Load training data
        script_dir = Path(__file__).parent
        df = pd.read_csv(script_dir / self.training_csv)

        # Engineer features
        X_eng = engineer_classifier_features(df)

        # Drop non-feature columns
        drop_cols = [c for c in self.drop_cols if c in X_eng.columns]
        X_eng = X_eng.drop(columns=drop_cols)

        # One-hot encode categorical columns
        cat_cols = X_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        X_eng = pd.get_dummies(X_eng, columns=cat_cols)

        # Store feature columns for prediction
        self.feature_columns = X_eng.columns.tolist()

        # Prepare arrays
        X = X_eng.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Prepare labels
        y_raw = df["min_threshold"].astype(int).values
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)

        # Train model
        self.model = RandomForestClassifier(**CLASSIFIER_PARAMS)
        self.model.fit(X, y)

    def predict(self, qasm_path, processor, precision, conservative=True, confidence_threshold=0.6):
        """Predict threshold for a single circuit."""
        # Extract features from QASM file
        extractor = QASMFeatureExtractor(qasm_path)
        circuit_features = extractor.extract_all()

        # Create DataFrame row
        row = circuit_features.copy()
        row['backend'] = processor
        row['precision'] = precision

        input_df = pd.DataFrame([row])
        input_eng = engineer_classifier_features(input_df)

        # Drop non-feature columns
        for col in self.drop_cols:
            if col in input_eng.columns:
                input_eng = input_eng.drop(columns=[col])

        # One-hot encode
        cat_cols = input_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        input_eng = pd.get_dummies(input_eng, columns=cat_cols)

        # Align with training columns
        input_eng = input_eng.reindex(columns=self.feature_columns, fill_value=0)

        # Prepare array
        X_input = input_eng.values.astype(np.float32)
        X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)

        # Get prediction and probabilities
        proba = self.model.predict_proba(X_input)[0]
        pred_encoded = self.model.predict(X_input)[0]
        confidence = proba.max()

        # Conservative prediction: bump up if not confident (avoids underprediction)
        if conservative and confidence < confidence_threshold:
            classes = self.model.classes_
            new_idx = min(pred_encoded + 1, len(classes) - 1)
            pred_encoded = new_idx

        pred_threshold = self.label_encoder.inverse_transform([pred_encoded])[0]
        return int(pred_threshold)


class RuntimeRegressor:
    """Predicts forward runtime for a circuit."""

    def __init__(self, training_csv='training_data_99.csv'):
        self.training_csv = training_csv
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.drop_cols = [
            "forward_runtime", "file", "family",
            "max_fidelity_achieved", "forward_shots", "forward_peak_rss_mb", "n_thresholds_tested"
        ]

    def train(self):
        """Train the runtime regressor on training data."""
        script_dir = Path(__file__).parent
        df = pd.read_csv(script_dir / self.training_csv)

        # Engineer features
        X_eng = engineer_regressor_features(df)
        y_log = np.log1p(df['forward_runtime'].values)

        # Drop non-feature columns
        drop_cols = [c for c in self.drop_cols if c in X_eng.columns]
        X_eng = X_eng.drop(columns=drop_cols)

        # One-hot encode categorical columns
        cat_cols = X_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        X_eng = pd.get_dummies(X_eng, columns=cat_cols)

        # Select top 15 features (those that exist)
        self.feature_columns = [f for f in TOP_15_RUNTIME_FEATURES if f in X_eng.columns]

        # Prepare arrays
        X = X_eng[self.feature_columns].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = XGBRegressor(**REGRESSOR_PARAMS)
        self.model.fit(X_scaled, y_log)

    def predict(self, qasm_path, processor, precision, threshold):
        """Predict runtime for a single circuit."""
        # Extract features from QASM file
        extractor = QASMFeatureExtractor(qasm_path)
        circuit_features = extractor.extract_all()

        # Create DataFrame row
        row = circuit_features.copy()
        row['backend'] = processor
        row['precision'] = precision
        row['min_threshold'] = threshold

        input_df = pd.DataFrame([row])
        input_eng = engineer_regressor_features(input_df)

        # One-hot encode
        cat_cols = input_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        input_eng = pd.get_dummies(input_eng, columns=cat_cols)

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in input_eng.columns:
                input_eng[col] = 0

        # Prepare array
        X = input_eng[self.feature_columns].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        # Predict (inverse log transform)
        y_log_pred = self.model.predict(X_scaled)[0]
        runtime = float(np.expm1(y_log_pred))

        # Ensure non-negative
        return max(runtime, 0.1)


def main():
    parser = argparse.ArgumentParser(description="Circuit Fingerprint Challenge Predictor")
    parser.add_argument("--tasks", type=str, required=True, help="Path to holdout tasks JSON")
    parser.add_argument("--circuits", type=str, required=True, help="Path to circuits directory")
    parser.add_argument("--id-map", type=str, required=True, help="Path to ID map JSON")
    parser.add_argument("--out", type=str, required=True, help="Output predictions JSON path")
    parser.add_argument("--no-conservative-bump", action="store_true",
                        help="Disable conservative threshold bumping")

    args = parser.parse_args()

    # Load tasks
    with open(args.tasks) as f:
        tasks_data = json.load(f)
        tasks = tasks_data["tasks"]

    # Load ID map
    with open(args.id_map) as f:
        id_map_data = json.load(f)
        id_map = {entry["id"]: entry["qasm_file"] for entry in id_map_data["entries"]}

    circuits_dir = Path(args.circuits)

    # Train models
    print("Training threshold classifier...")
    classifier = ThresholdClassifier()
    classifier.train()

    print("Training runtime regressor...")
    regressor = RuntimeRegressor()
    regressor.train()

    # Generate predictions
    print(f"Generating predictions for {len(tasks)} tasks...")
    predictions = []

    for task in tasks:
        task_id = task["id"]
        processor = task["processor"]
        precision = task["precision"]

        # Get QASM file path
        qasm_file = id_map[task_id]
        qasm_path = circuits_dir / qasm_file

        # Predict threshold
        predicted_threshold = classifier.predict(
            qasm_path, processor, precision,
            conservative=not args.no_conservative_bump
        )

        # Predict runtime (using predicted threshold as feature)
        predicted_runtime = regressor.predict(
            qasm_path, processor, precision, predicted_threshold
        )

        predictions.append({
            "id": task_id,
            "predicted_threshold_min": predicted_threshold,
            "predicted_forward_wall_s": round(predicted_runtime, 2)
        })

        print(f"  {task_id}: threshold={predicted_threshold}, runtime={predicted_runtime:.2f}s")

    # Write predictions
    output = {"predictions": predictions}
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nPredictions written to {args.out}")


if __name__ == "__main__":
    main()
