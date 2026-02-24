# Quantum Circuit Simulation Predictor
### iQuHACK 2026 — Quantum Rings Challenge Submission

Predict the simulation cost–accuracy trade-offs of quantum circuits **without running the simulator**. Given a raw OpenQASM 2.0 file, two gradient-boosting models jointly estimate:

1. **Minimum approximation threshold** — the smallest truncation rung (1, 2, 4, 8, 16, 32, 64) at which MPS-based simulation reaches fidelity ≥ 0.99
2. **Forward wall-clock runtime** — expected seconds for a 10,000-shot run at that threshold

Both predictions run in under 100 ms per circuit from a command-line interface.

---

## Why This Matters

Quantum circuit simulation using tensor-network (MPS) methods exposes a configurable truncation threshold that trades fidelity for speed. Choosing the right setting before pressing *Run* can mean the difference between an overnight job and a two-month job. This project builds that intuition into a machine-learned model so developers can plan resources automatically.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive session (guided prompts)
python predict_cli.py

# Single circuit
python predict_cli.py circuits_new/qft_indep_qiskit_30.qasm

# Single circuit, double precision, conservative mode
python predict_cli.py circuits_new/ae_indep_qiskit_36.qasm -p double --conservative

# All circuits in a directory, save results
python predict_cli.py circuits_new/ -p single --output my_results.json

# Official challenge batch format
python predict_cli.py --tasks data/holdout_public.json \
                      --circuits circuits_new/ \
                      --out predictions.json
```

> **Prerequisite:** Run `Classifier.ipynb` and `Regressor.ipynb` once to train and save the models to `models/`.

---

## Project Structure

```
2026-Quantum-Rings/
├── predict_cli.py             # CLI entrypoint — run predictions here
├── comprehensive_features.py  # 7-level QASM feature extractor (~90 features)
├── build_training_dataset.py  # Builds training_data.csv from data/data.json
├── Classifier.ipynb           # Threshold model: training, tuning, saving
├── Regressor.ipynb            # Runtime model:   training, tuning, saving
├── training_data.csv          # Extracted features + labels (1,107 samples)
├── requirements.txt
├── data/
│   └── data.json              # Labeled training data (576 circuits × 4 configs)
├── circuits_new/              # OpenQASM 2.0 circuit files (~576)
├── models/
│   ├── threshold_classifier.pkl  # Saved classifier artifact (9.7 MB)
│   └── runtime_model.pkl         # Saved regressor artifact  (3.0 MB)
└── docs/
    ├── CHALLENGE.md
    ├── DATA.md
    ├── SUBMISSION.md
    ├── CIRCUITS.md
    └── THIRD_PARTY.md
```

---

## CLI Reference

### `predict_cli.py`

```
usage: predict_cli.py [-h] [-p PRECISION] [--threshold T]
                      [--conservative] [-v] [-o FILE]
                      [--tasks TASKS_JSON] [--circuits DIR] [--out PREDICTIONS_JSON]
                      [CIRCUIT]
```

| Argument | Description |
|---|---|
| `CIRCUIT` | Path to a `.qasm` file **or** a directory of `.qasm` files. Omit for interactive mode. |
| `-p / --precision` | `single` or `double` (default: `single`) |
| `--threshold T` | Override threshold — skip the classifier, predict runtime only |
| `--conservative` | Bump threshold up one rung when classifier confidence < 60% |
| `-v / --verbose` | Print full probability distribution over all 7 threshold classes |
| `-o / --output FILE` | Save results to a JSON file |
| `--tasks TASKS_JSON` | Official challenge tasks file (batch mode) |
| `--circuits DIR` | Circuit directory for batch mode |
| `--out PREDICTIONS_JSON` | Output path for batch mode predictions |

### Interactive Mode

Running `python predict_cli.py` with no arguments launches a guided prompt session:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Quantum Circuit Simulation Predictor
  iQuHACK 2026 — Quantum Rings Challenge
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Path to .qasm file or directory: circuits_new/qft_indep_qiskit_30.qasm
  Precision (single/double) [single]:
  Conservative mode? (y/n) [n]:
  Show probability distribution? (y/n) [n]:
  Manually override the predicted threshold? (y/n) [n]:

┌────────────────────────────────────────────────────────┐
│  Circuit : qft_indep_qiskit_30.qasm                    │
├────────────────────────────────────────────────────────┤
│  Qubits  : 30     2Q gates : 435    Total : 901        │
│  Precision: single                                     │
├────────────────────────────────────────────────────────┤
│  Threshold: 2                                          │
│  Runtime  : 11.2h  (40482s)                            │
│  Confidence: ████████████████░░ 89%                   │
└────────────────────────────────────────────────────────┘
```

### Batch Example Output (`--verbose`)

```
  Threshold probability distribution:
    1: ░░░░░░░░░░░░░░░░░░░░░░  0.0%
    2: ██████████████████░░░░ 83.7%  <<
    4: ██░░░░░░░░░░░░░░░░░░░░  9.1%
    8: █░░░░░░░░░░░░░░░░░░░░░  4.4%
   16: ░░░░░░░░░░░░░░░░░░░░░░  2.1%
   32: ░░░░░░░░░░░░░░░░░░░░░░  0.7%
   64: ░░░░░░░░░░░░░░░░░░░░░░  0.0%
```

---

## Architecture

```
 .qasm file
     │
     ▼
 QASMFeatureExtractor          (~90 features, pure regex — no parser needed)
     │
     ├──► engineer_classifier_features  ──► StandardScaler ──► GradientBoostingClassifier
     │           (88 features)                                     │
     │                                                             ▼
     │                                                   predicted_threshold
     │                                                             │
     └──► engineer_regressor_features   ──► StandardScaler ──► GradientBoostingRegressor
                 (50 features)                                     │
                 + min_threshold ◄────────────────────────────────┘
                                                                   ▼
                                                         predicted_runtime (s)
```

Both models trained on **1,107 samples** from 576 unique circuits, validated with **5-fold GroupKFold** grouped by circuit file (prevents any same-circuit leakage across backends/precisions).

---

## Feature Engineering

Features are extracted in 7 layers of increasing complexity from raw QASM text.

### Layer 1 — Basic Counts
`n_qubits`, `n_classical_bits`, `n_measure`, `n_barrier`, `n_lines`

### Layer 2 — Gate Type Counts
All gate families counted individually:
- **2-qubit (entangling):** `n_cx`, `n_cz`, `n_cp`, `n_cy`, `n_ch`, `n_swap`
- **3-qubit:** `n_ccx` (Toffoli), `n_cswap` (Fredkin)
- **1-qubit:** `n_h`, `n_x`, `n_y`, `n_z`, `n_s`, `n_t`, `n_rx`, `n_ry`, `n_rz`, `n_u1/u2/u3`
- **Aggregates:** `n_1q_gates`, `n_2q_gates`, `n_3q_gates`, `n_total_gates`

### Layer 3 — Qubit Interaction Graph
From the 2-qubit gate adjacency graph:
`n_unique_edges`, `n_edge_repetitions`, `max_qubit_degree`, `avg_qubit_degree`, `qubit_degree_std`, `n_connected_components`

### Layer 4 — Depth Estimation
`crude_depth` (lines / qubits), `gates_per_layer_estimate`

### Layer 5 — Entanglement Proxies (physics-inspired)
- **Gate span:** `avg_gate_span`, `max_gate_span`, `std_gate_span` — distance between entangled qubit indices
- `entanglement_pressure` — 2Q gates per qubit
- `midpoint_cut_crossings` — gates crossing the circuit's midpoint (a cut-width proxy)

### Layer 6 — Algorithm Pattern Detection
Regex fingerprints for common circuit families:
`has_qft_pattern`, `has_iqft_pattern`, `has_grover_pattern`, `has_variational_pattern`, `has_ghz_pattern`, `n_custom_gates`, `n_opaque_gates`

### Layer 7 — Normalized Ratios
`ratio_2q_gates`, `ratio_1q_gates`, `gates_per_qubit`, `2q_gates_per_qubit`, `circuit_density`

### Engineered Features (added at training time)
Both models add interaction, polynomial, and log features:

| Feature | Formula | Why |
|---|---|---|
| `entanglement_per_qubit` | unique_edges / (n_qubits + 1) | Top classifier feature |
| `degree_squared` | avg_degree² | Non-linear connectivity signal |
| `complexity_score` | qubits × depth × avg_degree / 1000 | Combined difficulty proxy |
| `log_qubits` | log(1 + n_qubits) | Compress exponential scale |
| `log_threshold` | log₂(threshold + 1) | Regressor only |
| `threshold_x_qubits` | threshold × n_qubits | Regressor only |

---

## Models

### Threshold Classifier

| Property | Value |
|---|---|
| Algorithm | `GradientBoostingClassifier` (sklearn) |
| Target | min_threshold ∈ {1, 2, 4, 8, 16, 32, 64} |
| Features | 88 (all engineered features) |
| Validation | 5-fold StratifiedGroupKFold |
| Tuning | Optuna TPE, 40 trials |
| **Competition score** | **0.9250** (92.5% of perfect) |
| Accuracy | 90.97% |
| Under-prediction rate | 5.9% (threshold too low — penalized heavily) |
| Over-prediction rate | 3.2% (threshold too high — mild penalty) |

**Scoring function:** exact match = 1.0, over-predict = true/predicted, under-predict = 0.

**Tuned hyperparameters:**
```python
n_estimators   = 254
max_depth      = 8
learning_rate  = 0.020167
min_samples_split = 8
min_samples_leaf  = 6
subsample      = 0.947884
```

**Top 5 features by importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `entanglement_per_qubit` | 0.0428 |
| 2 | `avg_gate_span` | 0.0384 |
| 3 | `degree_squared` | 0.0372 |
| 4 | `avg_qubit_degree` | 0.0360 |
| 5 | `degree_x_qubits` | 0.0317 |

---

### Runtime Regressor

| Property | Value |
|---|---|
| Algorithm | `GradientBoostingRegressor` (sklearn) |
| Target | log(forward_runtime_seconds) — inverse-transformed for output |
| Features | 50 (top 50 by RF importance from 84 total) |
| Validation | 5-fold GroupKFold |
| Tuning | Optuna TPE, 200 trials |
| **MAPE** | **9.6%** |
| MedAPE | 6.9% |
| R² | 0.9666 |
| MAE | 31,261 s (~8.7 h) |

**Why log-transform?** Runtime spans 3,193 s (53 min) to 9,158,882 s (106 days) — a 2,900× range. Log-space regression dramatically improves MAPE and convergence.

**Tuned hyperparameters:**
```python
n_estimators   = 794
max_depth      = 5
learning_rate  = 0.011740
min_samples_split = 3
min_samples_leaf  = 1
subsample      = 0.572111
```

**Top 5 features by importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `max_gate_span` | 0.5690 |
| 2 | `log_qubits` | 0.0699 |
| 3 | `n_qubits` | 0.0631 |
| 4 | `n_measure` | 0.0463 |
| 5 | `circuit_density` | 0.0367 |

`max_gate_span` alone explains 57% of runtime variance — long-range entanglement is the dominant runtime driver in MPS simulation.

---

## Dataset

| Property | Value |
|---|---|
| Source | `data/data.json` |
| Circuits | 576 unique OpenQASM 2.0 files |
| Configurations | CPU × {single, double} = 2 per circuit |
| Total samples | 1,107 (after filtering incomplete entries) |
| Threshold ladder | 1, 2, 4, 8, 16, 32, 64 |
| Fidelity target | ≥ 0.99 (mirror-circuit benchmark) |

**Class distribution (threshold):**

| Threshold | Count | % |
|---|---|---|
| 1 | 434 | 39.2% |
| 2 | 336 | 30.4% |
| 4 | 114 | 10.3% |
| 8 | 80 | 7.2% |
| 16 | 56 | 5.1% |
| 32 | 58 | 5.2% |
| 64 | 29 | 2.6% |

---

## Reproducing Results

### 1. Build the feature dataset

```bash
python build_training_dataset.py
# outputs training_data.csv
```

### 2. Train and save models

Open and run all cells in:
- `Classifier.ipynb` → saves `models/threshold_classifier.pkl`
- `Regressor.ipynb` → saves `models/runtime_model.pkl`

Training takes ~2–3 hours total (dominated by Optuna hyperparameter tuning).

### 3. Run predictions

```bash
python predict_cli.py circuits_new/my_circuit.qasm
```

### 4. Generate challenge submission

```bash
python predict_cli.py \
  --tasks data/holdout_public.json \
  --circuits circuits_new/ \
  --out my_predictions.json
```

---

## Implementation Notes

- **No quantum simulator required** — all features are extracted via regex from raw QASM text.
- **Cross-validation discipline** — GroupKFold on circuit filename prevents (backend, precision) pairs of the same circuit from leaking across folds.
- **Conservative mode** — when classifier confidence < 60%, threshold is bumped up one rung. This trades a mild over-prediction penalty (partial credit) for avoiding a zero-score under-prediction.
- **Model artifacts** store everything needed for inference: model, scaler, label encoder, feature list, and CV metrics. Inference requires only `joblib` to load.
- **Numerical safety** — all feature arrays are NaN/inf-cleaned and clipped to ±10 standard deviations before model input.

---

## Dependencies

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
joblib>=1.3.0
scipy>=1.11.0
matplotlib>=3.10.8
```

Optional (only needed to retrain):
```
lightgbm
optuna
```

---

## Results Summary

| Metric | Value |
|---|---|
| Threshold competition score | **0.9250** |
| Threshold accuracy | 90.97% |
| Runtime MAPE | **9.6%** |
| Runtime R² | 0.9666 |
| Inference time | < 100 ms per circuit |
| Training samples | 1,107 |
| Threshold model size | 9.7 MB |
| Runtime model size | 3.0 MB |

---

## License

MIT — see `LICENSE`.
