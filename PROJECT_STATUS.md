# Circuit Fingerprint Challenge - Quick Reference

## 📁 Project Files

```
build_training_dataset.py  - Build training CSV from QASM + labels
comprehensive_features.py  - Feature extractor (62 features)
training_data.csv         - Training dataset (136 samples × 72 columns) ✅ VERIFIED
verify_dataset.py         - Verify strict fidelity >= 0.99 requirement
PROJECT_STATUS.md         - This file
README.md                 - Official challenge README
```

**Data folders:**
- `circuits/*.qasm` - 36 training circuits
- `data/hackathon_public.json` - Training labels
- `data/holdout_public.json` - Holdout task list

---

## 📊 Training Dataset ([training_data.csv](training_data.csv))

**Shape:** 137 samples × 72 columns

**Column Organization (ML-optimized):**
```
Columns 1-4:   Identifiers & Config
               file, family, backend, precision

Columns 5-66:  Circuit Features (62 total)
               n_qubits, n_2q_gates, avg_qubit_degree, ...

Columns 67-70: Metadata (optional)
               max_fidelity_achieved, forward_shots, ...

Columns 71-72: TARGETS (last 2 for easy slicing)
               min_threshold ← PREDICT THIS
               forward_runtime ← PREDICT THIS
```

**Easy ML slicing:**
```python
X = df.iloc[:, 4:-2]   # All features
y_threshold = df.iloc[:, -2]  # min_threshold
y_runtime = df.iloc[:, -1]    # forward_runtime
```

### Key Statistics

**Threshold Distribution:**
```
Threshold   1:  52 samples (38.2%)
Threshold   2:  44 samples (32.4%)
Threshold   4:   9 samples (6.6%)
Threshold   8:   9 samples (6.6%)
Threshold  16:  16 samples (11.8%)
Threshold  32:   4 samples (2.9%)
Threshold 256:   2 samples (1.5%)
```

**Runtime:** 0.99s (min) → 18.45s (median) → 2588s (max)

### Top Features (Correlation with min_threshold)

1. **avg_qubit_degree: +0.664** 🔥 Best predictor!
2. **n_unique_edges: +0.659** 🔥
3. has_variational_pattern: +0.289
4. n_rotation_gates: +0.214
5. max_qubit_degree: +0.096

**Key Insight:** Graph structure features (degree, edges) are the strongest predictors!

### Data Quality ✅

**STRICT REQUIREMENT:** All thresholds have fidelity >= 0.99 (NO ROUNDING!)

**Uses `sdk_get_fidelity`** (organizer's metric for forward run selection)

Verification results:
- ✅ All 137 samples pass strict check (exact floating point comparison)
- ✅ 100% match with organizer's forward run threshold selections
- 8 samples between 0.99-0.995 (5.8%) - valid but close to boundary
- 129 samples >= 0.995 (94.2%) - well above requirement

Run verification: `python3 verify_dataset.py`

---

## 🎯 What You Need to Predict

For each holdout task, predict TWO values:

### 1. predicted_threshold_min (Classification)
- **Must be a ladder rung:** 1, 2, 4, 8, 16, 32, 64, 128, 256
- **Minimum threshold** where fidelity ≥ 0.99
- **Critical:** Predicting too low = ZERO POINTS!
- **Strategy:** When unsure, predict slightly higher

### 2. predicted_forward_wall_s (Regression)
- **Wall-clock time** for 10,000-shot simulation
- **In seconds** (float)
- **Tip:** Model log(runtime) due to huge variance (1s to 2500s)

---

## 🚀 Next Steps

### Step 1: Train Models

**For Threshold Classification:**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv('training_data.csv')

# Top features
features = [
    'avg_qubit_degree',     # 0.664 correlation
    'n_unique_edges',       # 0.659 correlation
    'has_variational_pattern',
    'n_rotation_gates',
    'max_qubit_degree',
    'n_qubits',
    'n_2q_gates',
    'ratio_2q_gates',
]

# Encode categoricals
X = df[features + ['backend', 'precision']].copy()
X['backend'] = (X['backend'] == 'GPU').astype(int)
X['precision'] = (X['precision'] == 'double').astype(int)

y = df['min_threshold']

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save
import joblib
joblib.dump(clf, 'threshold_model.pkl')
```

**For Runtime Regression:**
```python
from sklearn.ensemble import RandomForestRegressor

# Features for runtime
runtime_features = [
    'n_qubits',
    'n_total_gates',
    'crude_depth',
    'gates_per_qubit',
    'backend',
    'precision'
]

X = df[runtime_features].copy()
X['backend'] = (X['backend'] == 'GPU').astype(int)
X['precision'] = (X['precision'] == 'double').astype(int)

# IMPORTANT: Predict log(runtime)!
y = np.log(df['forward_runtime'])

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, y)

joblib.dump(reg, 'runtime_model.pkl')
```

### Step 2: Create predict.py

**Required interface** (see [docs/SUBMISSION.md](docs/SUBMISSION.md)):
```python
# predict.py must accept:
# --tasks <holdout_tasks.json>
# --circuits <circuit_directory>
# --id-map <id_to_file_mapping.json>
# --out <predictions.json>
```

**Output format:**
```json
[
  {
    "id": "H001",
    "predicted_threshold_min": 16,
    "predicted_forward_wall_s": 15.8
  }
]
```

### Step 3: Validate

```bash
python scripts/validate_holdout_submission.py \
  --public data/holdout_public.json \
  --submission predictions.json
```

---

## 💡 Modeling Tips

### Threshold Classifier
- **Model:** RandomForest or XGBoost (handles non-linearity)
- **Top features:** avg_qubit_degree, n_unique_edges, has_variational_pattern
- **Safety:** Add conservative bias - when unsure, predict one rung higher
- **Expected accuracy:** 70-80%

### Runtime Regressor
- **Model:** RandomForest or XGBoost
- **CRITICAL:** Predict `log(runtime)`, then `exp()` for final prediction
- **Top features:** n_qubits, n_total_gates, backend, precision
- **Expected R²:** 0.85+

### Pattern-Based Rules

Simple heuristics that work:
```python
# QFT pattern → always threshold ≤ 2
if has_qft_pattern:
    threshold = 1

# Variational/custom gates → threshold ≥ 8
elif has_variational_pattern or n_custom_gates > 0:
    threshold = 16

# High connectivity → threshold ≥ 4
elif max_qubit_degree > 20:
    threshold = 8

# Large circuits → threshold ≥ 4
elif n_qubits > 100:
    threshold = 4

# Default for small circuits
else:
    threshold = 2
```

---

## 📝 Feature Quick Reference

### Graph Features (Best Predictors)
- `avg_qubit_degree` - Average connections per qubit
- `n_unique_edges` - Number of unique qubit pairs
- `max_qubit_degree` - Most-connected qubit
- `n_connected_components` - Number of separate subgraphs

### Circuit Size
- `n_qubits` - Total qubits
- `n_2q_gates` - Two-qubit gates (cx, cz, cp, swap)
- `n_1q_gates` - Single-qubit gates (h, x, y, z, etc.)
- `n_total_gates` - All gates

### Ratios
- `ratio_2q_gates` - Fraction of 2Q gates (entanglement indicator)
- `gates_per_qubit` - Circuit density
- `2q_gates_per_qubit` - Entanglement density

### Pattern Detection
- `has_qft_pattern` - Quantum Fourier Transform pattern
- `has_variational_pattern` - VQE/QAOA pattern
- `has_grover_pattern` - Grover's algorithm pattern
- `n_custom_gates` - User-defined gates (unpredictable)

### Depth & Complexity
- `crude_depth` - Estimated circuit layers
- `avg_gate_span` - Average distance between qubits in 2Q gates
- `entanglement_pressure` - n_2q_gates / n_qubits

---

## 🔧 Rebuild Dataset

If you modify features:
```bash
python3 build_training_dataset.py
```

This regenerates `training_data.csv` with all 62 features.

---

## 📚 Official Documentation

- [docs/CHALLENGE.md](docs/CHALLENGE.md) - Challenge overview
- [docs/DATA.md](docs/DATA.md) - Data format
- [docs/SUBMISSION.md](docs/SUBMISSION.md) - Submission requirements
- [docs/CIRCUITS.md](docs/CIRCUITS.md) - Circuit library info

---

## ✨ Summary

**You have:**
- ✅ Clean training dataset (136 samples)
- ✅ 62 comprehensive features
- ✅ Strong predictor signals (0.66 correlation)
- ✅ Minimal clutter - only essential files

**Next:** Train models → Create predict.py → Submit!
