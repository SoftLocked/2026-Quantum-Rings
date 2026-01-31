"""
Model Validation Script

This script loads the trained models and shows true vs predicted values
for both threshold and runtime on the test set.

Usage:
    python validate_models.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error

print("=" * 100)
print("MODEL VALIDATION - TRUE vs PREDICTED")
print("=" * 100)
print()

# Load training data
print("Loading training data...")
df = pd.read_csv('training_data.csv')
print(f"  ✓ Loaded {len(df)} samples")
print()

# Prepare features (same as training)
X = df.iloc[:, 2:-2].copy()
X['backend'] = (X['backend'] == 'GPU').astype(int)
X['precision'] = (X['precision'] == 'double').astype(int)

y_threshold = df.iloc[:, -2]
y_runtime = df.iloc[:, -1]

# Split data (same random_state as training)
X_train, X_test, y_thresh_train, y_thresh_test = train_test_split(
    X, y_threshold, test_size=0.2, random_state=42, stratify=y_threshold
)

X_train_rt, X_test_rt, y_rt_train, y_rt_test = train_test_split(
    X, y_runtime, test_size=0.2, random_state=42
)

# Get file names for test samples
df_with_split = df.copy()
df_with_split['split_idx'] = range(len(df))
test_indices = X_test.index

test_files = df.loc[test_indices, 'file'].values
test_backends = df.loc[test_indices, 'backend'].values
test_precisions = df.loc[test_indices, 'precision'].values

print(f"Test set: {len(X_test)} samples")
print()

# Load trained models
print("Loading trained models...")
clf_threshold = joblib.load('threshold_model.pkl')
reg_runtime = joblib.load('runtime_model.pkl')
print("  ✓ Loaded threshold_model.pkl")
print("  ✓ Loaded runtime_model.pkl")
print()

# ============================================================================
# THRESHOLD PREDICTIONS
# ============================================================================

print("=" * 100)
print("THRESHOLD CLASSIFIER - TEST SET PREDICTIONS")
print("=" * 100)
print()

# Make predictions (with conservative bias)
y_thresh_pred_raw = clf_threshold.predict(X_test)
y_thresh_proba = clf_threshold.predict_proba(X_test)

# Apply conservative bias (same as predict.py)
threshold_ladder = sorted(clf_threshold.classes_)
y_thresh_pred_conservative = []

for pred, proba_dist in zip(y_thresh_pred_raw, y_thresh_proba):
    max_prob = proba_dist.max()

    if max_prob < 0.70:  # Low confidence - bump up
        try:
            pred_idx = threshold_ladder.index(pred)
            if pred_idx < len(threshold_ladder) - 1:
                y_thresh_pred_conservative.append(threshold_ladder[pred_idx + 1])
            else:
                y_thresh_pred_conservative.append(pred)
        except ValueError:
            y_thresh_pred_conservative.append(pred)
    else:
        y_thresh_pred_conservative.append(pred)

y_thresh_pred = np.array(y_thresh_pred_conservative)

# Create detailed results table
print("Detailed Predictions:")
print("-" * 100)
print(f"{'File':<40} {'Config':<15} {'True':>6} {'Raw':>6} {'Final':>6} {'Conf':>6} {'Status':<10}")
print("-" * 100)

for i in range(len(X_test)):
    file = test_files[i]
    backend = test_backends[i]
    precision = test_precisions[i]
    config = f"{backend}/{precision}"

    true_thresh = y_thresh_test.iloc[i]
    raw_pred = y_thresh_pred_raw[i]
    final_pred = y_thresh_pred[i]
    confidence = y_thresh_proba[i].max()

    # Determine status
    if final_pred < true_thresh:
        status = "❌ UNDER"
    elif final_pred > true_thresh:
        status = "⚠️  OVER"
    else:
        status = "✓ EXACT"

    bumped = " [BUMP]" if final_pred != raw_pred else ""

    print(f"{file:<40} {config:<15} {true_thresh:>6} {raw_pred:>6} {final_pred:>6} {confidence:>6.1%} {status:<10}{bumped}")

print("-" * 100)
print()

# Metrics
accuracy = accuracy_score(y_thresh_test, y_thresh_pred)
under_preds = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred) if p < t)
over_preds = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred) if p > t)
exact_preds = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred) if p == t)

print("Summary Statistics:")
print(f"  Overall Accuracy:     {accuracy:.1%}")
print(f"  Exact Predictions:    {exact_preds}/{len(y_thresh_test)} ({100*exact_preds/len(y_thresh_test):.1f}%)")
print(f"  Over-predictions:     {over_preds}/{len(y_thresh_test)} ({100*over_preds/len(y_thresh_test):.1f}%) - SAFE")
print(f"  Under-predictions:    {under_preds}/{len(y_thresh_test)} ({100*under_preds/len(y_thresh_test):.1f}%) - DANGEROUS!")
print()

if under_preds == 0:
    print("  ✅ PERFECT: Zero under-predictions! No risk of automatic zeros.")
else:
    print(f"  ⚠️  WARNING: {under_preds} under-predictions would score ZERO points!")
print()

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_thresh_test, y_thresh_pred, labels=threshold_ladder)
print(f"\n{'True \\ Pred':<12}", end="")
for thresh in threshold_ladder:
    print(f"{thresh:>6}", end="")
print()
print("-" * (12 + 6 * len(threshold_ladder)))

for i, true_thresh in enumerate(threshold_ladder):
    print(f"{true_thresh:<12}", end="")
    for j in range(len(threshold_ladder)):
        count = cm[i, j]
        if count > 0:
            if i == j:
                print(f"{count:>6}", end="")  # Diagonal - correct predictions
            elif i < j:
                print(f"⚠{count:>5}", end="")  # Over-prediction
            else:
                print(f"❌{count:>5}", end="")  # Under-prediction
        else:
            print(f"{'':>6}", end="")
    print()
print()

# ============================================================================
# RUNTIME PREDICTIONS
# ============================================================================

print("=" * 100)
print("RUNTIME REGRESSOR - TEST SET PREDICTIONS")
print("=" * 100)
print()

# Make predictions (model trained on log(runtime))
y_rt_pred_log = reg_runtime.predict(X_test_rt)
y_rt_pred = np.exp(y_rt_pred_log)

# Runtime test files
test_files_rt = df.loc[X_test_rt.index, 'file'].values
test_backends_rt = df.loc[X_test_rt.index, 'backend'].values
test_precisions_rt = df.loc[X_test_rt.index, 'precision'].values

print("Detailed Predictions:")
print("-" * 100)
print(f"{'File':<40} {'Config':<15} {'True (s)':>10} {'Pred (s)':>10} {'Error (s)':>10} {'Error %':>10}")
print("-" * 100)

errors = []
for i in range(len(X_test_rt)):
    file = test_files_rt[i]
    backend = test_backends_rt[i]
    precision = test_precisions_rt[i]
    config = f"{backend}/{precision}"

    true_rt = y_rt_test.iloc[i]
    pred_rt = y_rt_pred[i]
    error = pred_rt - true_rt
    error_pct = 100 * error / true_rt if true_rt > 0 else 0

    errors.append(abs(error_pct))

    print(f"{file:<40} {config:<15} {true_rt:>10.2f} {pred_rt:>10.2f} {error:>+10.2f} {error_pct:>+9.1f}%")

print("-" * 100)
print()

# Metrics
r2 = r2_score(y_rt_test, y_rt_pred)
rmse = np.sqrt(mean_squared_error(y_rt_test, y_rt_pred))
mae = mean_absolute_error(y_rt_test, y_rt_pred)
mape = np.mean(errors)

print("Summary Statistics:")
print(f"  R² Score:             {r2:.3f}")
print(f"  RMSE:                 {rmse:.2f} seconds")
print(f"  MAE:                  {mae:.2f} seconds")
print(f"  MAPE:                 {mape:.1f}%")
print()

# Error analysis
print("Error Distribution:")
within_10pct = sum(1 for e in errors if e <= 10)
within_25pct = sum(1 for e in errors if e <= 25)
within_50pct = sum(1 for e in errors if e <= 50)

print(f"  Within ±10%:          {within_10pct}/{len(errors)} ({100*within_10pct/len(errors):.1f}%)")
print(f"  Within ±25%:          {within_25pct}/{len(errors)} ({100*within_25pct/len(errors):.1f}%)")
print(f"  Within ±50%:          {within_50pct}/{len(errors)} ({100*within_50pct/len(errors):.1f}%)")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)
print()

print("Threshold Classifier:")
print(f"  ✓ Accuracy: {accuracy:.1%}")
print(f"  ✓ Under-predictions: {under_preds} (CRITICAL: must be zero!)")
print(f"  ✓ Conservative bias: {sum(1 for r, c in zip(y_thresh_pred_raw, y_thresh_pred) if c != r)} predictions bumped up")
print()

print("Runtime Regressor:")
print(f"  ✓ R² Score: {r2:.3f}")
print(f"  ✓ MAPE: {mape:.1f}%")
print(f"  ✓ {within_25pct}/{len(errors)} predictions within ±25%")
print()

if under_preds == 0:
    print("✅ MODELS READY FOR SUBMISSION")
    print("   Conservative bias successfully prevents catastrophic under-predictions.")
else:
    print("⚠️  WARNING: Under-predictions detected!")
    print("   Consider increasing conservative bias threshold.")
print()
