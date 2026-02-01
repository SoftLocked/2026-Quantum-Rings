"""
Model Validation Script - Gradient Boosting & XGBoost

This script loads the trained GBM and XGBoost models and shows true vs predicted
values for both threshold and runtime on the test set.

Usage:
    python validate_models_gbm_xgb.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error

print("=" * 100)
print("MODEL VALIDATION - GRADIENT BOOSTING & XGBOOST")
print("=" * 100)
print()

# Load training data
print("Loading training data...")
df = pd.read_csv('training_data.csv')
print(f"  ✓ Loaded {len(df)} samples")
print()

# Prepare features (same as training)
# Prepare features - exclude columns not available at prediction time
exclude_cols = [
    # Identifiers and metadata already removed from CSV
    'min_threshold', 'forward_runtime'  # Targets
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
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
test_indices = X_test.index
test_files = df.loc[test_indices, 'file'].values
test_backends = df.loc[test_indices, 'backend'].values
test_precisions = df.loc[test_indices, 'precision'].values

print(f"Test set: {len(X_test)} samples")
print()

# Load trained models
print("Loading trained models...")
gbc = joblib.load('threshold_model_gbc.pkl')
xgbc = joblib.load('threshold_model_xgbc.pkl')
threshold_to_idx, idx_to_threshold = joblib.load('xgbc_encoding.pkl')
gbr = joblib.load('runtime_model_gbr.pkl')
xgbr = joblib.load('runtime_model_xgbr.pkl')
print("  ✓ Loaded threshold_model_gbc.pkl")
print("  ✓ Loaded threshold_model_xgbc.pkl")
print("  ✓ Loaded xgbc_encoding.pkl")
print("  ✓ Loaded runtime_model_gbr.pkl")
print("  ✓ Loaded runtime_model_xgbr.pkl")
print()

# ============================================================================
# GRADIENT BOOSTING CLASSIFIER - THRESHOLD PREDICTIONS
# ============================================================================

print("=" * 100)
print("GRADIENT BOOSTING CLASSIFIER - THRESHOLD PREDICTIONS")
print("=" * 100)
print()

y_thresh_pred_raw_gbc = gbc.predict(X_test)
y_thresh_proba_gbc = gbc.predict_proba(X_test)

# Apply conservative bias
threshold_ladder = sorted(gbc.classes_)
y_thresh_pred_gbc = []

for pred, proba_dist in zip(y_thresh_pred_raw_gbc, y_thresh_proba_gbc):
    max_prob = proba_dist.max()
    if max_prob < 0.70:
        pred_idx = threshold_ladder.index(pred)
        if pred_idx < len(threshold_ladder) - 1:
            y_thresh_pred_gbc.append(threshold_ladder[pred_idx + 1])
        else:
            y_thresh_pred_gbc.append(pred)
    else:
        y_thresh_pred_gbc.append(pred)

y_thresh_pred_gbc = np.array(y_thresh_pred_gbc)

# Detailed results
print("Sample Predictions:")
print("-" * 100)
print(f"{'File':<40} {'Config':<15} {'True':>6} {'Pred':>6} {'Conf':>6} {'Status':<10}")
print("-" * 100)

for i in range(min(10, len(X_test))):
    file = test_files[i]
    backend = test_backends[i]
    precision = test_precisions[i]
    config = f"{backend}/{precision}"

    true_thresh = y_thresh_test.iloc[i]
    pred = y_thresh_pred_gbc[i]
    confidence = y_thresh_proba_gbc[i].max()

    if pred < true_thresh:
        status = "❌ UNDER"
    elif pred > true_thresh:
        status = "⚠️  OVER"
    else:
        status = "✓ EXACT"

    print(f"{file:<40} {config:<15} {true_thresh:>6} {pred:>6} {confidence:>6.1%} {status:<10}")

print("-" * 100)
print()

# Metrics
accuracy_gbc = accuracy_score(y_thresh_test, y_thresh_pred_gbc)
under_preds_gbc = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred_gbc) if p < t)
over_preds_gbc = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred_gbc) if p > t)
exact_preds_gbc = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred_gbc) if p == t)

print("Summary Statistics:")
print(f"  Accuracy:             {accuracy_gbc:.1%}")
print(f"  Exact Predictions:    {exact_preds_gbc}/{len(y_thresh_test)} ({100*exact_preds_gbc/len(y_thresh_test):.1f}%)")
print(f"  Over-predictions:     {over_preds_gbc}/{len(y_thresh_test)} ({100*over_preds_gbc/len(y_thresh_test):.1f}%) - SAFE")
print(f"  Under-predictions:    {under_preds_gbc}/{len(y_thresh_test)} ({100*under_preds_gbc/len(y_thresh_test):.1f}%) - DANGEROUS!")
print()

# ============================================================================
# XGBOOST CLASSIFIER - THRESHOLD PREDICTIONS
# ============================================================================

print("=" * 100)
print("XGBOOST CLASSIFIER - THRESHOLD PREDICTIONS")
print("=" * 100)
print()

# Encode test labels for XGBoost
y_thresh_test_encoded = y_thresh_test.map(threshold_to_idx)

y_thresh_pred_raw_xgbc_encoded = xgbc.predict(X_test)
y_thresh_proba_xgbc = xgbc.predict_proba(X_test)

# Decode predictions back to threshold values
y_thresh_pred_raw_xgbc = np.array([idx_to_threshold[idx] for idx in y_thresh_pred_raw_xgbc_encoded])

# Apply conservative bias
y_thresh_pred_xgbc = []

for pred, proba_dist in zip(y_thresh_pred_raw_xgbc, y_thresh_proba_xgbc):
    max_prob = proba_dist.max()
    if max_prob < 0.70:
        try:
            pred_idx = threshold_ladder.index(pred)
            if pred_idx < len(threshold_ladder) - 1:
                y_thresh_pred_xgbc.append(threshold_ladder[pred_idx + 1])
            else:
                y_thresh_pred_xgbc.append(pred)
        except ValueError:
            y_thresh_pred_xgbc.append(pred)
    else:
        y_thresh_pred_xgbc.append(pred)

y_thresh_pred_xgbc = np.array(y_thresh_pred_xgbc)

# Metrics
accuracy_xgbc = accuracy_score(y_thresh_test, y_thresh_pred_xgbc)
under_preds_xgbc = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred_xgbc) if p < t)
over_preds_xgbc = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred_xgbc) if p > t)
exact_preds_xgbc = sum(1 for t, p in zip(y_thresh_test, y_thresh_pred_xgbc) if p == t)

print("Summary Statistics:")
print(f"  Accuracy:             {accuracy_xgbc:.1%}")
print(f"  Exact Predictions:    {exact_preds_xgbc}/{len(y_thresh_test)} ({100*exact_preds_xgbc/len(y_thresh_test):.1f}%)")
print(f"  Over-predictions:     {over_preds_xgbc}/{len(y_thresh_test)} ({100*over_preds_xgbc/len(y_thresh_test):.1f}%) - SAFE")
print(f"  Under-predictions:    {under_preds_xgbc}/{len(y_thresh_test)} ({100*under_preds_xgbc/len(y_thresh_test):.1f}%) - DANGEROUS!")
print()

# ============================================================================
# GRADIENT BOOSTING REGRESSOR - RUNTIME PREDICTIONS
# ============================================================================

print("=" * 100)
print("GRADIENT BOOSTING REGRESSOR - RUNTIME PREDICTIONS")
print("=" * 100)
print()

gbr_preds_log = gbr.predict(X_test_rt)
gbr_preds = np.expm1(gbr_preds_log)

# Runtime test files
test_files_rt = df.loc[X_test_rt.index, 'file'].values
test_backends_rt = df.loc[X_test_rt.index, 'backend'].values
test_precisions_rt = df.loc[X_test_rt.index, 'precision'].values

print("Sample Predictions:")
print("-" * 100)
print(f"{'File':<40} {'Config':<15} {'True (s)':>10} {'Pred (s)':>10} {'Error (s)':>10} {'Error %':>10}")
print("-" * 100)

for i in range(min(10, len(X_test_rt))):
    file = test_files_rt[i]
    backend = test_backends_rt[i]
    precision = test_precisions_rt[i]
    config = f"{backend}/{precision}"

    true_rt = y_rt_test.iloc[i]
    pred_rt = gbr_preds[i]
    error = pred_rt - true_rt
    error_pct = 100 * error / true_rt if true_rt > 0 else 0

    print(f"{file:<40} {config:<15} {true_rt:>10.2f} {pred_rt:>10.2f} {error:>+10.2f} {error_pct:>+9.1f}%")

print("-" * 100)
print()

# Metrics
r2_gbr = r2_score(y_rt_test, gbr_preds)
rmse_gbr = np.sqrt(mean_squared_error(y_rt_test, gbr_preds))
mae_gbr = mean_absolute_error(y_rt_test, gbr_preds)

errors_pct_gbr = [abs(100 * (p - t) / t) for t, p in zip(y_rt_test, gbr_preds) if t > 0]
mape_gbr = np.mean(errors_pct_gbr)

within_10_gbr = sum(1 for e in errors_pct_gbr if e <= 10)
within_25_gbr = sum(1 for e in errors_pct_gbr if e <= 25)

print("Summary Statistics:")
print(f"  R² Score:             {r2_gbr:.3f}")
print(f"  RMSE:                 {rmse_gbr:.2f} seconds")
print(f"  MAE:                  {mae_gbr:.2f} seconds")
print(f"  MAPE:                 {mape_gbr:.1f}%")
print(f"  Within ±10%:          {within_10_gbr}/{len(errors_pct_gbr)} ({100*within_10_gbr/len(errors_pct_gbr):.1f}%)")
print(f"  Within ±25%:          {within_25_gbr}/{len(errors_pct_gbr)} ({100*within_25_gbr/len(errors_pct_gbr):.1f}%)")
print()

# ============================================================================
# XGBOOST REGRESSOR - RUNTIME PREDICTIONS
# ============================================================================

print("=" * 100)
print("XGBOOST REGRESSOR - RUNTIME PREDICTIONS")
print("=" * 100)
print()

xgbr_preds_log = xgbr.predict(X_test_rt)
xgbr_preds = np.expm1(xgbr_preds_log)

# Metrics
r2_xgbr = r2_score(y_rt_test, xgbr_preds)
rmse_xgbr = np.sqrt(mean_squared_error(y_rt_test, xgbr_preds))
mae_xgbr = mean_absolute_error(y_rt_test, xgbr_preds)

errors_pct_xgbr = [abs(100 * (p - t) / t) for t, p in zip(y_rt_test, xgbr_preds) if t > 0]
mape_xgbr = np.mean(errors_pct_xgbr)

within_10_xgbr = sum(1 for e in errors_pct_xgbr if e <= 10)
within_25_xgbr = sum(1 for e in errors_pct_xgbr if e <= 25)

print("Sample Predictions:")
print("-" * 100)
print(f"{'File':<40} {'Config':<15} {'True (s)':>10} {'Pred (s)':>10} {'Error (s)':>10} {'Error %':>10}")
print("-" * 100)

for i in range(min(10, len(X_test_rt))):
    file = test_files_rt[i]
    backend = test_backends_rt[i]
    precision = test_precisions_rt[i]
    config = f"{backend}/{precision}"

    true_rt = y_rt_test.iloc[i]
    pred_rt = xgbr_preds[i]
    error = pred_rt - true_rt
    error_pct = 100 * error / true_rt if true_rt > 0 else 0

    print(f"{file:<40} {config:<15} {true_rt:>10.2f} {pred_rt:>10.2f} {error:>+10.2f} {error_pct:>+9.1f}%")

print("-" * 100)
print()

print("Summary Statistics:")
print(f"  R² Score:             {r2_xgbr:.3f}")
print(f"  RMSE:                 {rmse_xgbr:.2f} seconds")
print(f"  MAE:                  {mae_xgbr:.2f} seconds")
print(f"  MAPE:                 {mape_xgbr:.1f}%")
print(f"  Within ±10%:          {within_10_xgbr}/{len(errors_pct_xgbr)} ({100*within_10_xgbr/len(errors_pct_xgbr):.1f}%)")
print(f"  Within ±25%:          {within_25_xgbr}/{len(errors_pct_xgbr)} ({100*within_25_xgbr/len(errors_pct_xgbr):.1f}%)")
print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("=" * 100)
print("VALIDATION SUMMARY - COMPARISON")
print("=" * 100)
print()

print("THRESHOLD CLASSIFIERS:")
print("-" * 100)
print(f"{'Model':<30} {'Accuracy':<15} {'Under-Preds':<15} {'Status'}")
print("-" * 100)
print(f"{'Gradient Boosting':<30} {accuracy_gbc:.1%}{'':<7} {under_preds_gbc}/28{'':<10} {'✅ SAFE' if under_preds_gbc == 0 else '❌ DANGEROUS'}")
print(f"{'XGBoost':<30} {accuracy_xgbc:.1%}{'':<7} {under_preds_xgbc}/28{'':<10} {'✅ SAFE' if under_preds_xgbc == 0 else '❌ DANGEROUS'}")
print()

print("RUNTIME REGRESSORS:")
print("-" * 100)
print(f"{'Model':<30} {'R²':<10} {'MAPE':<10} {'Within ±25%':<15}")
print("-" * 100)
print(f"{'Gradient Boosting':<30} {r2_gbr:.3f}{'':<6} {mape_gbr:.1f}%{'':<6} {within_25_gbr}/28 ({100*within_25_gbr/len(errors_pct_gbr):.1f}%)")
print(f"{'XGBoost':<30} {r2_xgbr:.3f}{'':<6} {mape_xgbr:.1f}%{'':<6} {within_25_xgbr}/28 ({100*within_25_xgbr/len(errors_pct_xgbr):.1f}%)")
print(f"{'Random Forest (baseline)':<30} {'0.895':<10} {'38.6%':<10} {'15/28 (53.6%)':<15}")
print()

# Determine best models
best_thresh_acc = max(accuracy_gbc, accuracy_xgbc)
best_runtime_mape = min(mape_gbr, mape_xgbr)

print("RECOMMENDATIONS:")
print()

if under_preds_gbc == 0 and under_preds_xgbc == 0:
    print("✅ THRESHOLD SAFETY: Both models have zero under-predictions")
    if accuracy_gbc > accuracy_xgbc:
        print(f"   Best: Gradient Boosting ({accuracy_gbc:.1%} accuracy)")
    else:
        print(f"   Best: XGBoost ({accuracy_xgbc:.1%} accuracy)")
else:
    print("⚠️  WARNING: Some models have under-predictions!")

print()

if best_runtime_mape < 38.6:
    print(f"✅ RUNTIME: Better than baseline (best MAPE: {best_runtime_mape:.1f}% vs 38.6%)")
    if mape_gbr < mape_xgbr:
        print(f"   Best: Gradient Boosting ({mape_gbr:.1f}% MAPE)")
    else:
        print(f"   Best: XGBoost ({mape_xgbr:.1f}% MAPE)")
else:
    print(f"⚠️  RUNTIME: Baseline still better (38.6% vs best: {best_runtime_mape:.1f}%)")

print()
