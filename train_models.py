"""
Train Models for Circuit Fingerprint Challenge

This script trains:
1. Threshold classifier (predicts min_threshold)
2. Runtime regressor (predicts forward_runtime)

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib

print("="*80)
print("TRAINING MODELS FOR CIRCUIT FINGERPRINT CHALLENGE")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("Loading training data...")
df = pd.read_csv('training_data.csv')
print(f"  ✓ Loaded {len(df)} samples × {len(df.columns)} columns")
print()

# ============================================================================
# 2. PREPARE FEATURES AND TARGETS
# ============================================================================

print("Preparing features and targets...")

# Extract features (columns 4 to -2)
# Columns: backend, precision, n_qubits, n_2q_gates, ..., (62 circuit features)
X = df.iloc[:, 2:-2].copy()  # Include backend and precision

# Encode categorical features
print("  Encoding categorical features...")
X['backend'] = (X['backend'] == 'GPU').astype(int)  # GPU=1, CPU=0
X['precision'] = (X['precision'] == 'double').astype(int)  # double=1, single=0

# Extract targets
y_threshold = df.iloc[:, -2]  # min_threshold (classification)
y_runtime = df.iloc[:, -1]    # forward_runtime (regression)

print(f"  ✓ Features shape: {X.shape}")
print(f"  ✓ Threshold target: {len(y_threshold)} samples")
print(f"  ✓ Runtime target: {len(y_runtime)} samples")
print()

# Show feature names
print("Top features being used:")
for i, col in enumerate(X.columns[:10], 1):
    print(f"  {i:2d}. {col}")
print(f"  ... and {len(X.columns) - 10} more")
print()

# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================

print("Splitting data (80% train, 20% test)...")
X_train, X_test, y_thresh_train, y_thresh_test = train_test_split(
    X, y_threshold, test_size=0.2, random_state=42, stratify=y_threshold
)

X_train_rt, X_test_rt, y_rt_train, y_rt_test = train_test_split(
    X, y_runtime, test_size=0.2, random_state=42
)

print(f"  ✓ Train set: {len(X_train)} samples")
print(f"  ✓ Test set:  {len(X_test)} samples")
print()

# ============================================================================
# 4. TRAIN THRESHOLD CLASSIFIER
# ============================================================================

print("="*80)
print("TRAINING THRESHOLD CLASSIFIER")
print("="*80)
print()

print("Model: Random Forest Classifier")
print("  n_estimators: 100")
print("  random_state: 42")
print()

clf_threshold = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1
)

print("Training...")
clf_threshold.fit(X_train, y_thresh_train)
print("  ✓ Training complete")
print()

# Evaluate (with conservative bias analysis)
print("Evaluating on test set...")
y_thresh_pred_raw = clf_threshold.predict(X_test)

# CRITICAL: Analyze under/over predictions
# Predicting TOO LOW = AUTOMATIC ZERO!
errors = []
for true, pred in zip(y_thresh_test, y_thresh_pred_raw):
    if pred < true:
        errors.append(('UNDER', true, pred))
    elif pred > true:
        errors.append(('OVER', true, pred))
    else:
        errors.append(('EXACT', true, pred))

under = sum(1 for e in errors if e[0] == 'UNDER')
over = sum(1 for e in errors if e[0] == 'OVER')
exact = sum(1 for e in errors if e[0] == 'EXACT')

print(f"\n⚠️  CRITICAL: Under/Over Prediction Analysis")
print(f"  UNDER-predictions (TOO LOW → ZERO POINTS!): {under}/{len(errors)} ({100*under/len(errors):.1f}%)")
print(f"  OVER-predictions (SAFE, small penalty):     {over}/{len(errors)} ({100*over/len(errors):.1f}%)")
print(f"  EXACT predictions:                           {exact}/{len(errors)} ({100*exact/len(errors):.1f}%)")

if under > 0:
    print(f"\n  ⚠️  WARNING: {under} predictions are TOO LOW!")
    print("  These would score ZERO points in the challenge.")

# Apply CONSERVATIVE BIAS
# Strategy: When prediction confidence is low, choose next rung up
print(f"\nApplying CONSERVATIVE BIAS...")

# Get prediction probabilities
y_thresh_proba = clf_threshold.predict_proba(X_test)

# Conservative prediction: if top probability < 0.7, bump to next rung
threshold_ladder = sorted(clf_threshold.classes_)
y_thresh_pred_conservative = []

for i, (pred, proba_dist) in enumerate(zip(y_thresh_pred_raw, y_thresh_proba)):
    max_prob = proba_dist.max()

    if max_prob < 0.70:  # Low confidence - be conservative
        # Find next higher rung
        pred_idx = threshold_ladder.index(pred)
        if pred_idx < len(threshold_ladder) - 1:
            conservative_pred = threshold_ladder[pred_idx + 1]
            y_thresh_pred_conservative.append(conservative_pred)
        else:
            y_thresh_pred_conservative.append(pred)
    else:
        y_thresh_pred_conservative.append(pred)

y_thresh_pred = np.array(y_thresh_pred_conservative)

# Re-analyze with conservative predictions
errors_conservative = []
for true, pred in zip(y_thresh_test, y_thresh_pred):
    if pred < true:
        errors_conservative.append(('UNDER', true, pred))
    elif pred > true:
        errors_conservative.append(('OVER', true, pred))
    else:
        errors_conservative.append(('EXACT', true, pred))

under_cons = sum(1 for e in errors_conservative if e[0] == 'UNDER')
over_cons = sum(1 for e in errors_conservative if e[0] == 'OVER')
exact_cons = sum(1 for e in errors_conservative if e[0] == 'EXACT')

print(f"\nAfter conservative bias:")
print(f"  UNDER-predictions: {under_cons}/{len(errors_conservative)} ({100*under_cons/len(errors_conservative):.1f}%)")
print(f"  OVER-predictions:  {over_cons}/{len(errors_conservative)} ({100*over_cons/len(errors_conservative):.1f}%)")
print(f"  EXACT predictions: {exact_cons}/{len(errors_conservative)} ({100*exact_cons/len(errors_conservative):.1f}%)")

if under_cons < under:
    print(f"  ✓ Reduced under-predictions by {under - under_cons}")

accuracy = accuracy_score(y_thresh_test, y_thresh_pred)

print(f"  Accuracy: {accuracy:.1%}")
print()

print("Classification Report:")
print(classification_report(y_thresh_test, y_thresh_pred))

# Cross-validation
print("5-fold cross-validation...")
cv_scores = cross_val_score(clf_threshold, X_train, y_thresh_train, cv=5)
print(f"  CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
print()

# Feature importance
print("Top 10 most important features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf_threshold.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")
print()

# Save model
model_file = 'threshold_model.pkl'
joblib.dump(clf_threshold, model_file)
print(f"✓ Saved model to {model_file}")
print()

# ============================================================================
# 5. TRAIN RUNTIME REGRESSOR
# ============================================================================

print("="*80)
print("TRAINING RUNTIME REGRESSOR")
print("="*80)
print()

print("Model: Random Forest Regressor")
print("  Target: log(runtime) - to handle wide range (1s to 2500s)")
print("  n_estimators: 100")
print("  random_state: 42")
print()

# Log transform runtime (handle wide variance)
y_rt_train_log = np.log(y_rt_train)
y_rt_test_log = np.log(y_rt_test)

reg_runtime = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1
)

print("Training...")
reg_runtime.fit(X_train_rt, y_rt_train_log)
print("  ✓ Training complete")
print()

# Evaluate
print("Evaluating on test set...")
y_rt_pred_log = reg_runtime.predict(X_test_rt)
y_rt_pred = np.exp(y_rt_pred_log)  # Transform back

# Metrics
mse = mean_squared_error(y_rt_test, y_rt_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_rt_test, y_rt_pred)

print(f"  R² Score: {r2:.3f}")
print(f"  RMSE: {rmse:.2f} seconds")
print()

# Show some predictions
print("Sample predictions (first 5 test samples):")
print(f"{'Actual':>10} | {'Predicted':>10} | {'Error':>10}")
print("-" * 35)
for i in range(min(5, len(y_rt_test))):
    actual = y_rt_test.iloc[i]
    pred = y_rt_pred[i]
    error = pred - actual
    print(f"{actual:>10.2f} | {pred:>10.2f} | {error:>+10.2f}")
print()

# Feature importance
print("Top 10 most important features:")
feature_importance_rt = pd.DataFrame({
    'feature': X.columns,
    'importance': reg_runtime.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_rt.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")
print()

# Save model
model_file_rt = 'runtime_model.pkl'
joblib.dump(reg_runtime, model_file_rt)
print(f"✓ Saved model to {model_file_rt}")
print()

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print()

print("Models saved:")
print(f"  1. threshold_model.pkl - Threshold classifier ({accuracy:.1%} accuracy)")
print(f"  2. runtime_model.pkl   - Runtime regressor (R²={r2:.3f})")
print()

print("Next steps:")
print("  1. Create predict.py using these models")
print("  2. Test on holdout data (when provided)")
print("  3. Submit predictions + models")
print()

print("To use the models:")
print("  import joblib")
print("  clf = joblib.load('threshold_model.pkl')")
print("  reg = joblib.load('runtime_model.pkl')")
print()
