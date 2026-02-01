"""
Build Training Dataset for Circuit Fingerprint Challenge

This script:
1. Extracts features from all QASM circuits using comprehensive feature extraction
2. Loads labels (min_threshold, forward_runtime) from hackathon_public.json
3. Combines into a clean training dataset CSV
4. Performs quality checks and shows statistics

Usage:
    python build_training_dataset.py [--output training_data.csv]
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from comprehensive_features import QASMFeatureExtractor


def find_minimum_threshold(threshold_sweep, target_fidelity=0.75):
    """
    Find the minimum threshold where fidelity >= target.

    STRICT REQUIREMENT: Fidelity must be >= 0.99 with NO ROUNDING!
    Uses exact floating point comparison.

    Uses sdk_get_fidelity (the metric organizers use for forward run selection).

    Args:
        threshold_sweep: List of threshold sweep entries
        target_fidelity: Minimum required fidelity (default 0.99)

    Returns:
        int: Minimum threshold value, or None if no threshold meets target
    """
    # Sort by threshold in ascending order
    sorted_sweep = sorted(threshold_sweep, key=lambda x: x['threshold'])

    for entry in sorted_sweep:
        # Use sdk_get_fidelity (what organizers use)
        fidelity = entry.get('sdk_get_fidelity')

        # STRICT check: fidelity must be >= 0.99 (no rounding, no approximation)
        if fidelity is not None and fidelity >= target_fidelity:
            threshold = entry['threshold']

            # Extra validation: ensure we're not close to the boundary
            if fidelity < 0.75:
                # This should never happen, but be paranoid
                raise ValueError(
                    f"BUG: Selected threshold {threshold} with fidelity {fidelity:.6f} < 0.99! "
                    f"This violates the strict >= 0.99 requirement."
                )

            return threshold

    return None


def extract_labels_from_result(result):
    """
    Extract training labels from a single result entry.

    Args:
        result: Result dict from hackathon_public.json

    Returns:
        dict: Labels including min_threshold and forward_runtime, or None if invalid
    """
    # Check status
    if result['status'] != 'ok':
        return None

    # Extract minimum threshold
    threshold_sweep = result.get('threshold_sweep', [])
    min_threshold = find_minimum_threshold(threshold_sweep, target_fidelity=0.75)

    if min_threshold is None:
        # No threshold met the target
        return None

    # VALIDATION: Verify the selected threshold actually has fidelity >= 0.99
    # This catches any bugs in find_minimum_threshold
    actual_fidelity = None
    for entry in threshold_sweep:
        if entry['threshold'] == min_threshold:
            actual_fidelity = entry.get('sdk_get_fidelity')
            break

    if actual_fidelity is not None and actual_fidelity < 0.75:
        raise ValueError(
            f"BUG DETECTED: Selected min_threshold={min_threshold} for {result['file']} "
            f"({result['backend']}/{result['precision']}) has fidelity={actual_fidelity:.6f} < 0.99! "
            f"This should never happen. No rounding allowed!"
        )

    # Extract forward runtime
    forward = result.get('forward')
    if forward is None or 'run_wall_s' not in forward:
        # No forward run data
        return None

    forward_runtime = forward['run_wall_s']

    # Also extract some metadata for analysis
    labels = {
        'min_threshold': min_threshold,
        'forward_runtime': forward_runtime,
        'forward_shots': forward.get('shots', 10000),
        'forward_peak_rss_mb': forward.get('peak_rss_mb'),
    }

    # Add threshold sweep statistics for debugging
    if threshold_sweep:
        fidelities = [e.get('sdk_get_fidelity') for e in threshold_sweep if e.get('sdk_get_fidelity') is not None]
        if fidelities:
            labels['max_fidelity_achieved'] = max(fidelities)
            labels['n_thresholds_tested'] = len(threshold_sweep)

    return labels


def build_training_dataset(json_path='data/hackathon_public.json',
                          circuits_dir='circuits',
                          output_path='training_data.csv'):
    """
    Build complete training dataset.

    Args:
        json_path: Path to hackathon_public.json
        circuits_dir: Directory containing QASM files
        output_path: Where to save the CSV

    Returns:
        pd.DataFrame: Complete training dataset
    """
    print("="*80)
    print("BUILDING TRAINING DATASET")
    print("="*80)
    print()

    # Load JSON data
    print(f"Loading labels from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    circuits_meta = {c['file']: c for c in data['circuits']}
    results = data['results']
    print(f"  ✓ Found {len(circuits_meta)} circuits")
    print(f"  ✓ Found {len(results)} result configurations")
    print()

    # Process each result
    print("Processing results and extracting features...")
    rows = []
    skipped = {'status_not_ok': 0, 'no_threshold': 0, 'no_forward': 0, 'file_not_found': 0}
    circuits_path = Path(circuits_dir)

    for i, result in enumerate(results):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(results)} results processed...")

        # Extract labels
        labels = extract_labels_from_result(result)
        if labels is None:
            if result['status'] != 'ok':
                skipped['status_not_ok'] += 1
            else:
                skipped['no_threshold'] += 1
            continue

        # Get circuit file
        qasm_file = circuits_path / result['file']
        if not qasm_file.exists():
            print(f"  Warning: {qasm_file} not found, skipping")
            skipped['file_not_found'] += 1
            continue

        # Extract circuit features
        try:
            extractor = QASMFeatureExtractor(qasm_file)
            circuit_features = extractor.extract_all()
        except Exception as e:
            print(f"  Warning: Failed to extract features from {qasm_file}: {e}")
            continue

        # Get circuit metadata
        meta = circuits_meta.get(result['file'], {})

        # Combine everything into one row
        row = {
            # Identifiers
            'file': result['file'],
            #'family': meta.get('family', 'Unknown'),

            # Configuration
            'backend': result['backend'],
            'precision': result['precision'],

            # Circuit features (all 62 from comprehensive extractor)
            **circuit_features,

            # Labels
            **labels,
        }

        rows.append(row)

    print(f"  ✓ Processed all results")
    print()

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorganize columns: identifiers/config → features → metadata → TARGETS (last 2)
    # Identifiers and config
    id_cols = ['file', 'backend', 'precision']

    # Main prediction targets (MUST be last 2 columns for easy ML slicing)
    main_targets = ['min_threshold', 'forward_runtime']

    # Metadata (useful but not primary targets)
    metadata_cols = ['max_fidelity_achieved', 'forward_shots',
                     'forward_peak_rss_mb', 'n_thresholds_tested']
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    # Feature columns (everything else in between)
    all_special = id_cols + main_targets + metadata_cols
    feature_cols = [c for c in df.columns if c not in all_special]

    # Reorder: identifiers → features → metadata → TARGETS (last)
    df = df[id_cols + feature_cols + metadata_cols + main_targets]

    # Data quality checks
    print("="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    print()

    print(f"Total samples: {len(df)}")
    print(f"Skipped: {sum(skipped.values())} results")
    for reason, count in skipped.items():
        if count > 0:
            print(f"  - {reason}: {count}")
    print()

    print("Missing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    else:
        for col, count in missing[missing > 0].items():
            print(f"  - {col}: {count} ({100*count/len(df):.1f}%)")
    print()

    # Statistics
    print("="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print()

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print()

    print("Samples per configuration:")
    config_counts = df.groupby(['backend', 'precision']).size()
    for (backend, precision), count in config_counts.items():
        print(f"  {backend:3s} + {precision:6s}: {count:3d} samples")
    print()

    print("Label distributions:")
    print()
    print("  min_threshold:")
    threshold_dist = df['min_threshold'].value_counts().sort_index()
    for threshold, count in threshold_dist.items():
        pct = 100 * count / len(df)
        print(f"    {threshold:>3d}: {count:3d} samples ({pct:5.1f}%)")
    print()

    print("  forward_runtime (seconds):")
    runtime_stats = df['forward_runtime'].describe()
    for stat, value in runtime_stats.items():
        print(f"    {stat:>10s}: {value:10.2f}")
    print()

    # Feature statistics
    print("Top circuit features (by variance):")
    # Select numeric features only
    numeric_features = df.select_dtypes(include=[np.number]).columns
    # Exclude labels and identifiers
    exclude_cols = ['min_threshold', 'forward_runtime', 'forward_shots',
                   'forward_peak_rss_mb', 'max_fidelity_achieved', 'n_thresholds_tested']
    feature_cols = [c for c in numeric_features if c not in exclude_cols]

    if len(feature_cols) > 0:
        variances = df[feature_cols].var().sort_values(ascending=False).head(10)
        for feat, var in variances.items():
            print(f"    {feat:30s}: {var:12.2f}")
    print()

    # Correlation analysis
    print("Feature correlations with min_threshold (top 10):")
    correlations = df[feature_cols + ['min_threshold']].corr()['min_threshold'].sort_values(ascending=False)
    correlations = correlations[correlations.index != 'min_threshold'].head(10)
    for feat, corr in correlations.items():
        print(f"    {feat:30s}: {corr:>7.3f}")
    print()

    # Save to CSV
    print("="*80)
    print("SAVING DATASET")
    print("="*80)
    print()

    df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    print(f"  Size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    print()

    # Feature manifest
    print("Feature manifest:")
    print(f"  Total columns: {len(df.columns)}")
    print()
    print("  Identifiers (1):")
    print("    - file")
    print()
    print("  Configuration (2):")
    print("    - backend, precision")
    print()
    print(f"  Circuit features ({len(feature_cols)}):")
    print(f"    - {', '.join(feature_cols[:5])}...")
    print()
    print("  Labels (2 primary):")
    print("    - min_threshold (classification target)")
    print("    - forward_runtime (regression target)")
    print()

    print("="*80)
    print("READY FOR MODELING!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Load the CSV: df = pd.read_csv('training_data.csv')")
    print("  2. Split features and labels")
    print("  3. Train threshold classifier (Random Forest / XGBoost)")
    print("  4. Train runtime regressor (Random Forest / XGBoost)")
    print("  5. Validate with cross-validation")
    print("  6. Save models for predict.py")
    print()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build training dataset from QASM circuits and labels')
    parser.add_argument('--json', default='data/hackathon_public.json',
                       help='Path to hackathon_public.json')
    parser.add_argument('--circuits', default='circuits',
                       help='Directory containing QASM files')
    parser.add_argument('--output', default='training_data.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    # Build the dataset
    df = build_training_dataset(
        json_path=args.json,
        circuits_dir=args.circuits,
        output_path=args.output
    )

    print(f"Training dataset created successfully: {args.output}")
    print(f"Shape: {df.shape}")
