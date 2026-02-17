"""
Build Training Dataset for Circuit Fingerprint Challenge

This script:
1. Extracts features from all QASM circuits using comprehensive feature extraction
2. Loads labels (min_threshold, forward_runtime) from data/data.json
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

TARGET_FIDELITY = 0.99


def find_minimum_threshold(threshold_results, target_fidelity=TARGET_FIDELITY):
    """
    Find the minimum threshold where fidelity >= target.

    Args:
        threshold_results: Dict mapping threshold string keys to
                           {runtime_seconds, fidelity}
        target_fidelity: Minimum required fidelity (default 0.99)

    Returns:
        int: Minimum threshold value, or None if no threshold meets target
    """
    # Sort by threshold value ascending
    sorted_thresholds = sorted(threshold_results.items(), key=lambda x: int(x[0]))

    for t_str, vals in sorted_thresholds:
        if vals['fidelity'] >= target_fidelity:
            return int(t_str)

    return None


def build_training_dataset(json_path='data/data.json',
                           circuits_dir='circuits_new',
                           output_path='training_data.csv'):
    """
    Build complete training dataset.

    Args:
        json_path: Path to data.json
        circuits_dir: Directory containing QASM files
        output_path: Where to save the CSV

    Returns:
        pd.DataFrame: Complete training dataset
    """
    print("=" * 80)
    print("BUILDING TRAINING DATASET")
    print("=" * 80)
    print()

    # Load JSON data
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"  Found {len(data)} circuit entries")
    print()

    # Process each entry x backend x precision
    print("Processing entries and extracting features...")
    rows = []
    skipped = {'no_threshold': 0, 'file_not_found': 0, 'feature_error': 0}
    circuits_path = Path(circuits_dir)

    # Cache extracted features per circuit file (same circuit reused across configs)
    feature_cache = {}

    for i, entry in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(data)} circuits processed...")

        file_name = entry['file_name']
        n_qubits = entry['n_qubits']

        # Extract circuit features (once per file)
        if file_name not in feature_cache:
            qasm_file = circuits_path / file_name
            if not qasm_file.exists():
                print(f"  Warning: {qasm_file} not found, skipping")
                skipped['file_not_found'] += 1
                feature_cache[file_name] = None
                continue

            try:
                extractor = QASMFeatureExtractor(qasm_file)
                feature_cache[file_name] = extractor.extract_all()
            except Exception as e:
                print(f"  Warning: Failed to extract features from {qasm_file}: {e}")
                skipped['feature_error'] += 1
                feature_cache[file_name] = None
                continue

        circuit_features = feature_cache[file_name]
        if circuit_features is None:
            continue

        # Iterate over backends and precisions
        for backend in entry:
            if backend in ('file_name', 'n_qubits'):
                continue

            for precision, config in entry[backend].items():
                threshold_results = config.get('threshold_results', {})
                shots_results = config.get('shots_results')

                # Find minimum threshold meeting 0.99 fidelity
                min_threshold = find_minimum_threshold(threshold_results)
                if min_threshold is None:
                    skipped['no_threshold'] += 1
                    continue

                # Forward runtime from shots_results
                forward_runtime = shots_results['runtime_seconds'] if shots_results else None
                if forward_runtime is None:
                    continue

                # Threshold sweep metadata
                fidelities = [v['fidelity'] for v in threshold_results.values()]
                n_thresholds_tested = len(threshold_results)
                max_fidelity = max(fidelities) if fidelities else None

                # Runtime at the selected threshold
                threshold_runtime = threshold_results[str(min_threshold)]['runtime_seconds']

                row = {
                    # Identifiers
                    'file': file_name,

                    # Configuration
                    'backend': backend,
                    'precision': precision,
                    'n_qubits': n_qubits,

                    # Circuit features
                    **circuit_features,

                    # Metadata
                    'max_fidelity_achieved': max_fidelity,
                    'n_thresholds_tested': n_thresholds_tested,
                    'threshold_runtime': threshold_runtime,

                    # Labels
                    'min_threshold': min_threshold,
                    'forward_runtime': forward_runtime,
                }

                rows.append(row)

    print(f"  Processed all entries")
    print()

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorganize columns
    id_cols = ['file', 'backend', 'precision', 'n_qubits']
    main_targets = ['min_threshold', 'forward_runtime']
    metadata_cols = ['max_fidelity_achieved', 'n_thresholds_tested', 'threshold_runtime']
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    all_special = id_cols + main_targets + metadata_cols
    feature_cols = [c for c in df.columns if c not in all_special]

    df = df[id_cols + feature_cols + metadata_cols + main_targets]

    # Data quality checks
    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    print()

    print(f"Total samples: {len(df)}")
    print(f"Skipped: {sum(skipped.values())} configs")
    for reason, count in skipped.items():
        if count > 0:
            print(f"  - {reason}: {count}")
    print()

    print("Missing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values!")
    else:
        for col, count in missing[missing > 0].items():
            print(f"  - {col}: {count} ({100 * count / len(df):.1f}%)")
    print()

    # Statistics
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print()

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
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
    numeric_features = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['min_threshold', 'forward_runtime', 'n_qubits',
                    'max_fidelity_achieved', 'n_thresholds_tested', 'threshold_runtime']
    feat_cols = [c for c in numeric_features if c not in exclude_cols]

    if len(feat_cols) > 0:
        variances = df[feat_cols].var().sort_values(ascending=False).head(10)
        for feat, var in variances.items():
            print(f"    {feat:30s}: {var:12.2f}")
    print()

    # Correlation analysis
    print("Feature correlations with min_threshold (top 10):")
    correlations = df[feat_cols + ['min_threshold']].corr()['min_threshold'].sort_values(ascending=False)
    correlations = correlations[correlations.index != 'min_threshold'].head(10)
    for feat, corr in correlations.items():
        print(f"    {feat:30s}: {corr:>7.3f}")
    print()

    # Save to CSV
    print("=" * 80)
    print("SAVING DATASET")
    print("=" * 80)
    print()

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"  Size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    print()

    print(f"Feature manifest:")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Identifiers: {id_cols}")
    print(f"  Circuit features: {len(feat_cols)}")
    print(f"  Metadata: {metadata_cols}")
    print(f"  Labels: {main_targets}")
    print()

    print("=" * 80)
    print("READY FOR MODELING!")
    print("=" * 80)
    print()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build training dataset from QASM circuits and labels')
    parser.add_argument('--json', default='data/data.json',
                        help='Path to data.json')
    parser.add_argument('--circuits', default='circuits_new',
                        help='Directory containing QASM files')
    parser.add_argument('--output', default='training_data.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    df = build_training_dataset(
        json_path=args.json,
        circuits_dir=args.circuits,
        output_path=args.output
    )

    print(f"Training dataset created successfully: {args.output}")
    print(f"Shape: {df.shape}")
