"""
Verify Training Dataset Quality

This script verifies that all thresholds in the training dataset
strictly meet the fidelity >= 0.99 requirement (no rounding, no approximation).

Uses sdk_get_fidelity (the metric organizers use for forward run selection).
"""

import json
import pandas as pd

def verify_dataset():
    """Verify all training data entries meet strict fidelity requirements."""

    print("="*80)
    print("DATASET VERIFICATION - STRICT FIDELITY CHECK")
    print("="*80)
    print()

    # Load training data
    df = pd.read_csv('training_data.csv')
    print(f"Training data: {len(df)} samples")
    print()

    # Load raw data
    with open('data/hackathon_public.json', 'r') as f:
        raw_data = json.load(f)

    # Build verification map
    print("Verifying each sample...")
    print("-" * 80)
    print(f"{'File':<40} {'Config':<15} {'Threshold':>9} {'Fidelity':>10} {'Status'}")
    print("-" * 80)

    all_pass = True
    warnings = []

    for idx, row in df.iterrows():
        # Find corresponding raw data
        matching_result = None
        for result in raw_data['results']:
            if (result['file'] == row['file'] and
                result['backend'] == row['backend'] and
                result['precision'] == row['precision']):
                matching_result = result
                break

        if matching_result is None:
            print(f"ERROR: Could not find raw data for {row['file']}")
            all_pass = False
            continue

        # Find fidelity at selected threshold (using sdk_get_fidelity)
        fidelity_at_threshold = None
        for entry in matching_result['threshold_sweep']:
            if entry['threshold'] == row['min_threshold']:
                fidelity_at_threshold = entry.get('sdk_get_fidelity')
                break

        if fidelity_at_threshold is None:
            status = "ERROR: No fidelity data"
            all_pass = False
        elif fidelity_at_threshold < 0.99:
            status = f"❌ FAIL (< 0.99)"
            all_pass = False
        elif fidelity_at_threshold < 0.995:
            status = f"⚠️  WARN (close to boundary)"
            warnings.append((row['file'], row['backend'], row['precision'],
                           row['min_threshold'], fidelity_at_threshold))
        else:
            status = "✓ PASS"

        config = f"{row['backend']}+{row['precision']}"
        print(f"{row['file']:<40} {config:<15} {row['min_threshold']:>9} {fidelity_at_threshold:>10.6f} {status}")

    print("-" * 80)
    print()

    # Summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print()

    if all_pass:
        print("✅ ALL SAMPLES PASS STRICT FIDELITY CHECK")
        print()
        print("Every threshold in the training data has fidelity >= 0.99")
        print("No rounding, no approximation - exact numerical comparison.")
        print()
    else:
        print("❌ VERIFICATION FAILED")
        print()
        print("Some samples do not meet the strict fidelity >= 0.99 requirement!")
        print()

    if warnings:
        print(f"⚠️  {len(warnings)} samples have fidelity close to 0.99 boundary:")
        print()
        for file, backend, precision, threshold, fidelity in warnings:
            print(f"  {file} ({backend}/{precision}): T={threshold}, F={fidelity:.6f}")
        print()
        print("These are VALID (>= 0.99) but close to the boundary.")
        print()

    # Distribution of fidelities
    print("Fidelity distribution at selected thresholds:")
    print()

    fidelities = []
    for idx, row in df.iterrows():
        for result in raw_data['results']:
            if (result['file'] == row['file'] and
                result['backend'] == row['backend'] and
                result['precision'] == row['precision']):
                for entry in result['threshold_sweep']:
                    if entry['threshold'] == row['min_threshold']:
                        f = entry.get('sdk_get_fidelity')
                        if f is not None:
                            fidelities.append(f)

    if fidelities:
        print(f"  Min:    {min(fidelities):.6f}")
        print(f"  25th:   {sorted(fidelities)[len(fidelities)//4]:.6f}")
        print(f"  Median: {sorted(fidelities)[len(fidelities)//2]:.6f}")
        print(f"  75th:   {sorted(fidelities)[3*len(fidelities)//4]:.6f}")
        print(f"  Max:    {max(fidelities):.6f}")
        print()

        # Count how many are exactly at boundary
        at_boundary = sum(1 for f in fidelities if 0.99 <= f < 0.995)
        well_above = sum(1 for f in fidelities if f >= 0.995)

        print(f"  At boundary (0.99 - 0.995): {at_boundary:3d} samples ({100*at_boundary/len(fidelities):.1f}%)")
        print(f"  Well above (>= 0.995):       {well_above:3d} samples ({100*well_above/len(fidelities):.1f}%)")
        print()

    return all_pass


if __name__ == "__main__":
    success = verify_dataset()

    if success:
        print("="*80)
        print("✅ DATASET IS CLEAN AND READY FOR TRAINING")
        print("="*80)
        exit(0)
    else:
        print("="*80)
        print("❌ DATASET HAS ISSUES - FIX BEFORE TRAINING")
        print("="*80)
        exit(1)
